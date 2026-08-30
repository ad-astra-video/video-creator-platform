"""ID-V2V worker HTTP service (aiohttp).

A swappable worker container driven by the `live-runner` edge on the internal
Docker network. It does NOT register with the Livepeer Orchestrator or do
heartbeats — that is the live-runner's job. It exposes the control + inference
surface the live-runner needs:

    GET  /health          — liveness + model-loaded status
    POST /load            — build the model (int8 DiT+VACE, CPU offload)
    POST /evict           — drop the model, free GPU/CPU memory
    POST /v1/restyle      — accept a restylization job (base64 in -> base64 out)

Auth: every POST requires the shared `X-Worker-Token` header (WORKER_TOKEN env,
auto-generated if blank), which the live-runner attaches on every call.

Ported/adapted from the standalone id-v2v runner (`runner.py`) plus the control
surface /load and /evict.
"""

import asyncio
import math
import logging
import os
import sys
import time
import uuid

from aiohttp import web

from . import config
from . import run as run_mod
from . import gemma_forward
from . import bernini_io
from . import bernini as bernini_mod
from .model import ModelManager, health_check

logger = logging.getLogger("video_creator.runner.idv2v.server")


def _resolve_token() -> str:
    """Resolve the worker auth token, auto-generating a stable one if blank.

    When blank at first call, generates a token and persists it back to the env
    so every process in the container agrees on it.
    """
    if config.WORKER_TOKEN:
        return config.WORKER_TOKEN
    tok = config._random_token()
    os.environ["WORKER_TOKEN"] = tok
    config.WORKER_TOKEN = tok
    logger.info("WORKER_TOKEN was blank — auto-generated (won't be shown again)")
    return tok


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _require_token(request: web.Request) -> None:
    """Reject the request unless it carries the shared worker token."""
    expected = _resolve_token()
    provided = request.headers.get("X-Worker-Token", "")
    if not provided or provided != expected:
        raise web.HTTPForbidden(reason="missing/mismatched X-Worker-Token")


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------

# One ModelManager instance owned by this worker process.
_model: ModelManager | None = None
_model_lock = asyncio.Lock()
# Which model is resident: None | "idv2v" | "bernini-1.3b" | "bernini-14b".
# The worker can serve either family (diffsynth id-v2v pipe, or the Bernini
# subprocess), and /load places whichever the scheduler asks for on the
# assigned GPU — one resident video model per card at a time.
_resident_kind: str | None = None


def _get_model() -> ModelManager:
    global _model
    if _model is None:
        _model = ModelManager(device=config.GPU_DEVICE)
    return _model


def _device_from_body(device) -> int | None:
    """Return a validated CUDA index from a /load body ``device``, or None if
    missing/invalid. The device is authoritative — it comes from the live-runner
    scheduler's assignment, never inferred or defaulted here."""
    if device in (None, ""):
        return None
    try:
        idx = int(device)
    except (TypeError, ValueError):
        return None
    return idx if idx >= 0 else None


async def handle_load(request: web.Request) -> web.Response:
    _require_token(request)
    global _model, _resident_kind
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    device_idx = _device_from_body(body.get("device"))
    if device_idx is None:
        return web.json_response(
            {"error": "/load requires a valid 'device' (GPU index) from the "
                      "live-runner scheduler"},
            status=400,
        )
    device_str = f"cuda:{device_idx}"
    # The scheduler's device is AUTHORITATIVE for this worker — keep
    # config.GPU_DEVICE in sync so every downstream consumer (the SAM3
    # subprocess in run.py, gemma/bernini device selection) targets the
    # assigned card rather than the stale startup/env value (which pinned this
    # worker to GPU 0 and OOM'd SAM3 against the resident image-worker).
    config.GPU_DEVICE = device_str
    model_arg = body.get("model") or ""
    kind = config.resolve_model(model_arg)  # idv2v | bernini-1.3b | bernini-14b
    async with _model_lock:
        if kind in config.BERNINI_MODELS:
            # Load the Bernini subprocess resident on the assigned GPU. Free the
            # id-v2v diffsynth pipe first (one resident video model per card),
            # unless a restyle is actively generating on it.
            if _model is not None and not run_mod.generation_active():
                _model.evict()
            _model = None
            try:
                mgr = await bernini_mod.get_manager(model=kind, device=device_str)
                await asyncio.wait_for(mgr.ensure_loaded(), timeout=900)
            except bernini_mod.BerniniError as exc:
                return web.json_response({"error": str(exc)}, status=400)
            except asyncio.TimeoutError:
                return web.json_response({"error": "bernini load timed out"}, status=504)
            _resident_kind = kind
            return web.json_response(
                {"loaded": True, "model": kind, "device": mgr.device})

        # idv2v (default) — free a resident Bernini subprocess first.
        await bernini_mod.evict_manager()
        model = _get_model()
        if model.device != device_str:
            logger.info("idv2v-worker /load: relocating GPU %s -> %s",
                        model.device, device_str)
            model.set_device(str(device_idx))
        if model_arg and config.resolve_model(model_arg) == "idv2v":
            variant = config._norm_variant(model_arg)
            if model.variant != variant:
                model.set_variant(variant)
        if model.is_ready:
            _resident_kind = "idv2v"
            return web.json_response(
                {"loaded": True, "already_loaded": True,
                 "device": model.device, "model": "idv2v"})
        try:
            await asyncio.wait_for(model.load(), timeout=3600)
        except asyncio.TimeoutError:
            return web.json_response({"error": "model load timed out"}, status=504)
        _resident_kind = "idv2v"
        return web.json_response(
            {"loaded": True, "device": model.device, "model": "idv2v"})


def _resident_status() -> dict:
    """Health/info payload for whichever model family is resident."""
    global _resident_kind
    if _resident_kind and _resident_kind in config.BERNINI_MODELS:
        b = bernini_mod.resident_status()
        ready = bool(b and b.get("ready"))
        device = (b or {}).get("device") or config.GPU_DEVICE
        return {
            "status": "loaded" if ready else "loading",
            "model_loaded": ready,
            "device": device,
            "resident_kind": _resident_kind,
            "model": _resident_kind,
            "precision": "bf16",
            "offload": False,
        }
    model = _model
    base = health_check(model) if model is not None else {
        "status": "unloaded", "model_loaded": False, "device": config.GPU_DEVICE,
        "precision": config.IDV2V_QUANT, "offload": config.IDV2V_OFFLOAD,
    }
    base["resident_kind"] = _resident_kind
    base["model"] = "idv2v" if _resident_kind == "idv2v" else None
    return base


async def handle_evict(request: web.Request) -> web.Response:
    _require_token(request)
    global _model, _resident_kind
    async with _model_lock:
        if _model is not None:
            _model.evict()
        _model = None
        await bernini_mod.evict_manager()
        _resident_kind = None
    return web.json_response({"evicted": True})


async def handle_health(request: web.Request) -> web.Response:
    return web.json_response({"status": "ok", **_resident_status()})


def _device_index() -> int:
    """CUDA index this worker currently targets (for the scheduler's reconcile)."""
    global _resident_kind
    if _resident_kind and _resident_kind in config.BERNINI_MODELS:
        b = bernini_mod.resident_status()
        if b and b.get("device"):
            ds = str(b["device"])
            if ":" in ds:
                try:
                    return int(ds.split(":", 1)[1])
                except ValueError:
                    pass
        return 0
    if _model is not None:
        cur = str(_model.device)
        if ":" in cur:
            try:
                return int(cur.split(":")[1])
            except ValueError:
                pass
    ds = str(config.GPU_DEVICE)
    if ":" in ds:
        try:
            return int(ds.split(":", 1)[1])
        except ValueError:
            pass
    return int(ds) if ds.isdigit() else 0


async def _resolve_startup_device() -> None:
    """Pick this worker's GPU at startup, asking the live-runner scheduler.

    The live-runner (which owns the authoritative, /info-reconciled GPU map) knows
    which physical card is actually free, so on a live-runner restart this worker
    lands on a genuinely free GPU instead of blindly defaulting to GPU 0 (which
    the image worker may hold warm). An EXPLICIT GPU_DEVICE env is honored as a
    pin; when unset we consult ``LIVE_RUNNER_URL`` gpu-pick, falling back to the
    existing GPU_DEVICE / local behavior if the live-runner is unreachable. The
    resolved device is written back into ``config.GPU_DEVICE`` so every downstream
    consumer (ModelManager, /info, gemma_device) agrees on one card.
    """
    raw = os.environ.get("GPU_DEVICE", "").strip()
    if raw:
        return  # explicit pin -> honor it as-is
    base = config.LIVE_RUNNER_URL
    if not base:
        return  # no live-runner configured -> keep local default
    try:
        import aiohttp as _aio
        async with _aio.ClientSession() as _s:
            async with _s.post(
                f"{base.rstrip('/')}/video-creator/v1/gpu-pick",
                json={"worker": "idv2v-worker"},
                headers={"X-Worker-Token": config.worker_token()},
                timeout=_aio.ClientTimeout(total=8),
            ) as r:
                if r.status == 200:
                    gpu = (await r.json()).get("gpu")
                    if isinstance(gpu, int) and gpu >= 0:
                        config.GPU_DEVICE = f"cuda:{gpu}"
                        logger.info("idv2v-worker: live-runner assigned GPU %d", gpu)
                        return
        logger.warning("gpu-pick did not return a GPU; keeping GPU_DEVICE=%s",
                       config.GPU_DEVICE)
    except Exception as exc:
        logger.warning("gpu-pick failed (%s); keeping GPU_DEVICE=%s",
                       exc, config.GPU_DEVICE)


def _gpu_info() -> dict:
    """Report the GPU this worker renders on (torch primary, nvidia-smi fallback).

    Shape matches the ltx-worker's ``gpu`` block so the live-runner can merge all
    workers' GPU details uniformly into its advertised heartbeat metadata.
    """
    try:
        import torch
        idx = 0
        ds = str(config.GPU_DEVICE)
        if ":" in ds:
            try:
                idx = int(ds.split(":", 1)[1])
            except ValueError:
                idx = 0
        if torch.cuda.is_available() and idx < torch.cuda.device_count():
            props = torch.cuda.get_device_properties(idx)
            return {
                "gpu_name": torch.cuda.get_device_name(idx),
                "vram_gb": round(props.total_memory / (1024 ** 3), 1),
                "compute_cap": f"{props.major}.{props.minor}",
            }
    except Exception:
        logger.debug("torch GPU query failed; trying nvidia-smi", exc_info=True)
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        name, mem_mb, cc = (x.strip() for x in out.strip().splitlines()[0].split(","))
        return {"gpu_name": name, "vram_gb": round(int(mem_mb) / 1024.0, 1), "compute_cap": cc}
    except Exception:
        logger.debug("nvidia-smi GPU query failed", exc_info=True)
    return {"gpu_name": None, "vram_gb": None, "compute_cap": None}


async def handle_info(request: web.Request) -> web.Response:
    """GET /info — liveness + the GPU this worker renders on (for the live-runner)."""
    base = _resident_status()
    return web.json_response({
        "runner_id": "",
        "app": "idv2v",
        "capabilities": ["restyle", "sam3", "prompt-enhance",
                          "bernini-t2v", "bernini-v2v", "bernini-r2v"],
        "ready": base.get("model_loaded", False),
        "gpu": _gpu_info(),
        "device_in_use": _device_index(),
        **base,
    })


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

async def _run_gemma_stage(source_b64: str, prompt: str,
                           enhance_prompt: bool, seed) -> tuple:
    """Run the Gemma prompt stage (enhance + auto video caption) for a restyle.

    Prefers forwarding to the SHARED gemma-worker (config.GEMMA_FORWARD_URL, the
    llama.cpp Gemma 4 worker on the compose network): when it is configured and
    reachable, enhance/caption are sent there and the worker NEVER loads its own
    embedded Gemma 3 (no redundant ~24.5 GB model, no shared-GPU eviction dance).
    Falls back to the embedded Gemma 3 when forwarding is unavailable but the
    checkpoint is provisioned, and finally degrades to the unmodified prompt.
    Never raises: any failure is non-fatal so the restyle pipeline still runs.

    Returns ``(final_prompt, meta)`` where meta carries ``caption``,
    ``enhanced_prompt``, ``enhanced`` for the UI.
    """
    base = config.gemma_forward_base()
    if base:
        try:
            if await gemma_forward.gemma_worker_available(base, config.worker_token()):
                logger.info("Using shared gemma-worker (%s) for restyle enhance/caption", base)
                return await _gemma_stage_forward(base, source_b64, prompt,
                                                  enhance_prompt, seed)
            logger.warning("gemma-worker %s unreachable — falling back to embedded Gemma 3", base)
        except Exception as exc:
            logger.warning("gemma-worker forward failed — falling back to embedded Gemma 3: %s", exc)

    if config.gemma_enabled():
        from .gemma import get_enhancer

        def _gemma_wrapper():
            enhancer = get_enhancer()
            try:
                return run_mod._gemma_stage(
                    enhancer, source_b64, prompt, enhance_prompt, seed)
            finally:
                # Free VRAM so the video model can load on the shared GPU.
                enhancer.unload()

        return await asyncio.to_thread(_gemma_wrapper)

    return (prompt, {"caption": None, "enhanced_prompt": None, "enhanced": False})


async def _gemma_stage_forward(base: str, source_b64: str, prompt: str,
                               enhance_prompt: bool, seed) -> tuple:
    """Mirror of ``run._gemma_stage`` but via the shared gemma-worker.

    Uses the id-v2v worker's own DEDICATED system prompts so the restyle
    semantics are preserved (the color-fidelity RESTYLE_ENHANCE prompt, the
    CAPTION prompt) while the actual inference runs on the gemma-worker.
    """
    from .gemma import CAPTION_SYSTEM_PROMPT, RESTYLE_ENHANCE_SYSTEM_PROMPT

    want_caption = (not prompt.strip()
                    or prompt.strip().lower() == "restyle this video")
    final = prompt
    meta = {"caption": None, "enhanced_prompt": None, "enhanced": False}
    token = config.worker_token()
    if want_caption:
        frames = await asyncio.to_thread(run_mod._sample_source_frames, source_b64)
        if frames:
            context_frames = await asyncio.to_thread(_encode_frames_to_b64_jpeg, frames)
            logger.info("gemma-worker captioning %d sampled frame(s)", len(frames))
            caption_seed = None if seed is None else (seed + 1)
            captioned = await gemma_forward.forward_prompt_enhance(
                base, prompt="Describe this video clip.",
                system_prompt=CAPTION_SYSTEM_PROMPT,
                context_frames=context_frames, seed=caption_seed, token=token)
            if captioned:
                final = captioned
                meta["caption"] = captioned
                logger.info("gemma-worker auto-caption result: %r", final)
            else:
                logger.info("gemma-worker caption empty — keeping original prompt")
    if enhance_prompt and final.strip():
        enhanced = await gemma_forward.forward_prompt_enhance(
            base, prompt=final, system_prompt=RESTYLE_ENHANCE_SYSTEM_PROMPT,
            seed=seed, token=token)
        if enhanced:
            meta["enhanced_prompt"] = enhanced
            meta["enhanced"] = True
            final = enhanced
            logger.info("gemma-worker enhanced prompt: %r", final)
    return (final, meta)


def _encode_frames_to_b64_jpeg(frames, max_side: int = 896) -> list:
    """Encode PIL RGB frames to base64 JPEG strings (aspect-preserved).

    Matches the gemma-worker's expectation that ``context_frames`` are base64
    JPEG image strings (one per sampled video frame), and the id-v2v SigLIP
    caption convention of ~896 max-side.
    """
    import base64 as _b64
    import io as _io

    from PIL import Image

    out = []
    for frame in frames:
        img = frame.convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize(
                (max(2, int(round(w * scale))), max(2, int(round(h * scale)))),
                Image.BICUBIC,
            )
        buf = _io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        out.append(_b64.b64encode(buf.getvalue()).decode("ascii"))
    return out


async def handle_restyle(request: web.Request) -> web.Response:
    _require_token(request)
    global _resident_kind
    body = await request.json()
    job_id = body.get("job_id") or uuid.uuid4().hex[:12]

    # Gemma LLM stage (enhance + auto video caption). Prefers forwarding to the
    # shared gemma-worker (config.GEMMA_FORWARD_URL, the llama.cpp Gemma 4 worker
    # on the compose network) — the worker then does NOT load its own embedded
    # Gemma 3 (no redundant ~24.5 GB model, no shared-GPU eviction). It falls back
    # to the embedded Gemma 3 when forwarding is unavailable, and is non-fatal
    # either way (the original prompt survives). The enhanced prompt + LLM metadata
    # ride into process_job via body["prompt"] / body["_gemma_meta"].
    _prompt = str(body.get("prompt") or "").strip()
    _enhance = bool(body.get("enhance_prompt", False))
    _want_caption = (not _prompt or _prompt.lower() == "restyle this video")
    if _want_caption or _enhance:
        try:
            final, gemma_meta = await _run_gemma_stage(
                body.get("source_video", ""), _prompt, _enhance, body.get("seed"),
            )
            body["prompt"] = final
            body["_gemma_meta"] = gemma_meta
            run_mod.set_progress(job_id, 0.04, "preprocessing", "decoding source + conditioning")
        except Exception as exc:
            # Non-fatal: fall back to the original prompt; the pipeline still runs.
            logger.error("Gemma stage failed (falling back to original prompt): %s", exc, exc_info=True)

    # A restyle request can select which id-v2v model to run on via body["model"]
    # ("fast" = FusionX/`fusionx` subfolder, ~8 steps; "regular" = repo root, 30
    # steps). If that differs from the currently-loaded warm model, swap it (the
    # worker keeps ONE warm model; switching evicts + reloads under the lock, so
    # the cost only hits when the variant actually changes between batches).
    requested = config._norm_variant(body.get("model"))
    # Shared GPU: drop a resident Gemma (e.g. left warm by a prior
    # /prompt-enhance) before loading the video model so they never coexist.
    try:
        from .gemma import get_enhancer
        get_enhancer().unload()
    except Exception:
        pass
    model = _get_model()
    async with _model_lock:
        # A restyle needs the id-v2v pipe (never bernini) — free a resident
        # Bernini subprocess first so they don't share the GPU.
        await bernini_mod.evict_manager()
        if not model.is_ready:
            model.set_variant(requested)
            try:
                await asyncio.wait_for(model.load(), timeout=3600)
            except asyncio.TimeoutError:
                return web.json_response({"error": "model load timed out", "job_id": job_id}, status=504)
        elif model.variant != requested:
            logger.info("Switching idv2v model variant %s -> %s", model.variant, requested)
            model.evict()
            model.set_variant(requested)
            try:
                await asyncio.wait_for(model.load(), timeout=3600)
            except asyncio.TimeoutError:
                return web.json_response({"error": "model load timed out", "job_id": job_id}, status=504)
        _resident_kind = "idv2v"
    try:
        result = await run_mod.process_job(model, body, job_id=job_id)
        run_mod.set_progress(job_id, 1.0, "complete", "restyle done")
        result["job_id"] = job_id
        result["model"] = model.variant
    except Exception as exc:
        run_mod.set_progress(job_id, -1, "failed", str(exc))
        logger.error("Restyle job failed: %s", exc, exc_info=True)
        return web.json_response({"error": str(exc), "job_id": job_id}, status=500)
    return web.json_response(result)


async def handle_progress(request: web.Request) -> web.Response:
    _require_token(request)
    job_id = request.match_info.get("job_id", "")
    info = run_mod.get_progress(job_id)
    if info is None:
        return web.json_response({"job_id": job_id, "found": False,
                                  "progress": None, "stage": "unknown"})
    return web.json_response({"job_id": job_id, "found": True, **info})


# ---------------------------------------------------------------------------
# Bernini (wan-worker t2v / v2v / r2v)
# ---------------------------------------------------------------------------

_BERNINI_TASKS = {"t2v", "v2v", "r2v"}


def _berni_task_from_path(path: str) -> str:
    """Map a request path to a Bernini task id (t2v/v2v/r2v) or ''."""
    seg = path.rstrip("/").rsplit("/", 1)[-1]
    if seg in _BERNINI_TASKS:
        return seg
    if seg.startswith("bernini-"):
        maybe = seg[len("bernini-"):]
        return maybe if maybe in _BERNINI_TASKS else ""
    return ""


async def handle_bernini(request: web.Request) -> web.Response:
    """Run a Bernini generation/edit job (native 480p/16 @ max 848px).

    Dispatched by the ``{task}`` path segment (t2v / v2v / r2v). Body JSON:

        t2v : {prompt, num_frames?, fps?, num_inference_steps?, seed?, height?, width?}
        v2v : {prompt, video: <base64> (or list), ...}
        r2v : {prompt, images: [<base64>...], ...}

    The frontend decides the engine (berini vs ltx vs idv2v); this backend
    is strictly route-based — the task name IS the intent. The worker renders
    at native resolution; delivery above native goes through the vp-worker
    post rails orchestrated by the live-runner.
    """
    _require_token(request)
    global _model, _resident_kind
    # Routes are STATIC paths (aiohttp match_info has no {task} placeholder),
    # so derive the task from the request path. Handles every alias:
    #   /v1/t2v            -> t2v
    #   /video-creator/v1/t2v -> t2v
    #   /video-creator/v1/bernini-t2v -> t2v  (live-runner ROUTES id)
    task = _berni_task_from_path(request.path)
    if task not in _BERNINI_TASKS:
        return web.json_response({"error": f"unsupported Bernini task '{task}'"},
                                 status=404)
    if not config.bernini_enabled():
        return web.json_response(
            {"error": "Bernini weights not provisioned (BERNINI_ROOT missing)"},
            status=503)

    job_id = request.query.get("job_id") or uuid.uuid4().hex[:12]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)

    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return web.json_response({"error": "missing 'prompt'"}, status=400)

    # Resident/reuse the Bernini manager on the scheduler-assigned GPU. The
    # live-runner's proxy forwards the assigned card as X-Worker-Device (see
    # _post_worker); passing it here lets get_manager reuse a resident manager
    # on that card OR create it on the spot — so the generation request itself
    # can resident Bernini even if no /load landed (e.g. after a worker-container
    # recreate the live-runner's residency tracking can be stale and never
    # re-issues /load). Model comes from the body (resp. 14b).
    bernini_model = config.resolve_model(body.get("model"))
    if bernini_model not in config.BERNINI_MODELS:
        bernini_model = "bernini-1.3b"
    _dev_header = request.headers.get("X-Worker-Device", "")
    manager = await bernini_mod.get_manager(
        model=bernini_model,
        device=_dev_header.strip() if _dev_header.strip() else None,
    )
    # One resident video model per card: if the id-v2v diffsynth pipe is
    # resident, drop it before the Bernini subprocess allocates the shared GPU.
    if _model is not None and not run_mod.generation_active():
        _model.evict()
        _model = None
    _resident_kind = manager.model
    run_mod.set_progress(job_id, 0.02, "bernini", "decoding media")
    with bernini_io.tmpdir_context() as tmpdir:
        media = bernini_io.decode_source_media(body, tmpdir)
        if task == "v2v" and not media.get("video"):
            return web.json_response(
                {"error": "v2v requires 'video' (base64)"}, status=400)
        if task == "r2v" and not media.get("images"):
            return web.json_response(
                {"error": "r2v requires 'images' (base64 list)"}, status=400)

        out_path = os.path.join(tmpdir, "output.mp4")
        job: dict = {
            "prompt": prompt,
            "output": out_path,
            "task_name": task,
            **media,
        }
        for k in ("num_frames", "max_image_size", "height", "width",
                  "num_inference_steps", "fps", "seed", "guidance_mode",
                  "omega_vid", "omega_img", "omega_txt", "omega_scale",
                  "flow_shift", "eta", "momentum", "system_prompt", "turbo"):
            if body.get(k) is not None:
                job[k] = body[k]

        run_mod.set_progress(job_id, 0.05, "bernini", "generating")
        # Chunk-aware outer timeout: long v2v sources are split by the manager
        # into ~NATIVE_FRAMES chunks, each rendered natively; the outer wait
        # must cover ALL chunk renders, so scale it with the estimated chunk
        # count (900s single-shot baseline + ~420s headroom per extra chunk).
        _wanted = body.get("num_frames")
        try:
            _nf = int(_wanted) if _wanted else config.BERNINI_NATIVE_FPS * 5
        except (TypeError, ValueError):
            _nf = config.BERNINI_NATIVE_FPS * 5
        _extra_chunks = max(0, math.ceil(_nf / bernini_mod.NATIVE_FRAMES) - 1)
        _gen_timeout = 900 + _extra_chunks * 420

        # Per-step Bernini progress (mirrors the idv2v rail): the CLI emits one
        # {"type":"progress"} line per denoise step (plus per-chunk frames for
        # long v2v); we fold those into run_mod.set_progress so the live-runner
        # /progress/{job_id} poll and the SSE rail see real 0..1 progress +
        # step/total while "generating".
        def _prog(info):
            if not isinstance(info, dict):
                return
            if isinstance(info.get("step"), int) and                     isinstance(info.get("total"), int) and info.get("total"):
                frac = 0.05 + 0.90 * (min(info["step"], info["total"]) / info["total"])
                run_mod.set_progress(
                    job_id, round(min(max(frac, 0.05), 0.95), 4), "bernini",
                    f"step {info['step']}/{info['total']}",
                    step=info["step"], total=info["total"])
            elif isinstance(info.get("chunk"), int) and                     isinstance(info.get("chunks"), int):
                run_mod.set_progress(
                    job_id, 0.05, "bernini",
                    f"chunk {info['chunk']}/{info['chunks']} "
                    f"(frames {info.get('frames_done', '?')})")

        try:
            result = await asyncio.wait_for(
                manager.generate(job, progress_cb=_prog), timeout=_gen_timeout)
        except bernini_mod.BerniniError as exc:
            run_mod.set_progress(job_id, -1, "failed", str(exc))
            return web.json_response({"error": str(exc), "job_id": job_id},
                                     status=500)

        run_mod.set_progress(job_id, 0.97, "bernini", "encoding")
        try:
            out_b64 = bernini_io.encode_video_b64(out_path)
        except Exception as exc:  # noqa: BLE001
            run_mod.set_progress(job_id, -1, "failed", str(exc))
            return web.json_response(
                {"error": f"output encode failed: {exc}", "job_id": job_id},
                status=500)

        run_mod.set_progress(job_id, 1.0, "complete", "done")
        return web.json_response({
            "job_id": job_id,
            "task": task,
            "model": manager.model,
            "output_video": out_b64,
            "frames": result.get("frames"),
            "resolution": f"{body.get('width', 848)}x{body.get('height', 480)}",
            "fps": result.get("out_fps") or body.get("fps", config.BERNINI_NATIVE_FPS),
        })


async def handle_bernini_evict(request: web.Request) -> web.Response:
    """Evict the resident Bernini subprocess (frees its GPU)."""
    _require_token(request)
    await bernini_mod.evict_manager()
    return web.json_response({"evicted": True})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

async def handle_sam3(request: web.Request) -> web.Response:
    """Segment the object-to-keep from a single image via SAM3 (text prompt).

    Body JSON: {image: <base64 PNG/JPEG>, mode?: "auto"|"text", prompt?: str}.
    Runs SAM3 in a subprocess (memory-isolated, off the resident id-v2v model) and
    returns the object mask as a base64 binary PNG at the ORIGINAL resolution.
    The caller inverts it so the image edit regenerates everything except the
    detected subject.
    """
    _require_token(request)
    body = await request.json()
    image_b64 = body.get("image")
    if not image_b64:
        return web.json_response({"error": "missing 'image' (base64)"}, status=400)
    mode = str(body.get("mode", "auto"))
    if mode not in ("auto", "text"):
        return web.json_response(
            {"error": "mode must be 'auto' or 'text' (point/box prompts coming soon)"},
            status=400)
    prompt = str(body.get("prompt") or config.SAM_PROMPT)

    import base64
    import shutil
    import subprocess
    import sys
    import tempfile
    import PIL.Image as PILImage

    tmpdir = tempfile.mkdtemp(prefix="sam3_")
    try:
        img_path = os.path.join(tmpdir, "input.png")
        mask_path = os.path.join(tmpdir, "mask.png")
        with open(img_path, "wb") as fh:
            fh.write(base64.b64decode(image_b64))

        gpu = "0"
        device = config.GPU_DEVICE
        if str(device).startswith("cuda:"):
            gpu = str(device).split(":", 1)[1] or "0"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)

        cmd = [sys.executable, "-m", "runner.idv2v.segment_single",
               "--image", img_path, "--prompt", prompt,
               "--model_path", config.SAM3_CKPT, "--out_mask", mask_path]
        proc = await asyncio.to_thread(
            subprocess.run, cmd, capture_output=True, text=True, env=env, timeout=300,
        )
        if proc.returncode != 0:
            return web.json_response(
                {"error": "sam3 segmentation failed",
                 "detail": (proc.stderr or proc.stdout)[-2000:]}, status=500)

        with open(mask_path, "rb") as fh:
            mask_b64 = base64.b64encode(fh.read()).decode("ascii")
        with PILImage.open(img_path) as im:
            width, height = im.size
        return web.json_response({
            "mask_b64": mask_b64, "width": width, "height": height,
            "mode": mode, "prompt": prompt,
        })
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


async def handle_prompt_enhance(request: web.Request) -> web.Response:
    """Self-contained Gemma prompt enhancement on the id-v2v worker.

    Body JSON: {prompt: str, image_base64?: str (optional, for image-edit/i2v
    enhancement via the vision tower), seed?: int}.

    Prefers forwarding to the SHARED gemma-worker (config.GEMMA_FORWARD_URL)
    using this worker's own DEDICATED system prompts (text ->
    ENHANCE_T2V_SYSTEM_PROMPT, image -> IMAGE_ENHANCE_SYSTEM_PROMPT), so the
    actual inference runs on the shared llama.cpp Gemma 4 worker and the embedded
    Gemma 3 is never loaded. Falls back to the embedded Gemma 3 when the
    gemma-worker is unreachable, and to a 503 when neither is available.

    Returns {enhanced_prompt: str, image: bool}.
    """
    _require_token(request)
    body = await request.json()
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return web.json_response({"error": "missing 'prompt'"}, status=400)
    image_b64 = body.get("image_base64")
    has_image = bool(image_b64)
    seed = body.get("seed")

    base = config.gemma_forward_base()
    if base:
        try:
            if not await gemma_forward.gemma_worker_available(base, config.worker_token()):
                raise RuntimeError("gemma-worker unreachable")
            from .gemma import ENHANCE_T2V_SYSTEM_PROMPT, IMAGE_ENHANCE_SYSTEM_PROMPT
            system = IMAGE_ENHANCE_SYSTEM_PROMPT if has_image else ENHANCE_T2V_SYSTEM_PROMPT
            enhanced = await gemma_forward.forward_prompt_enhance(
                base, prompt=prompt, system_prompt=system,
                image_base64=image_b64 or None, seed=seed,
                token=config.worker_token())
            return web.json_response({"enhanced_prompt": enhanced, "image": has_image})
        except Exception as exc:
            logger.warning(
                "gemma-worker /prompt-enhance forward failed (using embedded Gemma 3): %s", exc)

    if not config.gemma_enabled():
        return web.json_response(
            {"error": "Gemma not provisioned (GEMMA_ROOT missing)"}, status=503)

    from .gemma import get_enhancer
    enhancer = get_enhancer()

    def _run():
        if has_image:
            import base64 as _b64
            import io as _io
            from PIL import Image as _Image
            img = _Image.open(
                _io.BytesIO(_b64.b64decode(image_b64))).convert("RGB")
            return enhancer.enhance_image(prompt, img, seed=seed)
        return enhancer.enhance_text(prompt, seed=seed)

    try:
        enhanced = await asyncio.to_thread(_run)
    except Exception as exc:
        logger.error("Prompt enhance failed: %s", exc, exc_info=True)
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"enhanced_prompt": enhanced, "image": has_image})


def create_app() -> web.Application:
    app = web.Application(client_max_size=config.MAX_BODY_BYTES)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/info", handle_info)
    app.router.add_get("/video-creator/v1/info", handle_info)
    app.router.add_post("/load", handle_load)
    app.router.add_post("/evict", handle_evict)
    app.router.add_post("/v1/restyle", handle_restyle)
    # The live-runner proxies every inference endpoint under
    # /video-creator/v1/{endpoint} (uniform across all workers); the idv2v
    # worker historically served /v1/restyle only, which made restyle
    # 404 via the live-runner. Register both so it works either way.
    app.router.add_post("/video-creator/v1/restyle", handle_restyle)
    # Bernini rail (wan-worker): t2v / v2v / r2v, both raw + proxied paths.
    app.router.add_post("/v1/t2v", handle_bernini)
    app.router.add_post("/video-creator/v1/t2v", handle_bernini)
    app.router.add_post("/v2v", handle_bernini)
    app.router.add_post("/v1/v2v", handle_bernini)
    app.router.add_post("/video-creator/v1/v2v", handle_bernini)
    app.router.add_post("/r2v", handle_bernini)
    app.router.add_post("/v1/r2v", handle_bernini)
    app.router.add_post("/video-creator/v1/r2v", handle_bernini)
    # Live-runner ROUTES ids (bernini-* — endpoint name reaches the worker as-is).
    app.router.add_post("/video-creator/v1/bernini-t2v", handle_bernini)
    app.router.add_post("/video-creator/v1/bernini-v2v", handle_bernini)
    app.router.add_post("/video-creator/v1/bernini-r2v", handle_bernini)
    app.router.add_post("/v1/bernini/evict", handle_bernini_evict)
    app.router.add_post("/video-creator/v1/bernini/evict", handle_bernini_evict)
    app.router.add_post("/video-creator/v1/bernini-evict", handle_bernini_evict)
    app.router.add_get("/progress/{job_id}", handle_progress)
    app.router.add_get("/video-creator/v1/progress/{job_id}", handle_progress)
    app.router.add_post("/v1/sam3", handle_sam3)
    app.router.add_post("/video-creator/v1/sam3", handle_sam3)
    app.router.add_post("/v1/prompt-enhance", handle_prompt_enhance)
    app.router.add_post("/video-creator/v1/prompt-enhance", handle_prompt_enhance)
    return app


async def _run() -> None:
    # Ask the live-runner for a free GPU before serving so a cold/recovered
    # worker never blindly lands on GPU 0 (mirrors ltx-worker; explicit
    # GPU_DEVICE is still honored as a pin).
    await _resolve_startup_device()
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.HOST, config.PORT)
    await site.start()
    logger.info("ID-V2V worker listening on %s:%d", config.HOST, config.PORT)
    await asyncio.Event().wait()


def _evict_video_for_gemma() -> None:
    """Evict the resident id-v2v model so Gemma can load on the shared GPU.

    Gemma runs on the SAME GPU as the video model (config.gemma_device() -> the
    worker's GPU_DEVICE). The two can't coexist on one 32 GB card, so before
    Gemma allocates VRAM we drop the video model. Refuse when a restyle is
    actively generating (the running job still owns the model + GPU).
    """
    global _model
    if run_mod.generation_active():
        raise RuntimeError(
            "cannot evict id-v2v model while a restyle is generating"
        )
    m = _model
    if m is not None and m.is_ready:
        m.evict()
        logger.info("Evicted id-v2v model (shared GPU) to load Gemma LLM")




def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stdout)

    # The live-runner + health probes poll /health, /info and /progress many
    # times a second; demote those three access-log lines to VERBOSE/DEBUG so the
    # INFO log isn't flooded with polling noise (job / bernini / evict request
    # logs stay at INFO). Only visible when the aiohttp.access logger is lowered
    # to DEBUG.
    _QUIET_ACCESS_PATHS = ("/health", "/info", "/progress")

    class _QuietAccessFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:  # noqa: BLE001 - never drop logs on a format error
                return True
            return not any(p in msg for p in _QUIET_ACCESS_PATHS)

    logging.getLogger("aiohttp.access").addFilter(_QuietAccessFilter())

    # Resolve auth token eagerly so a blank one is generated + logged once.
    _resolve_token()
    # Gemma shares the video model's GPU -> wire eviction into the enhancer so
    # every Gemma use (restyle stage + /prompt-enhance) evicts the resident video
    # model first. Gemma stays ON-DEMAND here (the .151 ltx worker preloads its
    # own Gemma at startup; the id-v2v worker's shares a GPU with its video model
    # and must not hold both resident).
    from .gemma import configure_evict_cb
    configure_evict_cb(_evict_video_for_gemma)
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
