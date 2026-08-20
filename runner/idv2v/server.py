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
import logging
import os
import sys
import time
import uuid

from aiohttp import web

from . import config
from . import run as run_mod
from . import gemma_forward
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


def _get_model() -> ModelManager:
    global _model
    if _model is None:
        _model = ModelManager(device=config.GPU_DEVICE)
    return _model


async def handle_load(request: web.Request) -> web.Response:
    _require_token(request)
    global _model
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    device = body.get("device")
    async with _model_lock:
        model = _get_model()
        if device is not None:
            try:
                device = int(device)
            except (TypeError, ValueError):
                device = None
            cur = model.device
            cur_idx = int(cur.split(":")[1]) if ":" in str(cur) else 0
            if device is not None and device >= 0 and cur_idx != device:
                logger.info("idv2v-worker /load: relocating GPU %d -> %d",
                            cur_idx, device)
                model.set_device(str(device))
        if model.is_ready:
            return web.json_response(
                {"loaded": True, "already_loaded": True, "device": model.device})
        try:
            await asyncio.wait_for(model.load(), timeout=3600)
        except asyncio.TimeoutError:
            return web.json_response({"error": "model load timed out"}, status=504)
    return web.json_response({"loaded": True, "device": model.device})


async def handle_evict(request: web.Request) -> web.Response:
    _require_token(request)
    global _model
    async with _model_lock:
        if _model is not None:
            _model.evict()
        _model = None
    return web.json_response({"evicted": True})


async def handle_health(request: web.Request) -> web.Response:
    model = _model
    info = health_check(model) if model is not None else {
        "status": "unloaded", "model_loaded": False, "device": config.GPU_DEVICE,
        "precision": config.IDV2V_QUANT, "offload": config.IDV2V_OFFLOAD,
    }
    return web.json_response({"status": "ok", **info})


def _device_index() -> int:
    """CUDA index this worker currently targets (for the scheduler's reconcile)."""
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
    model = _model
    base = health_check(model) if model is not None else {
        "status": "unloaded", "model_loaded": False, "device": config.GPU_DEVICE,
        "precision": config.IDV2V_QUANT, "offload": config.IDV2V_OFFLOAD,
    }
    return web.json_response({
        "runner_id": "",
        "app": "idv2v",
        "capabilities": ["restyle", "sam3", "prompt-enhance"],
        "ready": model is not None,
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
