"""LTX worker server — serves /video-creator/v1/* + token-gated /load /evict

Pure internal worker behind the live-runner edge. Does NOT register with the
Orchestrator (that is live-runner's job); it only serves the inference surface
and the worker control plane over the Docker network.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import tempfile
import threading
import uuid

from aiohttp import web

from runner.common import engineproc
from runner.ltx import enhance_forward
from runner.ltx.config import (
    LIVE_RUNNER_URL,
    ENHANCE_FORWARD_API_KEY,
    IDV2V_WORKER_URL,
    ENHANCE_FORWARD_MODEL,
    ENHANCE_FORWARD_TIMEOUT,
    ENHANCE_FORWARD_URL,
    ENHANCE_GPU_DEVICE,
    GPU_DEVICE,
    GPU_NAME,
    GPU_VRAM_GB,
    HOST,
    MODEL_CHECKPOINT,
    PORT,
    TEXT_ENCODER_ROOT,
    UPSCALER_PATH,
    WARMUP,
    worker_token,
)
from runner.ltx.gpu_profile import build_profile, STREAMING_MIN_GB
from runner.ltx.loracache import LoraCache

logger = logging.getLogger(__name__)
APP_ID = "video-creator"

# Effective default system prompts for forwarded enhancement: an env override
# wins, otherwise the built-in default used by enhance_forward (mirrors the
# desktop's generic free-rewrite fallback).
_ENHANCE_T2V_DEFAULT = enhance_forward.DEFAULT_T2V_SYSTEM_PROMPT
_ENHANCE_I2V_DEFAULT = enhance_forward.DEFAULT_I2V_SYSTEM_PROMPT

# Cap for the aiohttp request body. Default matches go-livepeer's declared
# max AI request size (server/ai_http.go: MaxAIRequestSize = 3000000000 // 3GB),
# so an attached base64 image for i2v or full base64 source video for
# extend/retake isn't rejected beneath what the orchestrator allows.
# Override with MAX_BODY_BYTES if a deployment needs more.
_MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(3000000000)))

# Global state
# ``engine`` is a subprocess-backed proxy (``_EngineProxy``) while a model is
# resident, or None after /evict / at boot. The real engine + GPU live ONLY in
# the child process; the proxy serializes method calls over JSONL.
engine: "_EngineProxy | None" = None
gpu_profile = None
registration = None
ready = False
# CUDA device picked at startup for this worker (the default target for /load
# when the request does not specify one). Never a CUDA context in this process.
_chosen_video_device: int | None = None
# Serializes (re)creation / eviction of the model subprocess.
_engine_lock = asyncio.Lock()


async def _pick_video_device(preferred: int) -> int:
    """Choose the CUDA device the video pipeline warms up on.

    An explicit ``GPU_DEVICE`` env always wins (operator-pinned). Otherwise, ask
    the live-runner's authoritative ``/gpu-pick`` endpoint (which owns the GPU
    scheduler and reconciles real ownership from every worker's /info) which
    card is free for us — this is the source of truth, and it knows the image
    worker holds GPU 0 even though that model loads lazily (so a local nvidia-smi
    would wrongly see GPU 0 as "free" at ltx-worker boot time).

    Falls back to local ``nvidia-smi memory.free`` (idlest card) only if the
    live-runner isn't reachable / not configured. No CUDA context is allocated
    in either path, so it works while other workers have models resident.
    """
    env = os.environ.get("GPU_DEVICE", "").strip()
    if env:
        try:
            return int(env)
        except ValueError:
            logger.warning("GPU_DEVICE=%r not an int; will auto-select", env)

    # 1) Authoritative: ask the live-runner scheduler for a free GPU.
    lr = LIVE_RUNNER_URL
    if lr:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{lr}/video-creator/v1/gpu-pick",
                    json={"worker": "ltx-worker"},
                    headers={"X-Worker-Token": worker_token()},
                    timeout=aiohttp.ClientTimeout(total=20),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        gpu = data.get("gpu")
                        if isinstance(gpu, int) and gpu >= 0:
                            logger.info("Live-runner assigned GPU %d to ltx-worker",
                                        gpu)
                            return gpu
                    logger.warning(
                        "live-runner gpu-pick returned %s; falling back to local select",
                        resp.status,
                    )
        except Exception as exc:
            logger.warning("live-runner gpu-pick failed (%s); falling back to local select", exc)

    # 2) Fallback: pick the idlest (most free VRAM) card from nvidia-smi.
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            best_idx, best_free = None, -1
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                try:
                    idx = int(parts[0])
                    free_mib = int(parts[1])
                    total_mib = int(parts[2])
                except ValueError:
                    continue
                if total_mib < (STREAMING_MIN_GB * 1024):
                    logger.info("GPU %d: total %.1f GiB < %.0f GiB minimum -- skipping",
                                idx, total_mib / 1024, STREAMING_MIN_GB)
                    continue
                if free_mib > best_free:
                    best_idx, best_free = idx, free_mib
            if best_idx is not None:
                logger.info("Auto-selected GPU %d (%.0f MiB free) as the idlest video card",
                            best_idx, best_free)
                return best_idx
        logger.warning("nvidia-smi unusable for auto-select; falling back to GPU %d", preferred)
    except Exception as exc:
        logger.warning("Auto GPU selection failed (%s); falling back to GPU %d", exc, preferred)
    return preferred


# Serializes generation: the single GPU can only run one generation at a time,
# and with handlers now dispatching to a thread pool, concurrent requests must
# be queued rather than racing on the shared engine / CUDA stream.
_generation_lock = asyncio.Lock()


# ── Helpers ──────────────────────────────────────────────

def _read_file_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _require_token(request: web.Request) -> None:
    """Reject the request unless it carries the shared worker token.

    Control-plane endpoints (/load, /evict) REQUIRE the token so only the
    live-runner edge can drive the swap policy. Generation /v1/* endpoints are
    left open for orchestrator proxying in this transitional worker; the
    live-runner applies X-Worker-Token to every upstream call it makes.
    """
    expected = worker_token()
    provided = request.headers.get("X-Worker-Token", "")
    if not provided or provided != expected:
        raise web.HTTPForbidden(reason="missing/mismatched X-Worker-Token")


class _EngineProxy:
    """Subprocess-backed stand-in for ``VideoCreatorInferenceEngine``.

    Mirrors the engine's public surface used by the HTTP routes; each call is
    serialized to the model child over JSONL via ``engineproc.EngineProc`` and
    returns the SAME JSON-safe type the real method returns (file path strings,
    None, clamped resolution / enhanced prompt strings). The parent holds no
    CUDA context: the engine + GPU live ONLY in the ``runner.ltx.engine_cli``
    child, which this proxy spawns (on /load / first generation) and tears down
    (on /evict, by killing the process -> destroying its primary context).
    """

    def __init__(self, device_index: int) -> None:
        self._device_index = device_index
        self._proc = engineproc.EngineProc(
            "ltx",
            [sys.executable, "-m", "runner.ltx.engine_cli",
             "--device", str(device_index)],
        )

    # ── lifecycle ────────────────────────────────────────────────────────
    async def ensure_loaded(self) -> None:
        """Spawn the child (if not running) and wait for its ready handshake."""
        await self._proc.start()

    async def stop(self) -> None:
        """Kill the child -> destroys its CUDA primary context. Idempotent."""
        await self._proc.stop()

    @property
    def device_index(self) -> int:
        return self._device_index

    # ── engine-op proxy ──────────────────────────────────────────────────
    @staticmethod
    def _ser_loras(loras):
        # list[tuple[path, scale]] -> [[path, scale], ...]  (tuples are not JSON)
        if loras is None:
            return None
        return [[p, float(s)] for p, s in loras]

    async def generate_t2v(self, prompt, seed, width, height, num_frames,
                           fps, output_path, loras=None, model=""):
        await self._proc.run("generate_t2v", {
            "prompt": prompt, "seed": seed, "width": width,
            "height": height, "num_frames": num_frames, "fps": fps,
            "output_path": output_path,
            "loras": self._ser_loras(loras), "model": model,
        })
        return None

    async def generate_i2v(self, prompt, image_base64, seed, width, height,
                           num_frames, fps, output_path, loras=None, model=""):
        await self._proc.run("generate_i2v", {
            "prompt": prompt, "image_base64": image_base64, "seed": seed,
            "width": width, "height": height, "num_frames": num_frames,
            "fps": fps, "output_path": output_path,
            "loras": self._ser_loras(loras), "model": model,
        })
        return None

    async def generate_extend(self, *, prompt, video_base64, extend_frames,
                              mode, seed, fps, output_path, context_seconds=1.0,
                              model="", progress_cb=None):
        def _fwd(obj):
            # child progress line {"stage","message","progress"} -> user cb
            if progress_cb is not None:
                try:
                    progress_cb(obj.get("stage"), obj.get("message"),
                                obj.get("progress"))
                except Exception:  # noqa: BLE001 - never break on progress
                    pass

        await self._proc.run("generate_extend", {
            "prompt": prompt, "video_base64": video_base64,
            "extend_frames": extend_frames, "mode": mode, "seed": seed,
            "fps": fps, "output_path": output_path,
            "context_seconds": context_seconds, "model": model,
        }, progress_cb=_fwd)
        return None

    async def generate_retake(self, *, prompt, video_base64, start_time,
                              end_time, seed, fps, regenerate_video=True,
                              regenerate_audio=True, output_path=None):
        return await self._proc.run("generate_retake", {
            "prompt": prompt, "video_base64": video_base64,
            "start_time": start_time, "end_time": end_time, "seed": seed,
            "fps": fps, "regenerate_video": regenerate_video,
            "regenerate_audio": regenerate_audio, "output_path": output_path,
        })

    async def generate_image(self, *, prompt, width, height, num_steps=9,
                             seed=42, guidance_scale=None):
        return await self._proc.run("generate_image", {
            "prompt": prompt, "width": width, "height": height,
            "num_steps": num_steps, "seed": seed,
            "guidance_scale": guidance_scale,
        })

    async def edit_image(self, *, prompt, image_path, mask_path=None,
                         keep_subject=False, sam3_url=None,
                         sam3_prompt="person", keep_mask_b64=None,
                         worker_token="", strength=0.6, num_steps=9, seed=42,
                         guidance_scale=None):
        return await self._proc.run("edit_image", {
            "prompt": prompt, "image_path": image_path,
            "mask_path": mask_path, "keep_subject": keep_subject,
            "sam3_url": sam3_url, "sam3_prompt": sam3_prompt,
            "keep_mask_b64": keep_mask_b64, "worker_token": worker_token,
            "strength": strength, "num_steps": num_steps, "seed": seed,
            "guidance_scale": guidance_scale,
        })

    async def generate_ic_lora_full_video(self, *, prompt, control_video_path,
                                          seed, width, height, num_frames, fps,
                                          output_path,
                                          conditioning_strength=1.0,
                                          lora_path="", lora_strength=1.0,
                                          skip_stage_2=False,
                                          resolution_factor=2.0):
        await self._proc.run("generate_ic_lora_full_video", {
            "prompt": prompt, "control_video_path": control_video_path,
            "seed": seed, "width": width, "height": height,
            "num_frames": num_frames, "fps": fps,
            "output_path": output_path,
            "conditioning_strength": conditioning_strength,
            "lora_path": lora_path, "lora_strength": lora_strength,
            "skip_stage_2": skip_stage_2,
            "resolution_factor": resolution_factor,
        })
        return None

    async def enhance_prompt(self, prompt, image_base64=None, seed=None,
                             system_prompt=None):
        return await self._proc.run("enhance_prompt", {
            "prompt": prompt, "image_base64": image_base64, "seed": seed,
            "system_prompt": system_prompt,
        })

    async def clamp_resolution(self, resolution):
        return await self._proc.run("clamp_resolution", {
            "resolution": resolution,
        })

    async def warmup(self, output_path):
        await self._proc.run("warmup", {"output_path": output_path})
        return None


async def _ensure_engine(target_device: "int | None" = None) -> "_EngineProxy":
    """Return a loaded engine proxy, spawning the model child if needed.

    ``target_device`` is the CUDA index the caller wants (e.g. /load's body);
    when None the worker's startup-picked device is used. If a child is already
    resident on a DIFFERENT device it is stopped and respawned on the requested
    one (the subprocess model's equivalent of ``set_device``). Idempotent when
    already loaded on the target device.
    """
    global engine
    dev = target_device if target_device is not None else _chosen_video_device
    async with _engine_lock:
        if engine is None:
            engine = _EngineProxy(dev)
        elif dev is not None and engine.device_index != dev:
            logger.info("ltx-worker: relocating model GPU %d -> %d",
                        engine.device_index, dev)
            await engine.stop()
            engine = _EngineProxy(dev)
        await engine.ensure_loaded()
        return engine


def _build_info_profile(device_index: int):
    """Build the /info GPU profile WITHOUT creating a CUDA context.

    The engine's authoritative VRAM-aware profile is built inside the GPU child
    (it may query torch freely). Here we only need display info for /info, so
    we pin name/VRAM via env overrides and/or nvidia-smi (no torch) and reuse
    ``build_profile`` by passing BOTH overrides explicitly (which skips its
    torch query path entirely).
    """
    import subprocess
    name = GPU_NAME
    vram = GPU_VRAM_GB
    if not (name and vram):
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits", "-i", str(device_index)],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                parts = [p.strip() for p in r.stdout.strip().splitlines()[0].split(",")]
                if not name:
                    name = parts[0]
                if not vram and len(parts) > 1:
                    try:
                        vram = float(parts[1]) / 1024.0
                    except ValueError:
                        pass
        except Exception:
            pass
    if not name:
        name = "Unknown GPU"
    if not vram:
        vram = 0.0
    return build_profile(device_index, vram, name)


async def handle_load(req: web.Request) -> web.Response:
    """POST /load — spawn + keep the model subprocess resident (warm).

    Reads ``device`` (int CUDA index) from the body. The model now lives in a
    SEPARATE process (``runner.ltx.engine_cli``), so loading = spawning that
    child on the requested GPU and waiting for its ready handshake. Relocating
    to a different card = stop the current child + respawn it on the requested
    one. This is what lets the scheduler place video workers on ANY free card
    instead of pinning them to GPU 0; an idle (never-loaded / evicted) worker
    holds NO GPU because no child is running. Returns 200 + the active device.
    """
    _require_token(req)
    body = {}
    try:
        body = await req.json()
    except Exception:
        body = {}
    # ``model`` is accepted for parity with every other worker's /load; ltx-worker
    # is SINGLE-ENGINE (one LTX video model), so the scheduler's model hint is
    # advisory — only the device affects this engine.
    model = body.get("model")
    if model:
        logger.info("ltx-worker /load: model=%s (single-engine; advisory)", model)
    target = _chosen_video_device
    device = body.get("device")
    if device is not None:
        try:
            device = int(device)
        except (TypeError, ValueError):
            device = None
        if device is not None and device >= 0:
            target = device
    global engine
    engine = await _ensure_engine(target)
    assert engine is not None
    return web.json_response({
        "loaded": True, "ready": ready, "device": engine.device_index})


async def handle_evict(req: web.Request) -> web.Response:
    """POST /evict — tear the model subprocess down, freeing its CUDA context.

    Called by the live-runner before it swaps in another worker's model on the
    shared GPU. The GPU model now lives in a SEPARATE process, so eviction =
    killing that child, which destroys its CUDA primary context entirely (the
    in-process ``free()``/``empty_cache()`` could only return the caching
    allocator's pool while the process stayed attached to the GPU with a
    ~0.5-0.7 GB driver floor). The next /v1/* generation respawns the child
    lazily.

    Like handle_load, an optional JSON body may carry ``{"device": N}``, but
    the ltx-worker is SINGLE-ENGINE today (one global engine owns one card), so
    ``device`` is parsed and intentionally ignored — freeing the resident card
    is the only action. Guarded with ``if engine is not None`` (rather than an
    assert) so a double /evict is a harmless no-op instead of a 500.
    """
    _require_token(req)
    global engine
    body = {}
    try:
        body = await req.json()
    except Exception:
        body = {}
    # Accepted for request/response symmetry with /load; unused (single-engine).
    _ = body.get("device")
    if engine is not None:
        await engine.stop()
        engine = None
    return web.json_response({"evicted": True})


async def _run_generation(fn, *args, **kwargs):
    """Run an engine op off the shared generation, serialized by the lock.

    The engine's generate_* calls now run in the model SUBPROCESS (the proxy's
    ``EngineProc.run`` is a JSONL round-trip bound to the event loop, i.e.
    non-blocking from the aiohttp process's perspective — the LiveRunner
    heartbeat is an asyncio background task and is never starved). We still
    serialize with the generation lock so only one generation can be in flight
    on the single engine at a time.
    """
    async with _generation_lock:
        return await fn(*args, **kwargs)


class _LoraError(Exception):
    """Raised when a requested LoRA can't be resolved (unknown id/file / cache down)."""
    pass


_lora_cache = None
_lora_cache_lock = threading.Lock()


def _get_lora_cache():
    """Lazily build the LoraCache (loads the catalog on first use).

    Returns None on failure so the service keeps running without LoRA support;
    the single failure is logged and not retried until restart."""
    global _lora_cache
    if _lora_cache is None:
        with _lora_cache_lock:
            if _lora_cache is None:
                try:
                    _lora_cache = LoraCache()
                    logger.info(
                        "LoRA catalog loaded (%d entries) — cache=%s — budget=%.1f GiB",
                        len(_lora_cache.catalog), _lora_cache.cache_dir,
                        _lora_cache.budget_bytes / (1024 ** 3),
                    )
                except Exception as exc:
                    logger.error("LoRA cache init failed (LoRA support disabled): %s", exc)
                    _lora_cache = False
    return _lora_cache if _lora_cache else None


def _resolve_loras(loras_raw):
    """Resolve a request's lora entries -> ``(resolved, custom_paths)``.

    ``resolved`` is ``[(local_path, scale)]`` for the sampler. Each entry is
    either:

      - a catalog LoRA ``{id, filename?, scale}`` (must exist in the catalog;
        unknown id -> ``_LoraError`` -> HTTP 404), or
      - a custom LoRA ``{custom_url, scale?, sha256?, hf_token?}`` (Option A):
        downloaded from an allowlisted https URL into a one-shot temp file.
        ``custom_paths`` holds those temp files so the caller can remove them
        after generation (and the custom-loader TTL-sweep handles orphans).

    Raises ``_LoraError`` (HTTP 404) for validation/download failures."""
    if not loras_raw:
        return [], []
    if not isinstance(loras_raw, list):
        raise _LoraError("'loras' must be a list of {id, filename?, scale} or custom entries")
    custom = [e for e in loras_raw if isinstance(e, dict) and e.get("custom_url")]
    cache = _get_lora_cache()
    if cache is None and custom:
        # Custom downloads don't need a catalog; spin up a custom-only cache so a
        # bad catalog source doesn't take down the user-LoRA path too.
        cache = LoraCache(catalog={})
    if cache is None:
        raise _LoraError("LoRA support unavailable on this runner (catalog/cache failed to initialize)")
    out = []
    custom_paths = []
    for entry in loras_raw:
        if not isinstance(entry, dict):
            raise _LoraError("each 'loras' entry must be an object")
        scale = float(entry.get("scale", 1.0))
        if entry.get("custom_url"):
            try:
                path = cache.download_custom(
                    entry["custom_url"],
                    sha256=entry.get("sha256"),
                    token=entry.get("hf_token"),
                )
            except Exception as exc:
                raise _LoraError(str(exc))
            custom_paths.append(path)
            out.append((path, scale))
            continue
        if not entry.get("id"):
            raise _LoraError("each 'loras' entry must be {id, filename?, scale} or {custom_url,...}")
        try:
            path = cache.ensure(entry["id"], entry.get("filename"))
        except KeyError as exc:
            raise _LoraError(str(exc))
        out.append((path, scale))
    return out, custom_paths


# ── Routes ───────────────────────────────────────────────

async def handle_health(_req: web.Request) -> web.Response:
    return web.json_response({"ok": True, "ready": ready, "app": APP_ID})


async def handle_info(_req: web.Request) -> web.Response:
    profile_info = {}
    if gpu_profile is not None:
        profile_info = gpu_profile.info
    return web.json_response({
        "runner_id": registration.runner_id if registration else "",
        "app": APP_ID,
        "model": MODEL_CHECKPOINT,
        "capabilities": ["t2v", "i2v", "image", "extend", "retake", "prompt-enhance", "suggest-gap-prompt", "ic-lora-extract", "ic-lora-generate", "ic-lora-restyle"],
        "ready": ready,
        "gpu": profile_info,
        "device_in_use": engine.device_index if engine is not None else (GPU_DEVICE or 0),
    })


async def handle_t2v(req: web.Request) -> web.Response:
    """POST /video-creator/v1/t2v"""
    body = await req.json()
    engine = await _ensure_engine()
    prompt = body["prompt"]
    seed = body.get("seed", 42)
    resolution = body.get("resolution", "1080p")
    duration = body.get("duration", 5)
    fps = body.get("fps", 24)
    aspect_ratio = body.get("aspectRatio", "16:9")
    model = str(body.get("model", ""))

    # Clamp requested resolution to what the GPU can handle, then resolve size
    if resolution in ("540p", "720p", "1080p"):
        resolution = await engine.clamp_resolution(resolution)
    res_map = {"540p": (960, 544), "720p": (1280, 704), "1080p": (1920, 1088)}
    w, h = res_map.get(resolution, (1920, 1088))
    if aspect_ratio == "9:16":
        w, h = h, w
    w = round(w / 64) * 64
    h = round(h / 64) * 64
    num_frames = duration * fps

    try:
        loras, custom_paths = _resolve_loras(body.get("loras"))
    except _LoraError as exc:
        return web.json_response({"error": str(exc)}, status=404)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        await _run_generation(engine.generate_t2v, prompt, seed, w, h, num_frames, fps, tmp.name, loras, model)
        b64 = _read_file_b64(tmp.name)
        return web.json_response({
            "video_base64": b64,
            "content_type": "video/mp4",
            "generation_id": uuid.uuid4().hex[:8],
        })
    finally:
        os.unlink(tmp.name)
        for _p in custom_paths:
            cache.remove_custom(_p)


async def handle_i2v(req: web.Request) -> web.Response:
    """POST /video-creator/v1/i2v"""
    body = await req.json()
    engine = await _ensure_engine()
    prompt = body["prompt"]
    image_base64 = body["image_base64"]
    seed = body.get("seed", 42)
    resolution = body.get("resolution", "1080p")
    duration = body.get("duration", 5)
    fps = body.get("fps", 24)
    aspect_ratio = body.get("aspectRatio", "16:9")
    model = str(body.get("model", ""))

    res_map = {"540p": (960, 544), "720p": (1280, 704), "1080p": (1920, 1088)}
    # Clamp requested resolution to what the GPU can handle
    if resolution in ("540p", "720p", "1080p"):
        resolution = await engine.clamp_resolution(resolution)
    w, h = res_map.get(resolution, (1920, 1088))
    if aspect_ratio == "9:16":
        w, h = h, w
    w = round(w / 64) * 64
    h = round(h / 64) * 64
    num_frames = duration * fps

    try:
        loras, custom_paths = _resolve_loras(body.get("loras"))
    except _LoraError as exc:
        return web.json_response({"error": str(exc)}, status=404)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        await _run_generation(engine.generate_i2v, prompt, image_base64, seed, w, h, num_frames, fps, tmp.name, loras, model)
        b64 = _read_file_b64(tmp.name)
        return web.json_response({
            "video_base64": b64,
            "content_type": "video/mp4",
            "generation_id": uuid.uuid4().hex[:8],
        })
    finally:
        os.unlink(tmp.name)
        for _p in custom_paths:
            cache.remove_custom(_p)


async def handle_a2v(req: web.Request) -> web.Response:
    """POST /video-creator/v1/a2v — audio-to-video (not supported on runner)"""
    return web.json_response({
        "error": "A2V requires local inference or A2V-capable runner",
        "status": "not_supported",
    })


async def handle_image(req: web.Request) -> web.Response:
    """POST /video-creator/v1/image — text-to-image generation."""
    body = await req.json()
    engine = await _ensure_engine()
    prompt = body["prompt"]
    width = body.get("width", 768)
    height = body.get("height", 768)
    num_steps = body.get("numSteps", 9)
    seed = body.get("seed", 42)
    # Optional guidance scale — forwarded only if the client sent it, so the
    # pipeline default applies otherwise.
    guidance_scale = body.get("guidanceScale")
    if isinstance(guidance_scale, (int, float)) and not isinstance(guidance_scale, bool):
        guidance_scale = float(guidance_scale)
    else:
        guidance_scale = None

    # Snap to /16 grid
    width = (width // 16) * 16
    height = (height // 16) * 16

    img_path = await _run_generation(
        engine.generate_image,
        prompt=prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        seed=seed,
        guidance_scale=guidance_scale,
    )
    try:
        b64 = _read_file_b64(img_path)
        return web.json_response({
            "image_base64": b64,
            "content_type": "image/png",
            "generation_id": uuid.uuid4().hex[:8],
        })
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)


async def handle_edit(req: web.Request) -> web.Response:
    """POST /video-creator/v1/edit — Z-Image img2img / masked-inpaint edit.

    Body: {image (b64), prompt, strength?, numSteps?, seed?, guidanceScale?,
           mask_image? (b64), keep_subject?: bool, sam3_prompt?: str}.

    - mask_image set            -> masked inpaint (only that region regenerates)
    - keep_subject True         -> object-to-keep from idv2v worker SAM3, inverted
                                   so everything EXCEPT the subject is regenerated
    - otherwise                 -> whole-frame img2img edit
    """
    body = await req.json()
    engine = await _ensure_engine()
    image_b64 = body.get("image")
    if not image_b64:
        return web.json_response({"error": "missing 'image' (base64)"}, status=400)
    prompt = str(body.get("prompt", ""))
    strength = float(body.get("strength", 0.6))
    num_steps = int(body.get("numSteps", 9))
    seed = int(body.get("seed", 42))
    gs = body.get("guidanceScale")
    guidance_scale = float(gs) if isinstance(gs, (int, float)) and not isinstance(gs, bool) else None
    keep_subject = bool(body.get("keep_subject", False))
    sam3_prompt = str(body.get("sam3_prompt", "person"))
    keep_mask_b64 = body.get("keep_mask")  # optional precomputed mask (white=subject)
    mask_b64 = body.get("mask_image")

    import base64 as _b64
    import tempfile as _tf
    import os as _os
    tmpdir = _tf.mkdtemp(prefix="zedit_")
    try:
        img_path = _os.path.join(tmpdir, "in.png")
        with open(img_path, "wb") as fh:
            fh.write(_b64.b64decode(image_b64))
        mask_path = None
        if mask_b64:
            mask_path = _os.path.join(tmpdir, "mask.png")
            with open(mask_path, "wb") as fh:
                fh.write(_b64.b64decode(mask_b64))
        out_path = await _run_generation(
            engine.edit_image,
            prompt=prompt,
            image_path=img_path,
            mask_path=mask_path,
            keep_subject=keep_subject,
            sam3_url=IDV2V_WORKER_URL,
            sam3_prompt=sam3_prompt,
            keep_mask_b64=keep_mask_b64,
            worker_token=worker_token(),
            strength=strength,
            num_steps=num_steps,
            seed=seed,
            guidance_scale=guidance_scale,
        )
        b64 = _read_file_b64(out_path)
        return web.json_response({
            "image_base64": b64,
            "content_type": "image/png",
            "generation_id": uuid.uuid4().hex[:8],
        })
    finally:
        try:
            if _os.path.exists(img_path):
                _os.unlink(img_path)
            if mask_path and _os.path.exists(mask_path):
                _os.unlink(mask_path)
            _os.rmdir(tmpdir)
        except Exception:
            pass


async def handle_extend(req: web.Request) -> web.Response:
    """POST /video-creator/v1/extend — extend a video by appending/prepending frames.

    With ``?sse=1`` the response is text/event-stream: `accepted` -> `progress`*
    (honest STAGE text: encoding / generating / decoding / finalizing; numeric
    ``progress`` is always null because the LTX denoise loop exposes no per-step
    callback -- no fabricated %) -> a single `complete` {video_base64,...} or `error`.
    The live-runner edge relays those events verbatim to the browser over the same
    paid connection, matching the image-worker /layer && /edit SSE surface.
    Without the flag the plain JSON response is returned (unchanged behaviour).
    """
    body = await req.json()
    engine = await _ensure_engine()
    prompt = body["prompt"]
    video_base64 = body["video_base64"]
    extend_frames = body.get("extendFrames", 120)
    mode = body.get("mode", "end")  # "start" or "end"
    seed = body.get("seed", 42)
    fps = body.get("fps", 24)
    model = str(body.get("model", ""))  # "ltx-2.5" picks the LTX-2.5 pipeline; else 2.3

    if req.query.get("sse") in ("1", "true", "yes"):
        return await _run_extend_sse(req, prompt, video_base64, extend_frames,
                                     mode, seed, fps, model)

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        await _run_generation(
            engine.generate_extend,
            prompt=prompt,
            video_base64=video_base64,
            extend_frames=extend_frames,
            mode=mode,
            seed=seed,
            fps=fps,
            output_path=tmp.name,
            model=model,
        )
        b64 = _read_file_b64(tmp.name)
        return web.json_response({
            "video_base64": b64,
            "content_type": "video/mp4",
            "generation_id": uuid.uuid4().hex[:8],
        })
    finally:
        os.unlink(tmp.name)


async def _run_extend_sse(req: web.Request, prompt: str, video_base64: str,
                          extend_frames: int, mode: str, seed: int,
                          fps: float, model: str) -> web.StreamResponse:
    """Serve /video-creator/v1/extend as text/event-stream when ``?sse=1``.

    Events: accepted -> progress* (STAGE text, no fabricated %) -> complete
    (video_base64 + content_type + generation_id), or error."""
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(req)
    engine = await _ensure_engine()

    async def _ev(event: str, data: dict) -> None:
        try:
            await resp.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
        except Exception:
            pass

    await _ev("accepted", {
        "endpoint": "extend",
        "model": model or "ltx-2.3",
        "extend_frames": extend_frames,
        "mode": mode,
    })
    loop = asyncio.get_running_loop()

    # generate_extend runs in a worker thread (run_in_executor), so progress
    # callbacks must hop back onto the loop before touching the stream.
    def _on_progress(stage: str, message: str, progress) -> None:
        loop.call_soon_threadsafe(
            asyncio.create_task,
            _ev("progress", {"stage": stage, "message": message, "progress": progress}),
        )

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        try:
            await _run_generation(
                engine.generate_extend,
                prompt=prompt,
                video_base64=video_base64,
                extend_frames=extend_frames,
                mode=mode,
                seed=seed,
                fps=fps,
                output_path=tmp.name,
                model=model,
                progress_cb=_on_progress,
            )
        except Exception as exc:
            logger.exception("extend SSE failed")
            await _ev("error", {"error": str(exc)})
            await resp.write_eof()
            return resp
        b64 = _read_file_b64(tmp.name)
        await _ev("complete", {
            "video_base64": b64,
            "content_type": "video/mp4",
            "generation_id": uuid.uuid4().hex[:8],
        })
    finally:
        os.unlink(tmp.name)
    await resp.write_eof()
    return resp


async def handle_retake(req: web.Request) -> web.Response:
    """POST /video-creator/v1/retake — regenerate a video segment with new prompt."""
    body = await req.json()
    engine = await _ensure_engine()
    prompt = body["prompt"]
    video_base64 = body["video_base64"]
    start_time = body.get("startTime", 0.0)
    duration = body.get("duration", 5.0)
    seed = body.get("seed", 42)
    fps = body.get("fps", 24)
    mode = body.get("mode", "replace_audio_and_video")

    regenerate_video = "video" in mode
    regenerate_audio = "audio" in mode

    end_time = start_time + duration

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        await _run_generation(
            engine.generate_retake,
            prompt=prompt,
            video_base64=video_base64,
            start_time=start_time,
            end_time=end_time,
            seed=seed,
            fps=fps,
            regenerate_video=regenerate_video,
            regenerate_audio=regenerate_audio,
            output_path=tmp.name,
        )
        b64 = _read_file_b64(tmp.name)
        return web.json_response({
            "video_base64": b64,
            "content_type": "video/mp4",
            "generation_id": uuid.uuid4().hex[:8],
        })
    finally:
        os.unlink(tmp.name)


async def handle_prompt_enhance(req: web.Request) -> web.Response:
    """POST /video-creator/v1/prompt-enhance

    Enhance a prompt either via the local Gemma text encoder or, when
    ENHANCE_FORWARD_URL is configured, by proxying to a shared OpenAI-compatible
    enhancer (e.g. one llama.cpp instance serving many runners). Accepts
    ``prompt`` plus optional ``image_base64`` (image-conditioned, i2v-style
    enhancement), ``seed``, and ``system_prompt`` overrides.
    """
    body = await req.json()
    raw_prompt = body.get("prompt")
    prompt = str(raw_prompt).strip() if raw_prompt is not None else ""
    if not prompt:
        return web.json_response({"error": "prompt is required"}, status=400)

    image_base64 = body.get("image_base64")
    if image_base64 is not None and not str(image_base64).strip():
        image_base64 = None
    seed = body.get("seed")
    seed = seed if isinstance(seed, int) else None
    sys_prompt = body.get("system_prompt")
    sys_prompt = sys_prompt if isinstance(sys_prompt, str) and sys_prompt.strip() else None

    # Forwarded mode: no local Gemma — proxy to the shared OpenAI-compatible
    # upstream so one enhancement model can serve many runners.
    if ENHANCE_FORWARD_URL:
        try:
            enhanced = await enhance_forward.forward_prompt_enhance(
                ENHANCE_FORWARD_URL,
                prompt=prompt,
                system_prompt=sys_prompt,
                image_base64=image_base64,
                seed=seed,
                model=ENHANCE_FORWARD_MODEL,
                api_key=ENHANCE_FORWARD_API_KEY,
                timeout=ENHANCE_FORWARD_TIMEOUT,
                default_t2v=_ENHANCE_T2V_DEFAULT,
                default_i2v=_ENHANCE_I2V_DEFAULT,
            )
        except Exception as exc:
            logger.exception("Forwarded prompt enhancement failed")
            return web.json_response({"error": f"Prompt enhancement failed: {exc}"}, status=500)
        return web.json_response({
            "enhanced_prompt": enhanced,
            "original_prompt": prompt,
        })

    # Local Gemma mode.
    if engine is None:
        return web.json_response({"error": "engine not ready"}, status=503)

    try:
        enhanced = await _run_generation(
            engine.enhance_prompt,
            prompt,
            image_base64=image_base64,
            seed=seed,
            system_prompt=sys_prompt,
        )
    except Exception as exc:
        logger.exception("Prompt enhancement failed")
        return web.json_response({"error": f"Prompt enhancement failed: {exc}"}, status=500)

    return web.json_response({
        "enhanced_prompt": enhanced,
        "original_prompt": prompt,
    })


async def handle_suggest_gap_prompt(req: web.Request) -> web.Response:
    """POST /video-creator/v1/suggest-gap-prompt — suggest a prompt to fill a gap between shots."""
    body = await req.json()
    before_prompt = body.get("before_prompt", "")
    after_prompt = body.get("after_prompt", "")
    gap_duration = body.get("gap_duration", 2.0)
    mode = body.get("mode", "text-to-video")

    # Simple heuristic: blend the two prompts
    parts = []
    if before_prompt:
        parts.append(before_prompt)
    if after_prompt:
        parts.append(after_prompt)

    suggested = " and ".join(parts) if len(parts) > 1 else (parts[0] if parts else "smooth transition")

    return web.json_response({
        "suggested_prompt": suggested,
        "gap_duration": gap_duration,
        "mode": mode,
    })


async def handle_extract_conditioning(req: web.Request) -> web.Response:
    """POST /video-creator/v1/extract-conditioning — extract conditioning frame from video."""
    body = await req.json()
    # Pure CPU (av + OpenCV + PIL) — no GPU engine needed.
    video_base64 = body["video_base64"]
    frame_time = body.get("frame_time", 0.0)
    conditioning_type = body.get("conditioning_type", "canny")

    # Decode video, extract frame at frame_time, apply conditioning
    import av
    src_bytes = base64.b64decode(video_base64)
    src_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    src_file.write(src_bytes)
    src_file.close()

    container = av.open(src_file.name)
    video_stream = container.streams.video[0]
    fps = video_stream.average_rate or 24
    target_frame = int(frame_time * fps)
    frames = []
    for i, frame in enumerate(container.decode(video_stream)):
        if i == target_frame:
            frames.append(frame.to_ndarray(format="rgb24"))
            break
    container.close()

    if not frames:
        os.unlink(src_file.name)
        return web.json_response({"error": f"Frame {target_frame} not found"}, status=400)

    from PIL import Image
    original_img = Image.fromarray(frames[0])
    original_buf = io.BytesIO()
    original_img.save(original_buf, format="JPEG", quality=85)
    original_b64 = base64.b64encode(original_buf.getvalue()).decode()

    # Apply conditioning (mirror of the desktop backend's VideoProcessor.apply_canny):
    # real OpenCV Canny so the preview matches the edge signal the IC-LoRA control videos
    # are trained on — the old PIL FIND_EDGES stand-in was too noisy and didn't look like
    # the actual canny conditioning. frames[0] is RGB (av decoded rgb24).
    if conditioning_type == "canny":
        import cv2
        import numpy as np
        img = frames[0]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        cond_img = Image.fromarray(np.concatenate([edges[:, :, None]] * 3, axis=2).astype(np.uint8))
    elif conditioning_type == "depth":
        # Real depth needs a MiDaS/DPT model that is not provisioned on the ltx-worker.
        # Keep the read-only reference's honest fallback (grayscale) rather than fabricate
        # a depth map. Only affects the preview, not the actual ic-lora-generate control.
        cond_img = Image.fromarray(frames[0]).convert("L").convert("RGB")
    else:
        cond_img = Image.fromarray(frames[0])

    cond_buf = io.BytesIO()
    cond_img.save(cond_buf, format="JPEG", quality=85)
    cond_b64 = base64.b64encode(cond_buf.getvalue()).decode()

    try:
        os.unlink(src_file.name)
    except Exception:
        pass

    return web.json_response({
        "conditioning": "data:image/jpeg;base64," + cond_b64,
        "original": "data:image/jpeg;base64," + original_b64,
        "conditioning_type": conditioning_type,
        "frame_time": frame_time,
    })


async def handle_ic_lora_generate(req: web.Request) -> web.Response:
    """POST /video-creator/v1/ic-lora-generate — IC-LoRA guided generation.

    Accepts the same payload as T2V/I2V but with conditioning images.
    Delegates to i2v pipeline using the conditioning frame as input image.
    """
    body = await req.json()
    engine = await _ensure_engine()
    prompt = body["prompt"]
    seed = body.get("seed", 42)
    resolution = body.get("resolution", {"width": 1280, "height": 720})
    fps = body.get("fps", 24)
    duration = body.get("duration", 5)

    # Parse resolution (may be a string "1080p" or dict with width/height)
    if isinstance(resolution, dict):
        w, h = resolution.get("width", 1280), resolution.get("height", 720)
    else:
        res_map = {"540p": (960, 544), "720p": (1280, 704), "1080p": (1920, 1088)}
        if resolution in ("540p", "720p", "1080p"):
            resolution = await engine.clamp_resolution(resolution)
        w, h = res_map.get(resolution, (1280, 704))

    w = round(w / 64) * 64
    h = round(h / 64) * 64
    num_frames = duration * fps

    # If a conditioning video/image is provided, use i2v with first frame
    video_b64 = body.get("video_base64")
    images = body.get("images", [])

    if video_b64:
        # Extract first frame from video as conditioning image
        import av
        src_bytes = base64.b64decode(video_b64)
        src_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        src_file.write(src_bytes)
        src_file.close()

        container = av.open(src_file.name)
        video_stream = container.streams.video[0]
        first_frame = None
        for frame in container.decode(video_stream):
            first_frame = frame.to_ndarray(format="rgb24")
            break
        container.close()

        try:
            os.unlink(src_file.name)
        except Exception:
            pass

        if first_frame is not None:
            from PIL import Image
            cond_img = Image.fromarray(first_frame)
            cond_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
            cond_img.save(cond_path)
            try:
                cond_b64 = _read_file_b64(cond_path)
            finally:
                os.unlink(cond_path)

            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            try:
                await _run_generation(engine.generate_i2v, prompt, cond_b64, seed, w, h, num_frames, fps, tmp.name)
                b64 = _read_file_b64(tmp.name)
                return web.json_response({
                    "video_base64": b64,
                    "content_type": "video/mp4",
                    "generation_id": uuid.uuid4().hex[:8],
                })
            finally:
                os.unlink(tmp.name)

    # No conditioning video — fall back to t2v
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        await _run_generation(engine.generate_t2v, prompt, seed, w, h, num_frames, fps, tmp.name)
        b64 = _read_file_b64(tmp.name)
        return web.json_response({
            "video_base64": b64,
            "content_type": "video/mp4",
            "generation_id": uuid.uuid4().hex[:8],
        })
    finally:
        os.unlink(tmp.name)


def _read_video_props(path: str) -> tuple[int, int, int, float]:
    """Return (width, height, frame_count, fps) for a video file."""
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, frame_count, fps


def _conditioning_video_frame(frame_bgr, conditioning_type: str):
    """Per-frame IC-LoRA conditioning of a BGR OpenCV frame -> 3-channel uint8 BGR.

    Full desktop parity: canny is real OpenCV Canny (mirror of the desktop
    VideoProcessor.apply_canny: gray -> cv2.Canny(100,200) -> 3-channel); depth keeps the
    honest grayscale fallback (no MiDaS/DPT provisioned on the ltx-worker) rather than
    fabricate a depth map.
    """
    import cv2
    import numpy as np
    if conditioning_type == "canny":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        return np.concatenate([edges[:, :, None]] * 3, axis=2)
    if conditioning_type == "depth":
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return np.concatenate([gray[:, :, None]] * 3, axis=2)
    raise ValueError(f"Unsupported conditioning_type for full-video restyle: {conditioning_type}")


def _build_control_video_from_source(src_path: str, dst_path: str, conditioning_type: str) -> tuple[int, int, int, float]:
    """Condition every frame of a source clip into a control video (desktop parity).

    Rewrites the whole clip frame-by-frame through canny/depth so the full-video
    IC-LoRA has a per-frame conditioning signal aligned to the source timeline.
    Returns (width, height, frame_count, fps) of the control video.
    """
    import cv2
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open source video: {src_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    writer = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(_conditioning_video_frame(frame, conditioning_type))
            n += 1
    finally:
        cap.release()
        writer.release()
    return w, h, n, fps


async def handle_ic_lora_restyle(req: web.Request) -> web.Response:
    """POST /video-creator/v1/ic-lora-restyle — full-video LTX IC-LoRA restyle.

    Runs the LTX ``ICLoraPipeline`` over the ENTIRE conditioning video via
    ``video_conditioning`` — frame-aligned restyle of the whole clip, NOT the
    first-frame-only i2v path (``ic-lora-generate``). Conditioning source:
      * canny/depth — per-frame cv2.Canny / grayscale preprocessing of the supplied
        source clip into a control video (full desktop parity);
      * custom — the supplied video IS the control video, used verbatim.
    Requires the IC-LoRA weights via a ``loras`` entry (catalog id or custom_url),
    resolved through the runner's LoraCache exactly like t2v/i2v loras — without them
    the pipeline has no IC-LoRA to condition on and the effect is inert.
    """
    body = await req.json()
    engine = await _ensure_engine()
    prompt = body["prompt"]
    video_base64 = body["video_base64"]
    conditioning_type = body.get("conditioning_type", "canny")
    conditioning_strength = float(body.get("conditioning_strength", 1.0))
    seed = int(body.get("seed", 42))
    skip_stage_2 = bool(body.get("skip_stage_2", False))
    resolution_factor = float(body.get("resolution_factor", 2.0))

    if conditioning_type not in ("canny", "depth", "custom"):
        return web.json_response(
            {"error": f"Unsupported conditioning_type: {conditioning_type} (expected canny|depth|custom)"},
            status=400,
        )

    # Resolve the IC-LoRA weights (exactly one) so the pipeline is actually loRA-driven.
    loras_raw = body.get("loras")
    if not loras_raw:
        return web.json_response(
            {"error": "IC-LoRA restyle requires a 'loras' entry (the IC-LoRA weights: catalog id or custom_url)"},
            status=400,
        )
    try:
        resolved, custom_paths = _resolve_loras(
            loras_raw if isinstance(loras_raw, list) else [loras_raw]
        )
    except _LoraError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    if not resolved:
        return web.json_response({"error": "Could not resolve IC-LoRA lora"}, status=400)
    lora_path, lora_strength = resolved[0]

    src_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    src_file.write(base64.b64decode(video_base64))
    src_file.close()
    cleanup = [src_file.name]
    control_path = src_file.name

    try:
        if conditioning_type == "custom":
            w, h, frame_count, fps = _read_video_props(control_path)
        else:
            control_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            cleanup.append(control_path)
            w, h, frame_count, fps = _build_control_video_from_source(src_file.name, control_path, conditioning_type)

        fps_override = body.get("fps")
        if fps_override is not None:
            fps = float(fps_override)

        # Target resolution: an explicit override wins; otherwise desktop parity (no
        # use_lora_in_stage_2) — width 768 by source aspect, snapped to 128.
        res = body.get("resolution")
        if isinstance(res, dict):
            tw, th = int(res.get("width", w)), int(res.get("height", h))
        elif isinstance(res, str):
            rmap = {"540p": (960, 544), "720p": (1280, 704), "1080p": (1920, 1088)}
            tw, th = rmap.get(res, (1280, 704))
        else:
            tw, th = 768, max(round(768 * h / w / 128) * 128, 128)
        width = max(128, (tw // 128) * 128)
        height = max(128, (th // 128) * 128)

        if (frame_count - 1) % 8 != 0:
            logger.warning(
                "[ic-lora-restyle] %d frames: (frames-1) %% 8 != 0 — pipeline may pad/trim and output could glitch",
                frame_count,
            )

        out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            engine.generate_ic_lora_full_video(
                prompt=prompt,
                control_video_path=control_path,
                seed=seed,
                width=width,
                height=height,
                num_frames=frame_count,
                fps=fps,
                output_path=out.name,
                conditioning_strength=conditioning_strength,
                lora_path=lora_path,
                lora_strength=lora_strength,
                skip_stage_2=skip_stage_2,
                resolution_factor=resolution_factor,
            )
            b64 = _read_file_b64(out.name)
            return web.json_response({
                "video_base64": b64,
                "content_type": "video/mp4",
                "generation_id": uuid.uuid4().hex[:8],
            })
        finally:
            os.unlink(out.name)
    finally:
        for p in cleanup:
            try:
                os.unlink(p)
            except Exception:
                pass
        for cp in custom_paths:
            try:
                os.unlink(cp)
            except Exception:
                pass


# ── Lifecycle ────────────────────────────────────────────

def _warn_if_low_host_ram() -> None:
    """Fail-fast guard for hosts whose system RAM can't map the checkpoint.

    ltx reads FP8 scales via safetensors.safe_open(device="cpu") which mmaps the
    entire checkpoint file; generation loads the transformer too. On boxes with
    little host RAM (e.g. a 5090 rig with 30 GB total) this OOMs with a cryptic
    "Cannot allocate memory" mid-load. Surface it clearly instead.
    """
    try:
        ckpt_size = os.path.getsize(MODEL_CHECKPOINT) if os.path.isfile(MODEL_CHECKPOINT) else 0
    except OSError:
        ckpt_size = 0
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        commit_limit = info.get("CommitLimit", 0)
    except OSError:
        commit_limit = 0
    if ckpt_size and commit_limit and ckpt_size > commit_limit:
        logger.error(
            "Host memory commit limit (~%.1f GiB) is smaller than the checkpoint "
            "(%.1f GiB). The model cannot be mmap'd on this host — add RAM or raise "
            "vm.overcommit_memory=1. Set GPU_VRAM_GB won't help; this is host RAM.",
            commit_limit / 2**30, ckpt_size / 2**30,
        )
    elif ckpt_size and commit_limit:
        logger.info("Host RAM commit limit ~%.1f GiB >= checkpoint %.1f GiB — OK",
                    commit_limit / 2**30, ckpt_size / 2**30)


def _memlog(tag: str) -> None:
    """Log the PARENT process's host RSS only (no CUDA — this process is
    CUDA-free; any GPU-memory logging lives in the model child)."""
    try:
        with open("/proc/self/status") as _f:
            for _line in _f:
                if _line.startswith("VmRSS:"):
                    _kb = int(_line.split()[1])
                    logger.info("MEMLOG %-24s rss=%dMiB", tag, _kb // 1024)
                    return
    except OSError:
        pass
    logger.info("MEMLOG %-24s rss=n/a", tag)


async def on_startup(_app: web.Application) -> None:
    global engine, ready, gpu_profile, _chosen_video_device
    _warn_if_low_host_ram()

    # Pick the CUDA device this worker warms up on. An explicit GPU_DEVICE env
    # always wins; otherwise auto-select the idlest (most-free-VRAM) card so a
    # video worker doesn't collide with the image worker's warm model on GPU 0.
    video_device = await _pick_video_device(GPU_DEVICE)

    # This aiohttp parent process stays CUDA-FREE. The real engine + GPU (and the
    # torch.cuda.set_device pin that used to run here) live ONLY in the
    # `runner.ltx.engine_cli` child, spawned lazily on /load / first generation.
    # Here we only record the picked device and build the VRAM-aware profile.
    _chosen_video_device = video_device

    # Detect GPU and build the VRAM-aware profile (4090 = streaming/24GB,
    # 5090 = full-resident/32GB, RTX PRO 6000 = full-resident/96GB).
    gpu_profile = build_profile(video_device, GPU_VRAM_GB, GPU_NAME)
    _memlog("after build_profile")
    if not gpu_profile.supports_generation:
        logger.error(
            "GPU[%d] %.1f GiB below the %d GiB minimum — generation will fail. "
            "Set GPU_VRAM_GB to bypass.",
            video_device, gpu_profile.vram_gb, 15,
        )
    logger.info("GPU: %s (%.1f GB VRAM, mode=%s)",
                gpu_profile.gpu_name, gpu_profile.vram_gb, gpu_profile.mode)

    # Subprocess-backed engine proxy (see _EngineProxy). No child is spawned
    # here: the model loads in the child on /load / first generation, and /evict
    # kills the child, destroying its CUDA primary context so this worker drops
    # to 0 MiB on every card it had touched.
    engine = _EngineProxy(video_device)
    _memlog("after engine proxy")
    logger.info("Engine proxy on GPU %d (mode=%s, max_res=%s)",
                video_device, gpu_profile.mode, gpu_profile.max_resolution)

    # Report which backend serves /prompt-enhance.
    if ENHANCE_FORWARD_URL:
        # Sharing one OPENAI-compatible enhancer across runners — the local
        # Gemma is never loaded, so a missing TEXT_ENCODER_ROOT is fine here.
        logger.info("Prompt enhancement: forwarded to %s/v1/chat/completions (local Gemma not loaded)",
                    ENHANCE_FORWARD_URL)
    else:
        # Prompt enhancement is served by the provisioned Gemma QAT q4_0 text
        # encoder at TEXT_ENCODER_ROOT (loaded lazily inside the engine child on
        # fallback use). Report availability at startup so an unprovisioned box
        # is obvious before /prompt-enhance returns 500s.
        _gemma_files = (
            [f.name for f in os.scandir(TEXT_ENCODER_ROOT) if f.is_file()]
            if os.path.isdir(TEXT_ENCODER_ROOT) else []
        )
        if any(n.startswith("model") and n.endswith(".safetensors") for n in _gemma_files):
            logger.info("Prompt-enhance Gemma (QAT q4_0) available at %s", TEXT_ENCODER_ROOT)
        else:
            logger.warning(
                "Prompt-enhance Gemma NOT found at TEXT_ENCODER_ROOT=%s "
                "(expected model*.safetensors) — /prompt-enhance will fail until "
                "provisioned (provision_models.py)", TEXT_ENCODER_ROOT
            )

    # Warmup (opt-in): spawn the child and run a warm generation.
    if WARMUP:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            await engine.ensure_loaded()
            await engine.warmup(tmp.name)
            _memlog("after warmup")
            logger.info("Warmup complete")
        finally:
            # warmup() already unlinks its own output; ignore if already gone.
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    # No startup warmup for the local prompt-enhance Gemma. /prompt-enhance is
    # routed to the dedicated gemma-worker (live-runner routing); the ltx worker
    # is only a cold fallback if that worker is down. Pre-warming its local
    # Gemma here would evict the just-warmed video pipeline (they share one GPU)
    # and cost startup for no benefit — it loads lazily on (rare) fallback use.
    _memlog("before ready")
    ready = True
    logger.info("Runner READY")
    _memlog("after ready assigned")


async def on_cleanup(_app: web.Application) -> None:
    # Nothing to unregister: this is a pure worker behind live-runner.
    pass


# ── App factory ──────────────────────────────────────────

def create_app() -> web.Application:
    # aiohttp's default client_max_size is 1 MB, which the runner routinely
    # exceeds: i2v sends an image (base64) and extend/retake send a full source
    # video (base64) in the request body. Raise it so attached media isn't
    # rejected with 413 "Maximum request body size ... exceeded".
    app = web.Application(client_max_size=_MAX_BODY_BYTES)
    # Worker control surface (token-gated) — drives the swap policy from the
    # live-runner edge. Served at ROOT /load /evict to match the swap contract
    # the live-runner's HttpWorkerTransport calls (base + "/load") and the
    # convention idv2v-worker already uses. (Generation routes live under
    # /video-creator/v1/*, so root control paths don't collide.)
    app.router.add_post("/load", handle_load)
    app.router.add_post("/evict", handle_evict)
    p = "/video-creator/v1"
    # Root /health alias so the live-runner's generic probe (base + "/health")
    # works for this worker too (idv2v-worker already serves root /health).
    app.router.add_get("/health", handle_health)
    app.router.add_get(f"{p}/health", handle_health)
    app.router.add_get(f"{p}/info", handle_info)
    app.router.add_post(f"{p}/t2v", handle_t2v)
    app.router.add_post(f"{p}/i2v", handle_i2v)
    app.router.add_post(f"{p}/a2v", handle_a2v)
    app.router.add_post(f"{p}/image", handle_image)
    app.router.add_post(f"{p}/edit", handle_edit)
    app.router.add_post(f"{p}/extend", handle_extend)
    app.router.add_post(f"{p}/retake", handle_retake)
    app.router.add_post(f"{p}/prompt-enhance", handle_prompt_enhance)
    app.router.add_post(f"{p}/suggest-gap-prompt", handle_suggest_gap_prompt)
    app.router.add_post(f"{p}/extract-conditioning", handle_extract_conditioning)
    app.router.add_post(f"{p}/ic-lora-generate", handle_ic_lora_generate)
    app.router.add_post(f"{p}/ic-lora-restyle", handle_ic_lora_restyle)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    _QUIET_ACCESS_PATHS = ("/health", "/info", "/progress")
    class _QuietAccessFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:  # noqa: BLE001 - never drop logs on a format error
                return True
            return not any(p in msg for p in _QUIET_ACCESS_PATHS)
    logging.getLogger("aiohttp.access").addFilter(_QuietAccessFilter())

    app = create_app()
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
