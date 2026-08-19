"""LTX worker server — serves /video-creator/v1/* + token-gated /load /evict

Pure internal worker behind the live-runner edge. Does NOT register with the
Orchestrator (that is live-runner's job); it only serves the inference surface
and the worker control plane over the Docker network.
"""
from __future__ import annotations

import asyncio
import base64
import functools
import io
import logging
import os
import tempfile
import threading
import uuid

import torch
from aiohttp import web

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
from runner.ltx.inference import VideoCreatorInferenceEngine
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
engine: VideoCreatorInferenceEngine | None = None
gpu_profile = None
registration = None
ready = False


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


async def handle_load(req: web.Request) -> web.Response:
    """POST /load — ensure the inference engine is resident (kept warm).

    Reads ``device`` (int CUDA index) from the body. If it differs from the GPU
    the engine currently targets, the engine is freed + relocated onto that GPU
    (weights stream from host RAM, so re-load is comparatively quick) and kept
    warm there. This is what lets the scheduler place video workers on ANY free
    card instead of pinning them to GPU 0. Returns 200 + the active device.
    """
    _require_token(req)
    global engine
    body = {}
    try:
        body = await req.json()
    except Exception:
        body = {}
    device = body.get("device")
    if device is not None and engine is not None:
        try:
            device = int(device)
        except (TypeError, ValueError):
            device = None
        if device is not None and device >= 0 and engine.device_index != device:
            logger.info("ltx-worker /load: relocating GPU %d -> %d",
                        engine.device_index, device)
            engine.set_device(device)
    assert engine is not None
    return web.json_response({
        "loaded": True, "ready": ready, "device": engine.device_index})


async def handle_evict(_req: web.Request) -> web.Response:
    """POST /evict — drop all resident pipelines + free GPU memory.

    Called by the live-runner before it swaps in another worker's model on the
    shared GPU. The next /v1/* generation reloads lazily.
    """
    _require_token(_req)
    assert engine is not None
    engine.free()
    return web.json_response({"evicted": True})


async def _run_generation(fn, *args, **kwargs):
    """Run a blocking engine generation off the event loop.

    The engine's generate_* calls are synchronous (GPU-bound), but the LiveRunner
    heartbeat is an asyncio background task — running generation directly on the
    loop would starve it and the orchestrator would drop the runner. Dispatch to
    a worker thread and serialize with the generation lock.
    """
    async with _generation_lock:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(fn, *args, **kwargs)
        )


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
        "capabilities": ["t2v", "i2v", "image", "extend", "retake", "prompt-enhance", "suggest-gap-prompt", "ic-lora-extract", "ic-lora-generate"],
        "ready": ready,
        "gpu": profile_info,
        "device_in_use": engine.device_index if engine is not None else (GPU_DEVICE or 0),
    })


async def handle_t2v(req: web.Request) -> web.Response:
    """POST /video-creator/v1/t2v"""
    body = await req.json()
    assert engine
    prompt = body["prompt"]
    seed = body.get("seed", 42)
    resolution = body.get("resolution", "1080p")
    duration = body.get("duration", 5)
    fps = body.get("fps", 24)
    aspect_ratio = body.get("aspectRatio", "16:9")
    model = str(body.get("model", ""))

    # Clamp requested resolution to what the GPU can handle, then resolve size
    if resolution in ("540p", "720p", "1080p"):
        resolution = engine.clamp_resolution(resolution)
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
    assert engine
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
        resolution = engine.clamp_resolution(resolution)
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
    assert engine
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
    assert engine
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
    """POST /video-creator/v1/extend — extend a video by appending/prepending frames."""
    body = await req.json()
    assert engine
    prompt = body["prompt"]
    video_base64 = body["video_base64"]
    extend_frames = body.get("extendFrames", 120)
    mode = body.get("mode", "end")  # "start" or "end"
    seed = body.get("seed", 42)
    fps = body.get("fps", 24)
    model = str(body.get("model", ""))  # "ltx-2.5" picks the LTX-2.5 pipeline; else 2.3

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


async def handle_upscale(req: web.Request) -> web.Response:
    """POST /video-creator/v1/upscale — upscale a finished video to a target
    resolution with the LTX-2.3 spatial upsampler (the 480->720 step in the
    restyle chain: ID-V2V generates at a box-fitting resolution, this restores
    full output resolution)."""
    body = await req.json()
    assert engine
    video_base64 = body["video_base64"]
    width = int(body.get("width", 1280))
    height = int(body.get("height", 720))
    fps = body.get("fps", 24)

    tmp_in = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_in.write(base64.b64decode(video_base64))
    tmp_in.close()
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out.close()
    try:
        await _run_generation(
            engine.upscale_video,
            video_path=tmp_in.name,
            output_path=tmp_out.name,
            target_width=width,
            target_height=height,
            fps=fps,
        )
        b64 = _read_file_b64(tmp_out.name)
        return web.json_response({
            "video_base64": b64,
            "content_type": "video/mp4",
            "generation_id": uuid.uuid4().hex[:8],
        })
    finally:
        os.unlink(tmp_in.name)
        os.unlink(tmp_out.name)


async def handle_retake(req: web.Request) -> web.Response:
    """POST /video-creator/v1/retake — regenerate a video segment with new prompt."""
    body = await req.json()
    assert engine
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
    assert engine
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

    # Apply conditioning
    if conditioning_type == "canny":
        # Simple edge detection via PIL
        from PIL import ImageFilter
        cond_img = Image.fromarray(frames[0]).convert("L")
        cond_img = cond_img.filter(ImageFilter.FIND_EDGES)
        cond_img = cond_img.convert("RGB")
    elif conditioning_type == "depth":
        # Depth conditioning requires local depth model — not feasible remotely
        # Fall back to grayscale
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
    assert engine
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
            resolution = engine.clamp_resolution(resolution)
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


async def on_startup(_app: web.Application) -> None:
    global engine, ready, gpu_profile
    _warn_if_low_host_ram()

    # Pick the CUDA device this worker warms up on. An explicit GPU_DEVICE env
    # always wins; otherwise auto-select the idlest (most-free-VRAM) card so a
    # video worker doesn't collide with the image worker's warm model on GPU 0.
    video_device = await _pick_video_device(GPU_DEVICE)

    # Detect GPU and build the VRAM-aware profile (4090 = streaming/24GB,
    # 5090 = full-resident/32GB, RTX PRO 6000 = full-resident/96GB).
    gpu_profile = build_profile(video_device, GPU_VRAM_GB, GPU_NAME)
    if not gpu_profile.supports_generation:
        logger.error(
            "GPU[%d] %.1f GiB below the %d GiB minimum — generation will fail. "
            "Set GPU_VRAM_GB to bypass.",
            video_device, gpu_profile.vram_gb, 15,
        )
    logger.info("GPU: %s (%.1f GB VRAM, mode=%s)",
                gpu_profile.gpu_name, gpu_profile.vram_gb, gpu_profile.mode)

    # Load inference engine. Prompt enhancement may run on a separate GPU
    # (ENHANCE_GPU_DEVICE); default to the video pipeline's GPU when unset.
    device = torch.device(f"cuda:{video_device}")
    enhance_device = torch.device(f"cuda:{ENHANCE_GPU_DEVICE}") if ENHANCE_GPU_DEVICE else device
    engine = VideoCreatorInferenceEngine(MODEL_CHECKPOINT, TEXT_ENCODER_ROOT, UPSCALER_PATH, device,
                                profile=gpu_profile, enhance_device=enhance_device)
    logger.info("Inference engine ready on %s (mode=%s, max_res=%s)",
                device, gpu_profile.mode, gpu_profile.max_resolution)

    # Report which backend serves /prompt-enhance.
    if ENHANCE_FORWARD_URL:
        # Sharing one OPENAI-compatible enhancer across runners — the local
        # Gemma is never loaded, so a missing TEXT_ENCODER_ROOT is fine here.
        logger.info("Prompt enhancement: forwarded to %s/v1/chat/completions (local Gemma not loaded)",
                    ENHANCE_FORWARD_URL)
    else:
        if enhance_device != device:
            logger.info("Prompt enhancement: local Gemma on %s (video pipeline stays on %s)",
                        enhance_device, device)
        # Prompt enhancement is served by the provisioned Gemma QAT q4_0 text
        # encoder at TEXT_ENCODER_ROOT. Report availability at startup so an
        # unprovisioned box is obvious before /prompt-enhance returns 500s.
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

    # Warmup
    if WARMUP:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            engine.warmup(tmp.name)
            logger.info("Warmup complete")
        finally:
            # warmup() already unlinks its own output; ignore if already gone.
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    # Load Gemma for /prompt-enhance at startup so it's warm (the first enhance
    # won't pay a multi-second cold load). Skips forwarded-enhance mode, where no
    # local Gemma is loaded. Loading Gemma may evict the video pipeline (they
    # share one GPU) — it reloads lazily on the next generation. Never crash the
    # runner if the warmup fails; enhance will load on demand.
    if not ENHANCE_FORWARD_URL and engine is not None:
        _gemma_files = (
            [f.name for f in os.scandir(TEXT_ENCODER_ROOT) if f.is_file()]
            if os.path.isdir(TEXT_ENCODER_ROOT) else []
        )
        if any(n.startswith("model") and n.endswith(".safetensors") for n in _gemma_files):
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, engine.enhance_prompt, "<startup warmup>", None, 0, None
                )
                logger.info("Gemma prompt-enhance model loaded at startup")
            except Exception as exc:
                logger.error("Gemma startup warmup failed (will load on demand): %s", exc)

    ready = True
    logger.info("Runner READY")


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
    app.router.add_post(f"{p}/upscale", handle_upscale)
    app.router.add_post(f"{p}/retake", handle_retake)
    app.router.add_post(f"{p}/prompt-enhance", handle_prompt_enhance)
    app.router.add_post(f"{p}/suggest-gap-prompt", handle_suggest_gap_prompt)
    app.router.add_post(f"{p}/extract-conditioning", handle_extract_conditioning)
    app.router.add_post(f"{p}/ic-lora-generate", handle_ic_lora_generate)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app()
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
