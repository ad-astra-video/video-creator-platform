"""image-worker server — Qwen-Image-Edit / Qwen-Image-Layered / Z-Image.

Serves the /video-creator/v1/* image surface (edit / layer / image) plus the
token-gated worker control plane (/load /evict /health /info) behind the
live-runner edge, mirroring the structure of the runner/ltx worker.

Pure internal worker behind the live-runner. Does NOT register with the
Orchestrator (that is live-runner's job); it only serves the inference surface
and the worker control plane over the Docker network.

The aiohttp route layer imports cleanly WITHOUT torch/diffusers — every GPU
dependency is reached lazily through the engine, so the routes are testable
standalone.
"""
from __future__ import annotations

import asyncio
import functools
import json
import logging
import os
import random
import uuid

from aiohttp import web

from runner.image.config import (
    APP_ID,
    DEFAULT_DEVICE,
    HOST,
    PORT,
    QWEN_LAYER_MAX_INPUT_SIDE,
    QWEN_LAYER_PREVIEW_SIDE,
    QWEN_LAYER_RESPONSE_CAP_BYTES,
    QWEN_MAX_LAYERS,
    worker_token,
)
from runner.image.inference import ImageInferenceEngine
from runner.ltx.gpu_profile import build_profile

logger = logging.getLogger(__name__)

# aiohttp's default client_max_size is 1 MB, which an attached base64 image
# (edit / layer inputs run up to QWEN_LAYER_MAX_INPUT_SIDE) routinely exceeds.
# 3 GB matches go-livepeer's declared MaxAIRequestSize and the ltx worker.
_MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(3 * 1024 ** 3)))

# ── Global state ───────────────────────────────────────────
engine: ImageInferenceEngine | None = None
gpu_profile = None
ready = False

# Serializes generation: the single GPU can only run one generation at a time,
# and with handlers dispatching to a thread pool, concurrent requests must be
# queued rather than racing on the shared engine / CUDA stream.
_generation_lock = asyncio.Lock()


# ── Helpers ────────────────────────────────────────────────

def _require_token(request: web.Request) -> None:
    """Reject the request unless it carries the shared worker token.

    Control-plane endpoints (/load, /evict) REQUIRE the token so only the
    live-runner edge can drive the swap policy. Generation /v1/* endpoints are
    left open for orchestrator proxying in this transitional worker; the
    live-runner applies X-Worker-Token to every upstream call it makes."""
    expected = worker_token()
    provided = request.headers.get("X-Worker-Token", "")
    if not provided or provided != expected:
        raise web.HTTPForbidden(reason="missing/mismatched X-Worker-Token")


def _devices_visible() -> int:
    """Number of visible CUDA devices, or 0 when torch/CUDA isn't available."""
    try:
        import torch  # lazy
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


def _estimate_bytes(obj) -> int:
    """Rough serialized-size upper bound of a JSON-able object: sum of the lengths
    of every string field (base64 payloads dominate), plus a small JSON envelope."""
    total = 0
    if isinstance(obj, dict):
        for v in obj.values():
            total += _estimate_bytes(v)
    elif isinstance(obj, list):
        for v in obj:
            total += _estimate_bytes(v)
    elif isinstance(obj, str):
        total += len(obj)
    return total


async def _run_generation(fn, *args, **kwargs):
    """Run a blocking engine generation off the event loop.

    The engine's generation methods are synchronous (GPU-bound). Dispatching to a
    worker thread (serialized by the generation lock) keeps the asyncio heartbeat
    responsive so the orchestrator never drops the runner."""
    async with _generation_lock:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, functools.partial(fn, *args, **kwargs)
        )


# ── Control-plane routes (token-gated) ─────────────────────

async def handle_load(req: web.Request) -> web.Response:
    """POST /load — set the active CUDA device + ensure the engine is resident.

    Reads ``device`` (int) from the body; when omitted falls back to DEFAULT_DEVICE.
    Kept warm by the live-runner's swap policy; models themselves load lazily on
    the first generation."""
    _require_token(req)
    global engine
    body = await req.json()
    device = body.get("device")
    if device is None:
        device = DEFAULT_DEVICE
    device = int(device)
    # Clamp the scheduler's device index to THIS container's visible devices
    # (e.g. a device_ids-pinned container exposes only cuda:0, so a scheduler
    # logical index of 1/2 must fall back to 0 rather than index-out-of-range).
    visible = _devices_visible()
    if visible > 0:
        device = max(0, min(device, visible - 1))
    if engine is not None:
        engine.current_device = device
    logger.info("image-worker /load: device=%s (visible=%d)", device, visible)
    return web.json_response({"loaded": True, "ready": ready, "device": device})


async def handle_evict(req: web.Request) -> web.Response:
    """POST /evict — drop all resident pipelines + free GPU memory.

    Called by the live-runner before it swaps in another worker's model on the
    shared GPU. The next /v1/* generation reloads lazily."""
    _require_token(req)
    assert engine is not None
    engine.free()
    return web.json_response({"evicted": True})


# ── Info / health ──────────────────────────────────────────

async def handle_health(_req: web.Request) -> web.Response:
    return web.json_response({"ok": True, "ready": ready, "app": APP_ID})


async def handle_info(_req: web.Request) -> web.Response:
    global engine
    profile_info = {}
    if gpu_profile is not None:
        profile_info = gpu_profile.info
    device_in_use = engine.current_device if engine is not None else None
    if device_in_use is None:
        device_in_use = DEFAULT_DEVICE
    return web.json_response({
        "app": APP_ID,
        "capabilities": ["image", "edit", "layer", "style-frame"],
        "models": ["z-image", "flux", "qwen", "hidream"],
        "ready": ready,
        "gpu": profile_info,
        "devices_visible": _devices_visible(),
        "device_in_use": device_in_use,
    })


# ── Generation routes ──────────────────────────────────────

async def _run_edit_sse(req: web.Request, body: dict) -> web.StreamResponse:
    """Serve /video-creator/v1/edit as text/event-stream when ``?sse=1``.

    Events: accepted -> progress* (per denoise step) -> complete ({image b64}),
    or error. Mirrors ``_run_layer_sse`` so the frontend can show a thin progress
    bar while a Qwen-Image-Edit (whole-frame or masked inpaint) runs."""
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(req)
    assert engine
    prompt = str(body.get("prompt", ""))
    engine_name = str(body.get("engine", "qwen-edit")).lower()

    async def _ev(event: str, data: dict) -> None:
        try:
            await resp.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
        except Exception:
            pass

    kw = {}
    for k in ("width", "height", "num_inference_steps", "guidance_scale", "seed"):
        if body.get(k) is not None:
            kw[k] = body.get(k)
    if "num_inference_steps" not in kw:
        q = str(body.get("quality") or "").strip().lower()
        if q in ("fast", "balanced", "high"):
            if engine_name == "hidream":
                from runner.image.inference import HIDREAM_STEP_PRESETS
                presets = HIDREAM_STEP_PRESETS
            else:
                presets = (QWEN_EDIT_STEP_PRESETS if engine_name == "qwen-edit"
                           else ZIMAGE_STEP_PRESETS)
            kw["num_inference_steps"] = presets[q]
    steps = kw.get("num_inference_steps") or QWEN_EDIT_STEP_PRESETS["balanced"]

    await _ev("accepted", {
        "engine": engine_name,
        "num_inference_steps": steps,
        "strength": body.get("strength", 0.6),
        "masked": bool(body.get("mask_image")),
    })
    loop = asyncio.get_running_loop()
    total = int(steps)

    def _progress(step: int, t: int):
        loop.call_soon_threadsafe(
            asyncio.create_task, _ev("progress", {"step": step, "total_steps": t})
        )

    async with _generation_lock:
        try:
            if engine_name == "hidream":
                img = await loop.run_in_executor(
                    None, functools.partial(
                        engine.hidream_edit, body["image"], prompt,
                        seed=kw.get("seed"),
                        num_inference_steps=kw.get("num_inference_steps"),
                        quality=body.get("quality"), progress_cb=_progress,
                    )
                )
            else:
                img = await loop.run_in_executor(
                    None, functools.partial(
                        engine.edit_image,
                        body["image"], prompt,
                        engine=engine_name,
                        mask=body.get("mask_image"),
                        keep_subject=bool(body.get("keep_subject", False)),
                        strength=float(body.get("strength", 0.6)),
                        padding_mask_crop=int(body.get("padding_mask_crop", 0) or 0),
                        progress_cb=_progress,
                        **kw,
                    )
                )
        except Exception as exc:
            logger.exception("edit SSE failed")
            await _ev("error", {"error": str(exc)})
            await resp.write_eof()
            return resp

    from runner.image.inference import _pil_to_b64  # cheap
    await _ev("complete", {
        "image": _pil_to_b64(img),
        "content_type": "image/png",
        "engine": engine_name,
    })
    await resp.write_eof()
    return resp


async def handle_edit(req: web.Request) -> web.Response:
    """POST /video-creator/v1/edit.

    Request:  {image: <b64 png>, prompt: str, engine?: "qwen-edit"|"zimage",
               mask_image?: <b64>, keep_subject?: bool, strength?: float,
               width?: int, height?: int, ...pipeline kwargs}
    Response: {image: <base64 png>, content_type: "image/png", engine}
    """
    body = await req.json()
    if not body.get("image"):
        return web.json_response({"error": "missing 'image' (base64)"}, status=400)
    if req.query.get("sse") in ("1", "true", "yes"):
        return await _run_edit_sse(req, body)
    assert engine
    image = body.get("image")
    prompt = str(body.get("prompt", ""))
    engine_name = str(body.get("engine", "qwen-edit")).lower()
    if engine_name not in ("qwen-edit", "zimage", "hidream"):
        return web.json_response(
            {"error": f"unknown edit engine '{engine_name}' (expected qwen-edit|zimage|hidream)"},
            status=400,
        )
    mask = body.get("mask_image")
    keep_subject = bool(body.get("keep_subject", False))
    strength = float(body.get("strength", 0.6))
    padding_mask_crop = int(body.get("padding_mask_crop", 0) or 0)
    kw = {}
    for k in ("width", "height", "num_inference_steps", "guidance_scale", "seed"):
        if body.get(k) is not None:
            kw[k] = body.get(k)
    # Bare quality name -> this engine's step count (client threads quality;
    # worker translates). Explicit num_inference_steps wins over quality.
    if "num_inference_steps" not in kw:
        q = str(body.get("quality") or "").strip().lower()
        if q in ("fast", "balanced", "high"):
            presets = (QWEN_EDIT_STEP_PRESETS if engine_name == "qwen-edit"
                       else ZIMAGE_STEP_PRESETS)
            kw["num_inference_steps"] = presets[q]

    from runner.image.inference import _pil_to_b64  # cheap, no torch
    if engine_name == "hidream":
        img = await _run_generation(
            engine.hidream_edit, image, prompt,
            seed=kw.pop("seed", None),
            num_inference_steps=kw.pop("num_inference_steps", None),
            quality=body.get("quality"),
        )
    else:
        img = await _run_generation(
            engine.edit_image,
            image, prompt,
            engine=engine_name,
            mask=mask,
            keep_subject=keep_subject,
            strength=strength,
            padding_mask_crop=padding_mask_crop,
            **kw,
        )
    return web.json_response({
        "image": _pil_to_b64(img),
        "content_type": "image/png",
        "engine": engine_name,
    })


def _parse_layer_body(body: dict):
    """Validate + coerce the /layer request body -> (layers, resolution, steps, preview_only).

    Raises ValueError on an invalid ``layers`` / ``num_inference_steps`` value so the
    caller can respond 400; returns defaults when fields are absent."""
    image = body.get("image")
    layers = body.get("layers")
    if layers is not None:
        try:
            layers = int(layers)
        except (TypeError, ValueError):
            raise ValueError("'layers' must be an integer")
        if layers < 2 or layers > QWEN_MAX_LAYERS:
            raise ValueError(f"'layers' must be in [2, {QWEN_MAX_LAYERS}]")
    steps = body.get("num_inference_steps")
    if steps is not None:
        try:
            steps = int(steps)
        except (TypeError, ValueError):
            raise ValueError("'num_inference_steps' must be an integer")
        if steps < 1:
            raise ValueError("'num_inference_steps' must be >= 1")
    resolution = body.get("resolution")
    preview_only = bool(body.get("preview_only", False))
    return layers, resolution, steps, preview_only


async def _run_layer_sse(req: web.Request, body: dict) -> web.StreamResponse:
    """Serve /layer as text/event-stream: accepted -> progress* (per denoise step)
    -> complete (full /layer contract JSON), or error."""
    layers, resolution, steps, preview_only = _parse_layer_body(body)
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(req)

    async def _ev(event: str, data: dict) -> None:
        try:
            await resp.write(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode("utf-8"))
        except Exception:
            pass

    await _ev("accepted", {
        "engine": "layer",
        "layers": layers,
        "num_inference_steps": steps,
        "resolution": resolution,
    })
    loop = asyncio.get_running_loop()
    total = steps or 30

    def _progress(step: int, t: int):
        loop.call_soon_threadsafe(
            asyncio.create_task, _ev("progress", {"step": step, "total_steps": t})
        )

    assert engine
    async with _generation_lock:
        try:
            result = await loop.run_in_executor(
                None, functools.partial(
                    engine.layered_decompose, body["image"], layers, resolution,
                    preview_only, steps, _progress,
                )
            )
        except Exception as exc:
            logger.exception("layer SSE failed")
            await _ev("error", {"error": str(exc)})
            await resp.write_eof()
            return resp

    if _estimate_bytes(result) > QWEN_LAYER_RESPONSE_CAP_BYTES:
        await _ev("error", {"error": "layer response exceeds QWEN_LAYER_RESPONSE_CAP_BYTES"})
    else:
        await _ev("complete", result)
    await resp.write_eof()
    return resp


async def handle_layer(req: web.Request) -> web.Response:
    """POST /video-creator/v1/layer.

    Request: {image: <b64 png>, layers?: int (clamped to [2, QWEN_MAX_LAYERS]),
              resolution?: str/int, num_inference_steps?: int, preview_only?: bool}
    Response: the /layer contract
              {layers: [{index, rgba_b64, preview_b64, alpha_b64, label}],
               composite: <b64>, width, height, layers_requested}
    When the query has ``?sse=1`` the response is a text/event-stream:
    accepted -> progress* (per denoise step) -> complete, or error.

    Validation: `layers` outside [2, QWEN_MAX_LAYERS] -> 400; a projected response
    larger than QWEN_LAYER_RESPONSE_CAP_BYTES -> 413. Input is clamped to
    QWEN_LAYER_MAX_INPUT_SIDE by the engine.
    """
    body = await req.json()
    if not body.get("image"):
        return web.json_response({"error": "missing 'image' (base64)"}, status=400)

    want_sse = req.query.get("sse") in ("1", "true", "yes")
    if want_sse:
        return await _run_layer_sse(req, body)

    try:
        layers, resolution, steps, preview_only = _parse_layer_body(body)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)

    result = await _run_generation(
        engine.layered_decompose, body["image"], layers, resolution, preview_only, steps
    )

    # Reject an oversized projected response (base64 payload dominates bytes).
    if _estimate_bytes(result) > QWEN_LAYER_RESPONSE_CAP_BYTES:
        return web.json_response(
            {"error": "layer response exceeds QWEN_LAYER_RESPONSE_CAP_BYTES"},
            status=413,
        )
    return web.json_response(result)


# Text-to-image quality presets -> FLUX.2 [klein] step counts. Klein is
# step-distilled at 4 steps (BFL); the "high" preset goes slightly above to give
# marginal refinement without the O(n^2) of a full-CFG run. These slot in under
# an explicit num_inference_steps override from the client.
KLEIN_STEP_PRESETS = {"fast": 4, "balanced": 8, "high": 12}

# Quality preset -> denoise steps, PER MODEL. The frontend threads a bare
# quality name ("fast"|"balanced"|"high") and the worker translates it here so
# clients don't hardcode engine-specific step counts.
#   qwen-edit:  Qwen-Image-Edit  (25/35/50)
#   klein:      FLUX.2 klein 4B  (step-distilled at 4; mild refinement above)
#   zimage:     Z-Image Turbo    (native distilled 4)
QWEN_EDIT_STEP_PRESETS = {"fast": 25, "balanced": 35, "high": 50}
KLEIN_STEP_PRESETS = {"fast": 4, "balanced": 8, "high": 12}
ZIMAGE_STEP_PRESETS = {"fast": 4, "balanced": 8, "high": 12}


def _quality_steps(quality, presets, fallback: int) -> int:
    """Resolve an explicit quality preset to steps; ``None`` when no preset maps."""
    q = str(quality or "").strip().lower()
    return presets.get(q, fallback) if q in presets else fallback

def _image_klein_steps(body) -> int:
    """Resolve the number of klein T2I steps for a request body.

    Priority: explicit num_inference_steps > the 'fast'/'balanced'/'high'
    'quality' preset > engine default (4). Returns an int.
    """
    if body.get("num_inference_steps") is not None:
        return int(body["num_inference_steps"])
    q = str(body.get("quality") or "").strip().lower()
    if q in KLEIN_STEP_PRESETS:
        return KLEIN_STEP_PRESETS[q]
    return 4


async def handle_image(req: web.Request) -> web.Response:
    """POST /video-creator/v1/image.

    Request:  {prompt: str, engine?: "zimage"|"klein", width?: int, height?: int,
               num_inference_steps?: int, quality?: "fast"|"balanced"|"high",
               guidance_scale?: float, seed?: int}
    Response: {image: <base64 png>, content_type: "image/png", engine,
               num_inference_steps?}
    """
    body = await req.json()
    assert engine
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        return web.json_response({"error": "missing 'prompt'"}, status=400)
    engine_name = str(body.get("engine", "zimage")).lower()
    if engine_name not in ("zimage", "klein", "hidream"):
        return web.json_response(
            {"error": f"unknown image engine '{engine_name}' (expected zimage|klein|hidream)"},
            status=400,
        )
    from runner.image import config as _cfg
    if engine_name == "klein" and not _cfg.klein4b_enabled():
        return web.json_response(
            {"error": "FLUX.2 Klein 4B not provisioned (KLEIN4B_MODEL missing)"},
            status=503,
        )
    kw = {}
    for k in ("width", "height"):
        if body.get(k) is not None:
            # diffusers ZImagePipeline (and other image backends) require
            # spatial dims divisible by 16; snap to the nearest multiple so
            # any client resolution (e.g. raw 1080p -> 1920x1080) just works
            # instead of 500ing with "Height must be divisible by 16".
            kw[k] = round(int(body.get(k)) / 16) * 16
    # Resolve the seed: client-supplied passes through (deterministic replay);
    # otherwise mint a fresh random one so each bare request differs. Forwarded
    # to BOTH engines so the response can truthfully report which seed ran.
    seed = int(body["seed"]) if body.get("seed") is not None else random.randrange(0, 2**31 - 1)
    kw["seed"] = seed
    # Classic Z-Image Turbo: default 4 steps (its distilled native). Forward
    # step/guidance to Z-Image when the client sends a snake-case field.
    if body.get("num_inference_steps") is not None:
        kw["num_inference_steps"] = int(body["num_inference_steps"])
    if body.get("guidance_scale") is not None:
        kw["guidance_scale"] = float(body["guidance_scale"])

    from runner.image.inference import _pil_to_b64  # cheap, no torch
    if engine_name == "hidream":
        kw.pop("num_inference_steps", None)
        kw.pop("guidance_scale", None)
        img = await _run_generation(
            engine.hidream_image, prompt,
            num_inference_steps=body.get("num_inference_steps"),
            guidance_scale=body.get("guidance_scale"),
            quality=body.get("quality"), progress_cb=None, **kw,
        )
        resp = {
            "image": _pil_to_b64(img),
            "content_type": "image/png",
            "engine": engine_name,
            "seed": seed,
        }
        if body.get("num_inference_steps") is not None:
            resp["num_inference_steps"] = int(body["num_inference_steps"])
        return web.json_response(resp)
    elif engine_name == "klein":
        steps = _image_klein_steps(body)
        # `steps` already consumes the body's num_inference_steps; don't also let
        # it ride in **kw or klein_image gets it twice (TypeError: multiple values).
        kw.pop("num_inference_steps", None)
        img = await _run_generation(
            engine.klein_image, prompt, num_inference_steps=steps, **kw,
        )
    else:
        # Z-Image Turbo: explicit num_inference_steps wins; else map the
        # client quality preset (fast|balanced|high) through the per-model
        # map so Fast/Balanced/High select 4/8/12 instead of always 4.
        if "num_inference_steps" not in kw:
            q = str(body.get("quality") or "").strip().lower()
            if q in ZIMAGE_STEP_PRESETS:
                kw["num_inference_steps"] = ZIMAGE_STEP_PRESETS[q]
        img = await _run_generation(engine.plain_image, prompt, **kw)
    resp = {
        "image": _pil_to_b64(img),
        "content_type": "image/png",
        "engine": engine_name,
        "seed": seed,
    }
    if engine_name == "klein":
        resp["num_inference_steps"] = _image_klein_steps(body)
    elif engine_name == "zimage" and "num_inference_steps" in kw:
        resp["num_inference_steps"] = kw["num_inference_steps"]
    return web.json_response(resp)

async def handle_style_frame(req: web.Request) -> web.Response:
    """POST /video-creator/v1/style-frame — style an image with FLUX.2 klein 4B.

    Request:  {image: <b64 png>, prompt: str, seed?: int, width?: int, height?: int,
               enhance_prompt?: bool}
    Response: {styled_image: <base64 png>, width, height, enhanced_prompt?}

    The restyle first-frame styling rail. ``enhance_prompt`` is accepted for the
    shared contract but the image-worker has no Gemma LLM, so the prompt is used
    verbatim (enhanced_prompt is omitted from the response).
    """
    body = await req.json()
    assert engine
    image = body.get("image")
    if not image:
        return web.json_response({"error": "missing 'image' (base64)"}, status=400)
    prompt = str(body.get("prompt", "")).strip()
    if not prompt:
        return web.json_response({"error": "missing 'prompt'"}, status=400)
    seed = int(body.get("seed", 123))
    width = body.get("width")
    height = body.get("height")
    steps = body.get("num_inference_steps")
    if steps is not None:
        steps = int(steps)
    else:
        q = str(body.get("quality") or "").strip().lower()
        if q in ("fast", "balanced", "high"):
            steps = KLEIN_STEP_PRESETS[q]

    # Composition-hold suffix so the styled frame keeps the source's layout.
    composition_hold = (
        ", keeping the exact same camera angle, framing, composition, subject scale "
        "and position as the input image — change only the style and materials"
    )

    from runner.image.inference import _pil_to_b64  # cheap, no torch
    from runner.image import config as _cfg
    if not _cfg.klein4b_enabled():
        return web.json_response(
            {"error": "FLUX.2 Klein 4B not provisioned (KLEIN4B_MODEL missing)"},
            status=503,
        )
    img = await _run_generation(
        engine.style_frame, image, prompt + composition_hold, seed, width, height,
        num_inference_steps=steps,
    )
    return web.json_response({
        "styled_image": _pil_to_b64(img),
        "width": img.size[0],
        "height": img.size[1],
    })


# ── Lifecycle ──────────────────────────────────────────────

async def on_startup(_app: web.Application) -> None:
    global engine, gpu_profile, ready
    # Detect GPU and build the VRAM-aware profile (reuses runner.ltx.gpu_profile,
    # which is pure-Python and builds fine even without torch/CUDA present).
    gpu_profile = build_profile(DEFAULT_DEVICE)
    engine = ImageInferenceEngine(profile=gpu_profile)
    if engine is not None:
        engine.current_device = DEFAULT_DEVICE
    logger.info(
        "image worker on GPU[%d] %s (%.1f GB VRAM, mode=%s)",
        DEFAULT_DEVICE,
        gpu_profile.gpu_name,
        gpu_profile.vram_gb,
        gpu_profile.mode,
    )
    ready = True
    logger.info("image worker READY")


async def on_cleanup(_app: web.Application) -> None:
    # Nothing to unregister: this is a pure worker behind live-runner.
    pass


# ── App factory ────────────────────────────────────────────

def create_app() -> web.Application:
    # client_max_size=3GB: an attached base64 image for /edit or /layer routinely
    # exceeds aiohttp's 1 MB default.
    app = web.Application(client_max_size=_MAX_BODY_BYTES)
    # Worker control surface (token-gated) — drives the swap policy from the
    # live-runner edge, served at root /load /evict like the ltx/idv2v workers.
    app.router.add_post("/load", handle_load)
    app.router.add_post("/evict", handle_evict)
    app.router.add_get("/health", handle_health)
    p = "/video-creator/v1"
    app.router.add_get(f"{p}/health", handle_health)
    app.router.add_get(f"{p}/info", handle_info)
    app.router.add_post(f"{p}/edit", handle_edit)
    app.router.add_post(f"{p}/layer", handle_layer)
    app.router.add_post(f"{p}/image", handle_image)
    app.router.add_post(f"{p}/style-frame", handle_style_frame)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app()
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
