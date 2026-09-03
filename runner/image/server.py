"""image-worker server — Qwen-Image-Edit / Qwen-Image-Layered / Z-Image.

Serves the /video-creator/v1/* image surface (edit / layer / image) plus the
token-gated worker control plane (/load /evict /health /info) behind the
live-runner edge, mirroring the structure of the runner/ltx worker.

Pure internal worker behind the live-runner. Does NOT register with the
Orchestrator (that is live-runner's job); it only serves the inference surface
and the worker control plane over the Docker network.

The aiohttp route layer imports cleanly WITHOUT torch/diffusers — every GPU
dependency lives in a child MODEL SUBPROCESS (``runner.image.engine_cli``),
so the routes are testable standalone and the PARENT process stays CUDA-free.

MODEL SUBPROCESS architecture: the real ``ImageInferenceEngine`` (and the
FLUX.2 klein singleton) run in a DEDICATED child python process that owns the
CUDA primary context. The parent holds one ``ImageEngineProxy`` per resident
CUDA device, each owning a child via ``runner.common.engineproc.EngineProc``.
/load spawns the child; /evict TERMINATES it — killing the subprocess is the
only way to destroy a CUDA context (PyTorch has no in-process teardown API),
so an evicted worker leaves NO primary-context floor on the GPU.

MULTI-ENGINE (one request per GPU at a time): each device's proxy has its own
serialization lock, so concurrent requests on DIFFERENT GPUs run in parallel
on their own subprocesses. The live-runner's scheduler picks the GPU and
forwards it via the ``X-Worker-Device`` header; when absent (backward compat)
the last /load'ed device is used.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import uuid

from aiohttp import web

from runner.common import engineproc
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
from runner.image.inference import _decoded_pil as _pil_from_b64
from runner.image.inference import _pil_to_b64
from runner.ltx.gpu_profile import build_profile

logger = logging.getLogger(__name__)

# aiohttp's default client_max_size is 1 MB, which an attached base64 image
# (edit / layer inputs run up to QWEN_LAYER_MAX_INPUT_SIDE) routinely exceeds.
# 3 GB matches go-livepeer's declared MaxAIRequestSize and the ltx worker.
_MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(3 * 1024 ** 3)))

# ── Global state ───────────────────────────────────────────
gpu_profile = None
ready = False

# MULTI-ENGINE: one ImageEngineProxy per resident CUDA device + one lock per
# device, so concurrent requests on DIFFERENT GPUs run in parallel (one request
# per GPU at a time). Each proxy owns a MODEL SUBPROCESS (runner.image.engine_cli)
# that holds the real ImageInferenceEngine and its CUDA context. Proxies are
# created lazily on /load; the child process is spawned on /load (ensure_loaded)
# or on the first generation (EngineProc.run auto-starts when absent). /evict
# TERMINATES the child, which is what actually destroys the CUDA context.
_engines: dict[int, "ImageEngineProxy"] = {}
_device_locks: dict[int, asyncio.Lock] = {}
_default_device: int | None = None

# FLUX.2 klein runs inside whatever child handles the klein request (the
# flux_edit singleton lives in the CHILD process, one GPU at a time) — so klein
# generation is serialized GLOBALLY across devices regardless of the per-device
# spread. qwen-edit / z-image / hidream engines are per-instance and DO spread
# across GPUs (each in its own child).
_klein_lock = asyncio.Lock()


class ImageEngineProxy:
    """Parent-side stand-in for ``ImageInferenceEngine`` that delegates to a
    model subprocess via ``engineproc.EngineProc`` (JSONL over the child's
    stdin/stdout).

    Generation methods mirror the engine's public signatures and return the SAME
    types the server handlers expect (``PIL.Image`` for the *_image / edit /
    style_frame methods, the pre-serialized /layer dict for layered_decompose),
    so the HTTP + SSE handlers are otherwise unchanged. The proxy NEVER touches
    CUDA — it only serializes JSON to/from the child. The child is where the GPU
    engine, the CUDA context, and the FLUX.2 klein singleton live.
    """

    _STARTUP_TIMEOUT = int(os.environ.get("IMAGE_CHILD_STARTUP_TIMEOUT", "900"))
    _JOB_TIMEOUT = int(os.environ.get("IMAGE_CHILD_JOB_TIMEOUT", "900"))

    def __init__(self, device: int) -> None:
        self.current_device = int(device)
        self.ready = False
        # ``sys.executable -m runner.image.engine_cli --device N`` — run through
        # the same interpreter that launched the server, so the child gets the
        # same venv, packages, and (via PYTHONPATH) the same repo root module
        # layout. In the container WORKDIR=/app is on sys.path for -m, but we
        # pass PYTHONPATH explicitly so the child also resolves ``runner.*``
        # when the server was launched from elsewhere.
        self._proc = engineproc.EngineProc(
            label=f"image-{device}",
            argv=[
                sys.executable, "-m", "runner.image.engine_cli",
                "--device", str(self.current_device),
            ],
            startup_timeout=self._STARTUP_TIMEOUT,
            job_timeout=self._JOB_TIMEOUT,
            env_extra={"PYTHONPATH": _child_pythonpath()},
        )

    # ── lifecycle ──────────────────────────────────────────────────────────
    @property
    def started(self) -> bool:
        """Whether the backing subprocess is alive (its CUDA context resident)."""
        return bool(getattr(self._proc, "_ready", False))

    async def ensure_loaded(self) -> None:
        """Spawn (or re-spawn) the model subprocess and wait for its ready
        handshake (engine shell built; weights still lazy)."""
        await self._proc.start()
        self.ready = True

    async def stop(self) -> None:
        """Terminate the model subprocess — destroying its CUDA primary
        context. This is the /evict mechanism. Idempotent."""
        await self._proc.stop()
        self.ready = False

    # ── dispatch ───────────────────────────────────────────────────────────
    async def _dispatch(self, op: str, args: dict | None,
                        progress_cb=None, timeout: float | None = None):
        """Send one op to the child. ``progress_cb`` (server's 2-arg SSE shape
        ``(step, total)``) is adapted to the child's progress line
        ``{"step":.., "total_steps":..}``."""
        child_pb = None
        if progress_cb is not None:
            def child_pb(line):
                try:
                    progress_cb(
                        int(line.get("step", 0)),
                        int(line.get("total_steps", 0)),
                    )
                except Exception:  # noqa: BLE001 - never break on progress
                    pass
        return await self._proc.run(op, args, timeout=timeout, progress_cb=child_pb)

    async def query_op(self, op: str, args: dict | None = None,
                       timeout: float | None = None):
        """Run an op but ONLY when the child is already resident (never spawns a
        child to answer a read, e.g. /info's klein residency probe)."""
        if not self.started:
            return None
        return await self._dispatch(op, args, timeout=timeout)

    # ── engine-method mirror (return the same types the handlers expect) ──
    async def edit_image(self, image, prompt, engine="qwen-edit", mask=None,
                         keep_subject=False, strength=0.6, padding_mask_crop=0,
                         mask_composite=True, progress_cb=None, **kw):
        res = await self._dispatch("edit_image", {
            "image": image, "prompt": prompt, "engine": engine, "mask": mask,
            "keep_subject": bool(keep_subject), "strength": float(strength),
            "padding_mask_crop": int(padding_mask_crop or 0),
            "mask_composite": bool(mask_composite), "kw": dict(kw),
        }, progress_cb)
        return _pil_from_b64(res["image_b64"])

    async def hidream_edit(self, image, prompt, seed=None, keep_original_aspect=True,
                           num_inference_steps=None, quality=None, progress_cb=None,
                           **kw):
        res = await self._dispatch("hidream_edit", {
            "image": image, "prompt": prompt, "seed": seed,
            "keep_original_aspect": bool(keep_original_aspect),
            "num_inference_steps": num_inference_steps, "quality": quality,
            "kw": dict(kw),
        }, progress_cb)
        return _pil_from_b64(res["image_b64"])

    async def hidream_image(self, prompt, width=1024, height=1024, seed=None,
                            num_inference_steps=None, guidance_scale=None,
                            quality=None, progress_cb=None, **kw):
        res = await self._dispatch("hidream_image", {
            "prompt": prompt, "width": width, "height": height, "seed": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale, "quality": quality,
            "kw": dict(kw),
        }, progress_cb)
        return _pil_from_b64(res["image_b64"])

    async def plain_image(self, prompt, **kw):
        res = await self._dispatch(
            "plain_image", {"prompt": prompt, "kw": dict(kw)})
        return _pil_from_b64(res["image_b64"])

    async def klein_image(self, prompt, seed=123, width=1024, height=1024,
                          num_inference_steps=None, **kw):
        res = await self._dispatch("klein_image", {
            "prompt": prompt, "seed": seed, "width": width, "height": height,
            "num_inference_steps": num_inference_steps, "kw": dict(kw),
        })
        return _pil_from_b64(res["image_b64"])

    async def style_frame(self, image, prompt, seed=123, width=None, height=None,
                          num_inference_steps=None):
        res = await self._dispatch("style_frame", {
            "image": image, "prompt": prompt, "seed": seed,
            "width": width, "height": height,
            "num_inference_steps": num_inference_steps,
        })
        return _pil_from_b64(res["image_b64"])

    async def layered_decompose(self, image, layers=None, resolution=None,
                                preview_only=False, num_inference_steps=None,
                                progress_cb=None):
        # The child returns the /layer contract dict already — pass straight
        # through (no image decode).
        return await self._dispatch("layered_decompose", {
            "image": image, "layers": layers, "resolution": resolution,
            "preview_only": bool(preview_only),
            "num_inference_steps": num_inference_steps,
        }, progress_cb)


def _child_pythonpath() -> str:
    """PYTHONPATH for the model child: the server's cwd (repo root / /app) plus
    any parent PYTHONPATH, so ``-m runner.image.engine_cli`` resolves ``runner``
    regardless of launch cwd."""
    parts = [p for p in (os.getcwd(), os.environ.get("PYTHONPATH", "")) if p]
    return os.pathsep.join(parts)


def _device_lock(device: int) -> asyncio.Lock:
    """Per-device lock (created on demand) serializing requests on that GPU."""
    return _device_locks.setdefault(device, asyncio.Lock())


def _resolve_device(req: web.Request) -> int:
    """Target CUDA device for a request.

    The live-runner's scheduler picks the GPU and forwards it as
    ``X-Worker-Device``; that is authoritative. When absent (backward compat:
    direct / single-device use) fall back to the last /load'ed device, then to
    DEFAULT_DEVICE.
    """
    hdr = req.headers.get("X-Worker-Device", "")
    if hdr.strip():
        try:
            return int(hdr.strip())
        except ValueError:
            pass
    if _default_device is not None:
        return _default_device
    return DEFAULT_DEVICE


def _engine_for(device: int) -> "ImageEngineProxy":
    """Return (creating on demand) the model-subprocess proxy for CUDA ``device``.

    Pure object construction + EngineProc handle — NO CUDA work happens here and
    nothing is spawned until /load (``ensure_loaded``) or the first generation
    (EngineProc.run auto-starts the child). The parent process never touches the
    GPU; the child owns the engine and its context.
    """
    device = int(device)
    if device not in _engines:
        e = ImageEngineProxy(device)
        e.current_device = device
        _engines[device] = e
    return _engines[device]


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
    """Number of visible CUDA devices, or 0 when torch/CUDA isn't available.

    ``torch.cuda.device_count()`` is a context-FREE driver query (it neither
    creates nor pins a CUDA primary context), used only to clamp the scheduler's
    device index to this container's visible cards. All context-bearing CUDA
    work happens in the model subprocess, never here.
    """
    try:
        import torch  # lazy
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


async def _klein_resident_device() -> int | None:
    """CUDA index the FLUX.2 klein singleton is resident on (None if not loaded).

    klein lives inside a MODEL SUBPROCESS (the ``flux_edit`` singleton exists
    only in the child that handled the last klein request), so we ask each
    already-resident proxy's child via the ``klein_resident_device`` op. The
    parent itself never imports the flux_edit module or touches
    CUDA. Proxies whose child isn't running are skipped (no klein there).
    """
    for device in sorted(_engines):
        proxy = _engines[device]
        try:
            res = await proxy.query_op("klein_resident_device", timeout=5)
        except Exception:
            continue
        if isinstance(res, int):
            return res
    return None


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


async def _run_generation(fn, device: int, *args, **kwargs):
    """Run a generation op on ``device`` through the model-subprocess proxy.

    ``fn`` is an async ``ImageEngineProxy`` method. The per-device lock
    serializes the requests on that GPU (two concurrent requests on one device
    must not both write the child's stdin); requests on DIFFERENT devices use
    DIFFERENT locks and therefore run concurrently, in parallel on their own
    subprocesses. The await itself is non-blocking (I/O over the child pipe), so
    the asyncio heartbeat stays responsive.
    """
    async with _device_lock(device):
        return await fn(*args, **kwargs)


# ── Control-plane routes (token-gated) ─────────────────────

async def handle_load(req: web.Request) -> web.Response:
    """POST /load — ensure an engine is resident for the CUDA ``device``.

    Reads ``device`` (int) from the body; when omitted falls back to DEFAULT_DEVICE.
    The engine instance is created/kept for that device (models themselves load
    lazily on the first generation). The live-runner's swap policy drives this
    per card, one engine per resident GPU.
    """
    _require_token(req)
    global _default_device
    body = await req.json()
    # ``model`` is an advisory warm/cache hint: image-worker selects the actual
    # model per-request from the body's ``engine`` (z-image / flux / qwen /
    # hidream) on the first generation, so /load only pins the GPU device here.
    model_hint = body.get("model")
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
    e = _engine_for(device)
    e.current_device = device
    # Spawn the model subprocess now so the card is actually claimed (the child
    # builds its engine shell; weights stay lazy until the first generation).
    # The engine/build no longer happens in THIS (parent) process at all.
    await e.ensure_loaded()
    _default_device = device
    logger.info("image-worker /load: device=%s model=%s (visible=%d, engines=%s)",
                device, model_hint, visible, sorted(_engines))
    return web.json_response({"loaded": True, "ready": ready, "device": device, "model": model_hint})


async def handle_evict(req: web.Request) -> web.Response:
    """POST /evict — KILL the model subprocess(es), destroying their CUDA
    contexts and freeing the GPU(s).

    ``{"device": N}`` frees only that ONE card of a multi-resident worker so the
    copies still warm on other cards survive; no device (legacy) frees ALL
    engines. Called by the live-runner before it swaps in another worker's model
    on that GPU. Killing the child is the ONLY way to destroy a CUDA primary
    context (PyTorch cannot do it in-process); the next /v1/* generation
    re-spawns a fresh child lazily. The FLUX.2 klein singleton lives inside the
    child too, so stopping it also frees klein's VRAM — no separate parent-side
    flux_edit eviction is needed.
    """
    _require_token(req)
    try:
        body = await req.json()
    except Exception:
        body = {}
    device = body.get("device")
    if device is None:
        for proxy in list(_engines.values()):
            await proxy.stop()
        _engines.clear()
        _device_locks.clear()
        logger.info("image-worker /evict: all model subprocesses stopped")
    else:
        device = int(device)
        proxy = _engines.pop(device, None)
        if proxy is not None:
            # Terminate the child -> its CUDA context is destroyed, freeing the
            # GPU (and any klein editor resident in it).
            await proxy.stop()
        _device_locks.pop(device, None)
        logger.info("image-worker /evict: device %d subprocess stopped", device)
    return web.json_response({"evicted": True})


# ── Info / health ──────────────────────────────────────────

async def handle_health(_req: web.Request) -> web.Response:
    return web.json_response({"ok": True, "ready": ready, "app": APP_ID})


async def handle_info(_req: web.Request) -> web.Response:
    profile_info = {}
    if gpu_profile is not None:
        profile_info = gpu_profile.info
    device_in_use = None
    if _engines:
        device_in_use = _default_device if _default_device is not None else min(_engines)
    if device_in_use is None:
        device_in_use = DEFAULT_DEVICE
    # The FLUX.2 klein editor is a process-global SINGLETON (one GPU at a time),
    # so its residency is NOT in the per-device _engines dict. Surface its actual
    # GPU so the scheduler's reconcile sees every card this worker owns — the
    # missing piece that let a style-frame park 18.6 GiB on a card the map
    # thought was free (co-residency when a video task later took it).
    klein_dev = await _klein_resident_device()
    if klein_dev is not None:
        device_in_use = device_in_use if device_in_use is not None else klein_dev
        devs = set(_engines) | {klein_dev}
    else:
        devs = set(_engines)
    return web.json_response({
        "app": APP_ID,
        "capabilities": ["image", "edit", "layer", "style-frame"],
        "models": ["z-image", "flux", "qwen", "hidream"],
        "ready": ready,
        "gpu": profile_info,
        "devices_visible": _devices_visible(),
        # Multi-resident: the scheduler's reconcile needs the full device SET so
        # it can mark several slots as owned by this worker.
        "device_in_use": device_in_use,
        "devices": sorted(devs),
        "klein_device": klein_dev,
    })


# ── Generation routes ──────────────────────────────────────

async def _run_edit_sse(req: web.Request, body: dict) -> web.StreamResponse:
    """Serve /video-creator/v1/edit as text/event-stream when ``?sse=1``.

    Events: accepted -> progress* (per denoise step) -> complete ({image b64}),
    or error. Mirrors ``_run_layer_sse`` so the frontend can show a thin progress
    bar while a Qwen-Image-Edit (whole-frame or masked inpaint) runs.
    """
    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })
    await resp.prepare(req)
    device = _resolve_device(req)
    e = _engine_for(device)
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
    # Echo the resolved seed in the complete event so the webapp can record the
    # exact seed that ran (see handle_edit).
    seed = int(kw["seed"]) if "seed" in kw else random.randrange(0, 2**31 - 1)
    kw.setdefault("seed", seed)
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

    async with _device_lock(device):
        try:
            if engine_name == "hidream":
                img = await e.hidream_edit(
                    body["image"], prompt,
                    seed=kw.get("seed"),
                    num_inference_steps=kw.get("num_inference_steps"),
                    quality=body.get("quality"), progress_cb=_progress,
                )
            else:
                img = await e.edit_image(
                    body["image"], prompt,
                    engine=engine_name,
                    mask=body.get("mask_image"),
                    keep_subject=bool(body.get("keep_subject", False)),
                    strength=float(body.get("strength", 0.6)),
                    padding_mask_crop=int(body.get("padding_mask_crop", 0) or 0),
                    progress_cb=_progress,
                    **kw,
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
        "seed": seed,
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
    device = _resolve_device(req)
    e = _engine_for(device)
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
    # Resolve the seed: client-supplied passes through (deterministic replay);
    # otherwise mint a fresh random one. Echoed back in the response metadata so
    # the webapp can record the exact seed that ran.
    seed = int(body["seed"]) if body.get("seed") is not None else random.randrange(0, 2**31 - 1)
    kw = {}
    for k in ("width", "height", "num_inference_steps", "guidance_scale", "seed"):
        if body.get(k) is not None:
            kw[k] = body.get(k)
    kw.setdefault("seed", seed)
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
            e.hidream_edit, device, image, prompt,
            seed=kw.pop("seed", None),
            num_inference_steps=kw.pop("num_inference_steps", None),
            quality=body.get("quality"),
        )
    else:
        img = await _run_generation(
            e.edit_image,
            device, image, prompt,
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
        "seed": seed,
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
    device = _resolve_device(req)
    e = _engine_for(device)

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

    async with _device_lock(device):
        try:
            result = await e.layered_decompose(
                body["image"], layers, resolution, preview_only, steps, _progress
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

    device = _resolve_device(req)
    e = _engine_for(device)
    result = await _run_generation(
        e.layered_decompose, device, body["image"], layers, resolution, preview_only, steps
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
    device = _resolve_device(req)
    e = _engine_for(device)
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
            kw[k] = body.get(k)
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
            e.hidream_image, device, prompt,
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
        async with _klein_lock:
            img = await _run_generation(
                e.klein_image, device, prompt, num_inference_steps=steps, **kw,
            )
    else:
        img = await _run_generation(e.plain_image, device, prompt, **kw)
    resp = {
        "image": _pil_to_b64(img),
        "content_type": "image/png",
        "engine": engine_name,
        "seed": seed,
    }
    if engine_name == "klein":
        resp["num_inference_steps"] = _image_klein_steps(body)
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
    device = _resolve_device(req)
    e = _engine_for(device)
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
    # klein (flux_edit singleton) serialized globally across devices.
    async with _klein_lock:
        img = await _run_generation(
            e.style_frame, device, image, prompt + composition_hold, seed, width, height,
            num_inference_steps=steps,
        )
    return web.json_response({
        "styled_image": _pil_to_b64(img),
        "width": img.size[0],
        "height": img.size[1],
    })


# ── Lifecycle ──────────────────────────────────────────────

async def on_startup(_app: web.Application) -> None:
    global gpu_profile, ready
    # Detect GPU and build the VRAM-aware profile (reuses runner.ltx.gpu_profile,
    # which is pure-Python and builds fine even without torch/CUDA present).
    gpu_profile = build_profile(DEFAULT_DEVICE)
    logger.info(
        "image worker on GPU[%d] %s (%.1f GB VRAM, mode=%s) — engines created "
        "per-device on /load",
        DEFAULT_DEVICE, gpu_profile.gpu_name, gpu_profile.vram_gb, gpu_profile.mode,
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
