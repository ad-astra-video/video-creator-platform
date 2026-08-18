"""Live-runner HTTP service — the single Livepeer-facing edge.

Registers + heartbeats with the Livepeer Orchestrator as app="video-creator",
owns the shared-GPU swap policy (ResidentWorkerManager), routes each
/video-creator/v1/* request to the LTX or ID-V2V worker, and proxies
request/response. Heartbeat metadata carries the warm model + worker up/down
status so the desktop can prefer a warm restyle runner.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid

import aiohttp
from aiohttp import web

from livepeer_gateway.live_runner import (
    LiveRunnerGPU,
    register_runner,
)

from . import config
from .routing import CAPABILITIES, MODELS, ROUTES, candidate_workers, proxy, proxy_layer_sse
from .scheduler import GPUScheduler, QueueTimeout
from .specs import build_model_specs
from .swap import HttpWorkerTransport, ResidentWorkerManager

logger = logging.getLogger("video_creator.runner.live_runner.server")

_max_body = int(os.environ.get("MAX_BODY_BYTES", "3000000000"))
_SKIP_UPSCALE = os.environ.get("IDV2V_SKIP_UPSCALE", "").lower() in ("1", "true", "yes")

# Global state
_session: aiohttp.ClientSession | None = None
_worker_manager: ResidentWorkerManager | None = None
_registration = None
_ready = False
_scheduler: GPUScheduler | None = None  # Plan B — one task -> one GPU, concurrent across GPUs
_in_flight = 0        # active request counter (idle-backfill gate)
_last_activity = 0.0  # monotonic ts of last request (idle-backfill grace)

# Plan B deployment guard: only workers that HONOR the device field from the
# /load body (they are on their own dedicated GPU and stay resident there) get
# the device-aware per-worker residency path. Legacy video workers (ltx/idv2v)
# that still share one physical GPU keep the proven evict-before-load swap so two
# models can never co-reside on the same card. image-worker + gemma each live on
# their own GPU, so device-aware residency is safe for them.
_DEVICE_AWARE_WORKERS = frozenset({"image-worker", "gemma-worker", "ltx-worker", "idv2v-worker"})


def _device_for(worker: str, gpu: int) -> int | None:
    """Effective device to send in /load: the scheduled GPU for device-aware
    workers, None (legacy shared-GPU evict) for shared-card workers."""
    return gpu if worker in _DEVICE_AWARE_WORKERS else None

# Advertised model-spec metadata (resolution/fps/duration) for the Livepeer
# video pipelines. Built from the GPU's ACTUAL VRAM detected at runtime from the
# GPU-visible workers (_poll_worker_gpu_info) and rebuilt if that VRAM changes.
# Detection can't run at import time, so we start from the conservative fallback
# and the first startup/heartbeat pass corrects it. (user-mandated 2026-08: use
# the real GPU VRAM, not the GPU_VRAM_MB env var.)
_MODEL_SPECS = build_model_specs(config.DEFAULT_VRAM_MB)
# Total VRAM (MiB) of the GPU the workers render on, as detected from the ltx
# worker's /info. None until the first successful detection.
_gpu_vram_mb: int | None = None


async def _fetch_worker_info(session: aiohttp.ClientSession, worker_url: str) -> dict | None:
    """GET {worker_url}/video-creator/v1/info — worker liveness + GPU/model detail.

    The thin live-runner edge has no GPU/torch and cannot read VRAM itself, so it
    learns each worker's GPU from that worker's ``/info`` (authoritative on the
    box: torch -> nvidia-smi). Returns None when the worker/report isn't ready.
    """
    url = f"{worker_url.rstrip('/')}/video-creator/v1/info"
    try:
        async with session.get(
            url,
            headers={"X-Worker-Token": config.worker_token()},
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                return None
            return await resp.json()
    except Exception:
        logger.debug("worker info fetch failed (%s)", url, exc_info=True)
        return None


async def _poll_worker_gpu_info(session: aiohttp.ClientSession) -> tuple[dict, int | None]:
    """Poll every configured worker's /info (polled at startup + each heartbeat).

    Returns ``(gpu_meta, ltx_vram_mb)``:
      * gpu_meta — SHORTHAND per-worker GPU summary ``{short_name: vram_mb}``
        (e.g. ``{"ltx": 32153, "idv2v": 32153, "gemma": 32153}``), folded into the
        advertised heartbeat metadata. Integer MiB only (no name/cc) so the JSON
        stays far under go-livepeer's 1024-byte metadata cap. The Worker control
        plane never reads ``meta.gpu`` — it's informational — so terse keys are safe.
      * ltx_vram_mb — the create (ltx) worker's total VRAM, which drives the
        video-create resolution/duration specs.
    """
    gpu_meta: dict = {}
    ltx_vram_mb: int | None = None
    for name, url in config.WORKERS.items():
        info = await _fetch_worker_info(session, url)
        g = (info or {}).get("gpu") or {}
        vram_gb = g.get("vram_gb")
        vram_mb = int(float(vram_gb) * 1024) if vram_gb else None
        gpu_meta[name.split("-", 1)[0]] = vram_mb  # ltx-worker -> ltx
        if name == "ltx-worker" and vram_gb:
            ltx_vram_mb = vram_mb
    return gpu_meta, ltx_vram_mb


def _apply_detected_vram(vram_mb: int) -> None:
    """Record detected VRAM and rebuild advertised model specs if it changed."""
    global _gpu_vram_mb, _MODEL_SPECS
    if vram_mb and vram_mb != _gpu_vram_mb:
        _gpu_vram_mb = vram_mb
        _MODEL_SPECS = build_model_specs(vram_mb)
        logger.info("Detected worker GPU VRAM %d MiB -> rebuilt model specs", vram_mb)


def _need(request: web.Request):
    """Return the (worker_manager, session, token) trio, raising 503 if not up."""
    if _worker_manager is None or _session is None or _scheduler is None:
        raise web.HTTPServiceUnavailable(reason="live-runner not ready")
    return _worker_manager, _session, config.worker_token()


async def _with_gpu(scheduler: GPUScheduler, worker_manager, session, token,
                    worker: str, endpoint: str, body: dict):
    """Plan B dispatch: acquire a single free GPU, ensure the worker is resident
    there (device in the /load body), run the task, then release the GPU.

    No GPU free -> the scheduler queues FIFO; on timeout it raises QueueTimeout
    which the caller maps to a retriable 503.
    """
    gpu = await scheduler.acquire(worker)
    try:
        return await proxy(worker_manager, session, token, worker, endpoint, body,
                           device=_device_for(worker, gpu))
    finally:
        await scheduler.release(worker, gpu)


async def handle_health(_req: web.Request) -> web.Response:
    return web.json_response({"ok": True, "ready": _ready, "app": config.APP_ID})


async def handle_info(_req: web.Request) -> web.Response:
    wm, session, token = _need(_req)
    meta = await wm.check_health() if wm else {}
    return web.json_response({
        "runner_id": _registration.runner_id if _registration else "",
        "app": config.APP_ID,
        "capabilities": CAPABILITIES,
        "ready": _ready,
        "gpu": {"name": config.GPU_NAME, "vram_mb": _gpu_vram_mb or config.DEFAULT_VRAM_MB},
        "metadata": {**meta, "capabilities": CAPABILITIES, "model_specs": _MODEL_SPECS,
                     "models": MODELS},
    })


async def handle_scheduler_status(req: web.Request) -> web.Response:
    """GET /video-creator/v1/scheduler/status — per-GPU owner snapshot.

    Worker-token protected (internal): only the operator / dashboard sees the
    advisory GPU map. Not part of the public runner surface.
    """
    if req.headers.get("X-Worker-Token", "") != config.worker_token():
        raise web.HTTPForbidden(reason="missing/mismatched X-Worker-Token")
    if _scheduler is None:
        raise web.HTTPServiceUnavailable(reason="live-runner not ready")
    return web.json_response(_scheduler.status())


async def handle_gpu_pick(req: web.Request) -> web.Response:
    """POST /video-creator/v1/gpu-pick — assign a free GPU to a worker at boot.

    Worker-token protected. The caller (a worker container, at startup) sends
    ``{"worker": "ltx-worker"}``; we ask the scheduler to authoritatively pick
    (and reserve) a free GPU for it using the same warm-reuse -> idle ->
    LRU-evict policy as request dispatch. This is the source of truth for which
    card is actually free: it's reconciled from every worker's /info, so it
    knows the image worker holds GPU 0 even though that model loads lazily (and
    would look "free" to a local nvidia-smi at ltx-worker boot time).
    Returns ``{"worker": ..., "gpu": <idx>}``.
    """
    if req.headers.get("X-Worker-Token", "") != config.worker_token():
        raise web.HTTPForbidden(reason="missing/mismatched X-Worker-Token")
    if _scheduler is None:
        raise web.HTTPServiceUnavailable(reason="live-runner not ready")
    try:
        body = await req.json()
    except Exception:
        body = {}
    worker = str(body.get("worker") or "").strip()
    if not worker:
        return web.json_response({"error": "missing 'worker'"}, status=400)
    try:
        gpu = await _scheduler.pick_for_worker(worker)
    except QueueTimeout as exc:
        return web.json_response({"error": str(exc)}, status=503)
    return web.json_response({"worker": worker, "gpu": gpu})


async def _client_stage(info: dict):
    """Map an idv2v worker progress payload to a client-facing stage/message pair."""
    st = info.get("stage", "generating")
    msg = info.get("message", "") or ""
    if st == "preprocessing":
        return "preprocessing", "Preparing frames..."
    if st == "decoding":
        return "decoding", "Decoding video..."
    if st == "complete":
        return "finalizing", "Finalizing output..."
    if st == "generating":
        return "generating", msg or "Generating..."
    return st, msg


async def _restyle_chain(wm, session, token, body, progress_cb=None) -> web.Response:
    """Restyle = idv2v generation at a GPU-fitting resolution (capped ~480), then
    LTX spatial upscaler restores the requested output resolution.

    The response keeps the downstream contract: ``output_video`` base64 +
    ``resolution``. A flag and the pre-upscale resolution are included so the
    desktop can tell a chained (upscaled) result from a native one.

    When ``progress_cb`` is supplied (SSE transport), the idv2v leg runs as a
    background task and its REAL per-step progress is polled from the worker's
    ``/progress/{job_id}`` and forwarded through the callback, and the upscale
    leg emits keep-alive text stages -- so the connection stays alive and the
    client sees live generation progress (fixes the Cloudflare-tunnel timeout
    that a silent multi-minute request was hitting). Without it the plain HTTP
    path returns the single JSON response unchanged.
    """
    headers = {"X-Worker-Token": token}
    idv2v_base = config.WORKERS["idv2v-worker"]
    idv2v_gpu = await _scheduler.acquire("idv2v-worker")
    await wm.ensure("idv2v-worker", device=_device_for("idv2v-worker", idv2v_gpu))

    job_id = str(body.get("job_id") or uuid.uuid4().hex[:12])
    rbody = {**body, "job_id": job_id}

    async def _send(ev: dict) -> None:
        if progress_cb is None:
            return
        try:
            await progress_cb(ev)
        except Exception:
            pass

    # 1) ID-V2V restyle (background task so we can poll /progress concurrently).
    async def _restyle():
        async with session.post(f"{idv2v_base}/video-creator/v1/restyle",
                                json=rbody, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=3600.0)) as r:
            if r.status >= 400:
                return {"_error": (await r.read())[:500].decode("utf-8", "replace")}
            return await r.json()

    task = asyncio.create_task(_restyle())
    try:
        while not task.done():
            try:
                async with session.get(
                    f"{idv2v_base}/video-creator/v1/progress/{job_id}",
                    headers=headers, timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    if r.status == 200:
                        info = await r.json()
                        stage, message = _client_stage(info)
                        prog = info.get("progress")
                        p = None
                        if stage == "generating":
                            try:
                                p = round(float(prog), 4)
                            except (TypeError, ValueError):
                                p = None
                        await _send({"stage": stage, "message": message, "progress": p})
            except Exception:
                pass
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=1.5)
            except asyncio.TimeoutError:
                pass
        data = task.result()
        if data.get("_error"):
            return web.Response(status=502, text=data["_error"],
                                content_type="application/json", charset="utf-8")
    except Exception as exc:
        logger.exception("restyle chain failed")
        return web.Response(status=500, text=str(exc),
                            content_type="application/json", charset="utf-8")

    await _scheduler.release("idv2v-worker", idv2v_gpu)
    out_b64 = data.get("output_video")
    if not out_b64:
        return web.json_response(data, status=502)

    if _SKIP_UPSCALE:
        # Diagnostic A/B: return the native idv2v output without the LTX
        # spatial upscale, to check whether the upscaler causes flicker.
        return web.json_response({
            "output_video": out_b64,
            "frames_generated": data.get("frames_generated"),
            "resolution": data.get("resolution"),
            "gen_resolution": data.get("resolution"),
            "upscaled": False,
            "video_caption": data.get("video_caption"),
            "enhanced_prompt": data.get("enhanced_prompt"),
            "used_prompt": data.get("used_prompt"),
        })

    # 2) LTX spatial upscale to the requested target resolution (keep-alive).
    target_w = max(16, int(body.get("width", 1280)))
    target_h = max(16, int(body.get("height", 720)))
    up_body = {"video_base64": out_b64, "width": target_w, "height": target_h,
               "fps": body.get("fps", 24)}
    ltx_base = config.WORKERS["ltx-worker"]
    ltx_gpu = await _scheduler.acquire("ltx-worker")
    await wm.ensure("ltx-worker", device=_device_for("ltx-worker", ltx_gpu))

    async def _upscale():
        async with session.post(f"{ltx_base}/video-creator/v1/upscale",
                                json=up_body, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=3600.0)) as r2:
            if r2.status >= 400:
                return {"_error": (await r2.read())[:500].decode("utf-8", "replace")}
            return await r2.json()

    utask = asyncio.create_task(_upscale())
    try:
        while not utask.done():
            await _send({"stage": "upscaling",
                         "message": "Upscaling to full resolution...", "progress": None})
            try:
                await asyncio.wait_for(asyncio.shield(utask), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        up = utask.result()
        if up.get("_error"):
            return web.Response(status=502, text=up["_error"],
                                content_type="application/json", charset="utf-8")
    except Exception as exc:
        logger.exception("restyle upscale failed")
        return web.Response(status=500, text=str(exc),
                            content_type="application/json", charset="utf-8")

    await _scheduler.release("ltx-worker", ltx_gpu)
    up_b64 = up.get("video_base64")
    if not up_b64:
        return web.json_response(up, status=502)
    return web.json_response({
        "output_video": up_b64,
        "frames_generated": data.get("frames_generated"),
        "resolution": f"{target_w}x{target_h}",
        "gen_resolution": data.get("resolution"),
        "upscaled": True,
        "video_caption": data.get("video_caption"),
        "enhanced_prompt": data.get("enhanced_prompt"),
        "used_prompt": data.get("used_prompt"),
    })


async def handle_ws(req: web.Request) -> web.Response:
    """WebSocket endpoint (orchestrator-proxied) for long-running generation jobs.

    The client opens a WS through the orchestrator to /video-creator/v1/ws and
    sends {"type":"<endpoint>","request_id":...,"body":{...}} where <endpoint>
    is any live-runner task (t2v, i2v, image, edit, extend, retake, prompt-enhance,
    ic-lora-generate, extract-conditioning, suggest-gap-prompt, restyle, sam3,
    upscale). The runner routes to the right worker and streams progress events
    and finally the result media over the socket — so long jobs don't hit the
    frontend's HTTP timeout and the client gets live progress.

    Message out:
      {"type":"accepted", request_id, job_id, progress:0}
      {"type":"progress", request_id, job_id, stage, message, progress, phase}
      {"type":"complete", request_id, job_id, payload:{output_video, resolution,...}}
      {"type":"error",    request_id, error}
    """
    ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=256 * 1024 * 1024)
    await ws.prepare(req)
    request_id = None
    try:
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                data = json.loads(msg.data)
            except Exception:
                await ws.send_json({"type": "error", "error": "bad JSON"})
                continue
            task_type = data.get("type")
            if task_type not in ROUTES:
                await ws.send_json({"type": "error", "error": f"unknown endpoint: {task_type}"})
                continue
            request_id = data.get("request_id")
            body = dict(data.get("body") or {})
            job_id = body.get("job_id") or request_id or uuid.uuid4().hex[:12]
            body["job_id"] = job_id
            await ws.send_json({"type": "accepted", "request_id": request_id,
                                "job_id": job_id, "progress": 0.0, "stage": "accepted"})
            wm, session, token = _need(req)
            global _in_flight, _last_activity
            _in_flight += 1
            _last_activity = time.monotonic()
            try:
                if task_type == "restyle":
                    # Restyle chain acquires/releases its own GPUs.
                    await _ws_restyle_chain(ws, wm, session, token, request_id, job_id, body)
                else:
                    worker = ROUTES[task_type]
                    try:
                        gpu = await _scheduler.acquire(worker)
                    except QueueTimeout as exc:
                        if not ws.closed:
                            await ws.send_json({"type": "error", "request_id": request_id,
                                                "error": str(exc)})
                        break
                    try:
                        await _ws_proxy(ws, wm, session, token, request_id, job_id,
                                        task_type, body, device=gpu)
                    finally:
                        await _scheduler.release(worker, gpu)
            finally:
                _in_flight -= 1
            break
    except Exception as exc:
        logger.exception("ws handler failed")
        try:
            if not ws.closed:
                await ws.send_json({"type": "error", "request_id": request_id,
                                    "error": str(exc)})
        except Exception:
            pass
    finally:
        if not ws.closed:
            await ws.close()
    return ws


async def _ws_restyle_chain(ws, wm, session, token, request_id, job_id, body) -> None:
    """Run idv2v then ltx upscale, streaming progress over the WS.

    Progress semantics:
      - Text updates at each process step (preprocessing / generating /
        decoding / upscaling / finalizing).
      - A numeric ``progress`` is sent ONLY for genuine backbone inference
        iterations (per idv2v denoise step, real 0..1 from the worker). Every
        other stage is text-only (progress = None) — no fabricated %.
      - Frames are pushed on a steady cadence so the socket never goes quiet.
    """
    headers = {"X-Worker-Token": token}
    idv2v_base = config.WORKERS["idv2v-worker"]
    idv2v_gpu = await _scheduler.acquire("idv2v-worker")
    await wm.ensure("idv2v-worker", device=_device_for("idv2v-worker", idv2v_gpu))

    def _client_stage(info: dict):
        st = info.get("stage", "generating")
        msg = info.get("message", "") or ""
        if st == "preprocessing":
            return "preprocessing", "Preparing frames..."
        if st == "decoding":
            return "decoding", "Decoding video..."
        if st == "complete":
            return "finalizing", "Finalizing output..."
        if st == "generating":
            return "generating", msg or "Generating..."
        return st, msg

    async def _send(ev: dict) -> None:
        if ws.closed:
            return
        try:
            await ws.send_json({"type": "progress", "request_id": request_id,
                                "job_id": job_id, **ev})
        except Exception:
            pass

    async def _restyle():
        # Restyle jobs routinely exceed aiohttp's 300s default total timeout
        # (id-v2v ~6.4min on .8). Without an explicit timeout the POST to the
        # worker aborts at 5:00, the live-runner sends an (empty-message) error
        # frame, and the desktop reports "runner websocket error" even though
        # the worker finishes fine. Give the backbone the full budget.
        async with session.post(f"{idv2v_base}/video-creator/v1/restyle",
                                json=body, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=3600.0)) as r:
            if r.status >= 400:
                return {"_error": (await r.read())[:500].decode("utf-8", "replace")}
            return await r.json()

    task = asyncio.create_task(_restyle())
    try:
        while not task.done():
            try:
                async with session.get(
                    f"{idv2v_base}/video-creator/v1/progress/{job_id}",
                    headers=headers, timeout=aiohttp.ClientTimeout(total=5)
                ) as r:
                    if r.status == 200:
                        info = await r.json()
                        stage, message = _client_stage(info)
                        prog = info.get("progress")
                        p = None
                        if stage == "generating":
                            try:
                                p = round(float(prog), 4)
                            except (TypeError, ValueError):
                                p = None
                        await _send({"stage": stage, "message": message, "progress": p})
            except Exception:
                pass
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=1.5)
            except asyncio.TimeoutError:
                pass
        data = task.result()
        if data.get("_error") or not data.get("output_video"):
            err = data.get("_error") or str(data)
            if not ws.closed:
                await ws.send_json({"type": "error", "request_id": request_id, "error": err})
            return
    except Exception as exc:
        if not ws.closed:
            await ws.send_json({"type": "error", "request_id": request_id, "error": str(exc)})
        return

    await _scheduler.release("idv2v-worker", idv2v_gpu)
    if _SKIP_UPSCALE:
        # Diagnostic A/B: return the native idv2v output without the LTX
        # spatial upscale, to check whether the upscaler causes flicker.
        payload = {
            "output_video": data["output_video"],
            "frames_generated": data.get("frames_generated"),
            "resolution": data.get("resolution"),
            "gen_resolution": data.get("resolution"),
            "upscaled": False,
            "job_id": job_id,
            "video_caption": data.get("video_caption"),
            "enhanced_prompt": data.get("enhanced_prompt"),
            "used_prompt": data.get("used_prompt"),
        }
        if not ws.closed:
            await ws.send_json({"type": "complete", "request_id": request_id,
                                "job_id": job_id, "payload": payload})
        return

    # LTX upscale (text-only stage; keep-alive so the socket stays alive).
    target_w = max(16, int(body.get("width", 1280)))
    target_h = max(16, int(body.get("height", 720)))
    ltx_base = config.WORKERS["ltx-worker"]
    ltx_gpu = await _scheduler.acquire("ltx-worker")
    await wm.ensure("ltx-worker", device=_device_for("ltx-worker", ltx_gpu))
    up_body = {"video_base64": data["output_video"], "width": target_w, "height": target_h,
               "fps": body.get("fps", 24)}

    async def _upscale():
        async with session.post(f"{ltx_base}/video-creator/v1/upscale",
                                json=up_body, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=3600.0)) as r2:
            if r2.status >= 400:
                return {"_error": (await r2.read())[:500].decode("utf-8", "replace")}
            return await r2.json()

    utask = asyncio.create_task(_upscale())
    try:
        while not utask.done():
            await _send({"stage": "upscaling", "message": "Upscaling to full resolution...",
                         "progress": None})
            try:
                await asyncio.wait_for(asyncio.shield(utask), timeout=2.0)
            except asyncio.TimeoutError:
                pass
        up = utask.result()
        if up.get("_error") or not up.get("video_base64"):
            err = up.get("_error") or "upscale produced no video"
            if not ws.closed:
                await ws.send_json({"type": "error", "request_id": request_id, "error": err})
            return
        up_b64 = up["video_base64"]
    except Exception as exc:
        if not ws.closed:
            await ws.send_json({"type": "error", "request_id": request_id, "error": str(exc)})
        return

    await _scheduler.release("ltx-worker", ltx_gpu)
    payload = {
        "output_video": up_b64,
        "frames_generated": data.get("frames_generated"),
        "resolution": f"{target_w}x{target_h}",
        "gen_resolution": data.get("resolution"),
        "upscaled": True,
        "job_id": job_id,
        "video_caption": data.get("video_caption"),
        "enhanced_prompt": data.get("enhanced_prompt"),
        "used_prompt": data.get("used_prompt"),
    }
    if not ws.closed:
        await ws.send_json({"type": "complete", "request_id": request_id,
                            "job_id": job_id, "payload": payload})


async def _ws_proxy(ws, wm, session, token, request_id, job_id, task_type, body,
                    device: int | None = None) -> None:
    """Run a non-restyle task over the WebSocket, streaming stage text + final result.

    Routes ``task_type`` to its worker via ROUTES, ensures the worker is resident,
    POSTs the body (full 3600s budget), streams text-only stage frames so the socket
    never goes quiet, then sends a ``complete`` frame with the worker's parsed JSON
    payload (media base64 for video/image tasks, enhanced_prompt for prompt-enhance).
    Workers that publish per-step progress (idv2v) could be polled here the same way
    the restyle chain does; LTX tasks return synchronously so we send stages around
    the call and the final result.
    """
    worker = ROUTES[task_type]
    headers = {"X-Worker-Token": token}
    candidates = candidate_workers(task_type, worker)
    async def _send(ev: dict) -> None:
        if ws.closed:
            return
        try:
            await ws.send_json({"type": "progress", "request_id": request_id,
                                "job_id": job_id, **ev})
        except Exception:
            pass
    await _send({"stage": "generating", "message": "Generating...", "progress": None})
    last_err: str | None = None
    data: dict | None = None
    for index, target in enumerate(candidates):
        try:
            base = config.WORKERS[target]
            await wm.ensure(target, device=_device_for(target, device))
            url = f"{base}/video-creator/v1/{task_type}"
            async with session.post(url, json=body, headers=headers,
                                    timeout=aiohttp.ClientTimeout(total=3600.0)) as r:
                if r.status >= 400:
                    last_err = (await r.read())[:500].decode("utf-8", "replace")
                    logger.warning("Worker %s/%s -> %s: %s", target, task_type, r.status, last_err)
                    continue
                data = await r.json()
            break
        except Exception as exc:
            last_err = str(exc)
            logger.warning("Worker %s/%s failed (%s)%s", target, task_type, exc,
                           " - falling back" if index < len(candidates) - 1 else "")
            data = None
    if data is None:
        if not ws.closed:
            await ws.send_json({"type": "error", "request_id": request_id,
                                "job_id": job_id,
                                "error": last_err or f"{task_type} failed on all workers"})
        return
    await _send({"stage": "finalizing", "message": "Finalizing output...", "progress": None})
    if not ws.closed:
        await ws.send_json({"type": "complete", "request_id": request_id,
                            "job_id": job_id, "payload": data})


async def handle_generic(req: web.Request) -> web.Response:
    """Proxy a /video-creator/v1/{endpoint} request to its worker.

    The endpoint name is the last non-empty path segment. Body is forward as-is
    (base64 in -> base64 out); the swap policy makes the right model resident.

    When the client requests ``?sse=1`` the response is served as text/event-stream
    (accepted -> progress* -> a single complete event carrying the worker's JSON
    payload with media base64, or error) so the browser can show live generation
    progress over the same paid HTTP connection the go-livepeer orchestrator
    reverse-proxies. Without the flag behavior is unchanged.
    """
    endpoint = req.match_info.get("endpoint", "")
    worker = ROUTES.get(endpoint)
    if worker is None:
        return web.json_response({"error": f"unknown endpoint: {endpoint}"}, status=404)

    wm, session, token = _need(req)
    body = await req.json()

    global _in_flight, _last_activity
    want_sse = req.query.get("sse") in ("1", "true", "yes")
    if not want_sse:
        _in_flight += 1
        _last_activity = time.monotonic()
        try:
            if endpoint == "restyle":
                # Restyle chain acquires/releases its own GPUs (idv2v then ltx).
                return await _restyle_chain(wm, session, token, body)
            try:
                return await _with_gpu(_scheduler, wm, session, token,
                                       worker, endpoint, body)
            except QueueTimeout as exc:
                return web.json_response({"error": str(exc)}, status=503)
        finally:
            _in_flight -= 1

    # SSE path: stream accepted + progress while the worker runs, then a single
    # complete event with the worker's JSON payload (media as base64).
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

    await _ev("accepted", {"endpoint": endpoint})

    # Keep-alive heartbeat: proxies (Cloudflare / go-livepeer) tear down an idle
    # SSE stream after ~30-60s with no bytes, and a cold model load (wm.ensure ->
    # /load) or any quiet leg produces NO progress frames. Emit an SSE comment
    # every 10s so the connection always has regular flow until the terminal event.
    stop_beat = asyncio.Event()
    async def _heartbeat() -> None:
        while not stop_beat.is_set():
            try:
                await asyncio.wait_for(stop_beat.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                pass
            if stop_beat.is_set():
                break
            try:
                await resp.write(b": keepalive\n\n")
            except Exception:
                return
    beat = asyncio.create_task(_heartbeat())

    _in_flight += 1
    _last_activity = time.monotonic()
    try:
        if endpoint == "restyle":
            async def _prog(ev): await _ev("progress", ev)
            out = await _restyle_chain(wm, session, token, body, progress_cb=_prog)
        elif endpoint == "layer":
            # Stream the image-worker's per-step SSE verbatim (accepted ->
            # progress* (per denoise step) -> complete).
            gpu = None
            try:
                gpu = await _scheduler.acquire("image-worker")
            except QueueTimeout as exc:
                await _ev("error", {"error": str(exc)})
                await resp.write_eof()
                return resp
            try:
                await proxy_layer_sse(wm, session, token, body, resp, device=gpu)
            except Exception as exc:
                logger.exception("layer SSE relay failed")
                await _ev("error", {"error": str(exc)})
            finally:
                await _scheduler.release("image-worker", gpu)
            await resp.write_eof()
            return resp
        else:
            await _ev("progress", {"stage": "generating", "message": "Generating...", "progress": None})
            try:
                out = await _with_gpu(_scheduler, wm, session, token,
                                      worker, endpoint, body)
            except QueueTimeout as exc:
                await _ev("error", {"error": str(exc)})
                await resp.write_eof()
                return resp
    except Exception as exc:
        logger.exception("sse %s failed", endpoint)
        await _ev("error", {"error": str(exc)})
        await resp.write_eof()
        return resp
    finally:
        _in_flight -= 1
        stop_beat.set()
        beat.cancel()

    if out.status >= 400:
        await _ev("error", {"error": out.text or f"worker error {out.status}"})
    else:
        try:
            data = json.loads(out.body.decode("utf-8"))
        except Exception:
            await _ev("error", {"error": "worker returned a non-JSON response"})
        else:
            await _ev("complete", data)
    await resp.write_eof()
    return resp


async def handle_namespaced(req: web.Request) -> web.Response:
    """Proxy an explicitly worker-routed /video-creator/v1/{worker}/{endpoint}.

    Lets callers force routing to a specific worker instead of relying on the
    capability table — e.g. the /idv2v/ prefix routes to the id-v2v worker even
    for an endpoint (like prompt-enhance) that the generic table would send to
    the LTX worker. Worker names: "ltx" | "idv2v".
    """
    worker = req.match_info.get("worker", "")
    endpoint = req.match_info.get("endpoint", "")
    full = {"ltx": "ltx-worker", "idv2v": "idv2v-worker"}.get(worker)
    if full is None:
        return web.json_response({"error": f"unknown worker: {worker}"}, status=404)

    wm, session, token = _need(req)
    body = await req.json()

    global _in_flight, _last_activity
    _in_flight += 1
    _last_activity = time.monotonic()
    try:
        try:
            return await _with_gpu(_scheduler, wm, session, token,
                                   full, endpoint, body)
        except QueueTimeout as exc:
            return web.json_response({"error": str(exc)}, status=503)
    finally:
        _in_flight -= 1


async def on_startup(_app: web.Application) -> None:
    global _session, _worker_manager, _registration, _ready, _scheduler
    _session = aiohttp.ClientSession()
    _worker_manager = ResidentWorkerManager(
        transport=HttpWorkerTransport(_session, config.worker_token()),
        workers=dict(config.WORKERS),
        pinned=frozenset(["gemma-worker"]) if config.LLM_PINNED else frozenset(),
    )
    # Plan B: the scheduler gates concurrency per GPU (one task -> one GPU). It
    # replaces the old single Semaphore(1) that serialized ALL tasks on one GPU.
    _scheduler = GPUScheduler.from_config()
    # Eviction hook: the scheduler evicts a worker's warm model from VRAM
    # before handing its GPU to a DIFFERENT worker's task, so two models never
    # co-reside on one card (the image-then-video OOM). Any worker can be the
    # victim; the generalized callback POST /evict frees its VRAM first.
    async def _evict_worker(name: str) -> None:
        try:
            await _worker_manager.evict(name)
        except Exception:
            logger.warning("worker evict failed (%s, may be down)", name, exc_info=True)
    _scheduler.evict_cb = _evict_worker

    # Learn the real GPU VRAM from the worker before registering, so the very
    # first heartbeat already advertises specs for the actual card (not the env
    # default). Falls back to DEFAULT_VRAM_MB if the worker isn't up yet -- the
    # heartbeat loop will correct it as soon as detection succeeds.
    gpu_meta, ltx_vram = await _poll_worker_gpu_info(_session)
    if ltx_vram:
        _apply_detected_vram(ltx_vram)
    gpu = LiveRunnerGPU(name=config.GPU_NAME, vram_mb=_gpu_vram_mb or config.DEFAULT_VRAM_MB)
    _registration = await register_runner(
        config.ORCHESTRATOR_URL,
        secret=config.ORCHESTRATOR_SECRET,
        runner_url=config.RUNNER_URL,
        app=config.APP_ID,
        mode="single-shot",
        price=config.PRICE,
        unit=config.PRICE_UNIT,
        currency=config.PRICE_CURRENCY,
        gpu=gpu,
        label="restyle",
        metadata=json.dumps({
            "capabilities": CAPABILITIES,
            "model_specs": _MODEL_SPECS,
            "models": MODELS,
            "gpu": gpu_meta,
            "ltx_up": False,
            "idv2v_up": False,
            "warm": None,
        }),
        heartbeat_interval_s=config.HEARTBEAT_INTERVAL_S,
    )
    # NOTE: `main`'s register_runner already calls .start() (returns a started
    # registration) — do NOT call .start() again or we'd duplicate the initial
    # heartbeat and spawn a second heartbeat loop.
    logger.info("Registered live-runner %s (app=%s)", _registration.runner_id, config.APP_ID)

    # Refresh heartbeat metadata each beat from the swap policy + live worker /health.
    asyncio.create_task(_refresh_metadata_loop())
    # Plan B: load gemma-worker resident on GEMMA_RESIDENT_GPU and keep it there;
    # the scheduler holds that GPU out of the task pool. Fail-safe: the edge
    # boots even if gemma is down.
    try:
        await _worker_manager.ensure("gemma-worker", device=config.GEMMA_RESIDENT_GPU)
    except Exception:
        logger.warning("initial gemma resident load failed (gemma-worker may be down)", exc_info=True)
    asyncio.create_task(_idle_backfill_loop())
    _ready = True
    logger.info("Live-runner READY")


async def _idle_backfill_loop() -> None:
    """Keep the LLM backend resident when the GPU is idle.

    Shared-GPU mode (blank LLM_GPU_DEVICE): after GEMMA_IDLE_GRACE_S with no
    request, replace the warm render worker with the gemma-worker so enhance /
    chat serve without a cold load. Pinned mode (dedicated GPU): a no-op once
    loaded. Gated on ``_in_flight == 0`` so it never evicts a busy worker.
    """
    while True:
        try:
            if _worker_manager is not None and _scheduler is not None:
                # Reload gemma only when its GPU is actually free (not currently
                # running a render task that co-opted it), then re-assert gemma
                # residency in the advisory map so the GPU is reserved for it again.
                if _scheduler.gemma_slot_free():
                    await _worker_manager.ensure(
                        "gemma-worker", device=config.GEMMA_RESIDENT_GPU)
                    _scheduler.mark_gemma_loaded()
        except Exception:
            logger.warning("idle LLM backfill failed", exc_info=True)
        await asyncio.sleep(2)


async def _refresh_metadata_loop() -> None:
    while True:
        try:
            if _worker_manager is not None and _registration is not None:
                # Keep advertised specs in sync with the actual GPU VRAM in case
                # the worker came up after startup (or its VRAM changed).
                gpu_meta, ltx_vram = await _poll_worker_gpu_info(_session)
                if ltx_vram:
                    _apply_detected_vram(ltx_vram)
                # Plan B: reconcile the advisory GPU map from each worker's
                # /info (device_in_use) so a crashed/restarted worker self-heals.
                if _scheduler is not None:
                    w_info = {}
                    for _name, _url in config.WORKERS.items():
                        _info = await _fetch_worker_info(_session, _url)
                        if _info is not None:
                            w_info[_name] = _info
                    await _scheduler.reconcile(w_info)
                meta = await _worker_manager.check_health()
                meta["capabilities"] = CAPABILITIES
                meta["model_specs"] = _MODEL_SPECS
                meta["models"] = MODELS
                meta["gpu"] = gpu_meta
                # The registration is an in-process object owned by this runner;
                # set its payload metadata so the next heartbeat advertises the
                # current warm-model + worker up/down status. (No SDK change needed.)
                _registration._metadata = json.dumps(meta)
        except Exception:
            logger.warning("metadata refresh failed", exc_info=True)
        await asyncio.sleep(config.HEARTBEAT_INTERVAL_S)


async def on_cleanup(_app: web.Application) -> None:
    global _registration, _session
    if _registration is not None:
        try:
            await _registration.close()
        except Exception:
            logger.debug("unregister failed", exc_info=True)
        _registration = None
    if _session is not None:
        await _session.close()
        _session = None


def create_app() -> web.Application:
    app = web.Application(client_max_size=_max_body)
    p = "/video-creator/v1"
    app.router.add_get(f"{p}/health", handle_health)
    app.router.add_get(f"{p}/info", handle_info)
    app.router.add_get(f"{p}/scheduler/status", handle_scheduler_status)
    app.router.add_post(f"{p}/gpu-pick", handle_gpu_pick)
    # One parameterized route so handle_generic can read the endpoint from
    # match_info["endpoint"] and look it up in ROUTES. (Previously these were
    # registered as static paths with no {endpoint} placeholder, so match_info
    # was empty and every proxied call returned 404 "unknown endpoint: ".)
    app.router.add_post(f"{p}/{{endpoint}}", handle_generic)
    # Explicit per-worker routing: /video-creator/v1/{worker}/{endpoint} with
    # worker in {"ltx","idv2v"} forces the request to that worker regardless of
    # the capability table (e.g. /idv2v/prompt-enhance must hit the id-v2v
    # worker, not the LTX one the generic table would pick).
    app.router.add_post(f"{p}/{{worker}}/{{endpoint}}", handle_namespaced)
    app.router.add_get(f"{p}/ws", handle_ws)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    # Resolve auth token eagerly so a blank one is generated + logged once.
    config.worker_token()
    app = create_app()
    web.run_app(app, host=config.HOST, port=config.PORT)


if __name__ == "__main__":
    main()
