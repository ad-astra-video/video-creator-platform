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
from .routing import CAPABILITIES, ROUTES, proxy
from .swap import HttpWorkerTransport, ResidentWorkerManager

logger = logging.getLogger("video_creator.runner.live_runner.server")

_max_body = int(os.environ.get("MAX_BODY_BYTES", "3000000000"))
_SKIP_UPSCALE = os.environ.get("IDV2V_SKIP_UPSCALE", "").lower() in ("1", "true", "yes")

# Global state
_session: aiohttp.ClientSession | None = None
_worker_manager: ResidentWorkerManager | None = None
_registration = None
_ready = False
_generation_sem = None  # asyncio.Semaphore(1) — single GPU, one inference at a time
_in_flight = 0        # active request counter (idle-backfill gate)
_last_activity = 0.0  # monotonic ts of last request (idle-backfill grace)


def _need(request: web.Request):
    """Return the (worker_manager, session, token) trio, raising 503 if not up."""
    if _worker_manager is None or _session is None:
        raise web.HTTPServiceUnavailable(reason="live-runner not ready")
    return _worker_manager, _session, config.worker_token()


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
        "gpu": {"name": config.GPU_NAME, "vram_mb": config.GPU_VRAM_MB},
        "metadata": meta,
    })


async def _restyle_chain(wm, session, token, body) -> web.Response:
    """Restyle = idv2v generation at a GPU-fitting resolution (capped ~480),
    then LTX spatial upscaler restores the requested output resolution.

    The response keeps the downstream contract: ``output_video`` base64 +
    ``resolution``. A flag and the pre-upscale resolution are included so the
    desktop can tell a chained (upscaled) result from a native one.
    """
    # 1) ID-V2V restyle (generates at the box-fitting ~480 cap internally).
    headers = {"X-Worker-Token": token}
    idv2v_base = config.WORKERS["idv2v-worker"]
    await wm.ensure("idv2v-worker")
    async with session.post(
        f"{idv2v_base}/video-creator/v1/restyle", json=body, headers=headers,
        timeout=aiohttp.ClientTimeout(total=3600.0),
    ) as resp:
        if resp.status >= 400:
            txt = (await resp.read())[:500].decode("utf-8", "replace")
            return web.Response(status=502, text=txt, content_type="application/json", charset="utf-8")
        data = await resp.json()

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

    # 2) LTX spatial upscale to the requested target resolution.
    target_w = max(16, int(body.get("width", 1280)))
    target_h = max(16, int(body.get("height", 720)))
    up_body = {
        "video_base64": out_b64,
        "width": target_w,
        "height": target_h,
        "fps": body.get("fps", 24),
    }
    ltx_base = config.WORKERS["ltx-worker"]
    await wm.ensure("ltx-worker")
    async with session.post(
        f"{ltx_base}/video-creator/v1/upscale", json=up_body, headers=headers,
        timeout=aiohttp.ClientTimeout(total=3600.0),
    ) as resp2:
        if resp2.status >= 400:
            txt = (await resp2.read())[:500].decode("utf-8", "replace")
            return web.Response(status=502, text=txt, content_type="application/json", charset="utf-8")
        up = await resp2.json()

    up_b64 = up.get("video_base64")
    if not up_b64:
        return web.json_response(data, status=502)
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
                async with _generation_sem:
                    if task_type == "restyle":
                        await _ws_restyle_chain(ws, wm, session, token, request_id, job_id, body)
                    else:
                        await _ws_proxy(ws, wm, session, token, request_id, job_id, task_type, body)
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
    await wm.ensure("idv2v-worker")

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
    await wm.ensure("ltx-worker")
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


async def _ws_proxy(ws, wm, session, token, request_id, job_id, task_type, body) -> None:
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
    base = config.WORKERS[worker]
    await wm.ensure(worker)
    url = f"{base}/video-creator/v1/{task_type}"
    async def _send(ev: dict) -> None:
        if ws.closed:
            return
        try:
            await ws.send_json({"type": "progress", "request_id": request_id,
                                "job_id": job_id, **ev})
        except Exception:
            pass
    await _send({"stage": "generating", "message": "Generating...", "progress": None})
    try:
        async with session.post(url, json=body, headers=headers,
                                timeout=aiohttp.ClientTimeout(total=3600.0)) as r:
            if r.status >= 400:
                text = (await r.read())[:500].decode("utf-8", "replace")
                if not ws.closed:
                    await ws.send_json({"type": "error", "request_id": request_id,
                                        "job_id": job_id, "error": text})
                return
            data = await r.json()
    except Exception as exc:
        logger.exception("ws proxy %s failed", task_type)
        if not ws.closed:
            await ws.send_json({"type": "error", "request_id": request_id,
                                "job_id": job_id, "error": str(exc)})
        return
    await _send({"stage": "finalizing", "message": "Finalizing output...", "progress": None})
    if not ws.closed:
        await ws.send_json({"type": "complete", "request_id": request_id,
                            "job_id": job_id, "payload": data})


async def handle_generic(req: web.Request) -> web.Response:
    """Proxy a /video-creator/v1/{endpoint} request to its worker.

    The endpoint name is the last non-empty path segment. Body is forward as-is
    (base64 in -> base64 out); the swap policy makes the right model resident.
    """
    endpoint = req.match_info.get("endpoint", "")
    worker = ROUTES.get(endpoint)
    if worker is None:
        return web.json_response({"error": f"unknown endpoint: {endpoint}"}, status=404)

    wm, session, token = _need(req)
    body = await req.json()

    global _in_flight, _last_activity
    _in_flight += 1
    _last_activity = time.monotonic()
    try:
        # Serialize inference: the shared GPU runs one generation at a time.
        async with _generation_sem:
            if endpoint == "restyle":
                return await _restyle_chain(wm, session, token, body)
            return await proxy(wm, session, token, worker, endpoint, body)
    finally:
        _in_flight -= 1


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
        # Serialize inference: the shared GPU runs one generation at a time.
        async with _generation_sem:
            return await proxy(wm, session, token, full, endpoint, body)
    finally:
        _in_flight -= 1


async def on_startup(_app: web.Application) -> None:
    global _session, _worker_manager, _registration, _ready, _generation_sem
    _session = aiohttp.ClientSession()
    _worker_manager = ResidentWorkerManager(
        transport=HttpWorkerTransport(_session, config.worker_token()),
        workers=dict(config.WORKERS),
        pinned=frozenset(["gemma-worker"]) if config.LLM_PINNED else frozenset(),
    )
    _generation_sem = asyncio.Semaphore(1)

    gpu = LiveRunnerGPU(name=config.GPU_NAME, vram_mb=config.GPU_VRAM_MB)
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
            "ltx_worker_up": False,
            "idv2v_worker_up": False,
            "warm_model": None,
        }),
        heartbeat_interval_s=config.HEARTBEAT_INTERVAL_S,
    )
    await _registration.start()
    logger.info("Registered live-runner %s (app=%s)", _registration.runner_id, config.APP_ID)

    # Refresh heartbeat metadata each beat from the swap policy + live worker /health.
    asyncio.create_task(_refresh_metadata_loop())
    # Load the LLM backend. Pinned (dedicated GPU) -> load once at boot and never
    # evict; blank (shared GPU) -> backfill the idle slot so enhance/chat serve
    # without a cold load. Fail-safe: the edge boots even if gemma is down.
    try:
        if config.LLM_PINNED:
            await _worker_manager.load_pinned()
        else:
            await _worker_manager.backfill("gemma-worker")
    except Exception:
        logger.warning("initial LLM backfill failed (gemma-worker may be down)", exc_info=True)
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
            if _worker_manager is not None:
                if config.LLM_PINNED:
                    # Dedicated GPU: retry the pin load until it lands (the boot-time
                    # load can lose a race against gemma-worker's own startup). Cheap
                    # no-op (lock + in-memory check) once it's loaded.
                    await _worker_manager.ensure("gemma-worker")
                elif _in_flight == 0:
                    if _last_activity and (time.monotonic() - _last_activity) >= config.GEMMA_IDLE_GRACE_S:
                        await _worker_manager.ensure("gemma-worker")
        except Exception:
            logger.warning("idle LLM backfill failed", exc_info=True)
        await asyncio.sleep(2)


async def _refresh_metadata_loop() -> None:
    while True:
        try:
            if _worker_manager is not None and _registration is not None:
                meta = await _worker_manager.check_health()
                meta["capabilities"] = CAPABILITIES
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
