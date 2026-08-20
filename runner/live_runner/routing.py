"""Routing: capability -> worker route table + proxy helper.

Maps each /video-creator/v1/{endpoint} to the worker that serves it, then
forwards the request body to that worker (after the swap policy makes its model
resident) and streams the response back.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiohttp import ClientTimeout, web

from . import config

if TYPE_CHECKING:
    from .swap import ResidentWorkerManager

logger = logging.getLogger("video_creator.runner.live_runner.routing")

# capability/endpoint -> worker container name.
ROUTES = {
    "t2v": "ltx-worker",
    "i2v": "ltx-worker",
    "a2v": "ltx-worker",            # LTX worker answers not_supported today
    "image": "image-worker",
    "extend": "ltx-worker",
    "retake": "ltx-worker",
    "prompt-enhance": "gemma-worker",
    "suggest-gap-prompt": "gemma-worker",
    "chat": "gemma-worker",
    "suggest-layers": "gemma-worker",
    "extract-conditioning": "ltx-worker",
    "ic-lora-generate": "ltx-worker",
    "edit": "image-worker",
    "layer": "image-worker",
    "restyle": "idv2v-worker",
    "style-frame": "image-worker",
    "sam3": "idv2v-worker",
}

# Routing capability ids advertised for the platform (each maps 1:1 via ROUTES
# to the worker container that serves it). `qwen-image-edit` is deliberately
# NOT here: it was an engine label, not a route. Available image MODELS are
# advertised separately (see MODELS) so consumers can tell z-image / flux /
# qwen apart without a per-engine capability entry.
CAPABILITIES = sorted({"restyle", "style-frame", "t2v", "i2v", "image", "edit", "layer",
                       "sam3", "extend", "retake", "prompt-enhance",
                       "suggest-gap-prompt", "chat", "suggest-layers", "extract-conditioning",
                       "ic-lora-generate"})

# Image models the image-worker can serve, by id. Consumed by the frontend to
# label models (z-image = Z-Image, flux = FLUX.2 klein 4B, qwen = Qwen-Image-Edit,
# hidream = HiDream-O1-Image 8B UiT).
MODELS = ["z-image", "flux", "qwen", "hidream"]

# Endpoints served at the idv2v-worker that do NOT need the 20 GB video model
# resident: SAM3 is image-only (subprocess) and its own internal lifecycle
# evicts whatever is resident. Forcing /load here would build the whole
# WanVideoPipeline (DiT+VACE) and co-resident it with SAM3 on the shared
# 32 GB card -> OOM. (style-frame used to be here too, but it now routes to the
# image-worker, whose /load is cheap + device-aware, so it goes through the
# normal scheduled-ensure path.) Real video jobs (restyle) still use ensure().
_IMAGE_ONLY_ENDPOINTS = frozenset({"sam3"})


class WorkerCallFailed(Exception):
    """A worker call failed at the transport/upstream level (HTTP >= 400 or the
    worker was unreachable). Carries the status + body text so a caller that
    tries multiple candidates can both decide to fall back and report the last
    failure if every candidate fails."""

    def __init__(self, status: int, text: str):
        super().__init__(text)
        self.status = status
        self.text = text


# For prompt-enhance we PREFER the dedicated gemma-worker (Gemma 4 12B via
# llama.cpp). When that worker is absent/down, fall back to the LTX pipeline's
# own Gemma text encoder — ltx-worker /prompt-enhance runs
# engine.enhance_prompt (the provisioned Gemma QAT q4_0 text encoder), which is
# exactly the "if there is no gemma worker, use the text encoder provided by
# the LTX pipeline" contract. ROUTES still points prompt-enhance at
# gemma-worker so the swap policy keeps Gemma resident; only the proxy-time
# fallback chain changes.
_PROMPT_ENHANCE_FALLBACK: dict[str, str] = {"gemma-worker": "ltx-worker"}


def candidate_workers(endpoint: str, worker: str) -> list[str]:
    """Ordered worker candidates for ``endpoint``, primary first.

    Only prompt-enhance gets a fallback today: gemma-worker -> ltx-worker.
    Everything else is a single-worker route (unchanged behaviour).
    """
    if endpoint == "prompt-enhance" and worker in _PROMPT_ENHANCE_FALLBACK:
        return [worker, _PROMPT_ENHANCE_FALLBACK[worker]]
    return [worker]


async def _post_worker(session, token: str, worker: str, endpoint: str, body: dict,
                       device: int | None = None) -> web.Response:
    """POST ``body`` to ``worker``'s inference endpoint and relay the response.

    Raises WorkerCallFailed on any upstream error (HTTP >= 400 or connection
    failure) so the caller can fall back to another worker; returns a web.Response
    on success. This is a single aiohttp HTTP request (no thread executor needed)
    so the live-runner's heartbeat asyncio tasks are never blocked.

    ``device`` (when set) is forwarded as ``X-Worker-Device`` so a multi-engine
    worker runs THIS request on the specific GPU the scheduler assigned (the
    go-livepeer proxy is only on the browser->runner leg; this internal hop
    carries the header untouched). When None (legacy shared-GPU), no header is
    sent and the worker falls back to its default device.
    """
    from . import config as cfg
    base = cfg.WORKERS[worker]
    url = f"{base}/video-creator/v1/{endpoint}"
    headers = {"X-Worker-Token": token}
    if device is not None:
        headers["X-Worker-Device"] = str(device)
    try:
        async with session.post(
            url, json=body, headers=headers,
            timeout=ClientTimeout(total=3600.0),
        ) as resp:
            # Read raw bytes so we relay the body byte-for-byte (the worker's
            # image results are base64 JSON, but future workers may return binary
            # media).
            body_bytes = await resp.read()
            # aiohttp parses the upstream Content-Type into media type + charset.
            # Passing the raw header into web.Response(content_type=...) explodes
            # with "charset must not be in content_type argument" when the worker
            # sends e.g. "application/json; charset=utf-8", so use the split parts.
            content_type = resp.content_type or "application/json"
            charset = resp.charset
            if resp.status >= 400:
                text = body_bytes[:500].decode(charset or "utf-8", "replace")
                raise WorkerCallFailed(resp.status, text)
            return web.Response(status=resp.status, body=body_bytes,
                                content_type=content_type, charset=charset)
    except WorkerCallFailed:
        raise
    except Exception as exc:
        # Connection-level failure (worker container down / unreachable).
        raise WorkerCallFailed(502, f"{worker} unreachable: {exc}") from exc


async def proxy(
    worker_manager: "ResidentWorkerManager",
    session,
    token: str,
    worker: str,
    endpoint: str,
    body: dict,
    device: int | None = None,
) -> web.Response:
    """Ensure ``worker`` is resident, forward the request, return the response.

    The live-runner's HTTP client is a single aiohttp session so connections
    to the worker containers are pooled. Every upstream call carries
    X-Worker-Token so the worker accepts the swap/inference request.

    For prompt-enhance the primary worker is gemma-worker; if it is unavailable
    (down / not deployed / errored) the request automatically falls back to the
    LTX pipeline's Gemma text encoder (ltx-worker), so the UI always gets an
    enhanced prompt without needing a dedicated gemma worker.
    """
    candidates = candidate_workers(endpoint, worker)
    last: WorkerCallFailed | None = None
    for index, target in enumerate(candidates):
        try:
            # Image-only endpoints must NOT force the 20 GB video model resident
            # (see _IMAGE_ONLY_ENDPOINTS). ensure(worker) POSTs /load -> builds
            # the full WanVideoPipeline; doing that for a style-frame would
            # co-resident the video model with the klein editor and OOM the
            # shared 32 GB card. Let the worker's own klein/sam3 eviction
            # lifecycle handle residency for these.
            if endpoint not in _IMAGE_ONLY_ENDPOINTS:
                await worker_manager.ensure(target, device=device)
            return await _post_worker(session, token, target, endpoint, body, device)
        except WorkerCallFailed as exc:
            last = exc
            logger.warning(
                "Worker %s/%s unavailable (%s)%s",
                target, endpoint, exc,
                " - falling back" if index < len(candidates) - 1 else "",
            )
        except Exception as exc:  # ensure() itself may fail for a down worker
            last = WorkerCallFailed(502, f"{target} unavailable: {exc}")
            logger.warning(
                "Worker %s/%s failed before call (%s)%s",
                target, endpoint, exc,
                " - falling back" if index < len(candidates) - 1 else "",
            )

    # Every candidate failed — relay the last failure exactly as the old proxy
    # did (status + raw body) so clients that read res.json().error keep working
    # and the actual underlying worker/connection error stays visible.
    if last is not None:
        return web.Response(
            status=last.status if last.status >= 400 else 502,
            text=last.text, content_type="application/json", charset="utf-8",
        )
    return web.json_response({"error": f"no worker available for {endpoint}"}, status=502)


async def proxy_worker_sse(
    worker_manager, session, token, worker: str, endpoint: str, body: dict,
    client_resp: "web.StreamResponse", device: int | None = None,
) -> None:
    """Relay ``{worker}/{endpoint}?sse=1`` SSE stream verbatim to the browser.

    Endpoints whose worker emits text/event-stream itself (image-worker ``/layer``
    and ``/edit``; ltx-worker ``/extend``) stream `accepted` -> `progress`* ->
    `complete` (or `error`). We fan the bytes straight through over the same paid
    HTTP connection so the client sees live progress without buffering the (large)
    base64 result.
    """
    import asyncio
    from . import config as cfg
    await worker_manager.ensure(worker, device=device)
    base = cfg.WORKERS[worker]
    url = f"{base}/video-creator/v1/{endpoint}?sse=1"
    headers = {"X-Worker-Token": token}
    if device is not None:
        headers["X-Worker-Device"] = str(device)
    async with session.post(
        url, json=body, headers=headers,
        timeout=ClientTimeout(total=3600.0),
    ) as resp:
        if resp.status >= 400:
            err = (await resp.read())[:500].decode("utf-8", "replace")
            try:
                await client_resp.write(
                    f"event: error\ndata: {err}\n\n".encode("utf-8"))
            except Exception:
                pass
            return

        # Keep-alive heartbeat on the RELAYED stream. The worker's own SSE
        # (ltx /extend, image /layer, /edit) does not self-heartbeat, and a long
        # quiet leg (big decode / multi-chunk denoise) can exceed the browser's
        # SSE idle watchdog. Inject the same `: keepalive` comment the live-runner
        # uses on its own streams so the relayed connection ALWAYS has byte flow
        # and the client only times out on a genuinely dead connection.
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
                    await client_resp.write(b": keepalive\n\n")
                except Exception:
                    return
        beat = asyncio.create_task(_heartbeat())

        try:
            # Relay the worker's SSE byte-for-byte.
            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                try:
                    await client_resp.write(chunk)
                except Exception:
                    break
        finally:
            stop_beat.set()
            beat.cancel()
            try:
                await beat
            except (asyncio.CancelledError, Exception):
                pass
