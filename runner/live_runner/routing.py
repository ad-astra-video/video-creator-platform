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
    "image": "ltx-worker",
    "extend": "ltx-worker",
    "retake": "ltx-worker",
    "prompt-enhance": "gemma-worker",
    "suggest-gap-prompt": "gemma-worker",
    "chat": "gemma-worker",
    "extract-conditioning": "ltx-worker",
    "ic-lora-generate": "ltx-worker",
    "edit": "ltx-worker",
    "restyle": "idv2v-worker",
    "sam3": "idv2v-worker",
}

CAPABILITIES = sorted({"restyle", "t2v", "i2v", "image", "edit", "sam3",
                       "extend", "retake", "prompt-enhance", "suggest-gap-prompt",
                       "chat", "extract-conditioning", "ic-lora-generate"})


async def proxy(
    worker_manager: "ResidentWorkerManager",
    session,
    token: str,
    worker: str,
    endpoint: str,
    body: dict,
) -> web.Response:
    """Ensure ``worker`` is resident, forward the request, return the response.

    The live-runner's HTTP client is a single aiohttp session so connections
    to the worker containers are pooled. Every upstream call carries
    X-Worker-Token so the worker accepts the swap/inference request.
    """
    from . import config as cfg
    base = cfg.WORKERS[worker]
    await worker_manager.ensure(worker)
    url = f"{base}/video-creator/v1/{endpoint}"

    # Live runner heartbeats are asyncio background tasks; a long restyle must
    # not block them, so the upstream call is an async aiohttp request (it
    # yields to the loop) — no thread executor needed for the network I/O.
    headers = {"X-Worker-Token": token}
    async with session.post(
        url, json=body, headers=headers,
        timeout=ClientTimeout(total=3600.0),
    ) as resp:
        # Read raw bytes so we relay the body byte-for-byte (the worker's image
        # results are base64 JSON, but future workers may return binary media).
        body_bytes = await resp.read()
        # aiohttp parses the upstream Content-Type into media type + charset.
        # Passing the raw header into web.Response(content_type=...) explodes
        # with "charset must not be in content_type argument" when the worker
        # sends e.g. "application/json; charset=utf-8", so use the split parts.
        content_type = resp.content_type or "application/json"
        charset = resp.charset
        if resp.status >= 400:
            text = body_bytes[:500].decode(charset or "utf-8", "replace")
            logger.error("Worker %s/%s -> %s: %s", worker, endpoint, resp.status, text)
            return web.Response(status=502, text=text,
                                content_type="application/json", charset="utf-8")
        return web.Response(status=resp.status, body=body_bytes,
                            content_type=content_type, charset=charset)
