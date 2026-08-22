"""Live-runner post-chain: orchestrate the vp-worker after a render.

When a generation request carries a ``post`` section:
    post: {fps_boost?: {target_fps, mode?}, upscale?: {final, scale?}}
the live-runner renders at the worker's native resolution first, then POSTs the
returned ``output_video`` to the vp-worker's combined ``/process`` stage
(RIFE -> FlashVSR -> ffmpeg finalize, one pass) and returns THAT result to the
browser.

Render workers are deliberately kept dumb — they never call vp-worker; the
live-runner is the single orchestrator (per the user's architecture decision).
The frontend decides the engine + whether post-processing is wanted; this stays
backend-route-based (no goal-intent logic).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from aiohttp import ClientTimeout, web

from . import config
from .routing import ROUTES

logger = logging.getLogger("video_creator.runner.live_runner.postchain")

# Render endpoints that are legitimately post-processable. Restyle chains its
# own two-worker flow and is NOT here (its result goes straight back).
_POSTABLE = {"t2v", "i2v", "bernini-t2v", "bernini-v2v", "bernini-r2v",
             "extend", "retake"}


def post_requested(body: dict) -> bool:
    """Does this request ask for a vp-worker post stage?"""
    post = body.get("post")
    return bool(post and (post.get("fps_boost") or post.get("upscale")))


async def apply(session, token: str, endpoint: str, body: dict,
                render_out: web.Response) -> Optional[web.Response]:
    """Run the vp-worker /process stage on ``render_out``'s video, if requested.

    ``render_out`` is the render worker's web.Response (its JSON body carries
    ``output_video`` in base64). If the request asked for post-processing and
    the render returned a video, forwards it to vp-worker /process and returns
    the final web.Response; otherwise returns None (caller keeps render_out).
    """
    if endpoint not in _POSTABLE or not post_requested(body):
        return None
    try:
        payload = json.loads(render_out.body.decode("utf-8"))
    except Exception:
        logger.warning("post-chain: render response not JSON, skipping")
        return None
    video = payload.get("output_video")
    if not video:
        logger.warning("post-chain: render returned no output_video, skipping")
        return None

    post = body.get("post") or {}
    proc_body = {"video": video,
                 "fps_boost": post.get("fps_boost"),
                 "upscale": post.get("upscale")}
    url = f"{config.VP_WORKER_URL.rstrip('/')}/video-creator/v1/process"
    headers = {"X-Worker-Token": token}
    try:
        async with session.post(url, json=proc_body, headers=headers,
                                timeout=ClientTimeout(total=3600.0)) as resp:
            raw = await resp.read()
            if resp.status >= 400:
                err = raw[:500].decode("utf-8", "replace")
                logger.warning("post-chain process failed (%s): %s", resp.status, err)
                return web.Response(status=resp.status, text=err,
                                    content_type="application/json", charset="utf-8")
            # Return the process result, tagged so the client knows the
            # resolution/fps came from the post stage.
            try:
                data = json.loads(raw.decode("utf-8"))
                data["post_processed"] = True
                return web.json_response(data)
            except Exception:
                return web.Response(status=resp.status, body=raw,
                                    content_type=resp.content_type or "application/json",
                                    charset=resp.charset)
    except Exception as exc:
        logger.exception("post-chain vp-worker unreachable")
        return web.json_response({"error": f"post processing failed: {exc}",
                                  "post_processed": False}, status=502)
