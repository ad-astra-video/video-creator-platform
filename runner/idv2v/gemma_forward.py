"""Forward id-v2v prompt-enhance + auto-caption to the shared gemma-worker.

The id-v2v worker historically embedded its own Gemma 3 (~24.5 GB) for restyle
prompt enhancement and source-video captioning, forcing an evict-negotiate
choreography with the resident id-v2v DiT on a shared GPU. This module lets it
instead forward those requests to the shared llama.cpp gemma-worker (Gemma 4 12B
QAT) over the compose network via that worker's native
``/video-creator/v1/prompt-enhance`` endpoint. When GEMMA_FORWARD_URL is set and
the gemma-worker is reachable, the id-v2v worker never loads its own Gemma 3.

Wire contract (runner/gemma/server.py ``_build_enhance_messages`` +
``handle_prompt_enhance``):

    POST {base}/video-creator/v1/prompt-enhance
    headers: X-Worker-Token: <shared WORKER_TOKEN>
    body:    {prompt, system_prompt?, context_frames?: [b64jpeg...],
              image_base64?: str, seed?: int, task?, direction?}
    ->       {enhanced_prompt: str, image: bool}

Every worker container shares the same WORKER_TOKEN (docker-compose), so the
id-v2v worker's own ``config.worker_token()`` is the correct X-Worker-Token to
present to the gemma-worker.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import aiohttp

logger = logging.getLogger("video_creator.runner.idv2v.gemma_forward")


async def gemma_worker_available(base_url: str, token: str | None = None,
                                 timeout: float = 5.0) -> bool:
    """Probe the gemma-worker ``/health`` endpoint (HTTP 200 == up).

    ``GET /health`` is public (no X-Worker-Token required) on the gemma-worker,
    so a missing/blank token still probes correctly.
    """
    url = base_url.rstrip("/") + "/health"
    headers = {"X-Worker-Token": token} if token else {}
    try:
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(url, headers=headers) as resp:
                return resp.status == 200
    except Exception:
        return False


async def forward_prompt_enhance(
    base_url: str,
    *,
    prompt: str,
    system_prompt: str,
    context_frames: list[str] | None = None,
    image_base64: str | None = None,
    seed: int | None = None,
    token: str | None = None,
    timeout: float = 90.0,
) -> str:
    """Forward one enhance/caption request to the gemma-worker.

    Args:
        base_url: gemma-worker base URL, e.g. ``http://gemma-worker:8993``.
        prompt: the user's prompt (for captioning, pass
            ``"Describe this video clip."``).
        system_prompt: the id-v2v worker's dedicated system prompt (color-fidelity
            restyle enhance, caption, image-edit, or text enhance) — this is what
            keeps the id-v2v restyle semantics while running on the shared worker.
        context_frames: base64 JPEG strings for multimodal captioning (sampled
            source-video frames). Mutually exclusive with ``image_base64``.
        image_base64: single base64 image for image-edit / i2v enhancement.
        seed: optional integer seed forwarded for deterministic sampling.
        token: the shared X-Worker-Token (``config.worker_token()``).

    Returns the enhanced/captioned prompt string. Raises on a non-200 response
    or an unexpected/empty response shape.
    """
    payload: dict[str, Any] = {"prompt": prompt, "system_prompt": system_prompt}
    if context_frames:
        payload["context_frames"] = context_frames
    elif image_base64:
        payload["image_base64"] = image_base64
    if seed is not None:
        payload["seed"] = seed

    url = base_url.rstrip("/") + "/video-creator/v1/prompt-enhance"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Worker-Token"] = token

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_obj) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            raw = await resp.text()
            if resp.status != 200:
                raise RuntimeError(
                    "gemma-worker prompt-enhance %s: %s" % (resp.status, raw[:500]))
            try:
                data = json.loads(raw)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError("gemma-worker non-JSON response: %s" % exc) from exc

    enhanced = data.get("enhanced_prompt")
    if not isinstance(enhanced, str) or not enhanced.strip():
        raise RuntimeError("gemma-worker returned empty enhanced_prompt")
    return enhanced.strip()
