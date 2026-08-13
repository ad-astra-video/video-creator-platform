"""gemma-worker HTTP service (aiohttp).

A swappable worker container driven by the `live-runner` edge on the internal
Docker network — it does NOT register with the Livepeer Orchestrator or do
heartbeats (that's the live-runner's job). It exposes the control + inference
surface the live-runner needs, mirroring ltx-worker / idv2v-worker:

    GET  /health                          — liveness + model-loaded status
    POST /load                            — load the Gemma GGUF (mmap)
    POST /evict                           — drop the model, free GPU layers
    POST /video-creator/v1/prompt-enhance — rewrite a generation prompt
    POST /video-creator/v1/chat           — general agent chat (future frontend)

Auth: every POST requires the shared `X-Worker-Token` header (WORKER_TOKEN env).
Concurrency: at most GEMMA_MAX_PARALLEL prompt executions admitted concurrently;
actual GPU eval is serialized inside GemmaLLM.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from typing import Any

from aiohttp import web

from . import config
from .model import GemmaLLM
from runner.ltx.enhance_forward import DEFAULT_I2V_SYSTEM_PROMPT, DEFAULT_T2V_SYSTEM_PROMPT

logger = logging.getLogger("video_creator.runner.gemma.server")

MODEL_NAME = "gemma-4-12B-it-qat-q4_0"

# ── Auth ─────────────────────────────────────────────────────────────────────

def _require_token(request: web.Request) -> None:
    """Reject the request unless it carries the shared worker token."""
    expected = config.worker_token()
    provided = request.headers.get("X-Worker-Token", "")
    if not provided or provided != expected:
        raise web.HTTPForbidden(reason="missing/mismatched X-Worker-Token")


# ── Model lifecycle ──────────────────────────────────────────────────────────

_llm: GemmaLLM | None = None
_llm_build_lock = threading.Lock()
# Admission cap: at most GEMMA_MAX_PARALLEL prompt executions in flight.
_parallel = asyncio.Semaphore(config.GEMMA_MAX_PARALLEL)


def _get_llm() -> GemmaLLM:
    global _llm
    if _llm is None:
        with _llm_build_lock:
            if _llm is None:
                _llm = GemmaLLM(
                    model_path=config.GEMMA_MODEL,
                    mmproj=config.GEMMA_MMPROJ or None,
                    n_gpu_layers=config.GEMMA_N_GPU_LAYERS,
                    main_gpu=config.gemma_device_index(),
                    n_ctx=config.N_CTX,
                )
    return _llm


async def handle_load(request: web.Request) -> web.Response:
    _require_token(request)
    llm = _get_llm()
    if not llm.is_ready:
        try:
            await asyncio.wait_for(asyncio.to_thread(llm.load), timeout=3600)
        except asyncio.TimeoutError:
            return web.json_response({"error": "model load timed out"}, status=504)
    return web.json_response({"loaded": True, "already_loaded": llm.is_ready})


async def handle_evict(request: web.Request) -> web.Response:
    _require_token(request)
    llm = _get_llm()
    await asyncio.to_thread(llm.evict)
    return web.json_response({"evicted": True})


async def handle_health(_request: web.Request) -> web.Response:
    llm = _get_llm()
    return web.json_response({
        "status": "ok",
        "model_loaded": llm.is_ready,
        "model": MODEL_NAME,
        "device": config.GEMMA_GPU_DEVICE or "shared",
        "main_gpu": config.gemma_device_index(),
        "n_gpu_layers": config.GEMMA_N_GPU_LAYERS,
        "max_parallel": config.GEMMA_MAX_PARALLEL,
        "dedicated": config.is_dedicated_gpu(),
    })


# ── Inference ────────────────────────────────────────────────────────────────

def _build_enhance_messages(body: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Build the OpenAI-style chat messages for prompt enhancement.

    Mirrors runner.ltx.enhance_forward: a system prompt + either a text user
    message, or an image+text user message when image_base64 is supplied (the
    mmproj vision path — only if GEMMA_MMPROJ is configured).
    """
    prompt = str(body.get("prompt") or "").strip()
    image_b64 = body.get("image_base64")
    has_image = bool(image_b64)
    system_prompt = body.get("system_prompt")
    if has_image:
        import base64 as _b64
        head = _b64.b64decode(image_b64[:64])
        mime = "image/jpeg" if head[:3] == b"\xff\xd8\xff" else "image/png"
        system = system_prompt or DEFAULT_I2V_SYSTEM_PROMPT
        user_content: Any = [
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
        ]
    else:
        system = system_prompt or DEFAULT_T2V_SYSTEM_PROMPT
        user_content = f"user prompt: {prompt}"
    return [{"role": "system", "content": system},
            {"role": "user", "content": user_content}], has_image


async def handle_prompt_enhance(request: web.Request) -> web.Response:
    _require_token(request)
    body = await request.json()
    prompt = str(body.get("prompt") or "").strip()
    if not prompt:
        return web.json_response({"error": "missing 'prompt'"}, status=400)
    messages, has_image = _build_enhance_messages(body)
    seed = body.get("seed")
    llm = _get_llm()
    if not llm.is_ready:
        return web.json_response({"error": "Gemma LLM not loaded"}, status=503)
    async with _parallel:
        try:
            enhanced = await asyncio.to_thread(
                llm.chat, messages, seed=seed if isinstance(seed, int) else None
            )
        except Exception as exc:
            logger.error("Prompt enhance failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"enhanced_prompt": enhanced, "image": has_image})


async def handle_chat(request: web.Request) -> web.Response:
    _require_token(request)
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return web.json_response({"error": "missing 'messages' (list)"}, status=400)
    llm = _get_llm()
    if not llm.is_ready:
        return web.json_response({"error": "Gemma LLM not loaded"}, status=503)
    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))
    seed = body.get("seed")
    async with _parallel:
        try:
            content = await asyncio.to_thread(
                llm.chat, messages, max_tokens=max_tokens,
                temperature=temperature,
                seed=seed if isinstance(seed, int) else None,
            )
        except Exception as exc:
            logger.error("Chat failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({"content": content, "model": MODEL_NAME})


# ── App factory ──────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application(client_max_size=config.MAX_BODY_BYTES)
    # Control surface at ROOT (the live-runner posts /load + /evict to {base}/).
    app.router.add_get("/health", handle_health)
    app.router.add_post("/load", handle_load)
    app.router.add_post("/evict", handle_evict)
    # Inference under the uniform /video-creator/v1/{endpoint} + legacy /v1.
    app.router.add_post("/v1/prompt-enhance", handle_prompt_enhance)
    app.router.add_post("/video-creator/v1/prompt-enhance", handle_prompt_enhance)
    app.router.add_post("/v1/chat", handle_chat)
    app.router.add_post("/video-creator/v1/chat", handle_chat)
    return app


async def _run() -> None:
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, config.HOST, config.PORT)
    await site.start()
    logger.info("gemma-worker listening on %s:%d", config.HOST, config.PORT)
    await asyncio.Event().wait()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    # Resolve the auth token eagerly so a blank one is generated + logged once.
    config.worker_token()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
