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
from runner.ltx.enhance_forward import (
    DEFAULT_I2V_SYSTEM_PROMPT,
    DEFAULT_T2V_SYSTEM_PROMPT,
    DEFAULT_EXTEND_SYSTEM_PROMPT,
    DEFAULT_RETAKE_SYSTEM_PROMPT,
)

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
                )  # flash_attn is always on (GemmaLLM default True; required)
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




# The rubric used to ask Gemma how many semantic layers an image decomposes into.
# The model is asked to THINK through the scene before committing to a count, then
# output only the number on the final line so the server's regex can extract it.
LAYER_SUGGEST_RUBRIC = (
    "You are a meticulous image-decomposition analyst. Analyze the image and "
    "select the appropriate Qwen-Image-Layered layer count.\n\n"
    "Use this rubric:\n"
    "- 2 = simple image with one main subject and simple background\n"
    "- 3 = main subject + background + one distinct secondary element\n"
    "- 4 = several distinct objects or regions\n"
    "- 5 = moderately complex scene with multiple overlapping objects\n"
    "- 6 = complex scene with many independently editable objects\n"
    "- 7 = very complex scene with many distinct overlapping elements\n"
    "- 8 = extremely complex scene where separating many objects is useful\n\n"
    "Think step by step before answering:\n"
    "1. Identify every semantically distinct element or region in the image.\n"
    "2. Consider background, foreground subjects, and any independent objects.\n"
    "3. Estimate whether each element is cleanly separable for editing.\n"
    "4. Choose the SMALLEST number that adequately represents the image.\n"
    "5. Show your reasoning in <think>...</think> tags, then answer with ONLY "
    "the final integer 2-8 on the very last line, inside a single pair of "
    "angle brackets, e.g. <5>.\n"
    "Put no other text after the bracketed number."
)

# ── Inference ────────────────────────────────────────────────────────────────

def _build_enhance_messages(body: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Build the OpenAI-style chat messages for prompt enhancement.

    Three shapes, mirroring runner.ltx.enhance_forward:
      - context_frames: a LIST of base64 frames sampled from the source video's context
        window (the edge an extend continuation attaches to) -> EXTEND system prompt and a
        multimodal user message with each frame as an image_url part.
      - image_base64 (single): i2v / image-edit enhancement -> DEFAULT_I2V_SYSTEM_PROMPT
        with one image_url part.
      - neither: plain text -> DEFAULT_T2V_SYSTEM_PROMPT.

    A caller-supplied `system_prompt` (body) always overrides the relevant default.
    """
    prompt = str(body.get("prompt") or "").strip()
    system_prompt = body.get("system_prompt")

    context_frames = body.get("context_frames")
    if isinstance(context_frames, list) and context_frames:
        import base64 as _b64
        user_content: Any = []
        for frame in context_frames:
            if not isinstance(frame, str) or not frame:
                continue
            head = _b64.b64decode(frame[:64]) if len(frame) >= 64 else b""
            mime = "image/jpeg" if head[:3] == b"\xff\xd8\xff" else "image/png"
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{frame}"},
            })
        task = str(body.get("task") or "").strip()
        if task == "retake":
            system = system_prompt or DEFAULT_RETAKE_SYSTEM_PROMPT
            note = " Task: re-render this selected segment."
        else:
            # extend (default for context_frames)
            system = system_prompt or DEFAULT_EXTEND_SYSTEM_PROMPT
            direction = str(body.get("direction") or "").strip()
            note = f" Extend direction: {direction}." if direction else ""
        user_content.append({
            "type": "text",
            "text": f"User Raw Input Prompt: {prompt}.{note}",
        })
        return [{"role": "system", "content": system},
                {"role": "user", "content": user_content}], True

    image_b64 = body.get("image_base64")
    has_image = bool(image_b64)
    if has_image:
        import base64 as _b64
        head = _b64.b64decode(image_b64[:64])
        mime = "image/jpeg" if head[:3] == b"\xff\xd8\xff" else "image/png"
        system = system_prompt or DEFAULT_I2V_SYSTEM_PROMPT
        user_content = [
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


async def handle_suggest_layers(request: web.Request) -> web.Response:
    """POST /video-creator/v1/suggest-layers.

    Request:  {image: <b64 png>}
    Response: {layers: int (2-8) | null, raw: str}

    Feeds the uploaded image + the Qwen-Image-Layered layer-count rubric to the
    multimodal Gemma chat and returns the parsed integer 2-8 (null when the LLM
    doesn't produce a parseable number, so the caller falls back to its default).
    """
    _require_token(request)
    body = await request.json()
    image = body.get("image")
    if not image:
        return web.json_response({"error": "missing 'image'"}, status=400)
    import base64 as _b64
    head = _b64.b64decode(image[:64])
    mime = "image/jpeg" if head[:3] == b"\xff\xd8\xff" else "image/png"
    messages = [
        {"role": "system", "content": LAYER_SUGGEST_RUBRIC},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image}"}},
            {"type": "text", "text": "How many layers should this image be decomposed into?"},
        ]},
    ]
    llm = _get_llm()
    if not llm.is_ready:
        return web.json_response({"error": "Gemma LLM not loaded"}, status=503)
    async with _parallel:
        try:
            content = await asyncio.to_thread(llm.chat, messages, max_tokens=512,
                                              temperature=0.6)
        except Exception as exc:
            logger.error("suggest-layers failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)
    # Parse the bracketed <N> (2-8) out of the response; fall back to the first
    # lone integer 2-8 if the model didn't bracket it.
    import re
    m = re.search(r"<([2-8])>", content) or re.search(r"\b([2-8])\b", content)
    layers = int(m.group(1)) if m else None
    return web.json_response({"layers": layers, "raw": content})


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
    app.router.add_post("/video-creator/v1/suggest-layers", handle_suggest_layers)
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
