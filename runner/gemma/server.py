"""gemma-worker HTTP service (aiohttp) — front + control surface over llama-server.

The native llama.cpp ``llama-server`` is the SINGLE resident model host (built
from upstream ggml-org/llama.cpp; reasoning + tool-calling). This aiohttp worker
is the swappable front the live-runner drives, mirroring ltx-worker /
idv2v-worker. It exposes the root control + inference surface and proxies every
inference endpoint to the managed llama-server subprocess:

    GET  /health                          — liveness + llama-server status
    POST /load                            — spawn llama-server (model resident)
    POST /evict                           — stop llama-server (free GPU)
    GET  /video-creator/v1/info           — uniform worker info for the scheduler
    POST /video-creator/v1/prompt-enhance — proxy: rewrite a generation prompt
    POST /video-creator/v1/chat           — proxy: general agent chat
    POST /video-creator/v1/suggest-layers — proxy: Qwen layer-count rubric
    POST /video-creator/v1/suggest-gap-prompt — gap-fill clip prompt (prebuilt messages)

Auth: every POST requires the shared `X-Worker-Token` header (WORKER_TOKEN env),
which is also forwarded to llama-server as a Bearer token. Concurrency: at most
GEMMA_MAX_PARALLEL prompt executions admitted; llama-server serializes actual GPU
eval (--parallel 1).
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from typing import Any

from aiohttp import web

from . import config
from . import llama_server as llms
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


# ── Admission ────────────────────────────────────────────────────────────────

# Admission cap: at most GEMMA_MAX_PARALLEL prompt executions in flight.
_parallel = asyncio.Semaphore(config.GEMMA_MAX_PARALLEL)


async def handle_load(request: web.Request) -> web.Response:
    _require_token(request)
    # ``model`` accepted for symmetry with every worker's /load; gemma-worker
    # is single-model (MODEL_NAME), so a hint that differs is logged + ignored.
    model = None
    try:
        model = (await request.json()).get("model")
    except Exception:
        pass
    if model and model != MODEL_NAME:
        logger.info("gemma-worker /load: model=%s (serving %s; ignoring)", model, MODEL_NAME)
    already = await llms.is_running()
    try:
        await asyncio.wait_for(llms.ensure_running(), timeout=3600)
    except asyncio.TimeoutError:
        return web.json_response({"error": "llama-server start timed out"}, status=504)
    return web.json_response({"loaded": True, "already_loaded": already})


async def handle_evict(request: web.Request) -> web.Response:
    _require_token(request)
    await llms.stop()
    return web.json_response({"evicted": True})


async def handle_health(_request: web.Request) -> web.Response:
    running = await llms.is_running()
    return web.json_response({
        "status": "ok",
        "model_loaded": running,
        "model": MODEL_NAME,
        "device": config.GEMMA_GPU_DEVICE or "shared",
        "main_gpu": config.gemma_device_index(),
        "n_gpu_layers": config.GEMMA_N_GPU_LAYERS,
        "max_parallel": config.GEMMA_MAX_PARALLEL,
        "dedicated": config.is_dedicated_gpu(),
        "agent": {
            "running": running,
            "base_url": llms.agent_base_url() if running else None,
            "port": llms.AGENT_PORT,
        },
    })


async def handle_info(_request: web.Request) -> web.Response:
    """GET /video-creator/v1/info — worker liveness + GPU ownership.

    Uniform with the image/ltx/idv2v workers so the live-runner's scheduler can
    reconcile the advisory GPU map from every worker. ``device_in_use`` is the
    PHYSICAL GPU index this container is pinned to (from GEMMA_RESIDENT_GPU):
    because CUDA_VISIBLE_DEVICES pins the container to one card, the in-container
    index is always 0, so the LOCAL index would be wrong for the scheduler. When
    gemma is a shared/evictable idle slot (GEMMA_GPU_DEVICE blank, no dedicated
    card), it owns no single GPU -> device_in_use is None so the scheduler never
    pins a card to it.
    """
    running = await llms.is_running()
    return web.json_response({
        "app": "video-creator",
        "model": MODEL_NAME,
        "ready": running,
        "model_loaded": running,
        "device": config.GEMMA_GPU_DEVICE or "shared",
        "main_gpu": config.gemma_device_index(),
        "dedicated": config.is_dedicated_gpu(),
        "devices_visible": 1,
        # Physical card owned (dedicated pin) or None (shared/evictable).
        "device_in_use": config.GEMMA_PHYSICAL_GPU if config.is_dedicated_gpu() else None,
        # Native OpenAI-compatible agent endpoint provided by llama-server.
        "agent_base_url": llms.agent_base_url() if running else None,
        "agent_port": llms.AGENT_PORT,
    })


# ── Inference (proxied to llama-server) ──────────────────────────────────────

# LEGACY local rubric — the client (webapp lib/llm-messages.ts) is the single source
# of truth and sends prebuilt `messages`; this constant is kept only for internal
# forwards that send raw {image} instead.
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
    "5. Show your reasoning in  thinking... response tags, then answer with ONLY "
    "the final integer 2-8 on the very last line, inside a single pair of "
    "angle brackets, e.g. <5>.\n"
    "Put no other text after the bracketed number."
)


def _messages_have_image(messages: list[dict[str, Any]]) -> bool:
    """True when any message carries an image_url part (multimodal)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


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
    llama-server accepts these OpenAI-format messages directly (incl. multimodal
    image_url data-URL parts).

    LEGACY: message building now happens client-side (webapp's lib/llm-messages.ts,
    the single source of truth). This builder is kept ONLY for internal/older
    forwards that send raw fields instead of prebuilt `messages`.
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
    seed = body.get("seed")
    messages = body.get("messages")
    prebuilt = isinstance(messages, list) and len(messages) > 0
    if prebuilt:
        # Client-authored messages (webapp lib/llm-messages.ts) execute verbatim —
        # the browser is the single source of truth for every prompt.
        has_image = _messages_have_image(messages)
    else:
        # LEGACY: build messages here for internal/older forwards only.
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            return web.json_response({"error": "missing 'prompt'"}, status=400)
        messages, has_image = _build_enhance_messages(body)
        seed = body.get("seed")
    async with _parallel:
        try:
            reasoning, enhanced = await llms.chat_with_reasoning(
                messages, max_tokens=4096, temperature=0.7,
                seed=seed if isinstance(seed, int) else None,
            )
        except Exception as exc:
            logger.error("Prompt enhance failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)
    return web.json_response({
        "enhanced_prompt": enhanced,
        "reasoning_content": reasoning,
        "image": has_image,
    })


async def handle_chat(request: web.Request) -> web.Response:
    _require_token(request)
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return web.json_response({"error": "missing 'messages' (list)"}, status=400)
    max_tokens = int(body.get("max_tokens", 512))
    temperature = float(body.get("temperature", 0.7))
    seed = body.get("seed")
    async with _parallel:
        try:
            content = await llms.chat(
                messages, max_tokens=max_tokens, temperature=temperature,
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
    multimodal llama-server chat and returns the parsed integer 2-8 (null when the
    LLM doesn't produce a parseable number, so the caller falls back to its default).
    Prebuilt client `messages` (webapp lib/llm-messages.ts) execute verbatim when
    present — the local RUBRIC builder below is the LEGACY path for internal forwards.
    """
    _require_token(request)
    body = await request.json()
    messages = body.get("messages")
    prebuilt = isinstance(messages, list) and len(messages) > 0
    if not prebuilt:
        image = body.get("image")
        if not image:
            return web.json_response({"error": "missing 'image'"}, status=400)
        import base64 as _b64
        image_bytes = _b64.b64decode(image)
        # Normalize to PNG before handing to llama-server. Its multimodal reader
        # only reliably decodes PNG (JPEG is flaky) and rejects WebP/other formats
        # with "Failed to load image or audio file"; the worker's own JPEG-vs-PNG
        # sniff would mislabel WebP bytes as PNG. Pillow (bundled) re-encodes any
        # supported format, so suggest-layers works for browser-native WebP/JPEG
        # imports too, not just generated PNGs.
        try:
            from io import BytesIO as _BytesIO
            from PIL import Image as _PIL
            with _PIL.open(_BytesIO(image_bytes)) as im:
                with _BytesIO() as buf:
                    im.convert("RGB").save(buf, format="PNG")
                    image_bytes = buf.getvalue()
            mime = "image/png"
        except Exception:
            # Not Pillow-decodable (unexpected bytes): keep the raw image and fall
            # back to the old sniff so llama-server rejects cleanly rather than us
            # 500ing on a corrupt/unsupported image here.
            mime = "image/jpeg" if image_bytes[:3] == b"\xff\xd8\xff" else "image/png"
        image_b64 = _b64.b64encode(image_bytes).decode()
        messages = [
            {"role": "system", "content": LAYER_SUGGEST_RUBRIC},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                {"type": "text", "text": "How many layers should this image be decomposed into?"},
            ]},
        ]
    async with _parallel:
        try:
            reasoning, content = await llms.chat_with_reasoning(
                messages, max_tokens=512, temperature=0.6,
            )
        except Exception as exc:
            logger.error("suggest-layers failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)
    # The rubric asks for step-by-step thinking and llama-server runs with
    # --reasoning on --reasoning-format deepseek, so the model's whole reply
    # (sometimes including the final count) can land in the reasoning channel
    # with `content` left empty. Parse both channels together.
    blob = f"{reasoning}\n{content}" if reasoning else content
    import re
    m = re.search(r"<([2-8])>", blob) or re.search(r"\b([2-8])\b", blob)
    layers = int(m.group(1)) if m else None
    return web.json_response({"layers": layers, "raw": blob})


async def handle_suggest_gap_prompt(request: web.Request) -> web.Response:
    """POST /video-creator/v1/suggest-gap-prompt.

    Request:  {messages: [...]}            (client-authored, see lib/llm-messages.ts)
    Response: {suggested_prompt: str}

    Runs the prebuilt messages through the multimodal llama-server chat and returns
    the suggested gap-fill clip prompt, stripping any inline thinking/reasoning tags
    (Gemma may land the whole reply in the reasoning channel with content empty, or
    emit <start_of_thinking>...</end_of_thinking> inline in content).
    """
    _require_token(request)
    body = await request.json()
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return web.json_response({"error": "missing 'messages' (list)"}, status=400)
    async with _parallel:
        try:
            reasoning, content = await llms.chat_with_reasoning(
                messages, max_tokens=2048, temperature=0.7,
            )
        except Exception as exc:
            logger.error("suggest-gap-prompt failed: %s", exc, exc_info=True)
            return web.json_response({"error": str(exc)}, status=500)
    import re
    blob = f"{reasoning}\n{content}" if reasoning else (content or "")
    blob = re.sub(r"<start_of_thinking>.*?</end_of_thinking>", "", blob, flags=re.S)
    blob = blob.strip()
    return web.json_response({"suggested_prompt": blob})


# ── App factory ──────────────────────────────────────────────────────────────

def create_app() -> web.Application:
    app = web.Application(client_max_size=config.MAX_BODY_BYTES)
    # Control surface at ROOT (the live-runner posts /load + /evict to {base}/).
    app.router.add_get("/health", handle_health)
    app.router.add_post("/load", handle_load)
    app.router.add_post("/evict", handle_evict)
    # Uniform /info for the live-runner's scheduler reconcile (reports the
    # physical pinned GPU as device_in_use; root + namespaced aliases).
    app.router.add_get("/video-creator/v1/info", handle_info)
    # Inference under the uniform /video-creator/v1/{endpoint} + legacy /v1.
    app.router.add_post("/v1/prompt-enhance", handle_prompt_enhance)
    app.router.add_post("/video-creator/v1/prompt-enhance", handle_prompt_enhance)
    app.router.add_post("/v1/chat", handle_chat)
    app.router.add_post("/video-creator/v1/chat", handle_chat)
    app.router.add_post("/video-creator/v1/suggest-layers", handle_suggest_layers)
    app.router.add_post("/v1/suggest-gap-prompt", handle_suggest_gap_prompt)
    app.router.add_post("/video-creator/v1/suggest-gap-prompt", handle_suggest_gap_prompt)
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

    _QUIET_ACCESS_PATHS = ("/health", "/info", "/progress")
    class _QuietAccessFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
            except Exception:  # noqa: BLE001 - never drop logs on a format error
                return True
            return not any(p in msg for p in _QUIET_ACCESS_PATHS)
    logging.getLogger("aiohttp.access").addFilter(_QuietAccessFilter())
    # Resolve the auth token eagerly so a blank one is generated + logged once.
    config.worker_token()
    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
