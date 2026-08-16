"""Forwarded (remote) prompt enhancement via an OpenAI-compatible endpoint.

By default prompt enhancement runs the runner's local Gemma encoder. This module
backstops an alternate deployment where a fleet of runners share ONE enhancement
model — e.g. a single llama.cpp / vLLM instance holding the Gemma encoder — so
each runner doesn't have to load the encoder into its own VRAM (or host RAM).

When ``ENHANCE_FORWARD_URL`` is configured the runner's ``/prompt-enhance``
handler proxies to ``<url>/v1/chat/completions`` (the OpenAI wire format) instead
of running Gemma locally. This module is the translation layer: it takes the
runner's custom ``{prompt, image_base64?, seed, system_prompt?}`` contract and
builds an OpenAI chat-completions request, then extracts the enhanced text from
the response.
"""
from __future__ import annotations

import base64
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# Default system prompts, mirroring the desktop app's generic free-rewrite
# fallback (services/prompt_enhancement/system_prompt.py). Overridable per-
# (system prompts are fixed constants — not env-overridable on the runner).
_OUTPUT_FORMAT_INSTRUCTION = (
    "Respond with ONLY the rewritten prompt, as a single continuous paragraph of "
    "natural language. No titles, headings, prefaces, explanations, quotes, or "
    "markdown formatting (no **bold**, no bullet points) — just the prompt text itself."
)

DEFAULT_T2V_SYSTEM_PROMPT = (
    "You are a prompt engineer for a text-to-video generation model. Rewrite the "
    "user's prompt into a single, vivid, concrete visual description — subject, "
    "action, setting, lighting, and camera framing — preserving the user's "
    "original subject and intent without inventing an unrelated scene. "
    + _OUTPUT_FORMAT_INSTRUCTION
)

DEFAULT_I2V_SYSTEM_PROMPT = (
    "You are a prompt engineer for a text-to-video generation model. The user has also "
    "provided a reference IMAGE that must drive the result: carefully inspect that image and "
    "rewrite their prompt into a single, vivid, concrete description of the result video — "
    "subject identity, appearance and pose, action, setting, composition, lighting, color "
    "palette, and camera framing. Ground every detail in what the image actually shows: "
    "faithfully preserve the image's subject and its visual appearance, surrounding scene, "
    "composition, and style, while describing the desired motion on top of it. Do not invent "
    "a different subject, character, or setting that contradicts the reference image. "
    + _OUTPUT_FORMAT_INSTRUCTION
)

DEFAULT_EXTEND_SYSTEM_PROMPT = (
    "You are a prompt engineer for a video EXTENSION (continuation) model. The user is "
    "extending an existing clip, and has provided a set of FRAMES sampled from the source "
    "video's context window — the stretch of footage right at the boundary the extension "
    "attaches to (the last second for an 'end' extension, the first second for a 'start' "
    "extension). Carefully inspect those frames and rewrite the user's short direction into "
    "a single, vivid continuation prompt that is fully grounded in the source: continue the "
    "same subject(s) with their identity, appearance, clothing and pose, the same setting, "
    "composition, framing, lighting, color palette and overall style shown in the frames, "
    "and the same motion/action already in progress — while honoring the motion or scene "
    "change the user's direction requests. Do not invent a different subject, character, "
    "location, or a jarring visual style that contradicts the context frames. The result "
    "must read as the natural next moment of the supplied footage. "
    + _OUTPUT_FORMAT_INSTRUCTION
)

DEFAULT_RETAKE_SYSTEM_PROMPT = (
    "You are a prompt engineer for a video RETAKE (re-render) model. The user is "
    "re-rendering a selected segment of an existing clip and has provided FRAMES sampled "
    "from that selected segment. Carefully inspect those frames and rewrite the user's short "
    "instruction into a single, vivid prompt that re-renders the segment to match the source "
    "closely: the same subject(s) with their identity, appearance, clothing and pose, the same "
    "setting, composition, framing, lighting, color palette and style, and the same general "
    "activity. Change ONLY the one specific thing the user's direction asks to change (for "
    "example, if a jet is moving erratically, have it straighten out its flight path) and "
    "leave every other visual element exactly as the frames show. Do not alter anything the "
    "direction does not explicitly touch. The result must remain visually continuous with the "
    "supplied frames. "
    + _OUTPUT_FORMAT_INSTRUCTION
)

def _image_mime(base64_image: str) -> str:
    """Sniff an image's MIME type from its header bytes (base64 input)."""
    try:
        head = base64.b64decode(base64_image[:64])
    except Exception:
        return "image/png"
    if head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


async def forward_prompt_enhance(
    base_url: str,
    *,
    prompt: str,
    system_prompt: str | None = None,
    image_base64: str | None = None,
    seed: int | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
    default_t2v: str = DEFAULT_T2V_SYSTEM_PROMPT,
    default_i2v: str = DEFAULT_I2V_SYSTEM_PROMPT,
) -> str:
    """Proxy a prompt-enhance request to an OpenAI-compatible chat endpoint.

    Mirrors the local Gemma message structure (system prompt + a ``user prompt: …``
    message, or an image+text user message for i2v). Returns the enhanced prompt
    string, raising on non-200 or an unexpected response shape.
    """
    if image_base64:
        resolved_system = system_prompt or default_i2v
        user_content: Any = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{_image_mime(image_base64)};base64,{image_base64}",
                },
            },
            {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
        ]
    else:
        resolved_system = system_prompt or default_t2v
        user_content = f"user prompt: {prompt}"

    payload: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": resolved_system},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "stream": False,
    }
    if model:
        payload["model"] = model
    if seed is not None:
        payload["seed"] = seed

    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_obj) as session:
        async with session.post(url, json=payload, headers=headers) as resp:
            if resp.status != 200:
                body = (await resp.text())[:500]
                raise RuntimeError(f"enhancer upstream {resp.status}: {body}")
            data = await resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected enhancer response shape: {exc}") from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("enhancer returned empty content")
    return content.strip()
