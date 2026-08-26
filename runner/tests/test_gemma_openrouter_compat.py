"""GPU-independent tests for the gemma-worker OpenRouter-compat paths.

The browser now authors every LLM prompt client-side (webapp lib/llm-messages.ts)
and sends PREBUILT OpenAI `messages` to the runner, which must execute them
verbatim — pure execution, no server-side prompt building. These tests pin:

  - handle_prompt_enhance executes a passed `messages` list verbatim (no `prompt`
    field required), and keeps the raw-input builder as a legacy fallback.
  - handle_suggest_layers accepts prebuilt `messages` (no `image` field required).
  - handle_suggest_gap_prompt requires `messages` (missing -> 400) and returns
    `suggested_prompt`, stripping any inline thinking tags.

llms.chat_with_reasoning is stubbed (no model/GPU needed).
"""

import asyncio

import pytest
import aiohttp

from runner.gemma import config
from runner.gemma import server as gemma_server


PREBUILT_ENHANCE = [
    {"role": "system", "content": "Rewrite this creative prompt."},
    {"role": "user", "content": "User Raw Input Prompt: a dog running."},
]
PREBUILT_LAYERS = [
    {"role": "system", "content": "You are a layer analyst."},
    {"role": "user", "content": "How many layers?"},
]
PREBUILT_GAP = [
    {"role": "system", "content": "Write the clip that fills the timeline gap."},
    {"role": "user", "content": "gapDuration 2.5 mode text-to-video"},
]


@pytest.fixture
def gemma_client(monkeypatch):
    """Boot the aiohttp app with llms.chat_with_reasoning stubbed."""
    calls = {}

    async def fake_chat_with_reasoning(messages, **kwargs):
        calls["messages"] = messages
        calls["kwargs"] = kwargs
        outcome = calls["outcome"]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(gemma_server.llms, "chat_with_reasoning",
                        fake_chat_with_reasoning)
    return calls


def _run(coro):
    return asyncio.run(coro)


def _post(app_runner, path, body):
    async def _go():
        runner = aiohttp.web.AppRunner(app_runner)
        await runner.setup()
        site = aiohttp.web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with aiohttp.ClientSession() as s:
                headers = {"X-Worker-Token": config.worker_token()}
                async with s.post(
                    f"http://127.0.0.1:{port}{path}",
                    json=body, headers=headers,
                ) as r:
                    payload = await r.json()
                    status = r.status
        finally:
            await runner.cleanup()
        return status, payload
    return _run(_go())


def test_enhance_prebuilt_messages_execute_verbatim(gemma_client):
    """prompt-enhance with `messages` runs them verbatim (no `prompt` needed)."""
    gemma_client["outcome"] = ("rt", "a cinematic dog running")
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/prompt-enhance",
                         {"messages": PREBUILT_ENHANCE})
    assert status == 200
    assert gemma_client["messages"] == PREBUILT_ENHANCE
    assert body["enhanced_prompt"] == "a cinematic dog running"
    assert body["image"] is False


def test_enhance_legacy_prompt_path_still_works(gemma_client):
    """Legacy raw-input path (prompt, no messages) still builds + runs messages."""
    gemma_client["outcome"] = ("rt", "rewritten")
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/prompt-enhance",
                         {"prompt": "a cat"})
    assert status == 200
    assert gemma_client["messages"][1]["role"] == "user"
    assert body["enhanced_prompt"] == "rewritten"


def test_enhance_missing_prompt_and_messages_rejected(gemma_client):
    """Neither prebuilt messages nor a prompt -> 400."""
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/prompt-enhance", {})
    assert status == 400


def test_suggest_layers_prebuilt_messages(gemma_client):
    """suggest-layers accepts prebuilt messages without an `image` field."""
    gemma_client["outcome"] = ("", "<4>")
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/suggest-layers",
                         {"messages": PREBUILT_LAYERS})
    assert status == 200
    assert gemma_client["messages"] == PREBUILT_LAYERS
    assert body["layers"] == 4


def test_suggest_gap_prompt_requires_messages(gemma_client):
    "suggest-gap-prompt with no messages -> 400."
    app = gemma_server.create_app()
    status, _ = _post(app, "/video-creator/v1/suggest-gap-prompt", {})
    assert status == 400


def test_suggest_gap_prompt_returns_suggestion(gemma_client):
    """Returns the combined reasoning+content as suggested_prompt."""
    gemma_client["outcome"] = (
        "The gap follows a field scene...",
        "A slow aerial push-in over the golden field.",
    )
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/suggest-gap-prompt",
                         {"messages": PREBUILT_GAP})
    assert status == 200
    assert gemma_client["messages"] == PREBUILT_GAP
    assert "field" in body["suggested_prompt"]


def test_suggest_gap_prompt_strips_thinking_tags(gemma_client):
    """Inline <start_of_thinking>...</end_of_thinking> is stripped from content."""
    gemma_client["outcome"] = (
        "",
        "<start_of_thinking>The user wants a bridge. "
        "</end_of_thinking>A bridge spans the river at dusk.",
    )
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/suggest-gap-prompt",
                         {"messages": PREBUILT_GAP})
    assert status == 200
    assert "<start_of_thinking>" not in body["suggested_prompt"]
    assert "bridge spans the river at dusk" in body["suggested_prompt"]


def test_suggest_gap_prompt_reasoning_only(gemma_client):
    """Whole reply in reasoning channel (content empty) is still returned."""
    gemma_client["outcome"] = (
        "The next clip should be a close-up of the storm cloud.",
        "",
    )
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/suggest-gap-prompt",
                         {"messages": PREBUILT_GAP})
    assert status == 200
    assert "storm cloud" in body["suggested_prompt"]


def test_suggest_gap_prompt_route_registered():
    app = gemma_server.create_app()
    paths = sorted(r.resource.canonical for r in app.router.routes() if r.resource)
    assert "/video-creator/v1/suggest-gap-prompt" in paths
    assert "/v1/suggest-gap-prompt" in paths
