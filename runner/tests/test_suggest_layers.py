"""GPU-independent tests for the suggest-layers handler.

The handler proxies to the managed llama-server subprocess, which runs with
``--reasoning on --reasoning-format deepseek``. Because the rubric demands
step-by-step thinking, the model's entire reply (occasionally including the
final layer count) can land in ``reasoning_content`` with ``content`` empty —
which used to raise "llama-server returned empty content" as a 500. These
tests stub ``llms.chat_with_reasoning`` (no model/GPU) and pin the fix: the
count is parsed from BOTH channels.
"""

import asyncio
import base64
import struct
import zlib

import pytest
import aiohttp

from runner.gemma import config
from runner.gemma import server as gemma_server


def _png_1x1_b64() -> str:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff))
    raw = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89")
        + chunk(b"IDAT", zlib.compress(b"\x00\x00\x00\x00"))
        + chunk(b"IEND", b"")
    )
    return base64.b64encode(raw).decode()


@pytest.fixture
def suggest_client(monkeypatch):
    """Boot the aiohttp app with llms.chat_with_reasoning stubbed.

    Returns a client usable in the test's own event loop via ``async with``.
    """
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


def test_suggest_layers_reasoning_only_returns_count(suggest_client):
    """The reported bug: answer lives only in reasoning_content (content empty).

    The rubric pushes the model to think, and reasoning-mode llama-server strips
    the thinking out of ``content``. The count must still be found in the
    reasoning channel instead of a 500 empty-content error.
    """
    suggest_client["outcome"] = (
        "The scene has a single cat in the foreground and a simple sky "
        "background... I enumerate: 1) cat, 2) sky, 3) ground. Fewer than "
        "four clean separable elements warrant <3>.",
        "",  # content empty — this used to trigger RuntimeError
    )
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/suggest-layers",
                         {"image": _png_1x1_b64()})
    assert status == 200
    assert body["layers"] == 3
    assert "cat" in body["raw"]


def test_suggest_layers_content_answer(suggest_client):
    """Normal path: the final answer is in content, reasoning is separate."""
    suggest_client["outcome"] = (
        "Some reasoning text here...",
        "The image decomposes cleanly into <5> layers.",
    )
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/suggest-layers",
                         {"image": _png_1x1_b64()})
    assert status == 200
    assert body["layers"] == 5


def test_suggest_layers_no_count_returns_null(suggest_client):
    """Neither channel yields a 2-8 -> layers null (caller falls back), no 500."""
    suggest_client["outcome"] = (
        "I see many things but cannot decide.",
        "The output is complete.",
    )
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/suggest-layers",
                         {"image": _png_1x1_b64()})
    assert status == 200
    assert body["layers"] is None


def test_suggest_layers_out_of_range_rejected(suggest_client):
    """Out-of-range numbers (e.g. 10) must not leak through as a layer count."""
    suggest_client["outcome"] = (
        "",
        "This complex scene really needs <10> layers.",
    )
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/suggest-layers",
                         {"image": _png_1x1_b64()})
    assert status == 200
    assert body["layers"] is None


def test_suggest_layers_calls_reasoning_endpoint(suggest_client):
    """Assert the handler uses chat_with_reasoning, not the answer-only chat."""
    suggest_client["outcome"] = ("", "<7>")
    app = gemma_server.create_app()
    status, body = _post(app, "/video-creator/v1/suggest-layers",
                         {"image": _png_1x1_b64()})
    assert status == 200
    assert body["layers"] == 7
    assert "messages" in suggest_client
    assert suggest_client["messages"][0]["role"] == "system"
