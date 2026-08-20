"""Managed subprocess for the native llama.cpp ``llama-server`` (single model host).

Architecture (chosen): ``llama-server`` is the ONLY resident model instance; the
aiohttp worker (this package) is the HTTP front + control surface. Every
inference endpoint proxies to ``http://127.0.0.1:AGENT_PORT/v1``. This is
VRAM-safe: the Python binding at 128K ctx needs ~31.8 GB on the 32 GB card, so
two resident 12B instances cannot co-exist.

The worker spawns llama-server on ``/load`` and stops it on ``/evict``,
preserving the live-runner swap-policy contract (loaded = model resident, evict
= GPU freed for an incoming render worker).

llama-server is the upstream ggml-org/llama.cpp build (sm_80/89/120 CUDA). It
returns ``reasoning_content`` natively (``--reasoning on --reasoning-format
deepseek``) and supports OpenAI function/tool calling — so pydantic-ai (and
other agent clients) can build Gemma-4 agents against the same endpoint.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from typing import Any

import aiohttp

from . import config

logger = logging.getLogger("video_creator.runner.gemma.llama_server")

_BINARY = "/usr/local/bin/llama-server"

# OpenAI-compatible agent endpoint of the native server.
AGENT_HOST = os.environ.get("GEMMA_AGENT_HOST", "0.0.0.0")
AGENT_PORT = int(os.environ.get("GEMMA_AGENT_PORT", "8995"))
# Bounded agent context: keeps co-residency headroom and load times sane; agents
# rarely need >32K. (The old long-context python path is gone.)
AGENT_N_CTX = int(os.environ.get("GEMMA_AGENT_N_CTX", "32768"))
_STARTUP_TIMEOUT_S = int(os.environ.get("GEMMA_AGENT_START_TIMEOUT", "300"))

_session: aiohttp.ClientSession | None = None
_proc: asyncio.subprocess.Process | None = None
_proc_lock = asyncio.Lock()


def agent_base_url() -> str:
    """Public base URL (host:port) of the OpenAI-compatible endpoint."""
    return f"http://127.0.0.1:{AGENT_PORT}/v1"


async def is_running() -> bool:
    return _proc is not None and _proc.returncode is None


async def ensure_running() -> None:
    """Start (or reuse) the llama-server subprocess and wait until healthy."""
    global _proc
    async with _proc_lock:
        if _proc is not None and _proc.returncode is None:
            return
        binary = _BINARY if os.path.exists(_BINARY) else shutil.which("llama-server")
        if not binary:
            raise RuntimeError("llama-server binary not found in image")

        cmd = [
            binary,
            "--model", config.GEMMA_MODEL,
            "--host", AGENT_HOST,
            "--port", str(AGENT_PORT),
            "--ctx-size", str(AGENT_N_CTX),
            "--n-gpu-layers", str(config.GEMMA_N_GPU_LAYERS),
            "--flash-attn", "on",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--parallel", "1",
            # Enable Gemma thinking (always-on, no toggle). `--reasoning on` is the
            # modern llama.cpp way (the enable_thinking chat-template-kwargs route
            # is deprecated); `--reasoning-format deepseek` makes llama-server put
            # the thoughts in `message.reasoning_content` and strip them from
            # content — that natively-populated field is what the worker returns.
            "--reasoning", "on",
            "--reasoning-format", "deepseek",
            "--log-format", "json",
            "--log-file", "/tmp/llama-server.log",
        ]
        if config.GEMMA_MMPROJ:
            cmd += ["--mmproj", config.GEMMA_MMPROJ]
        token = config.worker_token()
        if token:
            cmd += ["--api-key", token]

        logger.info("spawning llama-server (agent port %d): %s", AGENT_PORT,
                    " ".join(cmd))
        _proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await _wait_ready()
        except Exception:
            # Don't leave a half-started process behind on boot failure.
            await stop()
            raise
        logger.info("llama-server ready on %s", agent_base_url())


async def _wait_ready() -> None:
    url = f"http://127.0.0.1:{AGENT_PORT}/health"
    async with aiohttp.ClientSession() as probe:
        for _ in range(int(_STARTUP_TIMEOUT_S)):
            if _proc is None or _proc.returncode is not None:
                raise RuntimeError(
                    f"llama-server exited during startup (rc="
                    f"{_proc.returncode if _proc else '?'})")
            try:
                async with probe.get(url, timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            await asyncio.sleep(1)
    raise RuntimeError("llama-server did not become healthy in time")


async def stop() -> None:
    """Stop llama-server, releasing GPU residency (drives the swap policy)."""
    global _proc
    async with _proc_lock:
        p, _proc = _proc, None
        if p is None:
            return
        if p.returncode is None:
            try:
                p.terminate()
                await asyncio.wait_for(p.wait(), timeout=10)
            except (asyncio.TimeoutError, ProcessLookupError):
                try:
                    p.kill()
                except Exception:
                    pass
        logger.info("llama-server stopped")


async def _post(messages: list[dict[str, Any]], *, temperature: float,
                max_tokens: int, seed: int | None) -> tuple[str, str]:
    """POST messages to llama-server; return (reasoning_content, content)."""
    global _session
    if _session is None:
        _session = aiohttp.ClientSession()
    payload: dict[str, Any] = {
        "model": config.GEMMA_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if seed is not None:
        payload["seed"] = seed
    headers = {"Authorization": f"Bearer {config.worker_token()}"}
    async with _session.post(
        f"{agent_base_url()}/chat/completions",
        json=payload, headers=headers, timeout=600,
    ) as r:
        if r.status != 200:
            txt = (await r.text())[:500]
            raise RuntimeError(f"llama-server HTTP {r.status}: {txt}")
        data = await r.json()
    try:
        msg = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected llama-server response: {exc}") from exc
    content = msg.get("content")
    reasoning = (msg.get("reasoning_content") or "").strip()
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("llama-server returned empty content")
    return reasoning, content.strip()


async def chat(messages: list[dict[str, Any]], *, max_tokens: int = 512,
               temperature: float = 0.7, seed: int | None = None) -> str:
    """OpenAI-style chat completion proxied to llama-server (answer only)."""
    _, content = await _post(messages, temperature=temperature,
                             max_tokens=max_tokens, seed=seed)
    return content


async def chat_with_reasoning(messages: list[dict[str, Any]], *,
                              max_tokens: int = 512, temperature: float = 0.7,
                              seed: int | None = None) -> tuple[str, str]:
    """Like ``chat`` but also returns llama-server's native ``reasoning_content``.

    Returns ``(reasoning_content, content)`` — the model's chain-of-thought in
    its own field and the cleaned answer as content (llama-server splits the
    thinking out natively; no regex here).
    """
    return await _post(messages, temperature=temperature,
                       max_tokens=max_tokens, seed=seed)
