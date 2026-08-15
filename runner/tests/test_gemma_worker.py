"""GPU-independent unit tests for the gemma-worker (llama.cpp Gemma 4 LLM).

Stubs the heavy ``llama_cpp`` native module so no model/GPU is needed. Any
change that touches resident VRAM / n_gpu_layers stays out of these tests (the
swap-level residency behavior lives in test_live_runner_router.py).
"""

import base64
import struct
import sys
import types
import zlib

import pytest

from runner.gemma import config
from runner.gemma.model import GemmaLLM
from runner.gemma.server import _build_enhance_messages


class FakeLlama:
    """Records kwargs; returns a fixed chat completion."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        return {"choices": [{"message": {"content": "  vibrant cinematic   "}}]}


def _stub_llama(monkeypatch):
    """Install a fake llama_cpp.Llama factory; returns a handle to the instance."""
    mod = types.ModuleType("llama_cpp")
    holder = {}
    def factory(**kw):
        obj = FakeLlama(**kw)
        holder["obj"] = obj
        return obj
    mod.Llama = factory
    monkeypatch.setitem(sys.modules, "llama_cpp", mod)
    return holder


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


def test_device_index_parsing():
    old = config.GEMMA_GPU_DEVICE
    try:
        config.GEMMA_GPU_DEVICE = "cuda:3"
        assert config.gemma_device_index() == 3
        config.GEMMA_GPU_DEVICE = "1"
        assert config.gemma_device_index() == 1
        config.GEMMA_GPU_DEVICE = ""
        assert config.gemma_device_index() == 0
        config.GEMMA_GPU_DEVICE = "cuda:0"
        assert config.is_dedicated_gpu() is True  # non-blank = dedicated
        config.GEMMA_GPU_DEVICE = ""
        assert config.is_dedicated_gpu() is False
    finally:
        config.GEMMA_GPU_DEVICE = old


def test_load_chat_evict(monkeypatch):
    holder = _stub_llama(monkeypatch)
    llm = GemmaLLM(model_path="/m/gemma.gguf", mmproj="/m/mmproj.gguf",
                   n_gpu_layers=-1, main_gpu=1, n_ctx=8192)
    assert not llm.is_ready
    llm.load()
    fake = holder["obj"]
    assert llm.is_ready
    assert fake.kwargs["mmproj"] == "/m/mmproj.gguf"  # threaded to the binding
    assert fake.kwargs["main_gpu"] == 1
    assert fake.kwargs["n_gpu_layers"] == -1

    out = llm.chat([{"role": "user", "content": "hi"}], seed=7)
    assert out == "vibrant cinematic"
    assert fake.calls[0]["seed"] == 7
    assert fake.calls[0]["stream"] is False

    llm.evict()
    assert not llm.is_ready


def test_chat_requires_load():
    llm = GemmaLLM(model_path="/nope.gguf")
    with pytest.raises(RuntimeError):
        llm.chat([{"role": "user", "content": "hi"}])


def test_build_enhance_messages_text():
    msgs, has_image = _build_enhance_messages({"prompt": "a dog running"})
    assert has_image is False
    assert msgs[0]["role"] == "system"
    assert "user prompt: a dog running" in msgs[1]["content"]


def test_build_enhance_messages_image():
    msgs, has_image = _build_enhance_messages(
        {"prompt": "make it rain", "image_base64": _png_1x1_b64()})
    assert has_image is True
    user = msgs[1]["content"]
    assert isinstance(user, list)
    assert user[0]["type"] == "image_url"
    assert user[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert user[1]["type"] == "text"




def test_build_enhance_messages_context_frames():
    """Extend enhance: context_frames -> EXTEND system prompt + one image_url per frame
    + a text part carrying the direction note (multimodal)."""
    from runner.ltx.enhance_forward import DEFAULT_EXTEND_SYSTEM_PROMPT
    frames = [_png_1x1_b64(), _png_1x1_b64()]
    msgs, has_image = _build_enhance_messages({
        "prompt": "camera pulls back to reveal the city",
        "context_frames": frames,
        "direction": "end",
    })
    assert has_image is True
    user = msgs[1]["content"]
    assert isinstance(user, list)
    # 2 images + 1 text part
    img_parts = [p for p in user if p["type"] == "image_url"]
    assert len(img_parts) == 2
    for p in img_parts:
        assert p["image_url"]["url"].startswith("data:image/png;base64,")
    text_part = next(p for p in user if p["type"] == "text")["text"]
    assert "camera pulls back to reveal the city" in text_part
    assert "Extend direction: end" in text_part
    # EXTEND system prompt is selected (not the t2v/i2v default)
    assert msgs[0]["content"] == DEFAULT_EXTEND_SYSTEM_PROMPT
    assert "EXTENSION" in msgs[0]["content"]


def test_build_enhance_messages_context_empty_text_fallback():
    """Empty/absent context_frames falls through to plain-text enhance (no multimodal)."""
    msgs, has_image = _build_enhance_messages({"prompt": "hello", "context_frames": []})
    assert has_image is False
    assert msgs[1]["content"] == "user prompt: hello"


def test_create_app_registers_routes():
    from runner.gemma.server import create_app
    app = create_app()
    paths = sorted(r.resource.canonical for r in app.router.routes() if r.resource)
    assert "/load" in paths
    assert "/evict" in paths
    assert "/health" in paths
    assert "/video-creator/v1/prompt-enhance" in paths
    assert "/video-creator/v1/chat" in paths
