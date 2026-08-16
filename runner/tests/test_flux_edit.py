"""FLUX.2 [klein] 4B — image-worker host — GPU-independent regression tests.

FLUX.2 klein lives ENTIRELY on the image-worker (it serves both /style-frame and
text-to-image via /image with engine='klein'). It was removed from the id-v2v
worker.

Covers:
  * the pure helpers (output dims resolution, CUDA device normalization);
  * config default parsing (distilled klein = 4 steps / guidance 1.0);
  * the /image handler's klein dispatch: engine='klein' routes to
    engine.klein_image and maps quality presets (fast/balanced/high) to step
    counts, while engine='zimage' stays on plain_image.

All heavy model imports are avoided (the editor's flux2/torch imports are lazy),
so this runs torch-free.
"""

import asyncio
import io

import pytest
from PIL import Image

from runner.image import flux_edit
from runner.image import server as image_server


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_resolve_dims_preserves_aspect_and_caps_long_edge():
    # 1920x1080 styled at max_side 1024 -> 1024x576 (both /16, aspect-kept).
    assert flux_edit.resolve_dims(1920, 1080, 1024) == (1024, 576)
    # No cap -> passthrough (already /16 multiples).
    assert flux_edit.resolve_dims(1280, 720, 0) == (1280, 720)
    # Portrait 813x1219 @1024 -> long edge 1024, short edge /16.
    w, h = flux_edit.resolve_dims(813, 1219, 1024)
    assert h == 1024 and w % 16 == 0 and abs(w / h - 813 / 1219) < 0.03
    # Never returns sub-16 dims.
    assert flux_edit.resolve_dims(20, 20, 1024) == (16, 16)


def test_normalize_device():
    assert flux_edit._normalize_device("0") == "cuda:0"
    assert flux_edit._normalize_device("cuda:1") == "cuda:1"
    assert flux_edit._normalize_device("cpu") == "cpu"
    assert flux_edit._normalize_device("") == "cuda:0"


# ---------------------------------------------------------------------------
# Config defaults (distilled klein contract) — image-worker config
# ---------------------------------------------------------------------------

def test_klein_config_distilled_defaults(monkeypatch):
    import runner.image.config as cfg
    for var in ("KLEIN4B_STEPS", "KLEIN4B_GUIDANCE", "KLEIN4B_MAX_SIDE",
                "KLEIN4B_GPU_DEVICE", "KLEIN4B_ENABLED", "KLEIN4B_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cfg, "KLEIN4B_STEPS", 4)
    monkeypatch.setattr(cfg, "KLEIN4B_GUIDANCE", 1.0)
    monkeypatch.setattr(cfg, "KLEIN4B_MAX_SIDE", 1024)
    # Distilled klein is guidance+step distilled: 4 steps, guidance 1.0.
    assert cfg.klein4b_steps() == 4
    assert cfg.klein4b_guidance() == 1.0
    # Device defaults to the image model's GPU when KLEIN4B_GPU_DEVICE is unset.
    monkeypatch.setattr(cfg, "KLEIN4B_GPU_DEVICE", "")
    monkeypatch.setattr(cfg, "DEFAULT_DEVICE", 0)
    assert cfg.klein4b_device() == "cuda:0"
    # ...and honors an explicit separate GPU.
    monkeypatch.setattr(cfg, "KLEIN4B_GPU_DEVICE", "cuda:1")
    assert cfg.klein4b_device() == "cuda:1"


# ---------------------------------------------------------------------------
# /image handler klein dispatch + quality->step presets
# ---------------------------------------------------------------------------

def test_image_klein_steps_presets():
    # Explicit override wins.
    assert image_server._image_klein_steps({"num_inference_steps": 20}) == 20
    # Quality presets: fast=4 (distilled native), balanced=8, high=12.
    assert image_server._image_klein_steps({"quality": "fast"}) == 4
    assert image_server._image_klein_steps({"quality": "balanced"}) == 8
    assert image_server._image_klein_steps({"quality": "high"}) == 12
    assert image_server.KLEIN_STEP_PRESETS == {"fast": 4, "balanced": 8, "high": 12}
    # Unknown / blank -> distilled default 4.
    assert image_server._image_klein_steps({}) == 4
    assert image_server._image_klein_steps({"quality": "bogus"}) == 4


def test_zimage_call_kw_seed_to_generator(monkeypatch):
    """Z-Image pipelines take a seeded torch.Generator, not a bare 'seed' key.

    `_zimage_call_kw` must pop `seed` and build a generator; leaving `seed` in
    the kwargs would make ZImagePipeline.__call__ raise TypeError (the live 500
    this guards against). Torch isn't installed in the test env, so we stub it.
    """
    import sys

    calls = {}

    class _FakeGenerator:
        def __init__(self, device): self.device = device; self._seed = None
        def manual_seed(self, s): self._seed = s; return self

    class _FakeTorch:
        @staticmethod
        def Generator(device=None):
            calls["device"] = device
            return _FakeGenerator(device)

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch())

    from runner.image.inference import ImageInferenceEngine
    eng = ImageInferenceEngine.__new__(ImageInferenceEngine)
    eng.current_device = 2

    # seed present -> popped, generator built on that device
    out = eng._zimage_call_kw({"seed": 42, "width": 512})
    assert "seed" not in out
    assert "generator" in out and out["width"] == 512
    assert out["generator"]._seed == 42
    assert calls.get("device") == "cuda:2"

    # no seed -> untouched
    assert eng._zimage_call_kw({"width": 512}) == {"width": 512}


class _DummyImageEngine:
    """Never touches torch — records the dispatch and returns a tiny image.

    These are SYNCHRONOUS (the handler runs them through run_in_executor,
    exactly like the real engine methods).
    """
    def __init__(self):
        self.last = None

    def plain_image(self, prompt, **kw):
        self.last = ("plain_image", prompt, kw)
        return Image.new("RGB", (8, 8), (0, 0, 255))

    def klein_image(self, prompt, seed=123, width=1024, height=1024,
                    num_inference_steps=None, **kw):
        self.last = ("klein_image", prompt, kw, width, height, seed)
        return Image.new("RGB", (8, 8), (255, 0, 0))


def _b64_of(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    import base64
    return base64.b64encode(buf.getvalue()).decode()


def test_handle_image_routes_klein(monkeypatch):
    eng = _DummyImageEngine()
    monkeypatch.setattr(image_server, "engine", eng)
    # Klein is provisioned (the handler reads runner.image.config locally).
    import runner.image.config as _img_cfg
    monkeypatch.setattr(_img_cfg, "klein4b_enabled", lambda: True)

    import aiohttp
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    async def run():
        app = web.Application()
        app.router.add_post("/image", image_server.handle_image)
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            resp = await client.post("/image", json={
                "prompt": "a red sports car on a coastal road",
                "engine": "klein",
                "width": 1024,
                "height": 576,
                "quality": "high",
                "seed": 7,
            })
            assert resp.status == 200
            body = await resp.json()
            assert body["engine"] == "klein"
            assert body["num_inference_steps"] == 12  # quality=high preset
            assert body["seed"] == 7  # echoed back verbatim
            assert eng.last[0] == "klein_image"
            _, prompt, kw, width, height, seed = eng.last
            assert prompt == "a red sports car on a coastal road"
            assert width == 1024 and height == 576
            # The client seed reached the engine's klein_image seed param.
            assert seed == 7
        finally:
            await client.close()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# Qwen-Image-Layered — nested output extraction + RGBA input (the fixes)
# ---------------------------------------------------------------------------

class _Out:
    """Mimics diffusers QwenImageLayeredPipeline's nested return:
    out.images == [ [layer_0, ...] ] (outer = batches, inner = layer frames)."""
    def __init__(self, framesis):
        self.images = framesis


def test_extract_layers_handles_nested_diffusers_batch_output():
    """Regression: diffusers returns out.images = [[L0, L1, L2, L3]] (nested per
    batch). The old code treated it as a FLAT list, found no PILs, and fell back
    to the naive split. We must unwrap the first batch and return the layers."""
    from runner.image.inference import _extract_layers
    layers = [Image.new("RGBA", (8, 8), c) for c in
              ((255,0,0,255),(0,255,0,255),(0,0,255,255),(255,255,0,255))]
    out = _Out([layers])          # one batch -> nested list of 4 layers
    got = _extract_layers(out, 4)
    assert got is not None and len(got) == 4
    assert all(im.mode == "RGBA" for im in got)


def test_extract_layers_falls_back_when_fewer_frames_than_requested():
    from runner.image.inference import _extract_layers
    layers = [Image.new("RGBA", (8, 8))]
    out = _Out([layers])
    assert _extract_layers(out, 4) is None  # 1 < 4 -> caller uses naive fallback


def test_extract_layers_accepts_flat_list_too():
    from runner.image.inference import _extract_layers
    layers = [Image.new("RGBA", (8, 8)) for _ in range(4)]
    out = _Out(layers)             # some versions may return a flat list
    got = _extract_layers(out, 4)
    assert got is not None and len(got) == 4


def test_extract_layers_returns_none_when_not_introspectable():
    from runner.image.inference import _extract_layers
    class _Weird:
        def __init__(self): self.images = None
    assert _extract_layers(_Weird(), 4) is None


def test_layered_decompose_feeds_rgba_not_rgb(monkeypatch):
    """Regression: the layered VAE first-conv wants a 4-channel RGBA input, but
    _decoded_pil returns RGB (3ch) -> "expected ... 4 channels, but got 3".
    layered_decompose must convert the source to RGBA before the pipeline call."""
    from runner.image.inference import ImageInferenceEngine, _extract_layers
    import runner.image.config as _img_cfg

    captured = {}
    class _FakePipe:
        def __call__(self, **kw):
            captured.update(kw)
            # Return 4 nested RGBA layers so real extraction path runs
            layers = [Image.new("RGBA", (16, 16), c) for c in
                      ((255,0,0,255),(0,255,0,255),(0,0,255,255),(255,255,0,255))]
            return _Out([layers])

    eng = ImageInferenceEngine.__new__(ImageInferenceEngine)
    eng._qwen_layered = _FakePipe()
    eng._model_lock = __import__("threading").RLock()
    # _evict_other does `import torch`; not in the test env -> stub it.
    eng._evict_other = lambda keeper: None
    monkeypatch.setattr(_img_cfg, "QWEN_LAYERS", 4)
    monkeypatch.setattr(_img_cfg, "QWEN_MAX_LAYERS", 8)
    monkeypatch.setattr(_img_cfg, "QWEN_LAYER_MAX_INPUT_SIDE", 1024)
    monkeypatch.setattr(_img_cfg, "QWEN_LAYER_PREVIEW_SIDE", 8)

    src = Image.new("RGB", (16, 16), (200, 100, 50))  # 3-channel source
    res = eng.layered_decompose(src, layers=4, preview_only=False)
    # The pipeline must have received an RGBA (4-channel) image.
    assert "image" in captured
    assert captured["image"].mode == "RGBA"
    assert captured["layers"] == 4
    # Real layers (not naive): each layer RGBA, labels present, composite exists.
    assert len(res["layers"]) == 4
    assert [l["label"] for l in res["layers"]] == [
        "foreground", "midground", "midground", "background"]
    assert all(l["rgba_b64"] for l in res["layers"])
    assert res["composite"]
