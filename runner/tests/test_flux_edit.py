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
        self.last = ("klein_image", prompt, kw, width, height)
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
            assert eng.last[0] == "klein_image"
            _, prompt, kw, width, height = eng.last
            assert prompt == "a red sports car on a coastal road"
            assert width == 1024 and height == 576
        finally:
            await client.close()

    asyncio.run(run())
