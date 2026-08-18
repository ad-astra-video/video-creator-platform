"""Route-level tests for the image-worker (Qwen-Image-Edit / Layered / Z-Image).

Uses a fake engine so no torch/diffusers/GPU is needed — the aiohttp route layer
imports cleanly (all GPU work is reached lazily through the engine). Covers:
  - the route table (/edit /layer /image /load /evict /health /info)
  - /load honors the device field from the body
  - /info advertises image/edit/layer/qwen-image-edit + devices_visible/device_in_use
  - /edit rejects an unknown engine with 400
  - /layer clamps layers out of [2, QWEN_MAX_LAYERS] with 400
  - /layer rejects an oversized projected response with 413
  - /image rejects a missing prompt with 400
"""

import asyncio
import base64
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from runner.image import server as image_server  # noqa: E402
from runner.image.inference import _pil_to_b64  # noqa: E402


def _tiny_png_b64():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "green").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


class FakeEngine:
    """Stands in for ImageInferenceEngine — pure CPU, satisfies the route paths."""

    def __init__(self):
        self.current_device = 0
        self.edit_engine = None
        self.force_413 = False

    def edit_image(self, image, prompt, engine="qwen-edit", mask=None,
                   keep_subject=False, strength=0.6, padding_mask_crop=0,
                   progress_cb=None, **kw):
        self.edit_engine = engine
        self.edit_steps = kw.get("num_inference_steps")
        self.edit_strength = strength
        self.edit_padding = padding_mask_crop
        self.edit_mask = mask
        self.edit_progress_cb = progress_cb
        if progress_cb:
            progress_cb(1, self.edit_steps or 35)  # emit one progress tick
        from PIL import Image
        return Image.new("RGB", (8, 8), "red")

    def layered_decompose(self, image, layers, resolution, preview_only,
                          num_inference_steps=None, progress_cb=None):
        self.last_layers = layers
        self.last_steps = num_inference_steps
        if progress_cb:
            progress_cb(1, 4)  # emit one progress tick like a real first step
        from PIL import Image
        img = Image.new("RGBA", (8, 8), (0, 0, 0, 255))
        b64 = _pil_to_b64(img)
        if self.force_413:
            return {"composite": "x" * (10 * 1024 * 1024),
                    "layers": [{"rgba_b64": "x" * (5 * 1024 * 1024)}] * 8}
        return {
            "layers": [
                {"index": 0, "rgba_b64": b64, "preview_b64": b64,
                 "alpha_b64": b64, "label": "foreground"},
                {"index": 1, "rgba_b64": b64, "preview_b64": b64,
                 "alpha_b64": b64, "label": "background"},
            ],
            "composite": b64, "width": 8, "height": 8,
            "layers_requested": layers or 2,
        }

    def plain_image(self, prompt, **kw):
        from PIL import Image
        return Image.new("RGB", (8, 8), "blue")

    def free(self):
        pass


@pytest.fixture()
async def client(monkeypatch):
    engine = FakeEngine()
    app = image_server.create_app()
    # Don't run on_startup (it builds a real ImageInferenceEngine that needs
    # torch/GPU) — we test the route layer against the fake engine only.
    app.on_startup.clear()
    monkeypatch.setattr(image_server, "engine", engine)
    monkeypatch.setattr(image_server, "ready", True)
    monkeypatch.setattr(image_server, "gpu_profile", None)
    srv = TestServer(app)
    cli = TestClient(srv)
    await cli.start_server()
    try:
        yield cli, engine
    finally:
        await cli.close()


def _token_headers():
    from runner.image.config import worker_token
    return {"X-Worker-Token": worker_token()}


async def test_route_table(client):
    cli, _ = client
    routes = set()
    for r in cli.server.app.router.routes():
        if r.resource is not None:
            routes.add(r.resource.canonical)
    assert "/video-creator/v1/edit" in routes
    assert "/video-creator/v1/layer" in routes
    assert "/video-creator/v1/image" in routes
    assert "/load" in routes
    assert "/evict" in routes
    assert "/video-creator/v1/info" in routes


async def test_load_honors_device(client):
    cli, engine = client
    r = await cli.post("/load", json={"device": 2}, headers=_token_headers())
    assert r.status == 200
    body = await r.json()
    assert body["device"] == 2
    assert engine.current_device == 2


async def test_load_defaults_device(client):
    cli, engine = client
    r = await cli.post("/load", json={}, headers=_token_headers())
    assert r.status == 200
    body = await r.json()
    from runner.image.config import DEFAULT_DEVICE
    assert body["device"] == DEFAULT_DEVICE


async def test_info_advertises_capabilities_and_device(client):
    cli, engine = client
    engine.current_device = 1
    r = await cli.get("/video-creator/v1/info")
    assert r.status == 200
    body = await r.json()
    for cap in ("image", "edit", "layer", "style-frame"):
        assert cap in body["capabilities"]
    assert body["models"] == ["z-image", "flux", "qwen", "hidream"]
    assert body["device_in_use"] == 1
    assert "devices_visible" in body


async def test_edit_rejects_unknown_engine(client):
    cli, _ = client
    r = await cli.post("/video-creator/v1/edit",
                       json={"image": _tiny_png_b64(), "prompt": "hi",
                             "engine": "bogus"})
    assert r.status == 400
    body = await r.json()
    assert "unknown edit engine" in body["error"]


async def test_edit_qwen_engine_default(client):
    cli, engine = client
    r = await cli.post("/video-creator/v1/edit",
                       json={"image": _tiny_png_b64(), "prompt": "make it red"})
    assert r.status == 200
    body = await r.json()
    assert "image" in body
    assert body["content_type"] == "image/png"
    assert engine.edit_engine == "qwen-edit"  # default engine


async def test_edit_zimage_engine_selectable(client):
    cli, engine = client
    r = await cli.post("/video-creator/v1/edit",
                       json={"image": _tiny_png_b64(), "prompt": "keep subject",
                             "engine": "zimage", "keep_subject": True})
    assert r.status == 200
    assert engine.edit_engine == "zimage"


async def test_edit_missing_image_400(client):
    cli, _ = client
    r = await cli.post("/video-creator/v1/edit", json={"prompt": "hi"})
    assert r.status == 400


async def test_layer_clamps_out_of_range(client):
    cli, _ = client
    r = await cli.post("/video-creator/v1/layer",
                       json={"image": _tiny_png_b64(), "layers": 200})
    assert r.status == 400
    body = await r.json()
    assert "in [2," in body["error"]


async def test_layer_contract(client):
    cli, _ = client
    r = await cli.post("/video-creator/v1/layer",
                       json={"image": _tiny_png_b64(), "layers": 2})
    assert r.status == 200
    body = await r.json()
    assert len(body["layers"]) == 2
    assert body["layers"][0]["label"] == "foreground"
    assert body["layers"][0]["index"] == 0
    assert "composite" in body and "width" in body and "height" in body
    assert body["layers_requested"] == 2
    # preview_only strips full rgba/alpha (FakeEngine shortcuts; keep simple)
    assert isinstance(body["layers"][0].get("rgba_b64"), str)


async def test_layer_passes_steps(client):
    cli, engine = client
    r = await cli.post("/video-creator/v1/layer",
                       json={"image": _tiny_png_b64(), "layers": 2,
                             "num_inference_steps": 50})
    assert r.status == 200
    assert engine.last_steps == 50


async def test_layer_sse_streams_progress(client):
    cli, engine = client
    r = await cli.post("/video-creator/v1/layer?sse=1",
                       json={"image": _tiny_png_b64(), "layers": 3,
                             "num_inference_steps": 25})
    assert r.status == 200
    assert "text/event-stream" in r.headers.get("Content-Type", "")
    # First event is `accepted`, then a `progress` (FakeEngine emits one tick),
    # then `complete` with the full contract.
    text = await r.text()
    assert "event: accepted" in text
    assert '"engine": "layer"' in text
    assert "event: progress" in text
    assert '"step": 1' in text
    assert "event: complete" in text
    assert '"layers_requested"' in text


async def test_layer_413_on_oversize(client, monkeypatch):
    cli, engine = client
    engine.force_413 = True
    # Lower the response cap so the fake's large payload trips 413.
    monkeypatch.setattr(image_server, "QWEN_LAYER_RESPONSE_CAP_BYTES", 1024)
    r = await cli.post("/video-creator/v1/layer",
                       json={"image": _tiny_png_b64(), "layers": 2})
    assert r.status == 413


async def test_image_missing_prompt_400(client):
    cli, _ = client
    r = await cli.post("/video-creator/v1/image", json={})
    assert r.status == 400


async def test_edit_quality_translates_steps(client):
    """The worker maps a bare quality name to per-engine steps: qwen 25/35/50."""
    cli, engine = client
    r = await cli.post("/video-creator/v1/edit",
                       json={"image": _tiny_png_b64(), "prompt": "make it red",
                             "engine": "qwen-edit", "quality": "high"})
    assert r.status == 200
    assert engine.edit_steps == 50
    r = await cli.post("/video-creator/v1/edit",
                       json={"image": _tiny_png_b64(), "prompt": "make it red",
                             "engine": "qwen-edit", "quality": "fast"})
    assert engine.edit_steps == 25
    r = await cli.post("/video-creator/v1/edit",
                       json={"image": _tiny_png_b64(), "prompt": "make it red",
                             "engine": "qwen-edit", "quality": "balanced"})
    assert engine.edit_steps == 35


async def test_edit_masked_forwards_strength_padding(client):
    # A masked qwen edit threads strength + padding_mask_crop + quality->steps to
    # the engine (the QwenImageEditInpaintPipeline path).
    cli, engine = client
    r = await cli.post('/video-creator/v1/edit',
                       json={'image': _tiny_png_b64(), 'prompt': 'make the chair red',
                             'engine': 'qwen-edit', 'mask_image': _tiny_png_b64(),
                             'strength': 0.4, 'padding_mask_crop': 24,
                             'quality': 'fast'})
    assert r.status == 200
    assert engine.edit_mask is not None
    assert engine.edit_strength == 0.4
    assert engine.edit_padding == 24
    assert engine.edit_steps == 25

async def test_edit_sse_streams_progress(client):
    cli, engine = client
    r = await cli.post('/video-creator/v1/edit?sse=1',
                       json={'image': _tiny_png_b64(), 'prompt': 'make it red',
                             'engine': 'qwen-edit', 'quality': 'high'})
    assert r.status == 200
    assert 'text/event-stream' in r.headers.get('Content-Type', '')
    text = await r.text()
    assert 'event: accepted' in text
    assert 'event: progress' in text
    assert '"step": 1' in text
    assert '"total_steps": 50' in text
    assert 'event: complete' in text
    assert '"engine": "qwen-edit"' in text
    assert engine.edit_progress_cb is not None

async def test_image_returns_png(client):
    cli, _ = client
    r = await cli.post("/video-creator/v1/image", json={"prompt": "a cat"})
    assert r.status == 200
    body = await r.json()
    assert "image" in body
    assert body["content_type"] == "image/png"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
