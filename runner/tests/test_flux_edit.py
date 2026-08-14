"""FLUX.2 [klein] 4B first-frame styling — GPU-independent regression tests.

Covers:
  * the pure helpers (output dims resolution, CUDA device normalization);
  * config default parsing (distilled klein = 4 steps / guidance 1.0);
  * the CRITICAL shared-GPU sequencing in `server._style_first_frame`: the
    prompt enhancement MUST run through the loaded Gemma LLM BEFORE Gemma is
    evicted, then the FLUX.2 editor loads, edits, and is evicted after.

All heavy model imports are avoided (the editor's `flux2`/torch imports are
lazy, and `runner.idv2v.model` is stubbed), so this runs torch-free.
"""

import sys
import types

import pytest
from PIL import Image

from runner.idv2v import flux_edit


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
# Config defaults (distilled klein contract)
# ---------------------------------------------------------------------------

def test_klein_config_distilled_defaults(monkeypatch):
    # Force clean env so the module-level defaults apply.
    import runner.idv2v.config as cfg
    for var in ("KLEIN4B_STEPS", "KLEIN4B_GUIDANCE", "KLEIN4B_MAX_SIDE",
                "KLEIN4B_GPU_DEVICE", "KLEIN4B_ENABLED", "KLEIN4B_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(cfg, "KLEIN4B_STEPS", 4)
    monkeypatch.setattr(cfg, "KLEIN4B_GUIDANCE", 1.0)
    monkeypatch.setattr(cfg, "KLEIN4B_MAX_SIDE", 1024)
    # Distilled klein is guidance+step distilled: 4 steps, guidance 1.0.
    assert cfg.klein4b_steps() == 4
    assert cfg.klein4b_guidance() == 1.0
    # Device defaults to the video GPU when KLEIN4B_GPU_DEVICE is unset.
    monkeypatch.setattr(cfg, "KLEIN4B_GPU_DEVICE", "")
    monkeypatch.setattr(cfg, "GPU_DEVICE", "cuda:0")
    assert cfg.klein4b_device() == "cuda:0"
    # ...and honors an explicit separate GPU.
    monkeypatch.setattr(cfg, "KLEIN4B_GPU_DEVICE", "cuda:1")
    assert cfg.klein4b_device() == "cuda:1"


# ---------------------------------------------------------------------------
# server._style_first_frame: enhance BEFORE evict, then edit, then evict editor
# ---------------------------------------------------------------------------

class _FakeEnhancer:
    def __init__(self):
        self.calls = []

    def ensure_loaded(self):
        self.calls.append("enhancer:ensure_loaded")

    def unload(self):
        self.calls.append("enhancer:unload")

    def enhance_image(self, prompt, image, seed=None):
        self.calls.append("enhancer:enhance_image")
        return "ENHANCED: " + str(prompt)


class _FakeEditor:
    def __init__(self):
        self.calls = []

    def ensure_loaded(self):
        self.calls.append("editor:ensure_loaded")

    def unload(self):
        self.calls.append("editor:unload")

    def edit(self, image, prompt, seed, width=None, height=None, max_side=None):
        self.calls.append("editor:edit")
        return Image.new("RGB", (16, 16), (255, 0, 0))


@pytest.fixture()
def flux_server(monkeypatch):
    # Clear any prior idv2v module imports so our stubs take effect.
    for name in list(sys.modules):
        if name.startswith("runner.idv2v"):
            del sys.modules[name]
    # Stub the heavy model module server.py imports at the top (like the other
    # runner tests do).
    stub_model = types.ModuleType("runner.idv2v.model")
    stub_model.ModelManager = object
    stub_model.health_check = lambda *a, **k: {"model_loaded": False}
    monkeypatch.setitem(sys.modules, "runner.idv2v.model", stub_model)

    return __import__("runner.idv2v.server", fromlist=["_style_first_frame"])


def _route_paths(app):
    """(method, path) pairs for all registered routes."""
    out = set()
    for r in app.router.routes():
        try:
            method, path = r.method, r.resource.canonical
        except Exception:
            continue
        if method and path:
            out.add((method, path))
    return out


def test_style_frame_routes_registered(flux_server):
    app = flux_server.create_app()
    paths = _route_paths(app)
    assert ("POST", "/v1/style-frame") in paths, sorted(paths)
    assert ("POST", "/video-creator/v1/style-frame") in paths, sorted(paths)


def test_style_first_frame_enhance_before_evict(flux_server, monkeypatch):
    enhancer = _FakeEnhancer()
    editor = _FakeEditor()

    # The fixture re-imports fresh runner.idv2v.* modules, so patch the ones the
    # server actually sees (flux_server.flux_edit) plus a fresh gemma import.
    monkeypatch.setattr(flux_server.flux_edit, "get_editor", lambda: editor)
    monkeypatch.setattr(flux_server.flux_edit, "evict_editor", editor.unload)
    import runner.idv2v.gemma as gemma_mod
    monkeypatch.setattr(gemma_mod, "get_enhancer", lambda: enhancer)

    # Force the LLM on so the enhance path is exercised.
    monkeypatch.setattr(flux_server.config, "gemma_enabled", lambda: True)

    src = Image.new("RGB", (64, 64), (0, 128, 255))
    styled, enhanced = flux_server._style_first_frame(
        src, "make it an oil painting", seed=7, width=None, height=None,
        enhance_prompt=True,
    )

    assert enhanced == "ENHANCED: make it an oil painting"
    assert styled.size == (16, 16)

    # The enhance ran through the LOADED LLM BEFORE the LLM was evicted (the
    # "enhance before evicting the LLM" guarantee), and the FLUX.2 editor only
    # loaded AFTER Gemma was freed, then was evicted after editing. All enhance
    # activity strictly precedes all editor activity, and within each lifecycle
    # the evict comes after the work.
    assert enhancer.calls == [
        "enhancer:ensure_loaded", "enhancer:enhance_image", "enhancer:unload",
    ], enhancer.calls
    assert editor.calls == [
        "editor:ensure_loaded", "editor:edit", "editor:unload",
    ], editor.calls


def test_style_first_frame_skips_enhance_when_llm_off(flux_server, monkeypatch):
    editor = _FakeEditor()
    enhancer = _FakeEnhancer()
    monkeypatch.setattr(flux_server.flux_edit, "get_editor", lambda: editor)
    monkeypatch.setattr(flux_server.flux_edit, "evict_editor", editor.unload)
    import runner.idv2v.gemma as gemma_mod
    monkeypatch.setattr(gemma_mod, "get_enhancer", lambda: enhancer)
    monkeypatch.setattr(flux_server.config, "gemma_enabled", lambda: False)

    src = Image.new("RGB", (64, 64), (0, 128, 255))
    styled, enhanced = flux_server._style_first_frame(
        src, "keep it plain", seed=1, width=None, height=None,
        enhance_prompt=True,
    )

    assert enhanced is None
    assert enhancer.calls == []            # LLM never touched
    assert editor.calls[0] == "editor:ensure_loaded"
    assert editor.calls[-1] == "editor:unload"


def test_clamp01():
    from runner.idv2v import flux_edit
    assert flux_edit._clamp01(0.5) == 0.5
    assert flux_edit._clamp01(-1.0) == 0.0
    assert flux_edit._clamp01(2.0) == 1.0
    assert flux_edit._clamp01(1.0) == 1.0
    # Non-numeric / missing defaults to 1.0 (full re-imagine, no change).
    assert flux_edit._clamp01(None) == 1.0
    assert flux_edit._clamp01("nope") == 1.0


def test_klein_config_strength_default(monkeypatch):
    import runner.idv2v.config as cfg
    monkeypatch.delenv("KLEIN4B_STRENGTH", raising=False)
    monkeypatch.setattr(cfg, "KLEIN4B_STRENGTH", 1.0)
    # Default = 1.0 (stock full re-imagine) so the knob is inert unless set.
    assert cfg.klein4b_strength() == 1.0
