"""Regression test: ltx-worker serves root /load and /evict.

The live-runner's HttpWorkerTransport calls base + "/load" (swap.py), and the
idv2v-worker already serves root /load /evict. The ltx-worker previously
registered these under /video-creator/worker/*, so the live-runner's swap got
404 when trying to load the LTX model. This test boots the ltx worker's real
create_app (heavy model modules stubbed) and asserts the control routes are at
root, matching the swap contract.
"""

import importlib
import sys
import types

import pytest

# Stub out all heavy modules ltx.server imports so the test runs torch-free.
STUB = {
    "runner.ltx.enhance_forward": types.ModuleType("runner.ltx.enhance_forward"),
    "runner.ltx.inference": types.ModuleType("runner.ltx.inference"),
    "runner.ltx.loracache": types.ModuleType("runner.ltx.loracache"),
}

# Give the stubs the minimal attributes the server module reads at import time.
STUB["runner.ltx.enhance_forward"].DEFAULT_T2V_SYSTEM_PROMPT = "t2v"
STUB["runner.ltx.enhance_forward"].DEFAULT_I2V_SYSTEM_PROMPT = "i2v"
STUB["runner.ltx.inference"].VideoCreatorInferenceEngine = object  # type: ignore[attr-defined]
STUB["runner.ltx.loracache"].LoraCache = object  # type: ignore[attr-defined]


@pytest.fixture()
def ltx_server(monkeypatch):
    # Clear any prior imports of runner.ltx.* so stubs take effect.
    for name in list(sys.modules):
        if name.startswith("runner.ltx") and name != "runner.ltx.config":
            del sys.modules[name]
    for mod_name, mod in STUB.items():
        monkeypatch.setitem(sys.modules, mod_name, mod)

    # Provide a config with the constants server.py reads at import.
    cfg = types.ModuleType("runner.ltx.config")
    for k, v in {
        "ENHANCE_FORWARD_API_KEY": "", "ENHANCE_FORWARD_MODEL": "",
        "ENHANCE_FORWARD_TIMEOUT": 30, "ENHANCE_FORWARD_URL": "",
        "ENHANCE_GPU_DEVICE": "0", "ENHANCE_I2V_SYSTEM_PROMPT": "",
        "ENHANCE_T2V_SYSTEM_PROMPT": "", "GPU_DEVICE": "0", "GPU_NAME": "RTX 5090",
        "GPU_VRAM_GB": 0, "HOST": "0.0.0.0", "MODEL_CHECKPOINT": "/models/x",
        "PORT": 8991, "TEXT_ENCODER_ROOT": "/models/gemma", "UPSCALER_PATH": "",
        "WARMUP": False, "worker_token": lambda: "tok",
    }.items():
        setattr(cfg, k, v)
    monkeypatch.setitem(sys.modules, "runner.ltx.config", cfg)

    import runner.ltx.server as server
    return server


def _route_paths(app):
    """Extract (method, path) pairs for all routes from an aiohttp app.

    ``Application.router.routes()`` yields ``Route`` objects; ``.method`` and
    ``.resource.canonical`` are the public introspection surface.
    """
    out = set()
    for r in app.router.routes():
        try:
            method = r.method
            path = r.resource.canonical
        except Exception:
            continue
        if method and path:
            out.add((method, path))
    return out


def test_ltx_worker_load_evict_at_root(ltx_server):
    app = ltx_server.create_app()
    paths = _route_paths(app)
    assert ("POST", "/load") in paths, f"expected root /load, got {sorted(paths)}"
    assert ("POST", "/evict") in paths, f"expected root /evict, got {sorted(paths)}"


def test_ltx_worker_no_worker_prefix_routes(ltx_server):
    app = ltx_server.create_app()
    paths = _route_paths(app)
    # The old broken prefix must be gone.
    assert ("POST", "/video-creator/worker/load") not in paths
    assert ("POST", "/video-creator/worker/evict") not in paths


def test_ltx_worker_generation_routes_present(ltx_server):
    app = ltx_server.create_app()
    paths = _route_paths(app)
    for ep in ("t2v", "i2v", "image", "extend", "retake"):
        assert ("POST", f"/video-creator/v1/{ep}") in paths, ep
