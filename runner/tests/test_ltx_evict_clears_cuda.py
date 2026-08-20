"""Torch-free tests proving the ltx-worker's /evict clears CUDA context.

These tests exercise the REAL ``runner.ltx.inference.VideoCreatorInferenceEngine``
code path (not a route-level stub) by injecting a fake ``torch`` module into
``sys.modules`` before importing it. The fake records every
``torch.cuda.synchronize`` / ``torch.cuda.empty_cache`` / ``gc.collect`` call so
we can assert that ``engine.free()`` fully drains the GPU: fresh pipelines are
dropped AND the CUDA context is explicitly drained + released.

A second group exercises the real ``handle_evict`` route handler (imported the
same way ``test_ltx_worker_routes.py`` does, with the heavy sibling modules
stubbed) to prove the ROUTE is idempotent, still calls ``free()`` on the engine
after the ``assert engine is not None`` guard was removed, and returns 200 +
``{"evicted": True}`` even when no engine is resident. The handler is a coroutine,
so these tests call ``asyncio.run(...)`` from plain sync tests (the pattern the
repo's conftest explicitly blesses) to avoid depending on pytest-asyncio's mode.
"""

import asyncio
import json
import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Fake torch — CUDA-available, records synchronize/empty_cache/cache-alloc.
# ---------------------------------------------------------------------------
class _FakeCuda:
    def __init__(self) -> None:
        self.is_available = lambda: True
        self.device_count = lambda: 1
        self.memory_allocated = lambda i=0: 0
        self.memory_reserved = lambda i=0: 0
        self.synchronize_calls = []
        self.empty_cache_calls = 0

    def synchronize(self, device=None):
        self.synchronize_calls.append(device)

    def empty_cache(self):
        self.empty_cache_calls += 1


def _noop_decorator(*args, **kwargs):
    """Return an identity decorator (matches torch.inference_mode()/no_grad()).

    ``@torch.inference_mode()`` and ``@torch.no_grad()`` are applied at class
    definition time, so the fake torch must provide callables that return a
    working decorator.
    """

    def deco(fn):
        return fn

    return deco


class _FakeTorch(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.bfloat16 = object()
        self.cuda = _FakeCuda()
        self.device = lambda *a, **k: object()
        self.inference_mode = _noop_decorator
        self.no_grad = _noop_decorator


FAKE_TORCH = _FakeTorch()


def _reset_fakes() -> None:
    _FakeTorch.__init__(FAKE_TORCH)


class _FakeDevice:
    """Minimal stand-in for torch.device with a CUDA index."""

    def __init__(self, index):
        self.type = "cuda"
        self.index = index


class _FakeGC(types.ModuleType):
    """Records gc.collect() calls made through the module's `gc` binding."""

    def __init__(self) -> None:
        super().__init__("gc")
        self.collect_calls = 0

    def collect(self):
        self.collect_calls += 1
        return 0


@pytest.fixture()
def real_inference(monkeypatch):
    """Import the REAL engine module over a fake torch; reset fakes per test."""
    _reset_fakes()
    monkeypatch.setitem(sys.modules, "torch", FAKE_TORCH)
    # Remove any stale cached runner imports so nothing pre-imported leaks in.
    for name in list(sys.modules):
        if name.startswith("runner.ltx.inference"):
            del sys.modules[name]
    import runner.ltx.inference as inference

    inference.gc = _FakeGC()
    return inference


def _make_engine(inference, device):
    eng = inference.VideoCreatorInferenceEngine(
        checkpoint="/models/x",
        gemma_root="/models/gemma",
        upsampler_path="",
        device=device,
        profile=None,
    )
    eng._pipeline = object()
    eng._pipeline25 = object()
    eng._zimage_pipe = object()
    eng._zimg2img_pipe = object()
    eng._zinpaint_pipe = object()
    return eng


def _assert_fully_cleared(eng, inference, device):
    # All five resident pipelines dropped.
    assert eng._pipeline is None
    assert eng._pipeline25 is None
    assert eng._zimage_pipe is None
    assert eng._zimg2img_pipe is None
    assert eng._zinpaint_pipe is None
    cuda = FAKE_TORCH.cuda
    # CUDA context explicitly drained + released.
    assert len(cuda.synchronize_calls) >= 1
    assert cuda.empty_cache_calls >= 1
    # Python GC pass ran.
    assert inference.gc.collect_calls >= 1
    # sync targeted the actual GPU (index form when available).
    assert device.index is None or device.index in cuda.synchronize_calls


def test_free_drains_cuda_and_drops_pipelines(real_inference):
    device = _FakeDevice(index=0)
    eng = _make_engine(real_inference, device)
    eng.free()
    _assert_fully_cleared(eng, real_inference, device)


def test_free_synchronizes_bare_cuda_device_without_index(real_inference):
    # A bare `cuda` device (no index) must NOT crash _free_vram.
    device = _FakeDevice(index=None)
    eng = _make_engine(real_inference, device)
    eng.free()
    _assert_fully_cleared(eng, real_inference, device)


def test_free_is_idempotent(real_inference):
    device = _FakeDevice(index=1)
    eng = _make_engine(real_inference, device)
    eng.free()  # first call clears everything
    calls_after_first = (
        len(FAKE_TORCH.cuda.synchronize_calls),
        FAKE_TORCH.cuda.empty_cache_calls,
        real_inference.gc.collect_calls,
    )
    eng.free()  # second call must be a harmless no-op (no error raised)
    calls_after_second = (
        len(FAKE_TORCH.cuda.synchronize_calls),
        FAKE_TORCH.cuda.empty_cache_calls,
        real_inference.gc.collect_calls,
    )
    # Implicitly: calling free() the second time raised nothing (assert reached).
    assert calls_after_second == calls_after_first


def test_free_with_no_pipelines_loaded_is_safe(real_inference):
    # Fresh engine, nothing resident: free() must not error or demand CUDA work.
    device = _FakeDevice(index=0)
    eng = real_inference.VideoCreatorInferenceEngine(
        checkpoint="/models/x", gemma_root="/models/g", upsampler_path="",
        device=device, profile=None,
    )
    eng.free()  # no-op path (nothing loaded)
    assert eng._pipeline is None and eng._pipeline25 is None


# ---------------------------------------------------------------------------
# Route-level: real handle_evict is idempotent and still calls engine.free().
# ---------------------------------------------------------------------------
STUB = {
    "runner.ltx.enhance_forward": types.ModuleType("runner.ltx.enhance_forward"),
    "runner.ltx.inference": types.ModuleType("runner.ltx.inference"),
    "runner.ltx.loracache": types.ModuleType("runner.ltx.loracache"),
}
STUB["runner.ltx.enhance_forward"].DEFAULT_T2V_SYSTEM_PROMPT = "t2v"
STUB["runner.ltx.enhance_forward"].DEFAULT_I2V_SYSTEM_PROMPT = "i2v"
STUB["runner.ltx.inference"].VideoCreatorInferenceEngine = object  # type: ignore[attr-defined]
STUB["runner.ltx.loracache"].LoraCache = object  # type: ignore[attr-defined]


@pytest.fixture()
def evict_server(monkeypatch):
    """Import the real server module with heavy siblings stubbed (torch-free)."""
    _reset_fakes()
    monkeypatch.setitem(sys.modules, "torch", FAKE_TORCH)
    for name in list(sys.modules):
        if name.startswith("runner.ltx") and name != "runner.ltx.config":
            del sys.modules[name]
    for mod_name, mod in STUB.items():
        monkeypatch.setitem(sys.modules, mod_name, mod)

    cfg = types.ModuleType("runner.ltx.config")
    for k, v in {
        "LIVE_RUNNER_URL": "http://live-runner:8990",
        "ENHANCE_FORWARD_API_KEY": "", "ENHANCE_FORWARD_MODEL": "",
        "ENHANCE_FORWARD_TIMEOUT": 30, "ENHANCE_FORWARD_URL": "",
        "ENHANCE_GPU_DEVICE": "0", "GPU_DEVICE": "0", "GPU_NAME": "RTX 5090",
        "GPU_VRAM_GB": 0, "HOST": "0.0.0.0", "MODEL_CHECKPOINT": "/models/x",
        "PORT": 8991, "TEXT_ENCODER_ROOT": "/models/gemma", "UPSCALER_PATH": "",
        "WARMUP": False, "IDV2V_WORKER_URL": "http://idv2v-worker:8992",
        "worker_token": lambda: "tok",
    }.items():
        setattr(cfg, k, v)
    monkeypatch.setitem(sys.modules, "runner.ltx.config", cfg)

    import runner.ltx.server as server
    return server


def test_evict_route_calls_free_and_is_idempotent(evict_server):
    server = evict_server
    free_calls = {"n": 0}

    class _Engine:
        def free(self):
            free_calls["n"] += 1

    server.engine = _Engine()
    asyncio.run(server.handle_evict(_req()))
    assert free_calls["n"] == 1
    # Second evict on the SAME engine: still free()s (kept object) — no 500.
    # The optional {"device": N} body is parsed and ignored on the single engine.
    asyncio.run(server.handle_evict(_req(device=3)))
    assert free_calls["n"] == 2
    # Response shape preserved: 200 + {"evicted": True}.
    resp = asyncio.run(server.handle_evict(_req()))
    assert resp.status == 200
    assert json.loads(resp.text) == {"evicted": True}


def test_evict_route_with_no_engine_is_noop(evict_server):
    # Previously an `assert engine is not None` would 500 here. Now it's a no-op.
    evict_server.engine = None
    resp = asyncio.run(evict_server.handle_evict(_req()))
    assert resp.status == 200
    assert json.loads(resp.text) == {"evicted": True}


def _req(device=None):
    """Minimal aiohttp-web.Request stand-in; json() returns a dict (or raises)."""

    class _Headers:
        def get(self, key, default=None):
            # Worker token "tok" matches the config stub used in the fixture.
            return "tok" if key == "X-Worker-Token" else default

    class _Req:
        headers = _Headers()

        async def json(self):
            if device is None:
                raise Exception("no body")
            return {"device": device}

    return _Req()
