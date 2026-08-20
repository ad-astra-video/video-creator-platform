"""image-worker: /evict must clear the CUDA context from the GPU.

These tests are torch-free (no GPU): a FAKE ``torch`` module is injected into
sys.modules so the engine's lazily-imported ``import torch`` resolves to it and
records whether the CUDA-teardown sequence (synchronize + empty_cache) ran. We
verify:

  * engine.free() drops every resident pipeline -> None, marks ready=False,
    drains the device (torch.cuda.synchronize) and returns VRAM
    (torch.cuda.empty_cache), and is idempotent.
  * the /evict route frees ONLY the requested device for a multi-resident
    worker (other cards' engines stay warm), drops the device lock, and is
    idempotent, with no-device meaning "free all engines".
"""

import os
import sys
import types

import pytest

os.environ.setdefault("WORKER_TOKEN", "tok")


class _FakeCuda:
    def __init__(self):
        self.available = True
        self.sync_calls = []
        self.empty_calls = []

    def is_available(self):
        return self.available

    def synchronize(self, *a, **k):
        self.sync_calls.append((a, k))

    def empty_cache(self):
        self.empty_calls.append(1)


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()


@pytest.fixture()
def fake_torch(monkeypatch):
    ft = _FakeTorch()
    # The engine does `import torch` lazily inside free(); point it at the fake.
    monkeypatch.setitem(sys.modules, "torch", ft)
    return ft


@pytest.fixture()
def engine_cls():
    try:
        from runner.image import inference as _inf
    except Exception as exc:  # PIL/missing dep -> cannot test this host
        pytest.skip(f"runner.image.inference not importable: {exc}")
    return _inf.ImageInferenceEngine


def _seeded_engine(engine_cls):
    e = engine_cls(profile=None)
    e.current_device = 0
    e._qwen_edit = object()
    e._qwen_edit_inpaint = object()
    e._qwen_layered = object()
    e._zimage = object()
    e._hidream = object()
    e.ready = True
    return e


def test_image_engine_free_clears_all_pipelines_and_gpu(engine_cls, fake_torch):
    e = _seeded_engine(engine_cls)
    e.free()
    assert e._qwen_edit is None
    assert e._qwen_edit_inpaint is None
    assert e._qwen_layered is None
    assert e._zimage is None
    assert e._hidream is None
    assert e.ready is False
    assert fake_torch.cuda.sync_calls, "torch.cuda.synchronize must be called to drain in-flight kernels"
    assert fake_torch.cuda.empty_calls, "torch.cuda.empty_cache must be called to return VRAM to the driver"


def test_image_engine_free_is_idempotent(engine_cls, fake_torch):
    e = _seeded_engine(engine_cls)
    e.free()
    e.free()  # second /evict must not raise and must stay clean
    assert e._qwen_edit is None
    assert e.ready is False


# ── server /evict route: per-device registry removal ─────────────────────────

class _FakeReq:
    def __init__(self, device, headers=None):
        self._device = device
        self.headers = headers or {"X-Worker-Token": "tok"}

    async def json(self):
        return {"device": self._device} if self._device is not None else {}


@pytest.fixture()
def image_server(monkeypatch):
    # Reset any prior imports so a stale module doesn't leak engines.
    for name in list(sys.modules):
        if name.startswith("runner.image") and name != "runner.image.config":
            del sys.modules[name]
    import runner.image.server as server
    server._engines.clear()
    server._device_locks.clear()
    server._default_device = None
    return server


def _registered_engine(image_server, engine_cls, device):
    e = engine_cls(profile=None)
    e.current_device = device
    image_server._engines[device] = e
    image_server._device_locks[device] = __import__("asyncio").Lock()
    return e


def test_image_route_evict_frees_only_requested_device(image_server, engine_cls, fake_torch):
    e0 = _registered_engine(image_server, engine_cls, 0)
    e1 = _registered_engine(image_server, engine_cls, 1)

    import asyncio
    resp = asyncio.run(image_server.handle_evict(_FakeReq(0)))

    assert resp.status == 200
    assert 0 not in image_server._engines, "device 0 must be removed from the registry"
    assert 0 not in image_server._device_locks, "device 0 lock must be dropped"
    assert 1 in image_server._engines, "device 1 (other card) must stay warm"
    assert e1 is image_server._engines[1]
    assert e0._qwen_edit is None or e0.ready is False


def test_image_route_evict_idempotent_and_all_device(image_server, engine_cls, fake_torch):
    _registered_engine(image_server, engine_cls, 0)
    _registered_engine(image_server, engine_cls, 1)
    import asyncio
    asyncio.run(image_server.handle_evict(_FakeReq(0)))
    # Evicting the same device again (already gone) must not raise.
    asyncio.run(image_server.handle_evict(_FakeReq(0)))
    # No-device evict frees ALL remaining engines.
    asyncio.run(image_server.handle_evict(_FakeReq(None)))
    assert image_server._engines == {}
    assert image_server._device_locks == {}
