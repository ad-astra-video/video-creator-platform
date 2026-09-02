"""image-worker: the FLUX.2 klein editor must honor the engine's assigned GPU.

Regression test for the restyle "wrong GPU / didn't evict" bug. The klein
editor is a process-global SINGLETON whose device used to be fixed at
`klein4b_device()` = cuda:0 (GPU_DEVICE default) regardless of the live-runner
scheduler's per-task assignment. That silently parked ~18.6 GiB on a card the
scheduler's advisory map (from /info devices) didn't think was owned, so a
video task later took that same GPU and co-resided with it -> near-OOM.

Torch-free: a fake `torch` is injected; we import the REAL modules and verify:

  * `ImageInferenceEngine.klein_image` / `style_frame` call
    `editor.relocate(self._active_device())` BEFORE `ensure_loaded()` so klein
    lands on the engine's assigned GPU, not a hardcoded default.
  * `FluxKleinEditor.relocate()` normalizes 'N' -> 'cuda:N', no-ops when already
    on target, and unloads a resident model before moving to a different card.
  * `server._klein_resident_device()` reports the singleton's CUDA index so
    /info `devices` (the scheduler's reconcile source) includes klein's card.
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
    monkeypatch.setitem(sys.modules, "torch", ft)
    return ft


@pytest.fixture()
def image_server(monkeypatch):
    # Fresh import of the real server module so module-global singleton locks
    # / engines don't leak between tests.
    for name in [n for n in sys.modules if n.startswith("runner.image")]:
        del sys.modules[name]
    import runner.image.server as server
    return server


def test_klein_image_binds_editor_to_engine_device(fake_torch, monkeypatch):
    from runner.image import inference as _inf
    from runner.image import flux_edit as _fe

    calls = []

    class _FakeEditor:
        def __init__(self):
            self._device = "cuda:0"
            self._model = None

        def relocate(self, device):
            calls.append(device)

        def ensure_loaded(self):
            calls.append("ensure_loaded")

        def generate(self, *a, **k):
            from PIL import Image
            return Image.new("RGB", (32, 32))

    # Point the module singleton get_editor() at the fake.
    real_get_editor = _fe.get_editor
    monkeypatch.setattr(_fe, "get_editor", lambda: _FakeEditor())

    e = _inf.ImageInferenceEngine(profile=None)
    e.current_device = 2  # scheduler assigned GPU 2 for this request

    _inf.ImageInferenceEngine.klein_image(
        e, "a robot", seed=1, width=512, height=512)
    assert calls == [2, "ensure_loaded"], (
        "klein_image must relocate klein onto the engine's assigned GPU (2) "
        "before loading, so its residency is visible to the scheduler")


def test_style_frame_binds_editor_to_engine_device(fake_torch, monkeypatch):
    from runner.image import inference as _inf
    from runner.image import flux_edit as _fe
    from PIL import Image

    calls = []

    class _FakeEditor:
        def relocate(self, device):
            calls.append(device)

        def ensure_loaded(self):
            calls.append("ensure_loaded")

        def edit(self, image, prompt, seed, **kw):
            return image

    monkeypatch.setattr(_fe, "get_editor", lambda: _FakeEditor())

    e = _inf.ImageInferenceEngine(profile=None)
    e.current_device = 0

    src = Image.new("RGB", (64, 64))
    _inf.ImageInferenceEngine.style_frame(
        e, src, "a volcano", seed=1, width=64, height=64)
    assert calls == [0, "ensure_loaded"], (
        "style_frame must relocate klein onto the engine's assigned GPU (0) "
        "before loading")


def test_relocate_normalizes_and_moves(fake_torch):
    from runner.image import flux_edit as _fe

    ed = _fe.FluxKleinEditor(device=None)
    # _normalize_device('') -> 'cuda:0' default.
    assert ed.device == "cuda:0"

    # '1' is normalized to 'cuda:1'.
    ed.relocate("1")
    assert ed.device == "cuda:1"

    # Already on target -> no-op (model stays resident, no reload needed).
    ed.relocate("cuda:1")
    assert ed.device == "cuda:1"


def test_relocate_unloads_resident_model_before_moving(fake_torch):
    from runner.image import flux_edit as _fe

    ed = _fe.FluxKleinEditor(device=None)  # -> cuda:0
    ed.relocate("0")  # ensure cuda:0
    # Pretend a model is resident on GPU 0.
    ed._model = object()
    ed._ae = object()

    ed.relocate("cuda:2")
    assert ed.device == "cuda:2", "device must retarget to the new GPU"
    assert ed._model is None, "resident model must be unloaded before moving GPU"
    assert ed._ae is None


def test_klein_resident_device_reports_singleton_gpu(image_server, fake_torch):
    from runner.image import flux_edit as _fe

    # Singleton not loaded -> None.
    assert image_server._klein_resident_device() is None

    # Pretend klein is resident on cuda:2 -> must report index 2.
    ed = _fe.get_editor()
    ed._device = "cuda:2"
    ed._model = object()
    assert image_server._klein_resident_device() == 2


def test_info_devices_includes_klein_card(image_server, fake_torch):
    """/info devices must include klein's GPU so the scheduler sees it."""
    from runner.image import flux_edit as _fe
    from runner.image import server as _mod

    # Set up one per-device engine on GPU 1 + klein resident on GPU 0.
    _mod._engines.clear()
    _mod._device_locks.clear()
    _mod._default_device = 1
    e = _mod.ImageInferenceEngine(profile=None)
    e.current_device = 1
    _mod._engines[1] = e
    _mod._device_locks[1] = __import__("asyncio").Lock()

    ed = _fe.get_editor()
    ed._device = "cuda:0"
    ed._model = object()

    import asyncio
    resp = asyncio.run(image_server.handle_info(None))
    import json
    data = json.loads(resp.text)
    assert set(data["devices"]) == {0, 1}, (
        "devices must include klein's GPU 0 AND the engine's GPU 1")
    assert data["klein_device"] == 0


def test_evict_device_frees_klein_when_on_that_card(image_server, fake_torch):
    from runner.image import flux_edit as _fe
    from runner.image import server as _mod

    _mod._engines.clear()
    _mod._device_locks.clear()
    _mod._default_device = None

    ed = _fe.get_editor()
    ed._device = "cuda:0"
    ed._model = object()

    class _Req:
        headers = {"X-Worker-Token": "tok"}

        async def json(self):
            return {"device": 0}

    import asyncio
    resp = asyncio.run(image_server.handle_evict(_Req()))
    assert resp.status == 200
    # klein singleton must have been unloaded (it was resident on device 0).
    assert ed.is_ready is False, "klein resident on the evicted card must be freed"
