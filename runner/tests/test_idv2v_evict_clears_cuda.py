"""Torch-free regression test: ModelManager.evict() fully releases the GPU.

The idv2v-worker is SINGLE-ENGINE: runner/idv2v/server.py owns exactly one
module-global ModelManager, and /evict must drop the pipe AND return the card
to the live-runner pool. That means evict() has to drain in-flight kernels
(torch.cuda.synchronize) before emptying the CUDA cache -- otherwise a kernel
still queued after the tensors are freed + empty_cache() runs can re-pin VRAM
and the GPU never actually comes back.

This test runs WITHOUT a GPU or the real torch library: we inject a fake
``torch`` module into ``sys.modules`` that records cuda.synchronize /
cuda.empty_cache calls, import the REAL ``runner.idv2v.model`` module against
it, and prove evict():

  * nulls ``_pipe``,
  * calls ``torch.cuda.synchronize(self.device)`` (the new first CUDA step),
  * calls ``torch.cuda.empty_cache()``,
  * synchronizes BEFORE emptying the cache (drain first, then reclaim),
  * is idempotent (a second call on an already-evicted model raises nothing).

``config`` (the only other top-level import in model.py) reads env vars and
imports nothing heavy, so faking just ``torch`` is enough to import cleanly.

Run:
    python -m pytest runner/tests/test_idv2v_evict_clears_cuda.py -q
"""

import sys
import types

import pytest


class _FakeCuda:
    """Pretends a GPU is present and records every CUDA call."""

    def __init__(self):
        self.log = []  # ordered ("synchronize", device) / ("empty_cache", None)
        self.synchronize_calls = []
        self.empty_cache_calls = []

    def is_available(self):
        return True

    def synchronize(self, device=None):
        self.synchronize_calls.append(device)
        self.log.append(("synchronize", device))

    def empty_cache(self):
        self.empty_cache_calls.append(True)
        self.log.append(("empty_cache", None))


def _make_fake_torch():
    """Minimal torch stub sufficient to import runner.idv2v.model + evict()."""
    stub = types.ModuleType("torch")
    stub.cuda = _FakeCuda()
    # model.py defines `class _ChunkedFFN(torch.nn.Module)` at import time, so
    # torch.nn.Module must exist as a valid base class.
    stub.nn = types.SimpleNamespace(Module=type("_Module", (), {}))
    stub.bfloat16 = "bfloat16"
    stub.float32 = "float32"
    stub.cat = lambda *a, **k: a[0]
    stub.is_tensor = lambda x: False
    return stub


@pytest.fixture()
def model_with_fake_torch(monkeypatch):
    """Import the real runner.idv2v.model against a recording fake torch."""
    # Drop any previously-cached copy so the module re-imports against the fresh
    # fake torch below (the module binds `torch` at import time).
    for name in [n for n in sys.modules if n.startswith("runner.idv2v.model")]:
        del sys.modules[name]
    stub = _make_fake_torch()
    monkeypatch.setitem(sys.modules, "torch", stub)
    import runner.idv2v.model as model_mod
    return model_mod, stub.cuda


def test_evict_clears_pipe_and_cuda(model_with_fake_torch):
    model, cuda = model_with_fake_torch
    mgr = model.ModelManager(device="cuda:0")
    mgr._pipe = object()  # pretend a live pipeline is resident on the GPU

    mgr.evict()

    # The most important assertion: the pipeline handle is dropped so a future
    # /load starts from a clean slate.
    assert mgr._pipe is None
    # Drain in-flight kernels on THIS device first...
    assert cuda.synchronize_calls, "torch.cuda.synchronize must be called during evict"
    assert cuda.synchronize_calls == ["cuda:0"]
    # ...then reclaim the freed blocks.
    assert cuda.empty_cache_calls, "torch.cuda.empty_cache must be called during evict"
    # Order must be drain-before-reclaim, otherwise a queued kernel can re-pin
    # VRAM right after the cache is emptied.
    ops = [name for name, _ in cuda.log]
    assert ops[0] == "synchronize", "synchronize must be the first CUDA step"
    assert "empty_cache" in ops


def test_evict_is_idempotent(model_with_fake_torch):
    model, cuda = model_with_fake_torch
    mgr = model.ModelManager(device="cuda:0")

    # Two consecutive evicts with nothing loaded must not raise and must leave
    # the manager in the same unloaded state (safe under the server's re-run).
    mgr.evict()
    mgr.evict()

    assert mgr._pipe is None
    # Each evict still drains + empties so the GPU stays clean on repeat calls.
    assert cuda.synchronize_calls
    assert cuda.empty_cache_calls
