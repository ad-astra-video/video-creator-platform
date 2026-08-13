"""GPU-independent tests for the live-runner swap policy + routing.

Covers the ResidentWorkerManager evict-before-load invariant (only one worker
resident at a time) and the capability->worker route table. Uses an in-memory
fake transport — no aiohttp server, no GPU, no gateway SDK required.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402

from runner.live_runner.swap import ResidentWorkerManager, WorkerTransport  # noqa: E402


class InMemoryTransport(WorkerTransport):
    """Records calls; acts like a set of healthy fake workers."""

    def __init__(self):
        self.order = []       # ["load:ltx", "evict:ltx", ...]
        self.resident = None

    def _name(self, base):
        # The injected worker->base mapping points names at a base that encodes
        # the name itself ("ltx" -> "ltx", "idv2v" -> "idv2v").
        return base.rstrip("/").split("/")[-1]

    async def post(self, base, path, payload=None):
        name = self._name(base)
        kind = path.lstrip("/")  # "load" | "evict" | "restyle"...
        self.order.append(f"{kind}:{name}")
        if kind == "load":
            self.resident = name
        elif kind == "evict":
            self.resident = None
        return {}

    async def health(self, base):
        return {"status": "ok"}


@pytest.fixture
def workers_map():
    return {"ltx": "ltx", "idv2v": "idv2v"}


@pytest.fixture
def fake_workers():
    return InMemoryTransport()


def _mk(transport, workers_map):
    return ResidentWorkerManager(transport=transport, workers=workers_map)


async def _ensure_evicts_previous_resident(fake_workers, workers_map):
    w = _mk(fake_workers, workers_map)
    await w.ensure("ltx")
    await w.ensure("idv2v")
    assert fake_workers.order == ["load:ltx", "evict:ltx", "load:idv2v"]
    assert w.resident == "idv2v"


async def _ensure_same_resident_is_noop(fake_workers, workers_map):
    w = _mk(fake_workers, workers_map)
    await w.ensure("ltx")
    await w.ensure("ltx")
    assert fake_workers.order == ["load:ltx"]  # second ensure skipped load


async def _evict_all_releases(fake_workers, workers_map):
    w = _mk(fake_workers, workers_map)
    await w.ensure("ltx")
    await w.evict_all()
    assert fake_workers.order == ["load:ltx", "evict:ltx"]
    assert w.resident is None


async def _check_health_reports_workers_up(fake_workers, workers_map):
    w = _mk(fake_workers, workers_map)
    meta = await w.check_health()
    assert meta["ltx_up"] is True
    assert meta["idv2v_up"] is True


def test_resident_starts_none(fake_workers, workers_map):
    w = _mk(fake_workers, workers_map)
    assert w.resident is None


def test_ensure_evicts_previous_resident(fake_workers, workers_map):
    asyncio.run(_ensure_evicts_previous_resident(fake_workers, workers_map))


def test_ensure_same_resident_is_noop(fake_workers, workers_map):
    asyncio.run(_ensure_same_resident_is_noop(fake_workers, workers_map))


def test_evict_all_releases(fake_workers, workers_map):
    asyncio.run(_evict_all_releases(fake_workers, workers_map))


def test_check_health_reports_workers_up(fake_workers, workers_map):
    asyncio.run(_check_health_reports_workers_up(fake_workers, workers_map))


def test_routing_table_has_restyle_on_idv2v():
    from runner.live_runner.routing import ROUTES
    assert ROUTES["restyle"] == "idv2v-worker"
    # All the old LTX endpoints route to ltx-worker.
    for ep in ("t2v", "i2v", "retake", "extend", "image"):
        assert ROUTES[ep] == "ltx-worker"


if __name__ == "__main__":
    from asyncio import run

    async def _main():
        fw = InMemoryTransport()
        w = ResidentWorkerManager(transport=fw, workers={"ltx": "ltx", "idv2v": "idv2v"})
        await w.ensure("ltx")
        await w.ensure("idv2v")
        print(fw.order)
        assert fw.order == ["load:ltx", "evict:ltx", "load:idv2v"]
        print("PASS test_ensure_evicts_previous_resident")

    run(_main())


# ── Gemma LLM backend residency ──────────────────────────────────────────────

def _mk_wm(fake_workers, pinned=frozenset()):
    return ResidentWorkerManager(
        transport=fake_workers,
        workers={"ltx-worker": "ltx-worker", "idv2v-worker": "idv2v-worker",
                 "gemma-worker": "gemma-worker"},
        pinned=pinned,
    )


def test_routing_repoints_llm_endpoints_to_gemma():
    from runner.live_runner.routing import ROUTES
    assert ROUTES["prompt-enhance"] == "gemma-worker"
    assert ROUTES["suggest-gap-prompt"] == "gemma-worker"
    assert ROUTES["chat"] == "gemma-worker"
    assert ROUTES["t2v"] == "ltx-worker"
    assert ROUTES["restyle"] == "idv2v-worker"


def test_shared_gemma_evicted_for_render_worker(fake_workers):
    wm = _mk_wm(fake_workers)  # blank LLM_GPU_DEVICE -> shared slot, evictable

    async def _t():
        await wm.backfill("gemma-worker")       # idle backfill loads gemma
        assert wm.resident == "gemma-worker"
        await wm.ensure("ltx-worker")           # render request evicts gemma
        assert fake_workers.order == [
            "load:gemma-worker", "evict:gemma-worker", "load:ltx-worker"]
        assert wm.resident == "ltx-worker"

    asyncio.run(_t())


def test_backfill_only_when_slot_free(fake_workers):
    wm = _mk_wm(fake_workers)

    async def _t():
        await wm.ensure("ltx-worker")           # shared slot occupied
        await wm.backfill("gemma-worker")       # must NOT evict ltx
        assert fake_workers.order == ["load:ltx-worker"]
        assert wm.resident == "ltx-worker"
        await wm.evict_all()
        await wm.backfill("gemma-worker")       # slot free -> loads
        assert fake_workers.order == [
            "load:ltx-worker", "evict:ltx-worker", "load:gemma-worker"]
        assert wm.resident == "gemma-worker"

    asyncio.run(_t())


def test_pinned_gemma_never_evicted(fake_workers):
    wm = _mk_wm(fake_workers, pinned=frozenset({"gemma-worker"}))  # set GPU

    async def _t():
        await wm.load_pinned()
        assert "gemma-worker" in wm.pinned_resident
        assert fake_workers.order == ["load:gemma-worker"]
        await wm.ensure("ltx-worker")           # render loads; gemma NOT evicted
        assert fake_workers.order == ["load:gemma-worker", "load:ltx-worker"]
        assert wm.resident == "ltx-worker"
        assert "gemma-worker" in wm.pinned_resident
        await wm.evict_all()                    # evicts shared ltx only
        assert fake_workers.order == [
            "load:gemma-worker", "load:ltx-worker", "evict:ltx-worker"]
        assert "gemma-worker" in wm.pinned_resident

    asyncio.run(_t())


def test_pinned_load_is_idempotent(fake_workers):
    wm = _mk_wm(fake_workers, pinned=frozenset({"gemma-worker"}))

    async def _t():
        await wm.load_pinned()
        await wm.load_pinned()
        await wm.ensure("gemma-worker")
        assert fake_workers.order == ["load:gemma-worker"]  # loaded once

    asyncio.run(_t())


def test_health_reports_gemma_model(fake_workers):
    wm = _mk_wm(fake_workers, pinned=frozenset({"gemma-worker"}))

    async def _t():
        await wm.load_pinned()
        meta = await wm.check_health()
        assert meta["gemma_model_loaded"] is True
        assert meta["gemma-worker_up"] is True
        assert "gemma-worker" in meta["pinned_resident"]

    asyncio.run(_t())
