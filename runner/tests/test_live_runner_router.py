"""GPU-independent tests for the live-runner swap policy + routing.

Covers the ResidentWorkerManager evict-before-load invariant (only one worker
resident at a time), the capability->worker route table, and the prompt-enhance
fallback (gemma-worker -> LTX text encoder). Uses an in-memory fake transport —
no aiohttp server, no GPU, no gateway SDK required.
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
        assert meta["gmm"] is True
        assert meta["gemma_up"] is True
        assert "gemma-worker" in meta["pin"]

    asyncio.run(_t())


# ── Prompt-enhance fallback: gemma-worker -> LTX text encoder ─────────────────

def test_candidate_workers_only_prompt_enhance_has_fallback():
    from runner.live_runner.routing import candidate_workers
    # prompt-enhance prefers the dedicated gemma-worker, falls back to ltx-worker
    # (the LTX pipeline's own Gemma text encoder).
    assert candidate_workers("prompt-enhance", "gemma-worker") == [
        "gemma-worker", "ltx-worker"]
    # Everything else is single-worker (no fallback, unchanged behaviour).
    for ep in ("t2v", "i2v", "restyle", "extend", "retake", "chat", "image"):
        assert candidate_workers(ep, "ltx-worker") == ["ltx-worker"]
        assert candidate_workers(ep, "idv2v-worker") == ["idv2v-worker"]
    # A prompt-enhance NOT routed to gemma-worker also gets no fallback loop.
    assert candidate_workers("prompt-enhance", "ltx-worker") == ["ltx-worker"]


class _FakeResp:
    """aiohttp-style response object usable inside `async with`."""

    def __init__(self, status, body=b'{"enhanced_prompt":"rewritten"}'):
        self.status = status
        self._body = body
        self.content_type = "application/json"
        self.charset = "utf-8"

    async def read(self):
        return self._body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    """Mimics aiohttp ClientSession: post() returns a SYNC context manager whose
    __aenter__ either raises (worker down), returns an error response (>=400),
    or returns a success response."""

    def __init__(self):
        self.by_host = {}   # host -> status int | Exception instance
        self.seen = []

    def post(self, url, **_kw):
        host = url.split("//")[1].split(":")[0]
        self.seen.append(host)
        entry = self.by_host[host]

        class _CM:
            def __init__(self, entry):
                self._entry = entry

            async def __aenter__(self):
                if isinstance(self._entry, Exception):
                    raise self._entry
                if 400 <= self._entry <= 599:
                    return _FakeResp(self._entry, b'{"error":"bad"}')
                return _FakeResp(self._entry)

            async def __aexit__(self, *a):
                return False

        return _CM(entry)


class _FakeWM:
    def __init__(self):
        self.ensured = []

    async def ensure(self, name):
        self.ensured.append(name)


def _patch_workers():
    from runner.live_runner import config as cfg
    old = dict(cfg.WORKERS)
    cfg.WORKERS = {"gemma-worker": "http://gemma-worker:8993",
                   "ltx-worker": "http://ltx-worker:8991"}
    return cfg, old


def test_proxy_falls_back_when_gemma_worker_down():
    """proxy() must route prompt-enhance to ltx-worker when gemma-worker is
    unreachable, so the LTX pipeline text encoder is used as the fallback."""
    from runner.live_runner.routing import proxy
    cfg, old = _patch_workers()
    try:
        sess = _FakeSession()
        # gemma-worker container down -> connection error -> must fall back.
        sess.by_host = {"gemma-worker": ConnectionError("gemma down"),
                        "ltx-worker": 200}
        wm = _FakeWM()
        resp = asyncio.run(proxy(wm, sess, "tok", "gemma-worker",
                                 "prompt-enhance", {"prompt": "hi"}))
        assert resp.status == 200, resp
        assert sess.seen == ["gemma-worker", "ltx-worker"], sess.seen
        assert wm.ensured == ["gemma-worker", "ltx-worker"]
        assert b"enhanced_prompt" in resp.body
    finally:
        cfg.WORKERS = old


def test_proxy_falls_back_when_gemma_errors():
    """Even an HTTP error from gemma-worker (up but failing) falls back to the
    LTX text encoder rather than surfacing a hard failure."""
    from runner.live_runner.routing import proxy
    cfg, old = _patch_workers()
    try:
        sess = _FakeSession()
        sess.by_host = {"gemma-worker": 503, "ltx-worker": 200}
        resp = asyncio.run(proxy(_FakeWM(), sess, "tok", "gemma-worker",
                                 "prompt-enhance", {"prompt": "hi"}))
        assert resp.status == 200
        assert sess.seen == ["gemma-worker", "ltx-worker"]
    finally:
        cfg.WORKERS = old


def test_proxy_uses_gemma_only_when_available():
    """When gemma-worker is up, prompt-enhance must NOT touch ltx-worker at all."""
    from runner.live_runner.routing import proxy
    cfg, old = _patch_workers()
    try:
        sess = _FakeSession()
        sess.by_host = {"gemma-worker": 200}
        resp = asyncio.run(proxy(_FakeWM(), sess, "tok", "gemma-worker",
                                 "prompt-enhance", {"prompt": "hi"}))
        assert resp.status == 200
        assert sess.seen == ["gemma-worker"], sess.seen
    finally:
        cfg.WORKERS = old


def test_proxy_reports_error_when_all_candidates_fail():
    from runner.live_runner.routing import proxy
    cfg, old = _patch_workers()
    try:
        sess = _FakeSession()
        sess.by_host = {"gemma-worker": 503, "ltx-worker": 502}
        resp = asyncio.run(proxy(_FakeWM(), sess, "tok", "gemma-worker",
                                 "prompt-enhance", {"prompt": "hi"}))
        assert resp.status >= 500
        assert sess.seen == ["gemma-worker", "ltx-worker"]
    finally:
        cfg.WORKERS = old
