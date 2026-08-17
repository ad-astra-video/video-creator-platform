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
        self.payloads = []    # request bodies sent with post()
        self.resident = None
        self.model_loaded = None   # when set, /health reports this value

    def _name(self, base):
        # The injected worker->base mapping points names at a base that encodes
        # the name itself ("ltx" -> "ltx", "idv2v" -> "idv2v").
        return base.rstrip("/").split("/")[-1]

    async def post(self, base, path, payload=None):
        name = self._name(base)
        kind = path.lstrip("/")  # "load" | "evict" | "restyle"...
        self.order.append(f"{kind}:{name}")
        self.payloads.append(payload if payload is not None else {})
        if kind == "load":
            self.resident = name
        elif kind == "evict":
            self.resident = None
        return {}

    async def health(self, base):
        if self.model_loaded is not None:
            return {"status": "ok", "model_loaded": self.model_loaded}
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
    # LTX keeps the video endpoints.
    for ep in ("t2v", "i2v", "retake", "extend"):
        assert ROUTES[ep] == "ltx-worker"


def test_routing_routes_image_caps_to_image_worker():
    # image/edit/layer/style-frame move to the dedicated image-worker; sam3 and
    # the video workers are unchanged; new capabilities + model ids advertised.
    from runner.live_runner.routing import ROUTES, CAPABILITIES, MODELS
    for ep in ("image", "edit", "layer", "style-frame"):
        assert ROUTES[ep] == "image-worker"
    # sam3 (keep-subject segmentation) stays on idv2v-worker (no sam3 changes).
    assert ROUTES["sam3"] == "idv2v-worker"
    # New capabilities advertised in the heartbeat.
    assert "layer" in CAPABILITIES
    assert "style-frame" in CAPABILITIES
    # qwen-image-edit was an engine label, not a route — dropped; models advertised by id.
    assert "qwen-image-edit" not in CAPABILITIES
    assert MODELS == ["z-image", "flux", "qwen"]


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

def test_ensure_device_injects_device_and_does_not_evict_other_worker(fake_workers):
    # Plan B: ensure(name, device=N) sends device in the /load body and does
    # not evict a worker running on a different GPU (one task -> one GPU).
    wm = ResidentWorkerManager(
        transport=fake_workers,
        workers={"ltx-worker": "ltx-worker", "idv2v-worker": "idv2v-worker",
                 "gemma-worker": "gemma-worker", "image-worker": "image-worker"},
    )

    async def _t():
        await wm.ensure("ltx-worker", device=0)
        await wm.ensure("image-worker", device=2)
        # No cross-worker eviction: both stay loaded on their own GPUs.
        assert fake_workers.order == ["load:ltx-worker", "load:image-worker"]
        # The /load bodies carried the device index.
        assert fake_workers.payloads[0] == {"device": 0}
        assert fake_workers.payloads[1] == {"device": 2}
        # A second load of the same worker on the same device is a no-op.
        await wm.ensure("ltx-worker", device=0)
        assert fake_workers.order == ["load:ltx-worker", "load:image-worker"]

    asyncio.run(_t())


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
        self.devices = []

    async def ensure(self, name, device=None):
        self.ensured.append(name)
        self.devices.append(device)


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


# ── GPU co-opt scheduler: gemma pinned to one GPU, evict-on-need ─────────────
from runner.live_runner.scheduler import GPUScheduler  # noqa: E402


def test_coopt_scheduler_gemma_gets_own_gpu_first():
    """gemma always takes its own resident GPU while loaded."""
    async def _t():
        s = GPUScheduler(gemma_resident_gpu=2, gpu_count=3)
        assert await s.acquire("gemma-worker") == 2
        assert s.status()["gemma_evicted"] is False
    asyncio.run(_t())


def test_coopt_scheduler_render_task_uses_free_gpu():
    """A render task grabs a genuinely free (non-gemma) GPU first."""
    async def _t():
        s = GPUScheduler(gemma_resident_gpu=2, gpu_count=3)
        gpu = await s.acquire("ltx-worker")
        assert gpu in (0, 1)
        assert s.status()["gpus"][gpu]["worker"] == "ltx-worker"
        assert s.status()["gemma_evicted"] is False
    asyncio.run(_t())


def test_coopt_scheduler_evicts_gemma_when_all_busy():
    """When no free GPU and gemma is resident, a render task co-opts gemma's
    GPU: coopt_cb fires (evict), gemma's GPU is handed to the render task."""
    evicted = []
    async def _cb():
        evicted.append(True)

    async def _t():
        s = GPUScheduler(gemma_resident_gpu=2, gpu_count=3)
        # Fill GPUs 0 and 1 with render tasks.
        g0 = await s.acquire("ltx-worker")
        g1 = await s.acquire("image-worker")
        assert sorted((g0, g1)) == [0, 1]
        # Wire the co-opt hook (only then is co-opt active).
        s.coopt_cb = _cb
        # Third render task has no free GPU -> co-opts gemma's GPU 2.
        g2 = await s.acquire("idv2v-worker")
        assert g2 == 2
        assert evicted == [True]
        st = s.status()
        assert st["gpus"][2]["worker"] == "idv2v-worker"
        assert st["gpus"][2]["state"] == "busy"
        assert st["gpus"][2]["resident"] is False
        assert st["gemma_evicted"] is True
        # gemma itself now waits for its GPU to free (never borrows another).
        try:
            await asyncio.wait_for(s.acquire("gemma-worker"), timeout=0.2)
            raise AssertionError("gemma should block while its GPU is co-opted")
        except asyncio.TimeoutError:
            pass
    asyncio.run(_t())


def test_coopt_scheduler_no_coopt_without_callback():
    """Without coopt_cb wired, gemma's GPU is never handed to a render task
    (old behaviour): a third task on a 3-GPU box just times out."""
    async def _t():
        s = GPUScheduler(gemma_resident_gpu=2, gpu_count=3)
        await s.acquire("ltx-worker")
        await s.acquire("image-worker")
        try:
            await asyncio.wait_for(s.acquire("idv2v-worker"), timeout=0.2)
            raise AssertionError("should time out without co-opt")
        except asyncio.TimeoutError:
            pass
        st = s.status()
        assert st["gemma_evicted"] is False
        assert st["gpus"][2]["resident"] is True
    asyncio.run(_t())


async def _noop_cb():
    pass


def test_coopt_scheduler_gemma_reload_after_release():
    """After the co-opting task releases gemma's GPU, gemma can reclaim it, and
    gemma_slot_free()/mark_gemma_loaded() reflect residency restored."""
    async def _t():
        s = GPUScheduler(gemma_resident_gpu=2, gpu_count=3)
        s.coopt_cb = _noop_cb
        await s.acquire("ltx-worker")    # GPU 0
        await s.acquire("image-worker")  # GPU 1
        g2 = await s.acquire("idv2v-worker")  # co-opts gemma GPU 2
        assert g2 == 2
        assert s.gemma_slot_free() is False  # render task still on GPU 2
        await s.release("idv2v-worker", 2)
        assert s.gemma_slot_free() is True   # GPU 2 now free for gemma
        # gemma reloads and re-asserts residency.
        g = await s.acquire("gemma-worker")
        assert g == 2
        s.mark_gemma_loaded()
        st = s.status()
        assert st["gemma_evicted"] is False
        assert st["gpus"][2]["worker"] == "gemma-worker"
        assert st["gpus"][2]["resident"] is True
    asyncio.run(_t())


def test_coopt_release_does_not_free_resident_gemma():
    """A stale release of gemma's resident GPU while genuinely resident is a
    no-op (must not free gemma's reserved GPU into the pool)."""
    async def _t():
        s = GPUScheduler(gemma_resident_gpu=2, gpu_count=3)
        await s.release("gemma-worker", 2)
        st = s.status()
        assert st["gpus"][2]["resident"] is True
        assert st["gpus"][2]["state"] == "busy"
    asyncio.run(_t())


def test_reconcile_drops_stale_residency_and_reloads_pinned():
    """After a pinned worker's CONTAINER restarts (/health no longer reports
    model_loaded), check_health() must drop its residency so the next ensure()
    re-issues /load instead of routing to an unloaded worker (the 503
    'Gemma LLM not loaded' bug)."""
    async def _t():
        t = InMemoryTransport()
        w = ResidentWorkerManager(transport=t, workers={"gemma": "gemma"},
                                  pinned=frozenset({"gemma"}))
        # Real runtime loads pinned gemma via ensure(device=...) (on_startup),
        # which records BOTH _resident and _pinned_loaded.
        await w.ensure("gemma", device=0)
        assert t.order == ["load:gemma"]
        assert w._resident == {"gemma": 0}
        assert "gemma" in w._pinned_loaded
        # idle-backfill ensure is a no-op while residency is believed fresh
        await w.ensure("gemma", device=0)
        assert t.order == ["load:gemma"]
        # container restarts -> fresh process has no model
        t.model_loaded = False
        await w.check_health()
        assert w._resident == {}
        assert "gemma" not in w._pinned_loaded
        # next ensure re-loads
        await w.ensure("gemma", device=0)
        assert t.order == ["load:gemma", "load:gemma"]
    asyncio.run(_t())


def test_reconcile_drops_stale_residency_and_reloads_device():
    """Same fix for a non-pinned worker on the per-device residency path
    (_resident): a restarted worker's residency is dropped so ensure() reloads."""
    async def _t():
        t = InMemoryTransport()
        w = ResidentWorkerManager(transport=t, workers={"gemma": "gemma"})
        await w.ensure("gemma", device=0)
        assert t.order == ["load:gemma"]
        assert w._resident == {"gemma": 0}
        # idle-backfill ensure is a no-op (resident on that device)
        await w.ensure("gemma", device=0)
        assert t.order == ["load:gemma"]
        t.model_loaded = False
        await w.check_health()
        assert w._resident == {}
        await w.ensure("gemma", device=0)
        assert t.order == ["load:gemma", "load:gemma"]
    asyncio.run(_t())


def test_reconcile_keeps_residency_when_healthy():
    """A worker that /health reports as still loaded keeps its residency (no
    spurious reload churn)."""
    async def _t():
        t = InMemoryTransport()
        t.model_loaded = True
        w = ResidentWorkerManager(transport=t, workers={"gemma": "gemma"},
                                  pinned=frozenset({"gemma"}))
        await w.ensure("gemma", device=0)
        assert t.order == ["load:gemma"]
        await w.check_health()
        assert "gemma" in w._pinned_loaded
        assert w._resident == {"gemma": 0}   # still resident -> no reload
        await w.ensure("gemma", device=0)
        assert t.order == ["load:gemma"]
    asyncio.run(_t())
