"""Warm-resident GPU scheduler tests (pure asyncio, no GPU, no aiohttp server).

Covers the all-GPUs / keep-warm / LRU-evict policy, now MULTI-RESIDENT:
  - gemma's resident GPU is held out of the task pool
  - acquiring for gemma returns its resident GPU
  - a worker that is already warm reuses the same GPU (no reload)
  - a task with no warm model takes a FREE (unused) GPU — fully utilizes GPUs
  - when all GPUs hold warm models, the LRU warm model is EVICTED for the new task
  - THE regression: after an image gen, the image model must NOT land on the
    video card; with a free GPU available it takes a different card so a later
    video gen never co-resides/OOMs. When only the video card is left free, the
    video worker EVICTS whatever is warm there before loading.
  - MULTI-RESIDENT: two concurrent requests for the SAME worker spread across
    two DIFFERENT GPUs (parallel), not serialize on one warm card
  - MULTI-RESIDENT: after the burst both copies stay warm (idle+resident); the
    next request reuses an idle warm copy (no reload), keeping all cards hot
  - eviction passes the SPECIFIC device so a multi-resident worker only frees
    the one card being handed over (not all its copies)
  - reconcile() from worker /info (device LIST) marks several slots per worker
  - reconcile() self-heals a dead worker's slot
  - FIFO queue + timeout -> QueueTimeout (mapped to 503 by the server)
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402

from runner.live_runner.scheduler import GPUScheduler, QueueTimeout  # noqa: E402


def _mk(gpu_count: int = 3, gemma_gpu: int = 1, timeout: float = 10.0) -> GPUScheduler:
    return GPUScheduler(gemma_resident_gpu=gemma_gpu, gpu_count=gpu_count,
                        queue_timeout_s=timeout)


class _Harness:
    """Records (worker, device) evictions for assertions."""

    def __init__(self):
        self.evicted = []

    async def evict(self, name, device):
        self.evicted.append((name, device))


def test_gemma_gpu_reserved_at_boot():
    s = _mk(gpu_count=3, gemma_gpu=1)
    slots = s.status()["gpus"]
    by_id = {g["gpu_id"]: g for g in slots}
    assert by_id[1]["worker"] == "gemma-worker"
    assert by_id[1]["state"] == "idle"  # resident-warm & available, not in-flight
    assert by_id[1]["resident"] is True
    assert by_id[0]["state"] == "idle"
    assert by_id[2]["state"] == "idle"


def test_reconcile_keeps_idle_warm_slot_reusable_for_same_worker():
    """THE deadlock regression: after image runs warm on both non-gemma cards,
    a heartbeat reconcile must NOT flip those released (idle-warm) slots back to
    "busy". If it did, the same worker could neither reuse them (they read busy)
    nor evict them (its own slots are excluded from LRU eviction) -> it waits
    forever and times out with "no GPU free for image-worker" while every GPU is
    actually idle. Marking reported devices idle (preserving a genuine in-flight
    busy) keeps the warm copy reusable with no reload."""
    s = _mk(gpu_count=3, gemma_gpu=1, timeout=0.05)

    async def _t():
        g1 = await s.acquire("image-worker")
        g2 = await s.acquire("image-worker")  # concurrent peer -> 2nd card
        assert g1 != g2
        await s.release("image-worker", g1)
        await s.release("image-worker", g2)
        # Both copies warm+idle. A heartbeat reconcile re-reports residency.
        await s.reconcile({"image-worker": {"devices": [g1, g2]}})
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        # Reconcile must leave the warm cards reusable (idle), not busy.
        assert st[g1]["state"] == "idle" and st[g1]["resident"] is True
        assert st[g2]["state"] == "idle" and st[g2]["resident"] is True
        # A subsequent request for the SAME worker reuses a warm card (no timeout).
        g3 = await s.acquire("image-worker")
        assert g3 in (g1, g2)
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[g3]["state"] == "busy"  # now in flight again

    asyncio.run(_t())


def test_reconcile_preserves_inflight_busy_mark():
    """Reconcile must NOT downgrade a genuinely in-flight (busy) slot to idle, or
    a concurrent request could share a card mid-job (one-GPU-one-request)."""
    s = _mk(gpu_count=3, gemma_gpu=1, timeout=0.05)

    async def _t():
        gpu = await s.acquire("image-worker")  # in flight -> busy
        await s.reconcile({"image-worker": {"devices": [gpu]}})
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[gpu]["state"] == "busy"  # not downgraded while running

    asyncio.run(_t())


def test_acquire_takes_free_non_gemma_gpu():
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        gpu = await s.acquire("image-worker")
        assert gpu != 1  # never gemma's resident GPU
        st = s.status()["gpus"]
        mine = [g for g in st if g["gpu_id"] == gpu][0]
        assert mine["worker"] == "image-worker"
        assert mine["state"] == "busy"
        assert mine["resident"] is True  # model left warm

    asyncio.run(_t())


def test_release_keeps_model_warm_then_reuse():
    """A released worker stays warm on its GPU; the NEXT acquire for the same
    worker reuses that exact GPU (no reload), even though another is free."""
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        g1 = await s.acquire("image-worker")
        await s.release("image-worker", g1)
        # release keeps it warm: still owned, idle state
        mine = [g for g in s.status()["gpus"] if g["gpu_id"] == g1][0]
        assert mine["worker"] == "image-worker"
        assert mine["state"] == "idle"
        assert mine["resident"] is True
        # reuse -> same GPU, no eviction
        g2 = await s.acquire("image-worker")
        assert g2 == g1

    asyncio.run(_t())


def test_two_tasks_get_two_distinct_gpus():
    """Two held acquires of DIFFERENT workers land on two different GPUs
    (use ALL GPUs). Same-worker re-acquire is warm-reuse (see other test)."""
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        g1 = await s.acquire("image-worker")
        g2 = await s.acquire("idv2v-worker")  # a different worker -> cold -> free card
        assert g1 != g2
        assert {g1, g2} <= {0, 2}  # neither is gemma's GPU 1

    asyncio.run(_t())


def test_two_concurrent_same_worker_spread_to_distinct_gpus():
    """MULTI-RESIDENT: a second request for the SAME worker arriving while the
    first is still busy (not released) takes a DIFFERENT free GPU, so two
    concurrent requests run in parallel instead of serializing on one card."""
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        g1 = await s.acquire("image-worker")   # cold -> free card (0 or 2)
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[g1]["state"] == "busy"       # first is in flight
        g2 = await s.acquire("image-worker")   # no idle warm -> new free card
        assert g1 != g2                        # spread, not serialise
        assert {g1, g2} <= {0, 2}              # neither is gemma's
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[g1]["worker"] == "image-worker" and st[g1]["state"] == "busy"
        assert st[g2]["worker"] == "image-worker" and st[g2]["state"] == "busy"

    asyncio.run(_t())


def test_multi_resident_kept_warm_and_reused():
    """MULTI-RESIDENT: after a burst both copies stay warm (idle+resident).
    A subsequent single request reuses an IDLE warm copy (no reload, no new
    card) rather than spending a fresh load."""
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        g1 = await s.acquire("image-worker")
        g2 = await s.acquire("image-worker")
        assert g1 != g2
        await s.release("image-worker", g1)
        await s.release("image-worker", g2)
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        # both copies kept warm
        assert st[g1]["resident"] and st[g1]["state"] == "idle"
        assert st[g2]["resident"] and st[g2]["state"] == "idle"
        assert st[g1]["worker"] == "image-worker" and st[g2]["worker"] == "image-worker"
        # next request reuses one of the warm cards (no new load needed)
        g3 = await s.acquire("image-worker")
        assert g3 in (g1, g2)
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[g3]["state"] == "busy"       # the reused one is busy again
        # the OTHER copy is still warm+idle, ready for a concurrent peer
        other = (g1 if g3 == g2 else g2)
        assert st[other]["state"] == "idle" and st[other]["resident"] is True

    asyncio.run(_t())


def test_lru_eviction_when_all_gpus_warm():
    """3 free GPUs minus gemma leaves 2. After 2 different workers are warm on
    both, a 3rd worker must EVICT the LRU warm model to get a GPU."""
    h = _Harness()
    s = _mk(gpu_count=3, gemma_gpu=1, timeout=10.0)
    s.evict_cb = h.evict

    async def _t():
        a = await s.acquire("image-worker")   # GPU 0 (fastest free)
        b = await s.acquire("idv2v-worker")   # GPU 2
        assert a != b
        # Both are warm+resident. A new worker (ltx) needs a GPU -> evict LRU.
        ltx = await s.acquire("ltx-worker")
        assert ltx in (a, b)  # took a warm card
        assert len(h.evicted) >= 1
        # the LRU warm model was freed (worker gone from that slot)
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[ltx]["worker"] == "ltx-worker"
        assert st[ltx]["resident"] is True

    asyncio.run(_t())


def test_lru_eviction_passes_specific_device():
    """Eviction hands the scheduler the SPECIFIC GPU, so a multi-resident worker
    frees only the one card being handed over (not all its copies)."""
    h = _Harness()
    s = _mk(gpu_count=3, gemma_gpu=1, timeout=10.0)
    s.evict_cb = h.evict

    async def _t():
        a = await s.acquire("image-worker")
        b = await s.acquire("idv2v-worker")
        # third worker needs a card -> LRU warm (image or idv2v) is evicted
        ltx = await s.acquire("ltx-worker")
        assert len(h.evicted) >= 1
        victim, device = h.evicted[0]
        assert device in (a, b)              # the freed card
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[device]["worker"] == "ltx-worker"   # that card is now ltx's
        # the OTHER worker's card is untouched (still warm)
        kept = a if device == b else b
        assert st[kept]["worker"] != "ltx-worker"

    asyncio.run(_t())


def test_queue_timeout_raises_when_nothing_evictable():
    """If every GPU is warm and evict_cb is NOT wired, the scheduler cannot free
    a card -> waits and times out (no silent co-residency)."""
    s = _mk(gpu_count=3, gemma_gpu=1, timeout=0.05)  # no evict_cb

    async def _t():
        g1 = await s.acquire("image-worker")
        g2 = await s.acquire("idv2v-worker")
        assert g1 != g2
        with pytest.raises(QueueTimeout):
            await s.acquire("ltx-worker")  # a THIRD worker, none free -> timeout

    asyncio.run(_t())


def test_gemma_acquire_returns_resident_gpu():
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        gpu = await s.acquire("gemma-worker")
        assert gpu == 1

    asyncio.run(_t())


def test_reconcile_frees_slot_of_dead_worker():
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        gpu = await s.acquire("image-worker")
        assert [g for g in s.status()["gpus"] if g["gpu_id"] == gpu][0]["state"] == "busy"
        # image-worker is gone (not in workers_info); reconcile frees its slot
        await s.reconcile({"idv2v-worker": {"device_in_use": None}})
        freed = [g for g in s.status()["gpus"] if g["gpu_id"] == gpu][0]
        assert freed["state"] == "idle"
        assert freed["worker"] is None

    asyncio.run(_t())


def test_reconcile_multiple_devices_per_worker():
    """MULTI-RESIDENT: reconcile accepts a device LIST and marks several slots
    as owned by one worker; gemma's pinned card is never touched."""
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        await s.reconcile({"image-worker": {"devices": [0, 2]}})
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[0]["worker"] == "image-worker" and st[0]["resident"] is True
        assert st[2]["worker"] == "image-worker" and st[2]["resident"] is True
        # gemma's pinned card untouched
        assert st[1]["worker"] == "gemma-worker"

    asyncio.run(_t())


def test_reconcile_frees_multi_device_worker_when_gone():
    """MULTI-RESIDENT: when a multi-resident worker disappears, reconcile frees
    ALL of its slots, not just one."""
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        await s.reconcile({"image-worker": {"devices": [0, 2]}})
        # image-worker gone -> both its slots freed
        await s.reconcile({"idv2v-worker": {"device_in_use": None}})
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[0]["worker"] is None and st[0]["resident"] is False
        assert st[2]["worker"] is None and st[2]["resident"] is False
        assert st[1]["worker"] == "gemma-worker"

    asyncio.run(_t())


def test_video_takes_free_gpu_not_evicting_image():
    """THE regression: a video task arriving while image is warm on GPU 0 takes
    a FREE card (GPU 1) instead of evicting image — so image stays warm for a
    fast repeat and no model is dropped. Eviction only happens when NO GPU is
    free."""
    h = _Harness()
    s = _mk(gpu_count=3, gemma_gpu=2, timeout=10.0)
    s.evict_cb = h.evict

    async def _t():
        g_img = await s.acquire("ltx-worker")   # first worker warms a card
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[g_img]["worker"] == "ltx-worker"
        img_card = g_img
        free_card = 1  # gemma on 2; the other of {0,1} is free
        # image gen -> takes the FREE card, NOT evicting the video model
        g_img2 = await s.acquire("image-worker")
        assert g_img2 == free_card
        assert h.evicted == []  # nothing evicted — both stayed warm
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[img_card]["worker"] == "ltx-worker"
        assert st[free_card]["worker"] == "image-worker"

    asyncio.run(_t())


def test_video_and_image_land_on_distinct_free_gpus():
    """Video and image each take a DIFFERENT free GPU (all-GPUs placement):
    no reservation, no eviction, both stay warm."""
    s = _mk(gpu_count=3, gemma_gpu=2, timeout=10.0)

    async def _t():
        g_vid = await s.acquire("ltx-worker")     # first free card
        g_img = await s.acquire("image-worker")   # next free card
        assert g_vid != g_img
        assert {g_vid, g_img} <= {0, 1}  # gemma on 2
        await s.release("ltx-worker", g_vid)
        g_vid2 = await s.acquire("ltx-worker")    # warm-reuse same card
        assert g_vid2 == g_vid

    asyncio.run(_t())


def test_video_and_image_keep_warm_after_release():
    """After releasing both, image AND video stay resident on their own cards
    (idle+resident) so a repeat of either is a no-reload warm hit."""
    s = _mk(gpu_count=3, gemma_gpu=2, timeout=10.0)

    async def _t():
        g_vid = await s.acquire("ltx-worker")
        g_img = await s.acquire("image-worker")
        assert g_vid != g_img
        await s.release("ltx-worker", g_vid)
        await s.release("image-worker", g_img)
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[g_vid]["worker"] == "ltx-worker" and st[g_vid]["state"] == "idle"
        assert st[g_vid]["resident"] is True
        assert st[g_img]["worker"] == "image-worker" and st[g_img]["state"] == "idle"
        assert st[g_img]["resident"] is True

    asyncio.run(_t())


def test_reconcile_does_not_free_reachable_busy_worker():
    """A busy slot whose worker is still reachable must NOT be freed."""
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        gpu = await s.acquire("image-worker")
        await s.reconcile({"image-worker": {"device_in_use": None}})
        kept = [g for g in s.status()["gpus"] if g["gpu_id"] == gpu][0]
        assert kept["state"] == "busy"  # still busy (reachable) despite None
        assert kept["worker"] == "image-worker"

    asyncio.run(_t())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
