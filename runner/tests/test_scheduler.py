"""Warm-resident GPU scheduler tests (pure asyncio, no GPU, no aiohttp server).

Covers the all-GPUs / keep-warm / LRU-evict policy:
  - gemma's resident GPU is held out of the task pool
  - acquiring for gemma returns its resident GPU
  - a worker that is already warm reuses the same GPU (no reload)
  - a task with no warm model takes a FREE (unused) GPU — fully utilizes GPUs
  - when all GPUs hold warm models, the LRU warm model is EVICTED for the new task
  - THE regression: after an image gen, the image model must NOT land on the
    video card; with a free GPU available it takes a different card so a later
    video gen never co-resides/OOMs. When only the video card is left free, the
    video worker EVICTS whatever is warm there before loading.
  - reconcile() from worker /info self-heals a dead worker's slot
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


def test_gemma_gpu_reserved_at_boot():
    s = _mk(gpu_count=3, gemma_gpu=1)
    slots = s.status()["gpus"]
    by_id = {g["gpu_id"]: g for g in slots}
    assert by_id[1]["worker"] == "gemma-worker"
    assert by_id[1]["state"] == "busy"
    assert by_id[1]["resident"] is True
    assert by_id[0]["state"] == "idle"
    assert by_id[2]["state"] == "idle"


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


def test_lru_eviction_when_all_gpus_warm():
    """3 free GPUs minus gemma leaves 2. After 2 different workers are warm on
    both, a 3rd worker must EVICT the LRU warm model to get a GPU."""

    class _Harness:
        def __init__(self):
            self.evicted = []

        async def evict(self, name):
            self.evicted.append(name)

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


def test_queue_timeout_raises_when_nothing_evictable():
    """If every GPU is warm and evict_cb is NOT wired, the scheduler cannot free
    a card -> waits and times out (no silent co-residency)."""
    s = _mk(gpu_count=3, gemma_gpu=1, timeout=0.05)  # no evict_cb
    s.video_shared_gpu = None

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


def test_video_worker_evicts_image_from_video_card():
    """THE regression: a video task arriving while a DIFFERENT model is warm on
    the video card EVICTS it first — never co-resides -> no OOM. Here "image"
    is seeded directly on the video card (GPU 0) to model the case where only
    the video card is left free for image."""

    class _Harness:
        def __init__(self):
            self.evicted = []

        async def evict(self, name):
            self.evicted.append(name)

    h = _Harness()
    s = _mk(gpu_count=3, gemma_gpu=2, timeout=10.0)
    s.video_shared_gpu = 0
    s.evict_cb = h.evict

    async def _t():
        # seed a foreign warm model on the video card (image was forced there
        # because no normal card was free)
        s0 = [g for g in s.status()["gpus"] if g["gpu_id"] == 0][0]
        # acquire+release ltx to claim the base then evict-seed image on it is
        # awkward, so directly reconcile image-worker onto GPU 0:
        await s.reconcile({"image-worker": {"device_in_use": 0}})
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[0]["worker"] == "image-worker"
        # now a video task arrives -> ltx pinned to 0; evict image first
        g_ltx = await s.acquire("ltx-worker")
        assert g_ltx == 0
        assert "image-worker" in h.evicted
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        assert st[0]["worker"] == "ltx-worker"

    asyncio.run(_t())


def test_image_avoids_video_card_when_another_free():
    """When a normal GPU is free, image-worker does NOT take the video card —
    so a later restyle/video gen isn't forced to evict (and image stays warm
    on its own card for a fast repeat). GPU 0 reserved for video, GPU 1 for
    image, gemma on 2: all three GPUs utilized, no collision."""
    s = _mk(gpu_count=3, gemma_gpu=2, timeout=10.0)
    s.video_shared_gpu = 0

    async def _t():
        g_img = await s.acquire("ltx-worker")      # video holds GPU 0
        g2 = await s.acquire("image-worker")       # image takes GPU 1 (free)
        assert g2 == 1
        await s.release("ltx-worker", g_img)
        g3 = await s.acquire("ltx-worker")         # video still on 0
        assert g3 == 0
        await s.release("image-worker", g2)
        await s.release("ltx-worker", g3)

    asyncio.run(_t())


def test_video_and_image_are_concurrent_on_distinct_gpus():
    """Image on GPU 1 and video on GPU 0 run at the same time (the fix's goal):
    no co-residency on the video card -> no video-gen OOM after an image gen."""
    s = _mk(gpu_count=3, gemma_gpu=2, timeout=10.0)
    s.video_shared_gpu = 0

    async def _t():
        g_vid = await s.acquire("ltx-worker")
        g_img = await s.acquire("image-worker")
        assert g_vid == 0 and g_img == 1  # distinct cards, both busy
        await s.release("ltx-worker", g_vid)
        await s.release("image-worker", g_img)
        st = {g["gpu_id"]: g for g in s.status()["gpus"]}
        # both kept warm (idle+resident) on their own cards
        assert st[0]["worker"] == "ltx-worker" and st[0]["state"] == "idle"
        assert st[1]["worker"] == "image-worker" and st[1]["state"] == "idle"

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
