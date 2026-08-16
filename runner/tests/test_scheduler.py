"""Plan B — GPU scheduler tests (pure asyncio, no GPU, no aiohttp server).

Covers the one-task-one-GPU scheduler:
  - single-GPU acquire/release
  - two tasks acquire two DISTINCT GPUs (concurrency)
  - gemma's resident GPU is held out of the task pool
  - acquring for gemma returns its resident GPU
  - reconcile() from worker /info self-heals a restarted worker's slot
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


def test_acquire_returns_non_gemma_gpu_and_busy():
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        gpu = await s.acquire("ltx-worker")
        assert gpu != 1  # never the gemma-resident GPU
        st = s.status()["gpus"]
        mine = [g for g in st if g["gpu_id"] == gpu][0]
        assert mine["worker"] == "ltx-worker"
        assert mine["state"] == "busy"

    asyncio.run(_t())


def test_release_frees_the_gpu():
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        gpu = await s.acquire("ltx-worker")
        await s.release("ltx-worker", gpu)
        mine = [g for g in s.status()["gpus"] if g["gpu_id"] == gpu][0]
        assert mine["worker"] is None
        assert mine["state"] == "idle"

    asyncio.run(_t())


def test_two_tasks_get_two_distinct_gpus():
    """Two held acquires land on two different GPUs (concurrency across GPUs)."""
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        g1 = await s.acquire("ltx-worker")
        g2 = await s.acquire("idv2v-worker")
        assert g1 != g2
        assert {g1, g2} <= {0, 2}  # neither is gemma's GPU 1

    asyncio.run(_t())


def test_third_task_queues_then_runs_on_freed_gpu():
    """A 3rd task (after 2 busy GPUs, gemma holds the 3rd) queues FIFO; when one
    frees it proceeds."""

    async def _t():
        s = _mk(gpu_count=3, gemma_gpu=1, timeout=10.0)
        g1 = await s.acquire("ltx-worker")
        g2 = await s.acquire("idv2v-worker")
        # Both free task GPUs are busy -> a third acquire must wait.
        acquired = []
        t3 = asyncio.create_task(s.acquire("image-worker"))

        async def _release_after():
            await asyncio.sleep(0.05)
            await s.release("ltx-worker", g1)

        rel = asyncio.create_task(_release_after())
        g3 = await t3
        await rel
        assert g3 == g1  # took over the freed GPU
        assert g3 not in (g2,)

    asyncio.run(_t())


def test_queue_timeout_raises():
    s = _mk(gpu_count=3, gemma_gpu=1, timeout=0.05)

    async def _t():
        await s.acquire("ltx-worker")
        await s.acquire("idv2v-worker")
        with pytest.raises(QueueTimeout):
            await s.acquire("image-worker")  # no GPU free within 0.05s

    asyncio.run(_t())


def test_gemma_acquire_returns_resident_gpu():
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        gpu = await s.acquire("gemma-worker")
        assert gpu == 1  # gemma always gets its resident GPU

    asyncio.run(_t())


def test_reconcile_frees_slot_of_dead_worker():
    """A worker that dies mid-task leaks its busy slot; reconcile() from /info
    (with that worker absent/unreachable) frees it."""
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        gpu = await s.acquire("ltx-worker")
        assert [g for g in s.status()["gpus"] if g["gpu_id"] == gpu][0]["state"] == "busy"
        # ltx-worker crashed -> not in the reachable info set.
        await s.reconcile({"idv2v-worker": {"device_in_use": None}})
        freed = [g for g in s.status()["gpus"] if g["gpu_id"] == gpu][0]
        assert freed["state"] == "idle"
        assert freed["worker"] is None

    asyncio.run(_t())


def test_reconcile_does_not_free_reachable_busy_worker():
    """A busy slot whose worker is still reachable must NOT be freed (avoids
    double-assigning an in-flight GPU between heartbeats)."""
    s = _mk(gpu_count=3, gemma_gpu=1)

    async def _t():
        gpu = await s.acquire("ltx-worker")
        # ltx-worker still reachable (reports idle/None device) -> keep busy.
        await s.reconcile({"ltx-worker": {"device_in_use": None}})
        kept = [g for g in s.status()["gpus"] if g["gpu_id"] == gpu][0]
        assert kept["state"] == "busy"
        assert kept["worker"] == "ltx-worker"

    asyncio.run(_t())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
