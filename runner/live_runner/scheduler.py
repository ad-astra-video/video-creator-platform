"""GPU scheduler — warm-resident, all-GPUs, LRU-eviction (co-residency-safe).

Each worker container sees ALL GPUs (Docker `--gpus all`, CUDA_VISIBLE_DEVICES
unset). The live-runner schedules each task onto a SINGLE GPU, keeps that
worker's model WARM there (resident) across tasks, and — crucially — **evicts a
warm model before a different worker's task takes the same card**, so two models
can never co-reside in one GPU's VRAM (the image-then-video OOM).

Placement policy (per the operator's directive):
  1. One GPU serves ONE request at a time.
  2. A worker may be resident on SEVERAL GPUs at once (multi-resident), and the
     scheduler uses ALL GPUs as needed for the current request burst instead of
     parking cards idle — so concurrent requests for the SAME worker spread
     across its warm cards / free cards and run in parallel.
  3. If a worker has an IDLE warm copy somewhere -> reuse that card (no reload);
     a concurrent peer finds it busy and loads a new copy on a free GPU.
  4. Else if a GPU has nothing on it -> allocate there and leave it warm.
  5. Else (every GPU already holds a warm model) -> evict the least-recently-used
     warm model that is NOT needed for the incoming task, and hand its GPU over.

Eviction is VRAM-safe: ``evict_cb(worker, device)`` (POST /evict {device}) frees
the OLD model's VRAM on that ONE GPU BEFORE the new one loads, so two models
never transiently co-reside. Passing the device matters for multi-resident
workers: freeing card B must not tear down the copy still warm on card A.

Device awareness:
  * ALL workers (image, gemma, ltx, idv2v) are device-aware — they honor
    ``device`` in ``/load`` (free + relocate onto the given GPU) and can live
    on ANY GPU. No worker is pinned. Image AND video spread across all cards,
    and concurrent same-worker requests can occupy multiple cards at once.

The map is ADVISORY: reconciled from each worker's /info at heartbeat, so a
crashed/restarted worker self-heals between heartbeats.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from . import config

logger = logging.getLogger("video_creator.runner.live_runner.scheduler")


@dataclass
class GPUSlot:
    """Per-GPU scheduling state."""

    gpu_id: int
    worker: str | None = None      # worker warm/busy on this GPU
    state: str = "idle"            # "idle" | "busy"
    resident: bool = False         # True = a warm/pinned model owns this GPU's VRAM
    last_used: float = 0.0         # monotonic ts of last use (LRU eviction ordering)


@dataclass
class GPUScheduler:
    """Warm-resident, all-GPUs, LRU-eviction GPU map (serialized via a lock)."""

    gemma_resident_gpu: int | None = None
    gpu_count: int = 3
    queue_timeout_s: float = 600.0
    # Async hook that EVICTS a worker's model from a SPECIFIC GPU before another
    # task takes that card. ``(worker, device)`` so a multi-resident worker only
    # frees the one card being handed over, not all its copies. Generalized to
    # any worker; wired in server.py to POST /evict {device}. None = never
    # auto-evict (evictable victim search is skipped).
    evict_cb: Callable[[str, int], Awaitable[None]] | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _cond: asyncio.Condition = field(init=False)
    _slots: list[GPUSlot] = field(init=False)
    _gemma_evicted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._cond = asyncio.Condition(self._lock)
        self._slots = [GPUSlot(gpu_id=i) for i in range(self.gpu_count)]
        if self.gemma_resident_gpu is not None and 0 <= self.gemma_resident_gpu < self.gpu_count:
            s = self._slots[self.gemma_resident_gpu]
            s.worker, s.state, s.resident = "gemma-worker", "busy", True
            s.last_used = time.monotonic()
    @classmethod
    def from_config(cls) -> "GPUScheduler":
        return cls(
            gemma_resident_gpu=config.GEMMA_RESIDENT_GPU,
            gpu_count=config.GPU_COUNT,
            queue_timeout_s=config.SCHEDULER_QUEUE_TIMEOUT_S,
        )

    def _slot(self, gpu_id: int) -> GPUSlot:
        return self._slots[gpu_id]

    def _warm_gpu_of(self, worker: str) -> GPUSlot | None:
        for s in self._slots:
            if s.worker == worker:
                return s
        return None

    def devices(self, worker: str) -> list[int]:
        """Resident GPU indices for ``worker`` (multi-resident: may be several)."""
        return [s.gpu_id for s in self._slots if s.worker == worker]

    def _pick_idle_warm(self, worker: str) -> GPUSlot | None:
        """An IDLE (not busy) resident slot already warm for ``worker``.

        Multi-resident: picks the least-recently-used idle warm card so a lone
        request reuses a warm card (no reload) while concurrent peers - which
        find that card busy - naturally load onto the other/free cards. This is
        what lets a burst spread across ALL GPUs instead of serializing on one.
        """
        best: GPUSlot | None = None
        for s in self._slots:
            if s.worker == worker and s.state == "idle" and s.resident:
                if best is None or s.last_used < best.last_used:
                    best = s
        return best

    # -- eviction ---------------------------------------------------------

    async def _evict_slot(self, slot: GPUSlot, incoming: str) -> None:
        """Evict the warm model on ``slot`` if it isn't the incoming worker."""
        if slot.worker is None or slot.worker == incoming:
            return
        victim = slot.worker
        if slot.resident and self.evict_cb is not None:
            # Drop the lock across the HTTP evict (avoid holding it during a
            # network round-trip); re-acquire after. Pass the SPECIFIC GPU so a
            # multi-resident worker frees only this card, not all its copies.
            self._cond.release()
            try:
                await self.evict_cb(victim, slot.gpu_id)
            finally:
                await self._cond.acquire()
        slot.worker = None
        slot.state = "idle"
        slot.resident = False
        slot.last_used = 0.0
        logger.info("GPU evict: %s dropped from GPU %d", victim, slot.gpu_id)
        self._cond.notify_all()

    def _claim(self, slot: GPUSlot, worker: str) -> None:
        slot.worker = worker
        slot.state = "busy"
        slot.resident = True
        slot.last_used = time.monotonic()

    # -- gemma -----------------------------------------------------------

    def gemma_evicted(self) -> bool:
        return self._gemma_evicted

    def mark_gemma_loaded(self) -> None:
        if self.gemma_resident_gpu is None:
            return
        s = self._slot(self.gemma_resident_gpu)
        s.worker, s.state, s.resident = "gemma-worker", "busy", True
        s.last_used = time.monotonic()
        self._gemma_evicted = False

    def gemma_slot_free(self) -> bool:
        if self.gemma_resident_gpu is None:
            return False
        s = self._slot(self.gemma_resident_gpu)
        return s.worker in (None, "gemma-worker")

    # -- acquire / release -----------------------------------------------

    async def acquire(self, worker: str) -> int:
        """Reserve a single GPU for ``worker`` (idle-warm-reuse -> free -> LRU-evict).

        One GPU serves one request at a time. A worker may be resident on several
        GPUs at once (multi-resident), so concurrent requests for the SAME worker
        spread across its warm cards / free cards instead of serializing on one.
        """
        if worker == "gemma-worker" and self.gemma_resident_gpu is not None:
            if not self._gemma_evicted:
                self._slot(self.gemma_resident_gpu).last_used = time.monotonic()
                return self.gemma_resident_gpu
            async with self._cond:
                deadline = time.monotonic() + self.queue_timeout_s
                while not self.gemma_slot_free():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise QueueTimeout(f"gemma GPU busy within {self.queue_timeout_s:.0f}s")
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        raise QueueTimeout(f"gemma GPU busy within {self.queue_timeout_s:.0f}s")
                s = self._slot(self.gemma_resident_gpu)
                s.worker, s.state, s.resident = "gemma-worker", "busy", True
                s.last_used = time.monotonic()
                self._gemma_evicted = False
                return self.gemma_resident_gpu

        async with self._cond:
            deadline = time.monotonic() + self.queue_timeout_s
            while True:
                # 1) Reuse an IDLE warm copy of this worker (multi-resident):
                #    if a card already holds this worker warm and isn't busy, use
                #    it (no reload). Concurrent peers then find the other warm
                #    cards busy and naturally load onto free cards instead.
                idle_warm = self._pick_idle_warm(worker)
                if idle_warm is not None:
                    self._claim(idle_warm, worker)
                    logger.info("GPU schedule: %s -> GPU %d (warm)", worker, idle_warm.gpu_id)
                    return idle_warm.gpu_id

                # 2) A GPU with nothing on it (use ALL GPUs for the burst).
                idle = self._pick_idle_gpu()
                if idle is not None:
                    self._claim(idle, worker)
                    logger.info("GPU schedule: %s -> GPU %d", worker, idle.gpu_id)
                    return idle.gpu_id

                # 3) No free GPU -> evict the LRU warm model not needed here.
                victim = self._pick_lru_victim(exclude=worker)
                if victim is not None:
                    await self._evict_slot(self._slot(victim), worker)
                    continue

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise QueueTimeout(f"no GPU free for {worker} within {self.queue_timeout_s:.0f}s")
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise QueueTimeout(f"no GPU free for {worker} within {self.queue_timeout_s:.0f}s")

    def _pick_idle_gpu(self) -> GPUSlot | None:
        """A GPU with no warm model resident (nothing on it at all)."""
        idle = [s for s in self._slots if s.worker is None and not s.resident]
        if idle:
            idle.sort(key=lambda s: (s.gpu_id == self.gemma_resident_gpu,
                                     s.gpu_id))
            return idle[0]
        return None

    def _pick_lru_victim(self, exclude: str | None = None) -> int | None:
        """Least-recently-used warm slot that can be safely evicted. Returns
        None unless ``evict_cb`` is wired — without a real eviction hook we must
        NEVER clear a reservation we can't actually free from VRAM (that would
        silently cause co-residency / OOM)."""
        if self.evict_cb is None:
            return None
        best: GPUSlot | None = None
        for s in self._slots:
            if s.worker is None or s.worker == exclude:
                continue
            if s.gpu_id == self.gemma_resident_gpu:
                continue  # never defensively evict gemma's card
            if best is None or s.last_used < best.last_used:
                best = s
        return best.gpu_id if best is not None else None

    async def pick_for_worker(self, worker: str) -> int:
        """Authoritatively assign a GPU to ``worker`` at startup/warmup.

        Same policy as :meth:`acquire`, but called ONCE from the worker's own
        boot (via the live-runner's gpu-pick endpoint) so it lands on a
        genuinely free card instead of guessing from a local nvidia-smi (which
        races the lazy image-worker load). Marks the slot resident for
        ``worker``, so the first request-time ``acquire()`` reuses it (no
        reload). Raises ``QueueTimeout`` if nothing can be freed within the
        deadline.
        """
        async with self._cond:
            deadline = time.monotonic() + self.queue_timeout_s
            while True:
                warm = self._warm_gpu_of(worker)
                if warm is not None:
                    self._claim(warm, worker)
                    logger.info("GPU pick: %s reuses warm GPU %d", worker, warm.gpu_id)
                    return warm.gpu_id
                idle = self._pick_idle_gpu()
                if idle is not None:
                    self._claim(idle, worker)
                    logger.info("GPU pick: %s -> idle GPU %d", worker, idle.gpu_id)
                    return idle.gpu_id
                victim = self._pick_lru_victim(exclude=worker)
                if victim is not None:
                    await self._evict_slot(self._slot(victim), worker)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise QueueTimeout(
                        f"no GPU free for {worker} within {self.queue_timeout_s:.0f}s")
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise QueueTimeout(
                        f"no GPU free for {worker} within {self.queue_timeout_s:.0f}s")

    async def release(self, worker: str, gpu_id: int) -> None:
        """Task done: keep the model WARM on ``gpu_id`` (no co-load next time);
        the slot goes idle-but-resident so another worker can evict it if needed."""
        async with self._cond:
            slot = self._slot(gpu_id)
            if slot.worker != worker:
                return
            slot.state = "idle"
            slot.last_used = time.monotonic()
            logger.info("GPU release: %s idle on GPU %d (kept warm)", worker, gpu_id)
            self._cond.notify_all()

    async def reconcile(self, workers_info: dict[str, dict]) -> None:
        """Heal the map from each worker's /info (advisory). A worker may report
        a LIST of resident devices (``devices``, multi-resident) or a single
        ``device_in_use`` (back-compat). A slot whose worker is GONE (not in
        the reachable set) is freed even if resident — a live warm model can
        only exist while its process lives, so an unreachable worker can't
        still be holding VRAM. Gemma's pinned slot is never freed here."""
        async with self._cond:
            reachable = set(workers_info)
            for name, info in workers_info.items():
                devs = info.get("devices")
                if isinstance(devs, (list, tuple, set)):
                    devs = [int(d) for d in devs
                            if isinstance(d, int) and 0 <= d < self.gpu_count]
                else:
                    dev = info.get("device_in_use")
                    devs = [int(dev)] if (isinstance(dev, int)
                                          and 0 <= dev < self.gpu_count) else []
                for d in devs:
                    s = self._slot(d)
                    s.worker = name
                    s.state = "busy"
                    s.resident = True
                    s.last_used = time.monotonic()
            for s in self._slots:
                if (self.gemma_resident_gpu is not None
                        and s.gpu_id == self.gemma_resident_gpu
                        and s.worker == "gemma-worker"):
                    continue
                if s.worker is not None and s.worker not in reachable:
                    s.worker = None
                    s.state = "idle"
                    s.resident = False
                    s.last_used = 0.0
                    logger.info("GPU reconcile: freed GPU %d (%s gone)",
                                s.gpu_id, s.worker)

    def status(self) -> dict:
        return {
            "gpu_count": self.gpu_count,
            "gemma_resident_gpu": self.gemma_resident_gpu,
            "gemma_evicted": self._gemma_evicted,
            "gpus": [
                {"gpu_id": s.gpu_id, "worker": s.worker, "state": s.state,
                 "resident": s.resident}
                for s in self._slots
            ],
        }


class QueueTimeout(Exception):
    """Raised when a task waits too long for a free GPU (503, retriable)."""
