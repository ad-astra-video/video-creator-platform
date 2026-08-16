"""Plan B — GPU concurrency scheduler (one task -> one GPU, concurrent across GPUs).

Each worker container sees ALL GPUs (Docker `--gpus all`, CUDA_VISIBLE_DEVICES
unset). The live-runner schedules each task onto a SINGLE free GPU and delivers
the chosen device index in the ``/load`` request body; the worker pins its
pipeline (and ``enable_model_cpu_offload()``) to that one device before first
CUDA init. Different tasks therefore run concurrently on different GPUs, and a
single task never spans more than one GPU (single-task multi-GPU / model-parallel
is out of scope).

The gemma-worker normally loads once on ``GEMMA_RESIDENT_GPU`` and STAYS there;
the scheduler marks that GPU as gemma's and, as long as some OTHER GPU is free,
never hands it to a render task. It is CO-OPTABLE (optional, via ``coopt_cb``):
when a render task needs a GPU and none is free, the scheduler evicts gemma
(calls ``coopt_cb`` → POST /evict on gemma-worker), hands gemma's GPU to the
render task, and after that task releases the GPU a later idle backfill reloads
gemma. Gemma therefore surrenders its GPU to real work only under load, then
comes back — it is never a permanently-reserved idle slot.

The map is ADVISORY: it is reconciled from each worker's ``/info``
(``devices_visible`` / ``device_in_use``) at heartbeat, so a crashed/restarted
worker self-heals the map between heartbeats.
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
    worker: str | None = None      # worker currently using this GPU
    state: str = "idle"            # "idle" | "busy"
    resident: bool = False         # True = a pinned/resident worker owns it permanently


@dataclass
class GPUScheduler:
    """Holds the advisory GPU map; serializes acquire/release with an asyncio lock."""

    gemma_resident_gpu: int | None = None
    gpu_count: int = 3
    queue_timeout_s: float = 600.0
    # Optional async hook that evicts gemma (POST /evict on gemma-worker) so its
    # GPU can be handed to a render task under load. Wired in server.py; None =
    # gemma's GPU is never co-opted (always reserved for gemma).
    coopt_cb: Callable[[], Awaitable[None]] | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _cond: asyncio.Condition = field(init=False)
    _slots: list[GPUSlot] = field(init=False)
    _gemma_evicted: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._cond = asyncio.Condition(self._lock)
        self._slots = [
            GPUSlot(gpu_id=i)
            for i in range(self.gpu_count)
        ]
        # gemma's resident GPU is owned by gemma by default (busy+resident); it
        # can be co-opted (released to a render task) when coopt_cb is set.
        if self.gemma_resident_gpu is not None and 0 <= self.gemma_resident_gpu < self.gpu_count:
            self._slots[self.gemma_resident_gpu].worker = "gemma-worker"
            self._slots[self.gemma_resident_gpu].state = "busy"
            self._slots[self.gemma_resident_gpu].resident = True

    @classmethod
    def from_config(cls) -> "GPUScheduler":
        """Build the scheduler from live-runner config env (GPU_COUNT, ...)."""
        return cls(
            gemma_resident_gpu=config.GEMMA_RESIDENT_GPU,
            gpu_count=config.GPU_COUNT,
            queue_timeout_s=config.SCHEDULER_QUEUE_TIMEOUT_S,
        )

    def _slot(self, gpu_id: int) -> GPUSlot:
        return self._slots[gpu_id]

    def _free_gpu(self, worker: str, exclude_gpu: int | None = None) -> GPUSlot | None:
        """First genuinely idle GPU not reserved by a resident worker.

        A resident (gemma) slot is skipped so a render task on a busy box can
        never silently steal gemma's GPU while other work is running — co-opt is
        handled explicitly by ``acquire`` via ``coopt_cb``.
        """
        for s in self._slots:
            if exclude_gpu is not None and s.gpu_id == exclude_gpu:
                continue
            if s.state == "idle" and s.worker is None and not s.resident:
                return s
        return None

    def gemma_evicted(self) -> bool:
        """True once gemma's GPU has been co-opted (evicted) for a render task."""
        return self._gemma_evicted

    def mark_gemma_loaded(self) -> None:
        """Re-assert gemma residency after it (re)loads.

        Called by the server after ``ensure("gemma-worker")`` succeeds so the
        advisory map + ``_gemma_evicted`` flag reflect that gemma is resident
        again and its GPU is reserved for it.
        """
        if self.gemma_resident_gpu is None:
            return
        s = self._slot(self.gemma_resident_gpu)
        s.worker = "gemma-worker"
        s.state = "busy"
        s.resident = True
        self._gemma_evicted = False

    def gemma_slot_free(self) -> bool:
        """True when gemma's GPU is NOT occupied by a render task.

        The idle backfill loop reloads gemma only when this is true, so it can
        never reload gemma onto a GPU a render task is currently using. It is
        ALSO true when gemma is resident-but-not-yet-loaded (initial startup,
        or after the initial resident load failed and gemma is the slot owner),
        so the backfill retries loading gemma instead of deadlocking the model
        off just because the slot is marked resident.
        """
        if self.gemma_resident_gpu is None:
            return False
        s = self._slot(self.gemma_resident_gpu)
        # Free when owned by gemma (or nobody) — i.e. no render task holds it.
        return s.worker in (None, "gemma-worker")

    async def acquire(self, worker: str) -> int:
        """Reserve a single idle GPU for ``worker`` (FIFO wait up to timeout).

        Returns the device index (0-based) to send in the ``/load`` body. Raises
        ``QueueTimeout`` (503 retriable) if no GPU frees up within
        ``queue_timeout_s``.

        gemma-worker ALWAYS gets its resident GPU when gemma is loaded there.
        When a render task waits for a GPU and none is free, this scheduler will
        co-opt gemma's resident GPU (evict gemma via ``coopt_cb``, hand the GPU
        to the render task) so gemma is not a permanently-reserved idle slot on
        an otherwise fully-busy box. If gemma itself is called while its GPU is
        co-opted (in use by a render task), it waits for that GPU to free — it
        never borrows another card because it is hardware-pinned to its own.
        """
        if worker == "gemma-worker" and self.gemma_resident_gpu is not None:
            if not self._gemma_evicted:
                return self.gemma_resident_gpu
            # gemma was evicted: wait for its own GPU to free, then hand it back.
            async with self._cond:
                deadline = time.monotonic() + self.queue_timeout_s
                while not self.gemma_slot_free():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise QueueTimeout(
                            f"gemma GPU busy within {self.queue_timeout_s:.0f}s"
                        )
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        raise QueueTimeout(
                            f"gemma GPU busy within {self.queue_timeout_s:.0f}s"
                        )
                self._slot(self.gemma_resident_gpu).worker = "gemma-worker"
                self._slot(self.gemma_resident_gpu).state = "busy"
                self._slot(self.gemma_resident_gpu).resident = True
                self._gemma_evicted = False
                return self.gemma_resident_gpu

        async with self._cond:
            deadline = time.monotonic() + self.queue_timeout_s
            while True:
                slot = self._free_gpu(worker)
                if slot is not None:
                    slot.worker = worker
                    slot.state = "busy"
                    slot.resident = False
                    logger.info("GPU schedule: %s -> GPU %d", worker, slot.gpu_id)
                    return slot.gpu_id

                # No genuinely free GPU -> co-opt gemma's resident GPU if gemma
                # is currently resident and not already evicted.
                if self._try_coopt_gemma():
                    gpu = self.gemma_resident_gpu
                    assert gpu is not None
                    # Hand gemma's GPU to this render task so the slot shows a
                    # busy owner and the backfill never reloads gemma onto it.
                    self._slot(gpu).worker = worker
                    self._slot(gpu).state = "busy"
                    if self.coopt_cb is not None:
                        # Evict gemma BEFORE we hand its GPU to the render task.
                        await self.coopt_cb()
                    self._cond.notify_all()
                    logger.info("GPU co-opt: gemma on GPU %d handed to %s", gpu, worker)
                    return gpu

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise QueueTimeout(
                        f"no GPU free for {worker} within {self.queue_timeout_s:.0f}s"
                    )
                try:
                    await asyncio.wait_for(self._cond.wait(), timeout=remaining)
                except asyncio.TimeoutError:
                    raise QueueTimeout(
                        f"no GPU free for {worker} within {self.queue_timeout_s:.0f}s"
                    )

    def _try_coopt_gemma(self) -> bool:
        """Synchronously hand gemma's resident GPU to a waiting render task.

        Co-opt only fires when ``coopt_cb`` is wired (server.py), so a box that
        does not opt into co-opt keeps the old behaviour: gemma's GPU is never
        handed out, it stays resident forever.
        """
        if self.coopt_cb is None:
            return False
        if self.gemma_resident_gpu is None or self._gemma_evicted:
            return False
        gs = self._slot(self.gemma_resident_gpu)
        if not (gs.resident and gs.worker == "gemma-worker" and gs.state == "busy"):
            return False
        # Take gemma's GPU for the render task.
        self._gemma_evicted = True
        gs.worker = None
        gs.state = "idle"
        gs.resident = False
        logger.info("GPU co-opt: evicted gemma from GPU %d for render task", gs.gpu_id)
        return True

    async def release(self, worker: str, gpu_id: int) -> None:
        """Free ``gpu_id`` after ``worker``'s task finishes."""
        async with self._cond:
            slot = self._slot(gpu_id)
            if slot.worker == worker or slot.worker is None:
                if slot.resident:
                    # gemma's resident GPU is not released into the pool when it
                    # is genuinely resident (not co-opted).
                    return
                slot.worker = None
                slot.state = "idle"
                logger.info("GPU release: %s freed GPU %d", worker, gpu_id)
                self._cond.notify_all()

    async def reconcile(self, workers_info: dict[str, dict]) -> None:
        """Heal the map from each worker's /info (advisory, never destructive).

        ``workers_info`` maps worker name -> its /info dict (only REACHABLE
        workers are present). Uses ``device_in_use`` (the device index a worker
        reports it is pinned to) to re-assert ownership, and frees a busy slot
        only when its owner worker is no longer reachable (crashed/restarted) —
        it never frees a slot whose worker is still up, so an in-flight task's
        GPU can't be double-assigned between heartbeats.
        """
        async with self._cond:
            reachable = set(workers_info)
            for name, info in workers_info.items():
                dev = info.get("device_in_use")
                if isinstance(dev, int) and 0 <= dev < self.gpu_count:
                    s = self._slot(dev)
                    if s.resident:
                        continue  # gemma's resident GPU is only ever owned by gemma
                    s.worker = name
                    s.state = "busy"
            for s in self._slots:
                if s.resident:
                    continue
                if s.state == "busy" and s.worker not in reachable:
                    s.worker = None
                    s.state = "idle"
                    logger.info("GPU reconcile: freed GPU %d (%s gone)", s.gpu_id, s.worker)

    def status(self) -> dict:
        """Public (internal, worker-token protected) snapshot of the map."""
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
