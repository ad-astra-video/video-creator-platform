"""ResidentWorkerManager — owns GPU residency across the worker containers.

Only ONE worker's model is resident on a given GPU at a time. ``ensure(name)``
evicts the current resident (POST /evict) before loading the target (POST
/load), and records the new resident. Serialized by an asyncio lock so
concurrent requests can never race two models into one GPU's VRAM.

MULTI-RESIDENT (Plan B, device-aware): ``ensure(name, device)`` loads ``name``
resident on that single GPU index. A worker may be resident on SEVERAL GPUs at
once (``_resident`` maps a worker to the SET of devices it is warm on) so
concurrent requests for the same worker can run in parallel on different cards.
Only the same worker reloads (evict + rebuild) when asked to move off a device.

Pinned workers (dedicated GPU): a worker listed in ``pinned`` is loaded once (via
``load_pinned`` / ``ensure``) and is NEVER evicted — it lives on a separate GPU,
so it is tracked outside the shared ``_current`` slot and ignored by eviction.
The gemma-worker is pinned exactly when LLM_GPU_DEVICE is set (dedicated LLM GPU);
when blank (shared GPU) it is a normal evictable resident, backfilled when idle.

The ``transport`` is a small abstract over the HTTP calls so the TDD
ResidentWorkerManager test can inject an in-memory fake (see
runner/tests/test_live_runner_router.py).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

logger = logging.getLogger("video_creator.runner.live_runner.swap")


class WorkerTransport:
    """HTTP transport to worker containers (control + health calls).

    ``base`` is the worker's root URL (e.g. http://ltx-worker:8991). ``post``
    appends the path for POST /load and /evict; ``health`` GETs /health.
    """

    async def post(self, base: str, path: str, payload: dict | None = None) -> dict:
        raise NotImplementedError

    async def health(self, base: str) -> dict:
        raise NotImplementedError


class HttpWorkerTransport(WorkerTransport):
    """Real aiohttp transport: adds X-Worker-Token to control POSTs."""

    def __init__(self, session, token: str):
        self._session = session
        self._token = token

    def _headers(self, authed: bool) -> dict:
        if authed:
            return {"X-Worker-Token": self._token}
        return {}

    async def post(self, base: str, path: str, payload: dict | None = None) -> dict:
        async with self._session.post(
            base + path, json=payload or {}, headers=self._headers(True)
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"worker call {base}{path} -> {resp.status}: {text[:300]}")
            return await resp.json()

    async def health(self, base: str) -> dict:
        async with self._session.get(base + "/health") as resp:
            if resp.status >= 400:
                raise RuntimeError(f"worker health {base} -> {resp.status}")
            return await resp.json()


@dataclass
class ResidentWorkerManager:
    transport: WorkerTransport
    workers: dict[str, str] = field(default_factory=dict)  # worker name -> base URL
    pinned: frozenset[str] = field(default_factory=frozenset)
    _current: str | None = field(default=None, init=False)   # shared-GPU resident (never a pinned)
    _pinned_loaded: set[str] = field(default_factory=set, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    # Plan B — per-worker device residency: worker name -> SET of device indices
    # it is resident on (multi-resident). A worker loaded via ensure(name,
    # device=<idx>) stays resident on that GPU and is NEVER evicted by another
    # worker loading on a different GPU (each task owns one GPU; cross-GPU
    # eviction would tear down a concurrently-running task's model). Only the
    # SAME worker reloads (evict + rebuild) when asked to move off a device.
    # Eviction is per-device (POST /evict {device}) so freeing card B of a
    # multi-resident worker never drops the copy still warm on card A.
    _resident: dict[str, set[int]] = field(default_factory=dict, init=False)

    def _base(self, name: str) -> str:
        """Resolve a worker name to its base URL from the injected mapping."""
        try:
            return self.workers[name]
        except KeyError:
            raise KeyError(f"unknown worker: {name} (known: {list(self.workers)})")

    @property
    def resident(self) -> str | None:
        """Name of the shared-GPU worker whose model is currently resident."""
        return self._current

    @property
    def pinned_resident(self) -> list[str]:
        """Names of pinned (dedicated-GPU) workers currently loaded."""
        return sorted(self._pinned_loaded)

    def devices(self, name: str) -> set[int]:
        """Resident GPU indices for ``name`` (multi-resident: may be several)."""
        return set(self._resident.get(name, ()))

    async def load_pinned(self) -> None:
        """Load every pinned worker (dedicated GPUs) — run at startup, never evicted."""
        async with self._lock:
            for name in self.pinned:
                if name not in self._pinned_loaded:
                    await self.transport.post(self._base(name), "/load")
                    self._pinned_loaded.add(name)
                    logger.info("GPU pin: %s resident (dedicated)", name)

    async def ensure(self, name: str, device: int | None = None) -> None:
        """Make ``name`` resident: evict the current shared resident (if any), load ``name``.

        Pinned workers load (once) and are never evicted; shared-GPU workers (a
        pinned worker is never ``_current``) evict each other.

        Plan B (``device`` supplied): ``name`` is loaded resident on that single
        GPU index and stays there — it is tracked in ``_resident`` (a SET, so a
        worker may be resident on several cards at once) and is never evicted by
        another worker (which runs on a different GPU). If the same worker is
        asked to move to a different device, it evicts + reloads. This lets
        different tasks run concurrently on different GPUs.
        """
        async with self._lock:
            if device is not None:
                resident = self._resident.setdefault(name, set())
                if device in resident:
                    return  # already resident on that GPU
                # Load FIRST, then record residency ONLY after /load succeeds. The
                # prior code set residency BEFORE the POST, so a failed first load
                # (e.g. the worker not yet resolvable at startup) left the entry set
                # and every later ensure() short-circuited above — the worker's
                # model was NEVER (re)loaded ("warm but unloaded"). Assigning after
                # success lets the idle backfill keep retrying until the load lands.
                await self.transport.post(self._base(name), "/load",
                                          {"device": device})
                resident.add(device)
                if name in self.pinned:
                    self._pinned_loaded.add(name)
                    logger.info("GPU pin: %s resident (dedicated)", name)
                else:
                    if self._current == name:
                        self._current = None
                    logger.info("GPU pin: %s resident on GPU %d", name, device)
                return
            # Legacy shared-GPU path (no device): evict-before-load.
            if name in self.pinned:
                if name not in self._pinned_loaded:
                    await self.transport.post(self._base(name), "/load")
                    self._pinned_loaded.add(name)
                    logger.info("GPU pin: %s resident (dedicated)", name)
                return
            if self._current == name:
                return
            if self._current:
                await self.transport.post(self._base(self._current), "/evict")
            await self.transport.post(self._base(name), "/load")
            self._current = name
            logger.info("GPU swap: %s resident", name)

    async def evict(self, name: str, device: int | None = None) -> None:
        """Evict worker ``name``'s residency.

        ``device`` given (Plan B): frees only that ONE card of a multi-resident
        worker (POST /evict {device}) so the copies still warm on other cards
        survive. ``device`` None (legacy/manual): drops ALL residency for
        ``name`` (POST /evict {}) and clears the shared ``_current``.
        """
        async with self._lock:
            if device is not None:
                resident = self._resident.get(name)
                if resident is not None:
                    resident.discard(device)
                    if not resident:
                        del self._resident[name]
                    if self._current == name:
                        self._current = None
                self._pinned_loaded.discard(name)
                await self.transport.post(self._base(name), "/evict", {"device": device})
                logger.info("GPU swap: evicted %s on GPU %d", name, device)
            else:
                self._resident.pop(name, None)
                if self._current == name:
                    self._current = None
                await self.transport.post(self._base(name), "/evict")
                logger.info("GPU swap: evicted %s", name)

    async def backfill(self, name: str) -> None:
        """Load ``name`` only if it won't displace anything (idle-safe, no eviction)."""
        async with self._lock:
            if name in self.pinned:
                if name not in self._pinned_loaded:
                    await self.transport.post(self._base(name), "/load")
                    self._pinned_loaded.add(name)
                return
            if self._current is None and name != self._current:
                await self.transport.post(self._base(name), "/load")
                self._current = name
                logger.info("GPU backfill: %s resident", name)

    async def evict_all(self) -> None:
        """Evict the current shared-GPU resident (pinned workers stay)."""
        async with self._lock:
            if self._current:
                await self.transport.post(self._base(self._current), "/evict")
                logger.info("GPU swap: evicted %s", self._current)
                self._current = None

    async def _reconcile_loaded(self, health: dict[str, dict]) -> None:
        """Drop residency for any worker we believe is loaded but whose /health
        reports ``model_loaded`` is False (e.g. its container was restarted and
        the fresh process has no model). The next ``ensure()``/idle backfill
        will re-issue /load. Safe because every worker's load() is idempotent.

        Only workers whose /health exposes a ``model_loaded`` key are affected
        (today: gemma-worker) — others are ignored. Missing/error health is
        treated as "unknown" and never drops residency (avoid churn).
        """
        async with self._lock:
            for name, h in health.items():
                if h.get("model_loaded") is not False:
                    continue  # loaded, or worker doesn't report it -> leave alone
                dropped = False
                if name in self._pinned_loaded:
                    self._pinned_loaded.discard(name)
                    dropped = True
                if name in self._resident:
                    del self._resident[name]
                    dropped = True
                if self._current == name:
                    self._current = None
                    dropped = True
                if dropped:
                    logger.warning(
                        "worker %s reports model NOT loaded; dropped residency "
                        "(auto-reload on next ensure/backfill)", name)

    async def check_health(self) -> dict:
        """Live /health probe of every worker for heartbeat metadata.

        Returns a SHORTHAND status map (kept tiny for go-livepeer's 1024-byte
        heartbeat metadata cap): {<short>_up, warm, gmm, pin} where each worker
        name is reduced to its base token (ltx-worker -> ltx). None of these
        keys are read by the Worker control plane or the webapp -- they are
        informational -- so terse keys are safe. ``capabilities``/``model_specs``
        (the actual contract) are added in server.py, not here.
        """
        up = {name: False for name in self.workers}
        health: dict[str, dict] = {}
        for name in up:
            try:
                health[name] = await self.transport.health(self._base(name))
                up[name] = True
            except Exception:
                pass
        # A worker we believe is loaded that /health now reports as NOT loaded
        # (container restarted) has its residency dropped here so the next
        # ensure()/backfill re-loads it instead of routing to an unloaded worker.
        await self._reconcile_loaded(health)
        gemma = "gemma-worker"
        gemma_loaded = (gemma in self._pinned_loaded) or (self._current == gemma)
        return {
            **{f"{name.split('-', 1)[0]}_up": v for name, v in up.items()},
            "warm": self._current,
            "gmm": gemma_loaded,
            "pin": self.pinned_resident,
        }
