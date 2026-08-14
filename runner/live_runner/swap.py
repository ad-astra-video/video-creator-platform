"""ResidentWorkerManager — owns GPU residency across the worker containers.

Only ONE worker has its model resident on the shared 32 GB GPU at a time.
``ensure(name)`` evicts the current resident (POST /evict) before loading the
target (POST /load), and records the new resident. Serialized by an asyncio lock
so concurrent requests can never race two models into VRAM simultaneously.

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

    async def load_pinned(self) -> None:
        """Load every pinned worker (dedicated GPUs) — run at startup, never evicted."""
        async with self._lock:
            for name in self.pinned:
                if name not in self._pinned_loaded:
                    await self.transport.post(self._base(name), "/load")
                    self._pinned_loaded.add(name)
                    logger.info("GPU pin: %s resident (dedicated)", name)

    async def ensure(self, name: str) -> None:
        """Make ``name`` resident: evict the current shared resident (if any), load ``name``.

        Pinned workers load (once) and are never evicted; shared-GPU workers (a
        pinned worker is never ``_current``) evict each other.
        """
        async with self._lock:
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
        for name in up:
            try:
                await self.transport.health(self._base(name))
                up[name] = True
            except Exception:
                pass
        gemma = "gemma-worker"
        gemma_loaded = (gemma in self._pinned_loaded) or (self._current == gemma)
        return {
            **{f"{name.split('-', 1)[0]}_up": v for name, v in up.items()},
            "warm": self._current,
            "gmm": gemma_loaded,
            "pin": self.pinned_resident,
        }
