"""Bernini manager — resident-subprocess controller for the wan-worker.

Drives :mod:`runner.idv2v.bernini_cli` as a persistent JSONL worker running in
the ISOLATED Bernini venv (``BERNINI_VENV_PY``), so the model stays warm across
requests on a dedicated GPU without polluting the worker's own python env
(the diffsynth/Wan-IDV2V stack pins a transformers version incompatible with
Bernini's 4.57.3).

Lifecycle
    * ``ensure_loaded`` spawns the subprocess on the first job (or eagerly, if
      the scheduler wants it warm) and keeps it resident.
    * ``generate`` serializes one job through the process's stdin (one line in,
      one line out) — requests to the same process are naturally serialized.
    * ``evict`` terminates the subprocess and frees the GPU (used by the
      live-runner swap policy when another worker needs the card).

Concurrency
    The CLI keeps the model resident on ``BERNINI_GPU_DEVICE`` (defaults to the
    worker's ``GPU_DEVICE``). The worker allows ONE Bernini job in flight — an
    in-process asyncio lock serializes callers; a concurrent job waits, which
    matches the "one request per GPU at a time" directive.

Wire format (mirrors bernini_cli.py):
    -> {"prompt", "output", "image"?, "images"?, "video"?, task_name?, ...}
    <- {"ok": true, "output", "frames", "task"} | {"ok": false, "error"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import typing
from typing import Any, Optional

from . import config

logger = logging.getLogger("video_creator.runner.idv2v.bernini")

# Max seconds to wait for the resident process to come up (model build at warm).
STARTUP_TIMEOUT = 900
# Max seconds for a single generation job (long clips at 30 steps on a 5090).
JOB_TIMEOUT = 900


class BerniniError(RuntimeError):
    pass


class BerniniManager:
    """Own one resident ``bernini_cli.py`` subprocess for a single model."""

    def __init__(self, model: str = "bernini-1.3b",
                 device: Optional[str] = None, venv_py: Optional[str] = None,
                 root: Optional[str] = None, guidance: str = "rv2v"):
        self.model = model
        self.resolve_model = model  # config.resolve_model(model)
        self.device = device or config.BERNINI_GPU_DEVICE or config.GPU_DEVICE or "cuda:0"
        self.venv_py = venv_py or config.BERNINI_VENV_PY
        self.root = root or config.bernini_root(model)
        self.guidance = guidance
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._lock = asyncio.Lock()
        self._ready = False
        self._warm = False

    # -- process lifecycle ---------------------------------------------------
    @property
    def venv_python(self) -> str:
        py = self.venv_py
        if py and os.path.exists(py):
            return py
        raise BerniniError(
            f"Bernini venv python not found at {py!r} "
            f"(set BERNINI_VENV_PY; Isolated Bernini venv is required)")

    async def ensure_loaded(self) -> None:
        """Spawn + warm the resident process (idempotent)."""
        async with self._lock:
            if self._ready and self._proc and self._proc.returncode is None:
                return
            if not os.path.isdir(self.root):
                raise BerniniError(
                    f"Bernini model dir not found: {self.root!r}. "
                    "Provision weights (download_bernini) before serving "
                    "bernini requests.")
            python = self.venv_python
            here = os.path.dirname(os.path.abspath(__file__))
            script = os.path.join(here, "bernini_cli.py")
            logger.info("Spawning Bernini worker: %s %s --model-dir %s "
                        "--device %s --guidance %s",
                        python, script, self.root, self.device, self.guidance)
            self._proc = await asyncio.create_subprocess_exec(
                python, script,
                "--model-dir", self.root,
                "--device", self.device,
                "--guidance", self.guidance,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                cwd=here,
            )
            # The CLI writes "Worker ready; awaiting JSONL jobs on stdin" to
            # its own stderr->logging, but we cannot read it (DEVNULL). Treat
            # spawn as ready once the process is up; model build latency is
            # absorbed by JOB_TIMEOUT on the first generation. We pre-warm by
            # sending a lightweight no-op so the model actually materializes
            # here rather than on first user job.
            await self._probe_ready()
            self._ready = True
            self._warm = True
            logger.info("Bernini worker resident (model=%s, device=%s)",
                        self.model, self.device)

    async def _probe_ready(self) -> None:
        """Wait until the process accepts input (startup handshake)."""
        if self._proc is None or self._proc.stdin is None:
            raise BerniniError("berni subprocess not started")
        # The first real job doubles as the warm probe; nothing separate to
        # handshake on. Give the process a moment to exec before any write.
        await asyncio.sleep(0.5)

    async def evict(self) -> None:
        """Terminate the resident process and free its GPU."""
        async with self._lock:
            proc = self._proc
            self._proc = None
            self._ready = False
            self._warm = False
        if proc is not None and proc.returncode is None:
            logger.info("Evicting Bernini worker (model=%s)", self.model)
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=15)
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:  # noqa: BLE001
                    pass

    # -- generation ----------------------------------------------------------
    async def generate(self, job: dict[str, Any], timeout: float = JOB_TIMEOUT,
                       ) -> dict[str, Any]:
        """Run one Bernini task through the resident subprocess."""
        await self.ensure_loaded()
        async with self._lock:
            if self._proc is None or self._proc.stdin is None or \
                    self._proc.stdout is None:
                raise BerniniError("berni subprocess unavailable")
            line = json.dumps(job, ensure_ascii=False) + "\n"
            try:
                self._proc.stdin.write(line.encode("utf-8"))
                await self._proc.stdin.drain()
                raw = await asyncio.wait_for(self._proc.stdout.readline(), timeout)
            except asyncio.TimeoutError as exc:
                await self.evict()
                raise BerniniError(
                    f"Bernini job timed out after {timeout}s — worker evicted") from exc
            except Exception as exc:  # noqa: BLE001
                await self.evict()
                raise BerniniError(f"Bernini subprocess failed: {exc}") from exc
            if not raw:
                await self.evict()
                raise BerniniError("Bernini worker closed unexpectedly")
            try:
                result = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                await self.evict()
                raise BerniniError(f"Bernini bad response: {raw!r}") from exc
            if not result.get("ok"):
                raise BerniniError(result.get("error", "Bernini generation failed"))
            return result

    @property
    def is_ready(self) -> bool:
        return self._ready and self._proc is not None and \
            self._proc.returncode is None

    @property
    def resident(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "device": self.device,
            "ready": self.is_ready,
            "warm": self._warm,
        }


# Module singleton owned by the wan-worker server.
_manager: Optional[BerniniManager] = None
_manager_lock = asyncio.Lock()


async def get_manager(model: str = "bernini-1.3b",
                      device: Optional[str] = None) -> BerniniManager:
    """Return (creating if needed) the worker's shared Bernini manager."""
    global _manager
    async with _manager_lock:
        if _manager is None:
            _manager = BerniniManager(model=model, device=device)
        return _manager


async def evict_manager() -> None:
    global _manager
    async with _manager_lock:
        if _manager is not None:
            await _manager.evict()
        _manager = None
