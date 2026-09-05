"""Generic engine-subprocess harness.

Runs a heavy GPU model in a SEPARATE python process so its CUDA primary context
can be torn down on /evict. PyTorch has no in-process API to destroy a CUDA
context — ``empty_cache()`` only returns the caching-allocator pool to the
driver, while the process stays attached to the GPU (its nvidia-smi entry and a
~0.5-0.7 GB driver-reserved floor) for the whole process lifetime. Killing the
model subprocess destroys its context entirely, so the parent (aiohttp server)
can stay alive for lazy reload while the GPU is fully released.

Architecture
    * The PARENT is the worker's aiohttp server. It never touches CUDA: it holds
      an :class:`EngineProc` handle and proxies engine calls over JSONL.
    * The CHILD is ``run_child_loop`` in this module (a per-worker ``*_cli.py``
      calls it). It builds the real engine on the assigned GPU, then serves a
      JSONL command channel: one request line in, one result line out, with
      interim ``{"type":"progress", ...}`` lines forwarded to the parent's
      progress callback.

Wire format (both directions, one JSON object per line over the pipe)
    -> {"op": "<engine-method>", "args": {...}}          (request)
    <- {"type": "progress", ...}                         (optional, N times)
    <- {"type": "ready", "ok": true}                     (startup handshake)
    <- {"ok": true, "result": ...} | {"ok": false, "error": "..."}   (final)

The parent/<worker> proxy owns serialization of non-JSON types (PIL -> base64,
bytes -> base64, file path passthrough). The transport here is JSON-safe-only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import traceback
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("video_creator.runner.common.engineproc")


class EngineProcError(RuntimeError):
    pass


class EngineProc:
    """Own one resident model subprocess, proxy engine calls to it, and tear its
    CUDA context down on :meth:`stop` (process exit destroys the context)."""

    def __init__(self, label: str, argv: list[str],
                 startup_timeout: float = 900,
                 job_timeout: float = 900,
                 env_extra: Optional[dict] = None):
        self._label = label
        self._argv = list(argv)
        self._startup_timeout = startup_timeout
        self._job_timeout = job_timeout
        self._env_extra = env_extra or {}
        # CHILD I/O uses blocking subprocess.Popen in worker threads (piped to
        # asyncio queues) rather than asyncio's subprocess StreamReader. The
        # asyncio transport was reading a false EOF on the child's stdout in the
        # long-running aiohttp server even while the child was alive and its fd 1
        # was open — a reliable, verified fix here is Popen + a blocking-thread
        # pump (plain file readline has no asyncio 64 KB line limit and no
        # transport conn_lost race). Works uniformly for image/ltx/wan children.
        self._proc: Optional[subprocess.Popen] = None
        self._pump_threads: list[threading.Thread] = []
        self._stdout_q: Optional[asyncio.Queue] = None  # bytes lines + None EOF
        self._stderr_q: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ready = False
        self._lock = asyncio.Lock()

    # ── blocking-pipe helpers (run in worker threads) ─────────────────────
    @staticmethod
    def _pump_reader(stream, loop: asyncio.AbstractEventLoop,
                     q: asyncio.Queue, label: str) -> None:
        """Blocking ``readline`` from a Popen pipe -> push bytes/EOF sentinel
        onto an asyncio queue from a thread."""
        while True:
            try:
                raw = stream.readline()
            except Exception:  # noqa: BLE001 - stream gone
                raw = b""
            if not raw:
                try:
                    loop.call_soon_threadsafe(q.put_nowait, None)
                except Exception:  # noqa: BLE001 - loop closed
                    pass
                break
            try:
                loop.call_soon_threadsafe(q.put_nowait, raw)
            except Exception:  # noqa: BLE001 - loop closed
                break

    @staticmethod
    def _write_job(stream, data: bytes) -> None:
        stream.write(data)
        stream.flush()

    @staticmethod
    def _terminate_and_wait(proc: subprocess.Popen, timeout: float) -> int:
        if proc.poll() is not None:
            return proc.returncode
        proc.terminate()
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()

    async def _next_stdout_line(self) -> Optional[bytes]:
        """Await the next stdout line (bytes) or None on EOF/stopped."""
        q = self._stdout_q
        if q is None:
            return None  # proc stopped/not running -> treat as EOF
        return await q.get()

    # ── lifecycle ──────────────────────────────────────────────────────────
    async def start(self) -> None:
        """Spawn the child and wait for its ready handshake (idempotent)."""
        async with self._lock:
            await self._ensure_started_unlocked()

    async def _ensure_started_unlocked(self) -> None:
        """Spawn-ready logic; caller must hold ``self._lock``."""
        if self._ready and self._proc and self._proc.poll() is None:
            return
        logger.info("[%s] spawning model subprocess: %s",
                    self._label, " ".join(self._argv))
        loop = asyncio.get_running_loop()
        self._loop = loop
        self._stdout_q = asyncio.Queue()
        self._stderr_q = asyncio.Queue()
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env={**os.environ, "PYTHONUNBUFFERED": "1", **self._env_extra},
        )
        for _stream, _q in (
            (self._proc.stdout, self._stdout_q),
            (self._proc.stderr, self._stderr_q),
        ):
            _t = threading.Thread(
                target=self._pump_reader,
                args=(_stream, loop, _q, self._label), daemon=True)
            _t.start()
            self._pump_threads.append(_t)
        asyncio.create_task(self._drain_stderr())
        try:
            await asyncio.wait_for(self._probe_ready(), self._startup_timeout)
        except asyncio.TimeoutError:
            await self._stop_unlocked(5)
            raise EngineProcError(
                f"[{self._label}] model subprocess did not become ready "
                f"within {self._startup_timeout}s")
        self._ready = True
        logger.info("[%s] model subprocess ready (pid=%s)",
                    self._label, self._proc.pid)

    async def _probe_ready(self) -> None:
        """Consume stdout lines until the child's ready handshake."""
        if self._proc is None or self._stdout_q is None:
            raise EngineProcError(f"[{self._label}] subprocess not started")
        while True:
            raw = await self._next_stdout_line()
            if raw is None:
                raise EngineProcError(
                    f"[{self._label}] model subprocess exited before ready")
            text = raw.decode("utf-8", "replace").strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError:
                logger.info("[%s] pre-ready stdout: %s", self._label, text[:120])
                continue
            if isinstance(obj, dict) and obj.get("type") == "ready":
                if not obj.get("ok"):
                    raise EngineProcError(
                        f"[{self._label}] model subprocess failed to init: "
                        f"{obj.get('error', 'unknown')}")
                return
            # Not the handshake — ignore (e.g. an early progress/noise line).
            logger.info("[%s] pre-ready msg: %s", self._label, text[:120])

    async def _drain_stderr(self) -> None:
        """Forward the child's stderr (its logs) into our logger."""
        if self._stderr_q is None or self._loop is None:
            return
        # Capture the queue once: after a stop() swaps it to None (and the pump
        # pushes a None EOF sentinel), this task must keep draining the SAME
        # queue it started with rather than reading the (now-None) attribute.
        q = self._stderr_q
        while True:
            raw = await q.get()
            if raw is None:
                return
            text = raw.decode("utf-8", "replace").rstrip()
            if text:
                logger.info("[%s] %s", self._label, text)

    async def stop(self, timeout: float = 15) -> None:
        """Terminate the child (destroying its CUDA context). Idempotent."""
        async with self._lock:
            await self._stop_unlocked(timeout)

    async def _stop_unlocked(self, timeout: float = 15) -> None:
        """Non-lock-acquiring stop; caller must hold ``self._lock``."""
        proc = self._proc
        if proc is None:
            self._ready = False
            return
        self._proc = None
        self._ready = False
        self._stdout_q = None
        self._stderr_q = None
        try:
            if proc.poll() is None:
                logger.info("[%s] evicting model subprocess (pid=%s)",
                            self._label, proc.pid)
                await asyncio.to_thread(self._terminate_and_wait, proc, timeout)
            else:
                logger.info("[%s] subprocess already exited (code=%s)",
                            self._label, proc.returncode)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] error during evict: %s", self._label, exc)
        # Dissociate child stdio so the pump threads see EOF and exit.
        for _s in (proc.stdout, proc.stderr, proc.stdin):
            try:
                if _s is not None:
                    _s.close()
            except Exception:  # noqa: BLE001
                pass

    # ── request/response ───────────────────────────────────────────────────
    async def _drain_stale_stdout(self) -> None:
        """Drop orphaned result lines left by an aborted prior job so they can't
        be misread as the next job's result (mirrors bernini's guard)."""
        if self._stdout_q is None:
            return
        for _ in range(64):
            try:
                raw = await asyncio.wait_for(self._next_stdout_line(), 0.05)
            except asyncio.TimeoutError:
                return
            except Exception:  # noqa: BLE001 - stream unusable
                return
            if not raw:
                return
            text = raw.decode("utf-8", "replace").strip()
            if text:
                logger.info("[%s] drained stale stdout: %s",
                            self._label, text[:100])

    async def run(self, op: str, args: Optional[dict] = None,
                  timeout: Optional[float] = None,
                  progress_cb: Optional[Callable[[dict], None]] = None) -> Any:
        """Send one request to the child and return its result.

        Raises :class:`EngineProcError` on child failure/timeout (and evicts the
        child so a wedged state can never linger). The whole call holds
        ``self._lock`` so a concurrent :meth:`stop` cannot tear the child's
        streams out from under an in-flight result read."""
        async with self._lock:
            return await self._run_locked(op, args, timeout, progress_cb)

    async def _run_locked(self, op, args, timeout, progress_cb):
        """Lock-holding body of :meth:`run`; caller must hold ``self._lock``."""
        proc = self._proc
        try:
            if not (self._ready and proc is not None and proc.poll() is None):
                await self._ensure_started_unlocked()
            proc = self._proc
            if proc is None or proc.stdin is None or self._stdout_q is None:
                raise EngineProcError(f"[{self._label}] subprocess unavailable")
            await self._drain_stale_stdout()
            line = (json.dumps({"op": op, "args": args or {}},
                               ensure_ascii=False) + "\n").encode("utf-8")

            async def _read_result() -> Optional[bytes]:
                max_lines = 20000
                for _ in range(max_lines):
                    raw = await self._next_stdout_line()
                    if raw is None:
                        return raw  # EOF -> caller handles
                    text = raw.decode("utf-8", "replace").strip()
                    if not text:
                        continue
                    try:
                        obj = json.loads(text)
                    except json.JSONDecodeError:
                        logger.info("[%s] skip non-json stdout (%d B) %r",
                                    self._label, len(raw), raw[:120])
                        continue
                    if isinstance(obj, dict) and obj.get("type") == "progress":
                        if progress_cb is not None:
                            try:
                                progress_cb(obj)
                            except Exception:  # noqa: BLE001
                                pass
                        continue
                    if isinstance(obj, dict) and obj.get("type") == "ready":
                        continue
                    return raw
                raise EngineProcError(
                    f"[{self._label}] stdout produced no JSON result after "
                    f"{max_lines} lines (op={op})")

            await asyncio.to_thread(self._write_job, proc.stdin, line)
            raw = await asyncio.wait_for(
                _read_result(), timeout or self._job_timeout)
            if not raw:
                rc = getattr(self._proc, "returncode", None)
                if rc is None:
                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(self._proc.wait), 2)
                    except Exception:  # noqa: BLE001
                        pass
                    rc = getattr(self._proc, "returncode", None)
                mem_info = {}
                try:
                    for ln in open("/proc/meminfo"):
                        k, _, v = ln.partition(":")
                        mem_info[k] = v.strip()
                except Exception:  # noqa: BLE001
                    pass
                logger.error(
                    "[%s] child died mid-op (op=%s) returncode=%s signal=%s; "
                    "host MemAvailable=%s SwapFree=%s",
                    self._label, op, rc,
                    (-rc if (rc is not None and rc < 0) else None),
                    mem_info.get("MemAvailable"), mem_info.get("SwapFree"))
                raise EngineProcError(
                    f"[{self._label}] subprocess closed unexpectedly "
                    f"(op={op}, returncode={rc})")
            result = json.loads(raw.decode("utf-8"))
            if not result.get("ok"):
                raise EngineProcError(
                    f"[{self._label}] op '{op}' error: {result.get('error')}")
            return result.get("result")
        except asyncio.TimeoutError as exc:
            await self._stop_unlocked(5)
            raise EngineProcError(
                f"[{self._label}] op '{op}' timed out after "
                f"{timeout or self._job_timeout}s \u2014 subprocess evicted") from exc
        except EngineProcError:
            # A child-declared error may leave it wedged; evict so the next op
            # gets a fresh, clean process.
            await self._stop_unlocked(5)
            raise
        except Exception as exc:  # noqa: BLE001
            await self._stop_unlocked(5)
            raise EngineProcError(
                f"[{self._label}] op '{op}' failed: {exc}") from exc



def make_env(**overrides) -> dict:
    """Build a child environment with extra vars (thread-parallelism caps etc.
    can be overridden per worker)."""
    env = {**os.environ}
    env.update(overrides)
    return env


# ── child side ─────────────────────────────────────────────────────────────
def run_child_loop(device: str, build_engine: Callable[[], Any],
                   handlers: Dict[str, Callable[..., Any]],
                   ready_msg: Optional[str] = None) -> int:
    """Blocking JSONL command loop executed inside the model subprocess.

    Args
        device:  torch device string (``cuda:N``) to pin the child's active CUDA
                 device to BEFORE the engine builds, so no context leaks onto an
                 unassigned card.
        build_engine: zero-arg callable returning the real engine object.
        handlers:  ``{op: callable(engine, args, progress_cb) -> json-safe}``.
                   ``progress_cb`` is a small callable the handler may invoke
                   with a dict; it is emitted as a progress line to the parent.
        ready_msg: logger line printed (to stderr) once the engine is built.

    Returns the process exit code. Never raises out of the loop; per-request
    errors are reported on the wire so a bad op doesn't kill the child.
    """
    import torch  # noqa: PLC0415 - only imported inside the GPU process
    try:
        idx = int(str(device).replace("cuda:", ""))
        torch.cuda.set_device(idx)
    except Exception:  # noqa: BLE001
        pass  # best-effort pin; engine build asserts a working device
    engine = build_engine()
    if ready_msg:
        print(ready_msg, file=sys.stderr, flush=True)

    def _emit_progress(obj: dict) -> None:
        try:
            sys.stdout.write(json.dumps(
                {"type": "progress", **obj}, ensure_ascii=False) + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001
            pass

    # Ready handshake AFTER the engine shell is built (model weights may still
    # be lazy — the first request triggers them, matching /load semantics).
    sys.stdout.write(json.dumps({"type": "ready", "ok": True}) + "\n")
    sys.stdout.flush()

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            op = req.get("op")
            args = req.get("args") or {}
            handler = handlers.get(op)
            if handler is None:
                sys.stdout.write(json.dumps(
                    {"ok": False, "error": f"unknown op {op!r}"}) + "\n")
                sys.stdout.flush()
                continue
            try:
                result = handler(engine, args, _emit_progress)
                sys.stdout.write(json.dumps(
                    {"ok": True, "result": result}, ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc(limit=8)
                logger.error("[child][%s] op %s failed:\n%s", device, op, tb)
                sys.stdout.write(json.dumps(
                    {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                    ensure_ascii=False) + "\n")
            sys.stdout.flush()
    except Exception as exc:  # noqa: BLE001 - EOF / closed stdin
        logger.info("[child][%s] input closed (%s); exiting", device, exc)
    return 0
