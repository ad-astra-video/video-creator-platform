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
import base64
import json
import logging
import math
import os
import subprocess
import time
import typing
from typing import Any, Optional

from . import config

logger = logging.getLogger("video_creator.runner.idv2v.bernini")

# Max seconds to wait for the resident process to come up (model build at warm).
STARTUP_TIMEOUT = 900
# Max seconds for a single generation job (long clips at 30 steps on a 5090).
JOB_TIMEOUT = 900

# Native Bernini window (frames) — v2v inputs longer than this are split into
# consecutive chunks of this size, each rendered natively, then concatenated.
# Keep aligned with BERNINI_NATIVE_FRAMES in the frontend. Each chunk after
# the first is anchored to the previous chunk's last output frame (reference
# image) so the edit's scene content carries across chunk boundaries instead
# of being re-hallucinated per chunk.
NATIVE_FRAMES = 33

# Native Bernini cadence (fps) — the model renders at this fixed rate, and the
# pipeline derives each chunk's output frame count from the INPUT clip's
# duration at this fps (smart_video_nframes, vae_fps). So a chunk extracted at
# the SOURCE fps (e.g. 25) is seen as "81 frames @25fps = 3.24s" and returns
# only floor(3.24*16/4)*4+1 = 49 frames, not 81. To make every chunk come back
# with its full source frame count, we re-time the extracted window onto this
# native 16fps grid before the Bernini pass; the final concat re-time back to
# src_fps (setpts=N/(src_fps*TB)) restores the source rate, so the round-trip
# is speed-preserving and yields the correct duration. Keep aligned with
# BERNINI_NATIVE_FPS in the frontend.
NATIVE_FPS = 16

# Number of the PREVIOUS chunk's last OUTPUT frames to pass as reference images
# (`images=[...]`) to the next chunk. Bernini v2v is source-driven — the
# contiguous source window already carries MOTION across the boundary — so the
# seam is an APPEARANCE pop: each chunk independently re-restyles the boundary
# source frames, and a single still (prev_last.png) under-constrains the edit.
# Anchoring on the model's own last N output frames (an H3-style continuation
# window, e.g. 5 frames like MiniMax H3 "Fast") pins the restyled scene (the
# warehouse) across the boundary instead of re-hallucinating it. Keep the
# source `video` (motion) and single `image` slot intact; this widens only the
# `images` reference anchor.
CHUNK_REF_FRAMES = 4  # 14B: keep tight — each ref is a VAE latent tile and the 14B/32GB ceiling is why this is capped.

# 1.3B: widening the appearance anchor is VRAM-cheap, so pass more of the
# previous chunk's OUTPUT context forward. Each new chunk re-derives the
# restyle from its `images` refs; more well-conditioned frames of the already
# restyled scene means less per-chunk re-hallucination, tightening the seam
# (the reported "appearance jump"). 14B deliberately stays at CHUNK_REF_FRAMES
# (OOM ceiling). Resolved per-model in _generate_chunked via self.model.
CHUNK_REF_FRAMES_13B = 6

# Appearance-anchor weight used for cross-chunk reference frames. Anchoring on
# the previous chunk's OUTPUT frames also carries that chunk's synthesized
# pose/motion, which on 1.3B (softer motion control) nudges the new chunk's
# motion away from pure source-following. Dialed down from the default omega_img
# (6.5) so the source v2v control reasserts the intended motion, at a small
# appearance-carry cost at the seam.
CHUNK_ANCHOR_OMEGA_IMG = 4.5  # original bernini default omega_img (anchored chunks)

# Keep chunk 0's LAST restyled output frame as a persistent GLOBAL identity
# anchor (a "global sink"), re-fed into every subsequent chunk's `images`
# reference alongside the local seam tail. Purpose: lock long edits to the
# established restyle instead of letting the look drift chunk-to-chunk
# (JoyAI-Video-Edit "global sink"; MoCha-LD long-range context). The global
# frame trades out the OLDEST local-tail slot so the total `images` count stays
# at CHUNK_REF_FRAMES -> no extra VRAM on the 14B/32GB OOM ceiling.
CHUNK_GLOBAL_ANCHOR = True


class BerniniError(RuntimeError):
    pass


def _normalize_target_device(device) -> str:
    """Normalize a target device (int index, bare ``N``, or ``cuda:N``) to a full
    torch device string for the bernini subprocess. The device comes from the
    live-runner scheduler's ``/load`` (may arrive as an int, a bare ``N``, or a
    ``cuda:N`` string); ``None``/empty falls back to the worker's configured
    GPU_DEVICE. Does NOT invent a device — if the result is empty the caller
    (`/load`) rejects the request rather than defaulting to ``cuda:0``."""
    target = device or config.BERNINI_GPU_DEVICE or config.GPU_DEVICE
    if isinstance(target, bool):
        target = str(target)
    if isinstance(target, int):
        return f"cuda:{target}"
    target = str(target).strip()
    if target and not target.startswith(("cuda", "cpu", "meta", "xpu", "mps")):
        try:
            if str(int(target)) == target:  # bare index e.g. "3"
                return f"cuda:{target}"
        except ValueError:
            pass
    return target


def _dump_tasks(tag: str) -> None:
    """Log every live asyncio task's stack under ``tag`` (best-effort).
    An asyncio deadlock parks the loop in the selector so a faulthandler SIGALRM
    dump only shows the poll(); suspended coroutine frames come via Task.get_stack()."""
    import traceback as _tb
    try:
        tasks = asyncio.all_tasks()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[%s] asyncio.all_tasks() unavailable: %r", tag, exc)
        return
    logger.warning("[%s] dumping %d live task stack(s)", tag, len(tasks))
    for i, t in enumerate(tasks):
        try:
            frames = t.get_stack(limit=70)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] task %d stack unavailable: %r", tag, i, exc)
            continue
        text = "".join(_tb.format_stack(frames)) if frames else "<no frame>"
        logger.warning("[%s] --- task %d done=%s cancelled=%s ---\n%s",
                       tag, i, t.done(), t.cancelled(), text)

async def _dump_after_delay(seconds: float, tag: str) -> None:
    """Watchdog: after ``seconds`` dump all live task stacks under ``tag``.
    Caller MUST cancel when the guarded work completes."""
    try:
        await asyncio.sleep(seconds)
    except asyncio.CancelledError:
        return
    _dump_tasks(tag)


class BerniniManager:
    """Own one resident ``bernini_cli.py`` subprocess for a single model."""

    def __init__(self, model: str = "bernini-1.3b",
                 device: Optional[str] = None, venv_py: Optional[str] = None,
                 root: Optional[str] = None, guidance: str = "rv2v"):
        self.model = model
        self.resolve_model = model  # config.resolve_model(model)
        self.device = _normalize_target_device(
            device or config.BERNINI_GPU_DEVICE or config.GPU_DEVICE or "cuda:0")
        self.venv_py = venv_py or config.BERNINI_VENV_PY
        self.root = root or config.bernini_root(model)
        self.guidance = guidance
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        self._ready = False
        self._warm = False

    def _stage(self, msg: str) -> None:
        """[bernini-manager] one-line stage marker for deadlock triage."""
        logger.info("BERN_MGR %s", msg)

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
                stderr=asyncio.subprocess.PIPE,
                cwd=here,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            # bernini_cli writes ALL its logs to stderr (logging.basicConfig ->
            # stderr). Forward that stream into our logger so the wan-worker's
            # `docker logs` shows the CLI's build/job output — without this the
            # subprocess is a black box ("ran but no logs came out"). The model
            # build latency is absorbed by JOB_TIMEOUT on the first generation;
            # we pre-warm by sending a lightweight no-op so the model actually
            # materializes here rather than on first user job.
            self._stderr_task = asyncio.create_task(self._drain_stderr())
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

    async def _drain_stderr(self) -> None:
        """Forward the bernini_cli's stderr stream into our logger."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while True:
            try:
                raw = await proc.stderr.readline()
            except Exception:  # noqa: BLE001 - stream closed/unusable
                break
            if not raw:
                break
            text = raw.decode("utf-8", "replace").rstrip()
            if text:
                logger.info("[bernini] %s", text)

    async def evict(self) -> None:
        """Terminate the resident process and free its GPU."""
        async with self._lock:
            await self._evict_unlocked()

    async def _evict_unlocked(self) -> None:
        """NON-lock-acquiring evict (caller must already hold self._lock).

        Separated from ``evict()`` so error paths inside ``generate()`` -- which
        already hold the non-reentrant ``asyncio.Lock`` -- can terminate the
        process WITHOUT the classic asyncio.Lock self-deadlock (re-acquiring
        the same lock from inside its own critical section). Calling
        ``self.evict()`` from within generate()'s locked region previously
        blocked forever, so a failed job turned into a 900s hang + 504 instead
        of a clean evict.
        """
        proc = self._proc
        self._proc = None
        self._ready = False
        self._warm = False
        stderr_task = self._stderr_task
        self._stderr_task = None
        if stderr_task is not None:
            stderr_task.cancel()
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
                       progress_cb: Optional[Any] = None) -> dict[str, Any]:
        """Run one Bernini task through the resident subprocess.

        For ``v2v`` jobs whose source video exceeds the native window
        (:data:`NATIVE_FRAMES`), the source is split into consecutive
        no-overlap 81-frame chunks, each rendered natively, then concatenated
        back into a single clip covering the input's total frame count. Shorter
        inputs (and t2v/r2v) go through the single-shot path unchanged.
        """
        await self.ensure_loaded()
        self._stage("gen: ensure_loaded done")
        async with self._lock:
            self._stage("gen: lock acquired")
            total = self._resolve_total_frames(job)
            src_fps = self._probe_source_fps(job)
            if total is not None and total > NATIVE_FRAMES:
                self._stage("gen: chunked v2v total=%d native=%d src_fps=%s" %
                            (total, NATIVE_FRAMES, src_fps))
                return await self._generate_chunked(job, total, timeout,
                                                    src_fps=src_fps,
                                                    progress_cb=progress_cb)
            result = await self._run_one(job, timeout, progress_cb)
            if src_fps and job.get("task_name") == "v2v" and result.get("ok"):
                out = result.get("output") or job.get("output")
                if out and os.path.exists(out):
                    self._stage("gen: re-encode single-shot at src fps %s" % src_fps)
                    _reencode_fps(out, out, src_fps)
                    result["out_fps"] = src_fps
            return result

    # -- single-shot resident CLI call ---------------------------------------
    async def _drain_stale_stdout(self) -> None:
        """Discard orphaned result lines left in the resident CLI's stdout pipe.

        The resident subprocess is shared across jobs and chunks. If a prior
        request was aborted (e.g. the client 504'd) its CLI result line is
        never consumed and sits in the pipe; the next _run_one would otherwise
        misread it as ITS result, desyncing the chunk loop (unanchored chunks,
        missing src_chunk files, mismatched anchor exports). Drain before each
        write so only fresh results are seen. Short timeouts: we discard only
        what is ALREADY buffered from a prior write, never a result for the
        job that has not been sent yet.
        """
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for _ in range(64):
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), 0.05)
            except asyncio.TimeoutError:
                return
            except Exception:  # noqa: BLE001 - stream unusable
                return
            if not raw:
                return
            text = raw.decode("utf-8", "replace").strip()
            if text:
                self._stage("gen: drained stale stdout line (%d B) %r" %
                            (len(raw), text[:80]))

    async def _run_one(self, job: dict[str, Any], timeout: float = JOB_TIMEOUT,
                       progress_cb: Optional[Any] = None) -> dict[str, Any]:
        """Write one job to the resident CLI and read its JSONL result.

        Bounded by ``timeout``; evicts the worker (lock-free) on error/timeout
        so a failed job can never leave the CLI orphaned or deadlock the lock.
        """
        if self._proc is None or self._proc.stdin is None or \
                self._proc.stdout is None:
            raise BerniniError("berni subprocess unavailable")
        # Clear any orphaned result line from an aborted prior job before
        # writing ours, so it can't be misread as our result.
        await self._drain_stale_stdout()
        line = json.dumps(job, ensure_ascii=False) + "\n"

        async def _read_result() -> bytes:
            # stdout is the JSONL result channel. Skip non-JSON noise (CUDA
            # banners), AND interim {"type":"progress",...} JSON lines (the
            # CLI's per-denoise-step progress) — those are forwarded through
            # progress_cb and dropped, so we keep reading until the terminal
            # result line arrives. Cap iterations so a pure-noise flood can't
            # loop forever.
            for _ in range(5000):
                raw = await self._proc.stdout.readline()
                if not raw:
                    return raw  # b"" EOF -> handled by caller
                text = raw.decode("utf-8", "replace").strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except json.JSONDecodeError:
                    self._stage("gen: skip non-json stdout line (%d B) %r" %
                                (len(raw), raw[:120]))
                    continue
                if isinstance(obj, dict) and obj.get("type") == "progress":
                    if progress_cb is not None:
                        try:
                            progress_cb(obj)
                        except Exception:  # noqa: BLE001 - never break the job
                            pass
                    continue
                return raw
            raise BerniniError(
                "Bernini stdout produced no JSON result after 5000 lines")

        try:
            self._stage("gen: writing stdin")
            self._proc.stdin.write(line.encode("utf-8"))
            await self._proc.stdin.drain()
            self._stage("gen: stdin drained; awaiting result line(s)")
            _watch = asyncio.create_task(_dump_after_delay(
                480, "BERN_GEN_STALL:%s" % job.get("task_name", "?")))
            try:
                # Whole read loop bounded by JOB_TIMEOUT (not per line).
                raw = await asyncio.wait_for(_read_result(), timeout)
            finally:
                _watch.cancel()
            self._stage("gen: final result line len=%s raw=%r" % (
                len(raw) if raw else 0, (raw[:160] if raw else b"")))
            result = json.loads(raw.decode("utf-8"))
            if not result.get("ok"):
                self._stage("gen: result not ok -> raise %r" %
                            (result.get("error"),))
                raise BerniniError(result.get(
                    "error", "Bernini generation failed"))
            self._stage("gen: generate RETURNS ok=%s" % result.get("ok"))
            return result
        except asyncio.TimeoutError as exc:
            self._stage("gen: TIMEOUT awaiting result -> evict")
            await self._evict_unlocked()
            raise BerniniError(
                f"Bernini job timed out after {timeout}s — worker evicted") from exc
        except Exception as exc:  # noqa: BLE001
            self._stage("gen: EXC %r -> evict" % (exc,))
            await self._evict_unlocked()
            raise BerniniError(f"Bernini subprocess failed: {exc}") from exc
        if not raw:
            await self._evict_unlocked()
            raise BerniniError("Bernini worker closed unexpectedly")

    # -- no-overlap chunking for long v2v ------------------------------------
    def _is_chunkable(self, job: dict[str, Any]) -> bool:
        return job.get("task_name") == "v2v" and bool(job.get("video"))

    def _resolve_total_frames(self, job: dict[str, Any]) -> Optional[int]:
        """Total source frame count for chunking decisions, or None if N/A.

        For v2v we trust the actual source (ffprobe), falling back to the
        client-probed ``num_frames``. Non-video tasks (t2v/r2v) return None so
        they always take the single-shot path.
        """
        if not self._is_chunkable(job):
            return None
        video = job["video"]
        src = video[0] if isinstance(video, (list, tuple)) else video
        if not src or not os.path.exists(src):
            return None
        try:
            n = _probe_frames(src)
        except Exception:  # noqa: BLE001
            n = None
        if n is None or n <= 0:
            want = job.get("num_frames")
            try:
                return int(want) if want else None
            except (TypeError, ValueError):
                return None
        return n

    def _probe_source_fps(self, job: dict[str, Any]) -> Optional[float]:
        """Source fps for v2v output (so the edit plays at the input's rate/duration).

        Mirrors the ID-V2V rail: generation runs at the model's native cadence,
        then the output is re-encoded at the SOURCE's original fps (or explicit
        request). Returns None for non-v2v jobs or when the probe fails.
        """
        if not self._is_chunkable(job):
            return None
        video = job["video"]
        src = video[0] if isinstance(video, (list, tuple)) else video
        if not src or not os.path.exists(src):
            return None
        try:
            return _probe_fps(src)
        except Exception:  # noqa: BLE001
            return None

    async def _generate_chunked(self, job: dict[str, Any], total: int,
                                timeout: float,
                                src_fps: Optional[float] = None,
                                progress_cb: Optional[Any] = None) -> dict[str, Any]:
        """Split the source into NATIVE_FRAMES chunks, render each, concat.

        Each chunk renders its own source window [k*81, (k+1)*81). The window
        is re-timed onto the native 16fps grid (NATIVE_FPS) before the render so
        the Bernini pass returns all ``length`` frames (without this, a chunk
        extracted at a higher source fps returns only duration*16 frames — see
        NATIVE_FPS). The last chunk holds the remainder; a remainder that is not
        1 mod 4 (the VAE's 4-frame temporal grid) returns the nearest grid count
        (e.g. 30 -> 29), a sub-frame error. Outputs are concatenated back to
        src_fps by the final concat remap, restoring the source rate.

        Consistency across chunks: chunk 0 anchors on the raw source and its
        last restyled OUTPUT frame becomes a GLOBAL identity anchor; every
        later chunk is additionally conditioned on that persistent chunk-0
        frame PLUS the PREVIOUS chunk's last CHUNK_REF_FRAMES OUTPUT frames,
        passed as reference images (``images=[...]``), so scene content the
        edit introduces (e.g. a restyled background) carries across the
        boundary (local seam) and stays locked to the established look
        (long-range identity) instead of drifting per chunk. The chunk's own
        source window still drives motion via v2v control.
        """
        video = job["video"]
        src = video[0] if isinstance(video, (list, tuple)) else video
        out_path = job["output"]
        base = os.path.dirname(out_path) or "."
        frames_per = NATIVE_FRAMES
        n_chunks = math.ceil(total / frames_per)
        # Per-chunk generation budget scales with the chunk count (900s x the
        # number of chunks), so chunk-1's model-build time is absorbed without
        # a false manager eviction on a multi-chunk native render.
        timeout = JOB_TIMEOUT * n_chunks
        chunk_srcs: list[str] = []
        chunk_outs: list[str] = []
        ref_files: list[str] = []
        preview_files: list[str] = []
        listfile = os.path.join(base, "concat.list")
        try:
            self._stage("gen: chunks=%d total=%d fpc=%d src=%s" %
                        (n_chunks, total, frames_per, src))
            prev_out: Optional[str] = None
            prev_frames: int = 0
            global_ref: Optional[str] = None  # chunk-0 frame, persistent identity anchor
            # --- chunk-aggregate progress ------------------------------------
            # Multi-chunk v2v renders n_chunks chunks, each running its own
            # denoise loop. The CLI reports per-chunk step/total (resets to
            # 0 each chunk), which would make the frontend bar bounce 0..1 N
            # times. We instead relay a CUMULATIVE step/total spanning ALL
            # chunks (total = n_chunks * per-chunk steps, learned from the
            # per-chunk CLI events) plus a monotonic absolute fraction, so the
            # frontend shows one accurate progress bar for the whole job.
            _steps_per: list = [None]  # effective per-chunk denoise steps
            def _scale_frac(f):
                return 0.05 + 0.90 * min(max(float(f), 0.0), 1.0)
            for k in range(n_chunks):
                start = k * frames_per
                end = min((k + 1) * frames_per, total)
                length = end - start
                src_k = os.path.join(base, f"src_chunk_{k:03d}.mp4")
                out_k = os.path.join(base, f"out_chunk_{k:03d}.mp4")
                self._stage("gen: chunk %d/%d frames [%d,%d) len=%d" %
                            (k + 1, n_chunks, start, end, length))
                if progress_cb is not None:
                    try:
                        _ev = {"chunk": k + 1, "chunks": n_chunks,
                               "frames_done": min((k + 1) * frames_per, total)}
                        if _steps_per[0]:
                            _to = n_chunks * _steps_per[0]
                            _base = k * _steps_per[0]
                            _ev["step"] = _base
                            _ev["total"] = _to
                            _ev["fraction"] = round(_scale_frac(_base / _to), 4)
                        progress_cb(_ev)
                    except Exception:  # noqa: BLE001 - never break the job
                        pass
                # extract the exact source window, re-timed onto the native
                # 16fps grid so the Bernini pass returns all `length` frames
                # (see NATIVE_FPS). `N` here is the post-select output frame
                # counter, so the selected `length` frames are laid out at
                # N/16s -> src_k is `length` frames @ NATIVE_FPS.
                await asyncio.to_thread(_run_ffmpeg, [
                    "ffmpeg", "-y", "-i", src,
                    "-vf", f"select='between(n,{start},{end - 1})',"
                           f"setpts=N/({NATIVE_FPS}*TB)",
                    "-r", str(NATIVE_FPS),
                    "-an", src_k,
                ])
                chunk_srcs.append(src_k)
                chunk_outs.append(out_k)
                sub_job = dict(job)
                sub_job["video"] = [src_k]
                sub_job["num_frames"] = length
                sub_job["output"] = out_k
                if prev_out and prev_frames > 0:
                    # Anchor appearance to the PREVIOUS chunk's last
                    # CHUNK_REF_FRAMES OUTPUT frames (H3-style continuation
                    # window) so the restyled scene (e.g. the warehouse)
                    # persists across the boundary instead of being
                    # re-hallucinated. A window, not a single still, pins the
                    # edit's look AND its recent state. MOTION is not the
                    # concern here — the contiguous source window drives it via
                    # v2v control — so this widens only the appearance anchor.
                    # When CHUNK_GLOBAL_ANCHOR is on, the persistent chunk-0
                    # frame is prepended (first) and the OLDEST local tail
                    # frame is dropped, keeping the total `images` count at
                    # CHUNK_REF_FRAMES so the 14B/32GB VRAM budget is unchanged.
                    # 1.3B widens the anchor (VRAM-cheap); 14B stays at the
                    # tight CHUNK_REF_FRAMES cap to protect the 32GB ceiling.
                    _ref_cap = (CHUNK_REF_FRAMES_13B if "14" not in
                                (self.model or "") else CHUNK_REF_FRAMES)
                    nref = min(_ref_cap, max(int(prev_frames), 1))
                    refs: list[str] = []
                    try:
                        n_tail = nref
                        if global_ref and nref > 1:
                            refs.append(global_ref)   # long-range identity pin
                            n_tail = nref - 1         # drop oldest local tail frame
                        base_idx = max(int(prev_frames) - n_tail, 0)
                        for i in range(n_tail):
                            ref = os.path.join(
                                base, f"ref_chunk_{k:03d}_{i:02d}.png")
                            await asyncio.to_thread(
                                _export_frame, prev_out, base_idx + i, ref)
                            refs.append(ref)
                        sub_job["images"] = refs
                        sub_job["omega_img"] = CHUNK_ANCHOR_OMEGA_IMG
                        ref_files.extend(refs)
                        self._stage(
                            "gen: chunk %d anchored -> %d frames (global=%s "
                            "tail=%d out idx %d..%d)" %
                            (k + 1, len(refs), bool(global_ref), n_tail,
                             base_idx, max(base_idx + n_tail - 1, 0)))
                    except Exception as exc:  # noqa: BLE001
                        # A failed anchor export must not kill the whole
                        # multi-chunk job; render this chunk unanchored.
                        refs.clear()
                        self._stage("gen: anchor export failed (%r) - unanchored" %
                                    (exc,))
                def _chunk_prog(info, _k=k, _steps_per=_steps_per,
                                _scale_frac=_scale_frac, _n_chunks=n_chunks,
                                _cb=progress_cb):
                    if not isinstance(info, dict):
                        return
                    if isinstance(info.get("step"), int) and \
                            isinstance(info.get("total"), int) and info.get("total"):
                        if _steps_per[0] is None:
                            _steps_per[0] = info["total"]
                        _to = _n_chunks * _steps_per[0]
                        if _to <= 0:
                            _cb(info)
                            return
                        _cum = _k * _steps_per[0] + info["step"]
                        _cb({"type": "progress", "step": _cum, "total": _to,
                             "fraction": round(_scale_frac(_cum / _to), 4)})
                    else:
                        _cb(info)
                result = await self._run_one(
                    sub_job, timeout,
                    _chunk_prog if progress_cb is not None else None)
                prev_out = out_k
                prev_frames = result.get("frames") or length
                # Global identity anchor: keep chunk 0's FIRST restyled frame
                # as a persistent reference for every later chunk (identity /
                # long-range drift pin). Exported once from chunk 0, then
                # reused on all k>0 anchors.
                if global_ref is None and prev_frames > 0 and CHUNK_GLOBAL_ANCHOR:
                    try:
                        global_ref = os.path.join(base, "ref_global_000.png")
                        # Use chunk 0's LAST frame as the global identity pin:
                        # chunk 0 was rendered unanchored, so its own look can
                        # drift head->tail over its frames. Pinning on frame 0
                        # made the first anchored chunk pull against chunk 0's
                        # own tail (both refs from the same chunk), popping the
                        # 1->2 seam. The last frame agrees with the tail region
                        # that chunk and its successors actually continue.
                        await asyncio.to_thread(
                            _export_frame, prev_out,
                            max(int(prev_frames) - 1, 0), global_ref)
                        ref_files.append(global_ref)
                        self._stage(
                            "gen: global anchor set from chunk 0 last frame")
                    except Exception as exc:  # noqa: BLE001
                        global_ref = None
                        self._stage(
                            "gen: global anchor export failed (%r)" % (exc,))
                # Preview: after each chunk lands, surface a small ephemeral still
                # so the user sees every chunk complete in the chat card (each new
                # chunk's preview replaces the last; the client removes it on done).
                if progress_cb is not None:
                    try:
                        _pvf = os.path.join(base, f"pv_chunk_{k:03d}.jpg")
                        _pv = _export_preview_jpg(out_k, prev_frames, _pvf)
                        if _pv:
                            preview_files.append(_pvf)
                            progress_cb({"type": "preview", "chunk": k + 1,
                                         "chunks": n_chunks,
                                         "preview": _pv})
                        else:
                            try:
                                if os.path.exists(_pvf):
                                    os.remove(_pvf)
                            except OSError:
                                pass
                    except Exception as exc:  # noqa: BLE001
                        self._stage("gen: preview export failed (%r)" % (exc,))
            # concatenate the per-chunk outputs into the final file
            with open(listfile, "w") as f:
                for o in chunk_outs:
                    f.write(f"file '{o}'\n")
            concat_args = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
            ]
            if src_fps:
                # remap each frame to index/srcfps so `total` frames span
                # total/srcfps seconds (source timeline). -frames:v total makes
                # the remap exact (no trailing duplicate frame).
                concat_args += ["-vf", f"setpts=N/({src_fps:.6g}*TB)",
                                "-r", f"{src_fps:.6g}",
                                "-frames:v", str(total)]
            concat_args += ["-movflags", "+faststart", out_path]
            self._stage("gen: concatenating %d chunks @fps=%s -> %s" %
                        (n_chunks, src_fps, out_path))
            await asyncio.to_thread(_run_ffmpeg, concat_args)
            self._stage("gen: concatenated ok frames=%d" % total)
            return {"ok": True, "output": out_path,
                    "frames": total, "task": job.get("task_name"),
                    "out_fps": src_fps}
        finally:
            for pth in chunk_srcs + chunk_outs + ref_files + preview_files:
                try:
                    if os.path.exists(pth):
                        os.remove(pth)
                except OSError:
                    pass
            try:
                if os.path.exists(listfile):
                    os.remove(listfile)
            except OSError:
                pass


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
    """Return the worker's shared Bernini manager, loading ``model`` on ``device``.

    When ``device`` is None the existing resident manager is reused (or created
    on the worker's configured GPU) — inference simply uses whichever card the
    scheduler placed it on via ``/load``. When ``device`` is given (a ``/load``
    relocation), a manager that differs in model or device is evicted and a fresh
    one created on the new target, so the live-runner can steer Bernini onto the
    assigned GPU. A device that resolves to empty raises ``BerniniError`` — the
    scheduler must send one; we never default to ``cuda:0``.
    """
    global _manager
    async with _manager_lock:
        target = _normalize_target_device(device)
        if _manager is not None:
            if _manager.model == model and (_manager.device == target or not target):
                # Same model already resident (on this device, or on whatever the
                # scheduler placed) — reuse it. When no device was requested we
                # deliberately do NOT force a device match + evict: a resident
                # cuda:N manager would otherwise be torn down and the re-load
                # re-raise "requires a device" whenever GPU_DEVICE is unset.
                return _manager
            # Resident model/device differs from what's asked — evict and load the
            # requested model. If no device was provided, reuse the resident card's
            # device (the scheduler already placed one), so a request is NEVER
            # silently served by the wrong family (previously a 14b request with
            # device=None ran on a resident 1.3b manager).
            if not target:
                target = _normalize_target_device(_manager.device)
            await _manager.evict()
            _manager = None
        if not target:
            raise BerniniError(
                "bernini load requires a 'device' from the live-runner scheduler "
                "(none provided / GPU_DEVICE empty)"
            )
        _manager = BerniniManager(model=model, device=target)
        return _manager


async def evict_manager() -> None:
    global _manager
    async with _manager_lock:
        if _manager is not None:
            await _manager.evict()
        _manager = None


def resident_status() -> Optional[dict]:
    """Advisory dict describing the resident Bernini subprocess, or None if none
    is loaded (used by /health + /info so the live-runner sees Bernini residency
    like any other worker model)."""
    global _manager
    if _manager is None:
        return None
    return _manager.resident


def _run_ffmpeg(args: list, timeout: float = 300) -> None:
    """Run ffmpeg/ffprobe; raise BerniniError on non-zero exit."""
    try:
        proc = subprocess.run(args, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise BerniniError(f"ffmpeg timed out: {' '.join(args[:8])}") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "")[-800:]
        raise BerniniError(f"ffmpeg failed ({proc.returncode}): {detail}")


def _export_frame(src: str, idx: int, out_png: str) -> None:
    """Write frame index ``idx`` of ``src`` to ``out_png``.

    Used to build the multi-frame cross-chunk anchor: the previous chunk's last
    ``CHUNK_REF_FRAMES`` output frames become the ``images`` reference list for
    the next chunk, so the restyled scene carries across the boundary instead
    of being re-hallucinated per chunk.
    """
    _run_ffmpeg([
        "ffmpeg", "-y", "-i", src,
        "-vf", f"select='eq(n\\,{max(int(idx), 0)})'",
        "-frames:v", "1", out_png,
    ])


def _export_last_frame(src: str, nframes: int, out_png: str) -> None:
    """Write the last frame (index ``nframes-1``) of ``src`` to ``out_png``.

    Used to build the cross-chunk reference image (Option A anchor): the last
    OUTPUT frame of chunk k becomes the ``images`` reference for chunk k+1, so
    the edit's scene content (e.g. a restyled background) persists instead of
    being re-hallucinated. Mirrors the existing ``select='between(n,...)'``
    quoting style already used for chunk extraction.
    """
    _run_ffmpeg([
        "ffmpeg", "-y", "-i", src,
        "-vf", f"select='eq(n\,{max(nframes - 1, 0)})'",
        "-frames:v", "1", out_png,
    ])


def _export_preview_jpg(src: str, nframes: int, out_jpg: str,
                        max_w: int = 320) -> str:
    """Export a small JPEG preview (base64) - a mid-frame of (src) downscaled
    to (max_w) wide - used to surface each finished chunk to the frontend as
    an ephemeral still in the chat task card. Never raises; returns "" on any
    failure so a preview problem can never affect the render job."""
    try:
        n = max(int(nframes) - 1, 0)
        mid = n // 2
        _run_ffmpeg([
            "ffmpeg", "-y", "-i", src,
            "-vf", f"select='eq(n\\,{mid})',scale={max_w}:-2",
            "-frames:v", "1", "-q:v", "5", out_jpg,
        ])
        with open(out_jpg, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:  # noqa: BLE001
        return ""


def _probe_frames(path: str) -> Optional[int]:
    """Best-effort total frame count of a video file (nb_frames then count).

    Runs synchronously (a brief blocking ffprobe) and returns None on any
    failure so the caller can fall back to the client-probed num_frames.
    """
    def _try(extra: list) -> Optional[int]:
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_frames",
                 "-of", "default=noprint_wrappers=1:nokey=1", *extra, path],
                capture_output=True, text=True, timeout=90)
        except Exception:  # noqa: BLE001
            return None
        val = (proc.stdout or "").strip()
        try:
            v = int(val)
            return v if v > 0 else None
        except (TypeError, ValueError):
            return None
    n = _try([])
    if n is not None:
        return n
    # nb_frames may be N/A for some streams — fall back to counting.
    return _try(["-count_frames"])


def _probe_fps(path: str) -> Optional[float]:
    """Best-effort fps of a video's first stream (r_frame_rate then avg).

    Returns None on any failure (or a non-positive/absent rate).
    """
    def _parse(val: str) -> Optional[float]:
        val = (val or "").strip()
        if not val:
            return None
        try:
            if "/" in val:
                n, d = val.split("/", 1)
                n, d = float(n), float(d)
                return n / d if d else None
            return float(val)
        except (TypeError, ValueError):
            return None
    for entry in ("r_frame_rate", "avg_frame_rate"):
        try:
            proc = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", f"stream={entry}",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=90)
        except Exception:  # noqa: BLE001
            continue
        f = _parse(proc.stdout)
        if f and f > 0:
            return f
    return None


def _reencode_fps(src: str, dst: str, fps: float,
                  frames: Optional[int] = None) -> None:
    """Re-encode a video at the given fps (in place when src == dst).

    Pure fps remux/re-encode — does NOT interpolate; a higher output fps simply
    plays each existing frame on the source timeline (matches the ID-V2V rail's
    source-fps restore). ``setpts=N/(fps*TB)`` places frame i at time i/fps and
    ``-frames:v N`` caps to exactly the source frame count, so N frames span
    N/fps seconds with no duplication/off-by-one.
    """
    if frames is None:
        frames = _probe_frames(src)
    tmp = dst + ".fps.tmp.mp4"
    args = [
        "ffmpeg", "-y", "-i", src,
        "-vf", f"setpts=N/({fps:.6g}*TB)",
        "-r", f"{fps:.6g}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
    ]
    if frames:
        args += ["-frames:v", str(frames)]
    args += ["-movflags", "+faststart", tmp]
    _run_ffmpeg(args)
    os.replace(tmp, dst)
