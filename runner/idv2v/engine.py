"""Parent-side proxy for the id-v2v ModelManager running in a subprocess.

The worker's aiohttp server never constructs a ``ModelManager`` or touches
``torch.cuda`` for id-v2v. It holds an :class:`Idv2vEngine`, which owns an
``engineproc.EngineProc`` subprocess that hosts the real model on the assigned
GPU. The proxy adds the worker-specific ergonomics:

    * /load  -> ``ensure_loaded(variant)`` spawns the child and runs its
                ``load`` op (the actual diffsynth pipeline build).
    * /evict -> ``stop()`` terminates the subprocess. The child's process
                exit destroys its CUDA primary context (the only reliable way
                to release the GPU), unlike an in-process ``evict()`` which
                cannot drop the driver-reserved floor.
    * restyle-> ``infer_frames(...)`` proxies the GPU-only denoise step to the
                child and maps its progress lines onto the parent's SSE rail.

Variant/device swaps are handled by STOPPING the child and respawning a fresh
one on the next ``ensure_loaded`` (a process restart subsumes evict+reload).
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys

from runner.common import engineproc
from runner.idv2v import config

logger = logging.getLogger("video_creator.runner.idv2v.engine")

# Allow the (multi-minute) diffsynth build / a long clip-stitching denoise.
LOAD_TIMEOUT = 3600
INFER_JOB_TIMEOUT = 3600


def _load_frame_paths(paths, frame_dir=None):
    """Read PNG scratch files the child wrote back into PIL frames, then remove
    the files (and their scratch dir). Mirrors bernini_cli: the child persists
    its output frames to disk; the parent reads them back for MP4 encoding, so
    the child->parent pipe only ever carries small paths, never base64."""
    from PIL import Image
    frames = []
    for p in paths:
        try:
            im = Image.open(p)
            im.load()
            frames.append(im)
        except Exception:  # noqa: BLE001 - never drop a job over one bad frame
            logger.warning("idv2v frame read failed: %s", p, exc_info=True)
        finally:
            try:
                os.remove(p)
            except OSError:
                pass
    if frame_dir:
        try:
            shutil.rmtree(frame_dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
    return frames


def _child_argv(device_index: int) -> list:
    return [sys.executable, "-m", "runner.idv2v.engine_cli",
            "--device", f"cuda:{device_index}"]


class Idv2vEngine:
    """Proxy that spawns/kills the real id-v2v ModelManager subprocess."""

    def __init__(self, device: str = "", argv: list | None = None) -> None:
        self.device = str(device) or str(config.GPU_DEVICE)
        self.variant = config.DEFAULT_MODEL_VARIANT
        self._ready = False
        # Test seam: a caller may supply the exact child argv (e.g. a fake
        # script) instead of the real engine_cli.
        self._argv_override = argv
        self._proc: engineproc.EngineProc | None = None

    # -- cached state -------------------------------------------------------
    @property
    def is_ready(self) -> bool:
        """True while a live, loaded child is resident (the GPU is busy)."""
        return self._ready

    def _cuda_index(self) -> int:
        ds = str(self.device)
        if ":" in ds:
            tail = ds.split(":", 1)[1]
            if tail.isdigit():
                return int(tail)
        return 0

    def _new_proc(self) -> engineproc.EngineProc:
        if self._argv_override:
            argv = list(self._argv_override)
        else:
            argv = _child_argv(self._cuda_index())
        return engineproc.EngineProc(
            "idv2v", argv, startup_timeout=LOAD_TIMEOUT, job_timeout=INFER_JOB_TIMEOUT)

    # -- lifecycle ----------------------------------------------------------
    async def _stop_child(self) -> None:
        """Terminate the child (kills its CUDA context). Idempotent."""
        self._ready = False
        proc = self._proc
        self._proc = None
        if proc is not None:
            await proc.stop()

    async def set_device(self, device: str) -> None:
        """Retarget this engine to another CUDA card.

        If a child is resident it is stopped (context destroyed); the next
        ``ensure_loaded`` spawns a fresh child on the new card.
        """
        device = str(device)
        if device == self.device:
            return
        self.device = device
        await self._stop_child()

    async def ensure_loaded(self, variant: str | None = None) -> None:
        """Spawn the child (if needed) and load the model on the assigned GPU.

        A variant swap (or a prior stop/evict) respawns a FRESH child, so the
        GPU state carried by the old process never leaks in — this is exactly
        what makes /evict release the card.
        """
        if variant:
            v = config._norm_variant(variant)
            if self._ready and v != self.variant:
                await self._stop_child()
            self.variant = v
        if self._ready and self._proc is not None:
            return
        if self._proc is None:
            self._proc = self._new_proc()
        res = await self._proc.run("load", {"variant": self.variant},
                                   timeout=LOAD_TIMEOUT)
        self._ready = bool(res.get("ready"))
        self.variant = res.get("variant") or self.variant
        self.device = res.get("device") or self.device
        if not self._ready:
            self._ready = False
            raise engineproc.EngineProcError(
                "child model did not report ready after load")

    async def stop(self) -> None:
        """Evict: kill the subprocess -> destroy the CUDA primary context."""
        await self._stop_child()

    # -- inference ----------------------------------------------------------
    async def infer_frames(self, *, prompt, negative_prompt, input_image,
                           condition_frames, keyframes, width, height,
                           num_frames, max_frames, num_inference_steps,
                           cfg_scale, vace_scale, seed,
                           job_id=None) -> dict:
        """Run the GPU denoise step in the child and return the generated
        frames (base64 PNG) plus a count for the parent to encode to MP4.

        The conditioning inputs are already-prepared, JSON-safe base64 PNGs
        (SAM3/segmentation + anchor/keyframe decode happen in the parent).
        """
        from runner.idv2v import run as run_mod

        if self._proc is None:
            self._proc = self._new_proc()

        args = {
            "prompt": prompt, "negative_prompt": negative_prompt,
            "input_image": input_image, "condition_frames": condition_frames,
            "keyframes": keyframes,
            "width": width, "height": height, "num_frames": num_frames,
            "max_frames": max_frames, "num_inference_steps": num_inference_steps,
            "cfg_scale": cfg_scale, "vace_scale": vace_scale, "seed": seed,
        }

        def _prog(obj):
            if not isinstance(obj, dict) or obj.get("type") != "progress":
                return
            if obj.get("preview"):
                run_mod.set_preview(job_id, obj["preview"])
            kwargs = {}
            if obj.get("step") is not None:
                kwargs["step"] = obj["step"]
            if obj.get("total") is not None:
                kwargs["total"] = obj["total"]
            run_mod.set_progress(
                job_id, obj.get("progress", 0.0), obj.get("stage", "generating"),
                obj.get("message", ""), **kwargs)

        res = await self._proc.run("infer_frames", args,
                                   timeout=INFER_JOB_TIMEOUT, progress_cb=_prog)
        # The child wrote each frame to a scratch PNG; read them back as PIL for
        # MP4 encoding and clean the files up.
        return {"frames": _load_frame_paths(res.get("frames") or [],
                                            res.get("frame_dir")),
                "count": res.get("count") or 0}
