"""RIFE motion-preserving fps-boost rail for the vp-worker.

Delta over a naive 2x/drop: every ORIGINAL frame is anchored at its exact
timestamps and the network interpolates ONLY the gap-filler target frames at
arbitrary timestep ``t`` — no 2x-then-drop, no temporal smoothing after, so the
16 fps native aesthetic (which the user wants preserved) is kept and the motion
is not over-smoothed ("soap-opera").

Pipeline (weights: hzwer/RIFE -> RIFEv4.26_0921.zip -> flownet.pkl; code: the
vendored ComfyUI-VFI HDv3 trio in ``rife/`` — the known-good 4.26 pairing):

    frames [N,H,W,3] uint8
      -> _compute_target_timeline(source_fps, target_fps, N)  (anchored gaps)
      -> pad each pair to mult of 32, /255.0 CHW float tensors
      -> Model.inference_batch(I0, I1, timesteps, scale) in slices
      -> unpad, back to uint8 [M,H,W,3]

I/O is numpy frames; the server layer owns ffmpeg rawvideo stream transport.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger("video_creator.runner.vp.rife_post")

# RIFE pads to a multiple of this on the long side (model /32 requirement).
_PAD = 32

# Frame [N,H,W,3] uint8 (0..255). Timeline list-of-triples typed for clarity.
FrameArray = np.ndarray
TimelineEntry = Tuple[int, int, float]  # (source_idx1, source_idx2, interp_factor)


class RifeError(RuntimeError):
    pass


def compute_target_timeline(
    source_fps: float, target_fps: float, total_source_frames: int
) -> List[TimelineEntry]:
    """Anchored-gap frame timeline from source_fps -> target_fps.

    For every TARGET frame compute its (source_idx1, source_idx2, interp_factor)
    so that the FIRST source frame is always an exact anchor (t=0) and each gap
    is filled at the true sub-frame timestamps. When a target timestamp lands
    exactly on a source frame (interp_factor == 0) we anchor (copy) that source
    frame verbatim — this is the motion-preserving constraint.

    Ported from the verified ComfyUI-VFI reference
    (rife_comfyui_wrapper._calculate_target_frame_positions) so the timeline
    maths matches the known-good pipeline.
    """
    if source_fps <= 0 or target_fps <= 0:
        raise RifeError(f"fps must be > 0 (source={source_fps}, target={target_fps})")
    if target_fps < source_fps:
        # Never down-rate: clamp to the source rate (a boost rail only adds fps).
        target_fps = source_fps
    if total_source_frames < 2:
        raise RifeError("need >= 2 source frames to interpolate")

    duration = total_source_frames / source_fps
    total_target_frames = int(round(duration * target_fps))
    total_target_frames = max(total_target_frames, total_source_frames)

    timeline: List[TimelineEntry] = []
    for k in range(total_target_frames):
        target_time = k / target_fps
        source_position = target_time * source_fps
        i0 = int(source_position)
        i1 = min(i0 + 1, total_source_frames - 1)
        if i0 == i1:
            t = 0.0
        else:
            t = source_position - i0
        timeline.append((i0, i1, t))
    return timeline


def _pad_to_32(h: int, w: int) -> Tuple[int, int, Tuple[int, int, int, int]]:
    ph = ((h - 1) // _PAD + 1) * _PAD
    pw = ((w - 1) // _PAD + 1) * _PAD
    return ph, pw, (0, pw - w, 0, ph - h)


class FpsBooster:
    """Warm RIFE model; boost a numpy video by anchored-gap interpolation."""

    def __init__(self, model_path: str, device: Optional[str] = None,
                 fp16: bool = False):
        if not model_path or not os.path.exists(model_path):
            raise RifeError(f"RIFE weights not found: {model_path!r}")
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("cuda requested but unavailable; falling back to cpu")
            self.device = "cpu"
        self.fp16 = fp16 and self.device.startswith("cuda")
        self._dtype = torch.float16 if self.fp16 else torch.float32

        torch.set_grad_enabled(False)
        if self.device.startswith("cuda"):
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        from runner.vp.rife.train_log.RIFE_HDv3 import Model

        self.model = Model()
        self.model.load_model(model_path, -1)
        self.model.eval()
        self.model.device()
        if self.fp16:
            self.model.flownet = self.model.flownet.half()
        logger.info("RIFE %s loaded (device=%s, fp16=%s)",
                    os.path.basename(model_path), self.device, self.fp16)

    # -- internals -----------------------------------------------------------
    def _interp_tensor(
        self, frames: torch.Tensor, timeline: Sequence[TimelineEntry],
        batch_size: int,
    ) -> Tuple[torch.Tensor, List[int]]:
        """Run the anchor+interp jobs; returns (output tensor, out_index map)."""
        h, w = frames.shape[1:3]
        ph, pw, padding = _pad_to_32(h, w)
        dev = self.device
        dtype = self._dtype
        out: List[Optional[torch.Tensor]] = [None] * len(timeline)
        jobs: List[int] = []  # timeline indices needing interpolation
        for idx, (i0, i1, t) in enumerate(timeline):
            if t == 0.0 or i0 == i1:
                out[idx] = frames[i0]  # anchor (single HWC slice)
            else:
                jobs.append(idx)

        with torch.inference_mode():
            for start in range(0, len(jobs), batch_size):
                chunk = jobs[start:start + batch_size]
                bs = len(chunk)
                needed = set()
                for idx in chunk:
                    i0, i1, _ = timeline[idx]
                    needed.add(i0)
                    needed.add(i1)
                src_cache = {
                    s: frames[s].to(device=dev, dtype=dtype) for s in needed
                }
                I0 = torch.empty((bs, 3, ph, pw), dtype=dtype, device=dev)
                I1 = torch.empty((bs, 3, ph, pw), dtype=dtype, device=dev)
                ts = []
                for j, idx in enumerate(chunk):
                    i0, i1, t = timeline[idx]
                    a = src_cache[i0].permute(2, 0, 1).unsqueeze(0)
                    b = src_cache[i1].permute(2, 0, 1).unsqueeze(0)
                    I0[j] = torch.nn.functional.pad(a, padding)[0]
                    I1[j] = torch.nn.functional.pad(b, padding)[0]
                    ts.append(t)
                interp = self.model.inference_batch(I0, I1, ts, scale=1.0)
                for j, idx in enumerate(chunk):
                    res = interp[j, :, :h, :w].permute(1, 2, 0)
                    out[idx] = res.float().cpu()
                del I0, I1, interp, src_cache
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        return torch.stack(out)  # type: ignore[arg-type]

    # -- public API ----------------------------------------------------------
    def boost(self, frames: FrameArray, source_fps: float, target_fps: float,
              batch_size: int = 8) -> FrameArray:
        """Interpolate a [N,H,W,3] uint8 video to target_fps (>= source_fps)."""
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise RifeError(f"expected [N,H,W,3] frames, got {frames.shape}")
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)
        n = frames.shape[0]
        target_fps = max(float(target_fps), float(source_fps))
        need = int(math.ceil(n / source_fps * target_fps))
        if need <= n:
            return frames

        timeline = compute_target_timeline(source_fps, target_fps, n)
        frames_t = torch.from_numpy(frames.transpose(0, 3, 1, 2).copy())
        # [N,C,H,W] uint8 -> /255 float (RIFE wants 0..1 on [N,H,W,C]).
        frames_t = frames_t.permute(0, 2, 3, 1).float() / 255.0
        out = self._interp_tensor(frames_t, timeline, batch_size)
        out = out.numpy()
        return np.clip(out * 255.0, 0, 255).astype(np.uint8)
