"""ffmpeg rail — streamed transcode / upscale-downscale / encode for vp-worker.

This is the LAST stage of the combined post-chain: it takes either the raw
RIFE/FlashVSR frame stack or an intermediate encoded file and produces the
final deliverable at the target resolution/fps. Uses ffmpeg with ``rawvideo``
stdin transport for frame-in/encode-out, and the ``scale=lanczos`` +
``libx264 -preset slow -crf 15 -pix_fmt yuv420p`` recipe (quality-first, per
user preference).

final: "raw" -> returns the raw frame stack (no ffmpeg encode) so the caller
can hand back native 4x output. Otherwise -> encodes to the requested
resolution at the target fps.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("video_creator.runner.vp.ffmpeg_post")


class FfmpegError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def probe_video(path: str) -> Tuple[int, int, float, int]:
    """Return (width, height, fps, frame_count) via ffprobe."""
    if not shutil.which("ffprobe"):
        raise FfmpegError("ffprobe not available")
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
           "-show_entries", "stream=width,height,r_frame_rate,nb_frames",
           "-of", "default=noprint_wrappers=1", path]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    w = h = fps = n = 0
    for line in out.splitlines():
        k, _, v = line.partition("=")
        if k == "width":
            w = int(v)
        elif k == "height":
            h = int(v)
        elif k == "r_frame_rate" and "/" in v:
            num, den = v.split("/")
            fps = float(num) / (float(den) or 1.0)
        elif k == "nb_frames" and v != "N/A":
            try:
                n = int(v)
            except ValueError:
                n = 0
    return w, h, fps, n


def read_frames(path: str, fps: float) -> np.ndarray:
    """Decode a video to [F,H,W,3] uint8 numpy via ffmpeg rawvideo (rgb24)."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-"]
    p = subprocess.run(cmd, capture_output=True)
    arr = np.frombuffer(p.stdout, dtype=np.uint8)
    w, h, _f, _n = probe_video(path)
    if not w or not h or arr.size == 0 or arr.size % (w * h * 3) != 0:
        raise FfmpegError(f"rawvideo decode failed for {path} (got {arr.size} bytes)")
    return arr.reshape(-1, h, w, 3)


def encode_frames(frames: np.ndarray, out_path: str, fps: float,
                  width: Optional[int] = None, height: Optional[int] = None,
                  crf: int = 15, preset: str = "slow") -> str:
    """Encode [F,H,W,3] uint8 frames to H.264 mp4 via ffmpeg rawvideo stdin.

    4x upscale is done by FlashVSR/RIFE at render; here we only SIZE to the
    final requested resolution (SSAA downscale of the 4x output via lanczos)
    when width/height are given and differ from the frame stack size.
    """
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise FfmpegError(f"expected [F,H,W,3] frames, got {frames.shape}")
    fh, fw = frames.shape[1], frames.shape[2]
    tw, th = width or fw, height or fh
    vf = f"scale={tw}:{th}:flags=lanczos" if (tw, th) != (fw, fh) else "null"
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo",
           "-pixel_format", "rgb24", "-video_size", f"{fw}x{fh}",
           "-framerate", str(fps), "-i", "-",
           "-vf", vf,
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", out_path]
    p = subprocess.run(cmd, input=frames.tobytes(), capture_output=True)
    if p.returncode != 0 or not os.path.exists(out_path):
        raise FfmpegError(f"ffmpeg encode failed: {(p.stderr or p.stdout)[-2000:]}")
    return out_path


def finalize(frames: np.ndarray, out_fps: float, final: str,
             width: Optional[int] = None, height: Optional[int] = None,
             out_path: Optional[str] = None) -> tuple:
    """Tail of the post chain.

    final == "raw" -> return (frames_array, None) unchanged (no encode).
    else           -> transcode/encode to the requested resolution/fps -> (path, path).
    """
    if final == "raw":
        return frames, None
    path = out_path or "/tmp/vp_final.mp4"
    encode_frames(frames, path, out_fps, width=width, height=height)
    return path, path
