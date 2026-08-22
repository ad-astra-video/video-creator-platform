"""Bernini job I/O helpers for the wan-worker server.

The worker's HTTP surface is base64-in/base64-out (mirroring the restyle
handler): the request carries base64 source media, the handler writes them to
temp files for the Bernini subprocess (which needs real paths), and reads the
rendered mp4 back into base64 for the response.

Decoding uses the existing lightweight helpers (cv2 for video frame probing,
PIL for images) already available in the worker image; no Bernini-venv import
happens here (that only exists in the isolated subprocess).
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import tempfile
from typing import Any, Optional

logger = logging.getLogger("video_creator.runner.idv2v.bernini_io")


def write_b64(data_b64: str | bytes, path: str) -> str:
    """Decode base64 (str or already-decoded bytes) to ``path``; return path."""
    if isinstance(data_b64, str):
        raw = base64.b64decode(data_b64)
    else:
        raw = data_b64
    with open(path, "wb") as fh:
        fh.write(raw)
    return path


def decode_source_media(body: dict, tmpdir: str) -> dict[str, Any]:
    """Write the request's base64 media to real files under ``tmpdir``.

    Returns a mapping with only the keys the bernini_cli job accepts:
      * ``image``   -> single image file (i2i-style / first-frame)
      * ``images``  -> list of reference image paths (r2v / rv2v)
      * ``video``   -> list of source video paths (v2v)
    Keys whose base64 is absent are omitted entirely.
    """
    job: dict[str, Any] = {}

    video_b64 = body.get("video")
    if video_b64:
        if isinstance(video_b64, list):
            paths = []
            for i, v in enumerate(video_b64):
                if not v:
                    continue
                p = os.path.join(tmpdir, f"src_{i}.mp4")
                write_b64(v, p)
                paths.append(p)
            if paths:
                job["video"] = paths
        elif isinstance(video_b64, str) and video_b64:
            job["video"] = [write_b64(video_b64, os.path.join(tmpdir, "src.mp4"))]

    image_b64 = body.get("image") or body.get("image_base64")
    if image_b64:
        job["image"] = write_b64(image_b64, os.path.join(tmpdir, "first.png"))

    images_b64 = body.get("images") or body.get("references")
    if images_b64:
        refs = []
        for i, img in enumerate(images_b64):
            if not img:
                continue
            refs.append(write_b64(img, os.path.join(tmpdir, f"ref_{i}.png")))
        if refs:
            job["images"] = refs

    return job


def encode_video_b64(path: str) -> str:
    """Read a rendered mp4 back into base64 for the HTTP response."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def default_task_name(body: dict) -> str:
    """Resolve the Bernini task key from the request."""
    t = str(body.get("task") or body.get("task_type") or "").strip()
    if t in ("t2v", "v2v", "r2v", "i2v"):
        return {"i2v": "t2v"}.get(t, t)
    # Infer from which media is present.
    if body.get("video"):
        return "v2v"
    if body.get("images") or body.get("references"):
        return "r2v"
    return "t2v"


def tmpdir_context():
    return tempfile.TemporaryDirectory(prefix="bernini_")
