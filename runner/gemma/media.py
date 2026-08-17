"""PyAV-backed media helpers for the gemma copilot.

The gemma-worker image ships PyAV (>=13, FFmpeg built with Vulkan) and the
container is granted ``NVIDIA_DRIVER_CAPABILITIES=compute,graphics,video,utility``
so the host's Vulkan ICD + NVDEC/NVENC libs are mounted. The copilot uses these
helpers to *introspect* media (dimensions, duration, fps, codec) and to pull
downscaled frames out of a user's uploaded video or reference image so its
vision tools (``inspect_media`` / ``evaluate_output``) can ground on real media.

Vulkan: the frame paths here are intentionally correct software decoders (fast
for a few evenly-sampled frames and deterministic for tests). A GPU hw-decode
path can be layered on later via ``av.Codec(..., hw_device_ctx=...)`` when the
resident GPU is confirmed idle; every public helper degrades gracefully if PyAV
or Pillow is missing, so this module never blocks the worker from loading.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("video_creator.runner.gemma.media")

_av: Any | None = None
_pil: Any | None = None


def _libs() -> tuple[Any, Any]:
    """Lazily import PyAV + Pillow. Raises RuntimeError if unavailable."""
    global _av, _pil
    if _av is None:
        try:
            import av  # type: ignore
            _av = av
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(f"PyAV not available: {exc}") from exc
    if _pil is None:
        try:
            from PIL import Image  # type: ignore
            _pil = Image
        except Exception as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(f"Pillow not available: {exc}") from exc
    return _av, _pil


class MediaError(RuntimeError):
    """Raised when a media source can't be opened/probed/decoded."""


# ── Source handling ──────────────────────────────────────────────────────────

def _open_source(source: Any) -> Any:
    """Open an av.Container from a path, bytes, or an existing container."""
    av, _ = _libs()
    if isinstance(source, av.container.InputContainer):
        return source
    if isinstance(source, (str, Path)):
        return av.open(str(source))
    if isinstance(source, (bytes, bytearray, memoryview)):
        return av.open(io.BytesIO(bytes(source)))
    if hasattr(source, "read"):  # file-like
        return av.open(source)
    raise MediaError(f"unsupported media source type: {type(source)!r}")


def _close(container: Any, source: Any) -> None:
    """Close a container unless the caller handed us an existing one."""
    av, _ = _libs()
    if isinstance(source, av.container.InputContainer):
        return
    try:
        container.close()
    except Exception:  # pragma: no cover
        pass


# ── Probing ──────────────────────────────────────────────────────────────────

def probe(source: Any) -> dict[str, Any]:
    """Return media metadata: type, duration, resolution, fps, codec, frame count."""
    av, _ = _libs()
    container = None
    try:
        container = _open_source(source)
        vstream = next((s for s in container.streams if s.type == "video"), None)
        if vstream is None and container.streams:
            vstream = container.streams[0]
        if vstream is None:
            raise MediaError("no media streams found")

        fps = None
        avg_rate = getattr(vstream, "average_rate", None)
        if avg_rate:
            fps = float(avg_rate)

        # Stream duration is in the stream's time_base units; the container
        # duration is in microseconds. Prefer the container, else convert.
        duration_s = None
        cd = getattr(container, "duration", None)
        if cd is not None:
            duration_s = float(cd) / 1_000_000.0
        else:
            sd = getattr(vstream, "duration", None)
            if sd is not None:
                tb = getattr(vstream, "time_base", None)
                duration_s = float(sd) * float(tb) if tb else None

        cc = getattr(vstream, "codec_context", None)
        codec = getattr(vstream, "name", None) or (getattr(cc, "name", None)
                                                   if cc else None)
        nb_frames = None
        try:
            nb_frames = vstream.frames
        except Exception:
            nb_frames = None

        is_video = bool(duration_s and duration_s > 1e-6)
        return {
            "media_type": "video" if is_video else "image",
            "width": getattr(vstream, "width", None),
            "height": getattr(vstream, "height", None),
            "fps": round(fps, 3) if (is_video and fps) else None,
            "duration_s": round(duration_s, 3) if duration_s else None,
            "codec": codec,
            "nb_frames": nb_frames,
        }
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError(f"failed to probe media: {exc}") from exc
    finally:
        if container is not None:
            _close(container, source)


# ── Frame extraction ─────────────────────────────────────────────────────────

def extract_frames(
    source: Any,
    count: int = 4,
    max_side: int = 512,
    fmt: str = "jpeg",
    quality: int = 82,
) -> list[str]:
    """Extract up to ``count`` evenly-sampled, downscaled frames as base64 URLs.

    Decodes on CPU, downscales to ``max_side`` on the long edge, returns base64
    data URLs the copilot's vision tools can feed straight to Gemma. Sampling is
    deterministic (spread across the clip), friendly to tests and stability.
    """
    av, Image = _libs()
    if count < 1:
        count = 1
    container = None
    try:
        container = _open_source(source)
        vstream = next((s for s in container.streams if s.type == "video"), None)
        if vstream is None:
            raise MediaError("no video stream to extract frames from")

        positions = _sample_ranks(container, vstream, count)
        out: list[str] = []
        seen = 0
        for _frame in container.decode(video=0):
            if seen in positions:
                out.append(_frame_to_b64(_frame, max_side=max_side, fmt=fmt,
                                         quality=quality, Image=Image))
                if len(out) >= count:
                    break
            seen += 1
        return out
    except MediaError:
        raise
    except Exception as exc:
        raise MediaError(f"failed to decode frames: {exc}") from exc
    finally:
        if container is not None:
            _close(container, source)


def extract_preview(source: Any, max_side: int = 512, fmt: str = "jpeg",
                    quality: int = 82) -> str | None:
    """A single representative frame (middle) as base64, or None if unavailable."""
    frames = extract_frames(source, count=3, max_side=max_side,
                            fmt=fmt, quality=quality)
    if not frames:
        return None
    return frames[len(frames) // 2]


def _sample_ranks(container: Any, vstream: Any, count: int) -> set[int]:
    """Evenly-spaced decoded-frame ranks to keep (0-indexed)."""
    nb = getattr(vstream, "frames", None)
    if not nb:
        return set(range(min(count, 3)))
    if count <= 1:
        return {nb // 2}
    return {int(round(i * (nb - 1) / (count - 1))) for i in range(count)}


def _frame_to_b64(frame: Any, *, max_side: int, fmt: str, quality: int,
                  Image: Any) -> str:
    """Convert an av.VideoFrame to a downscaled base64 data URL."""
    img = frame.to_image()  # requires Pillow
    img.thumbnail((max_side, max_side))
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    if fmt.lower() == "png":
        img.save(buf, format="PNG")
        mime = "image/png"
    else:
        img.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"
