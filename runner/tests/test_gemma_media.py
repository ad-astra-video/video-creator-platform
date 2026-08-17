"""Tests for runner.gemma.media (PyAV-backed media introspection + frames).

PyAV (and optionally Pillow) are runtime deps of the gemma-worker image; skip
gracefully where they aren't installed (e.g. a bare dev interpreter) so the
suite stays green while still exercising real decode when available.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

av = pytest.importorskip("av")
PILImage = pytest.importorskip("PIL.Image")

from runner.gemma import media  # noqa: E402


def _make_video(seconds: float = 1.5, fps: int = 24,
                w: int = 320, h: int = 240) -> bytes:
    """Synthesize a small h264 mp4 in memory with a frame-varying red channel."""
    out = io.BytesIO()
    c = av.open(out, "w", format="mp4")
    st = c.add_stream("h264", rate=fps)
    st.width, st.height = w, h
    st.pix_fmt = "yuv420p"
    for i in range(int(seconds * fps)):
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = int(i * 8 % 256)
        arr[:, :, 1] = int(i * 4 % 256)
        arr[:, :, 2] = 120
        for pkt in st.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    return out.getvalue()


def _make_image(b64: bool = False) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", (640, 420), (200, 40, 40)).save(buf, format="PNG")
    return buf.getvalue()


def test_probe_video():
    info = media.probe(_make_video())
    assert info["media_type"] == "video"
    assert info["width"] == 320 and info["height"] == 240
    assert info["codec"] == "h264"
    assert abs(info["duration_s"] - 1.5) < 0.2
    assert info["fps"] == 24.0
    assert info["nb_frames"] == 36


def test_probe_image():
    info = media.probe(_make_image())
    assert info["media_type"] == "image"
    assert info["width"] == 640 and info["height"] == 420
    assert info["fps"] is None  # images must not report a bogus fps


def test_probe_by_path_matches_bytes(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(_make_video())
    assert media.probe(str(p)) == media.probe(p.read_bytes())


def test_probe_invalid_source_raises():
    with pytest.raises(media.MediaError):
        media.probe(b"this is not media")


def test_extract_frames_sampling_and_downscale():
    frames = media.extract_frames(_make_video(), count=4, max_side=256)
    assert len(frames) == 4
    for f in frames:
        assert f.startswith("data:image/")
        im = PILImage.open(io.BytesIO(base64.b64decode(f.split(",", 1)[1])))
        assert im.size[0] <= 256 and im.size[1] <= 256


def test_extract_preview():
    assert media.extract_preview(_make_video(), max_side=256) is not None
    # image input: PyAV still reports one video-ish frame -> preview may exist;
    # we only assert it doesn't raise.
    media.extract_preview(_make_image(), max_side=256)
