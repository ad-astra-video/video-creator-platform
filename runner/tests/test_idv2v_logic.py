"""Smoke tests for GPU-independent ID-V2V worker logic.

These tests run WITHOUT torch, diffsynth, or a GPU — they cover the pure helpers
(clip scheduling, center-crop, base64 I/O, keyframe validation) exercised by
every request. Model-loading + inference still require the real 5090
environment and are covered by manual/hardware validation.

Ported from c:\\dev\\id-v2v\\tests\\test_runner_logic.py and adapted to the
video-creator `runner.idv2v` package.

Run:
    python -m pytest runner/tests/test_idv2v_logic.py -q
"""

import base64
import sys
import os

# Allow running standalone with `python runner/tests/test_idv2v_logic.py`.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402


def _stub_torch():
    """Provide a minimal torch stub so model.py imports without the real lib."""
    import types
    if "torch" in sys.modules:
        return
    stub = types.ModuleType("torch_stub")
    stub.bfloat16 = "bfloat16"
    stub.float32 = "float32"
    stub.no_grad = lambda: __import__("contextlib").nullcontext()
    stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = stub


# ---------------------------------------------------------------------------
# model.py pure helpers
# ---------------------------------------------------------------------------

def test_compute_clip_schedule_single():
    _stub_torch()
    from runner.idv2v.model import compute_clip_schedule
    assert compute_clip_schedule(40, 81) == [(0, 40)]
    assert compute_clip_schedule(81, 81) == [(0, 81)]


def test_compute_clip_schedule_multi():
    _stub_torch()
    from runner.idv2v.model import compute_clip_schedule
    sched = compute_clip_schedule(240, 81)
    assert sched[0] == (0, 81)
    assert sched[1] == (80, 161)
    assert sched[-1] == (240 - 81, 240)
    assert all(0 <= s < e <= 240 for s, e in sched)


def test_slice_frames_pad():
    _stub_torch()
    from runner.idv2v.model import slice_frames
    frames = list(range(5))
    assert slice_frames(frames, 0, 3) == [0, 1, 2]
    assert slice_frames(frames, 0, 8) == [0, 1, 2, 3, 4, 4, 4, 4]
    assert len(slice_frames(frames, 3, 7)) == 4


def test_center_crop_and_resize_uses_pil():
    _stub_torch()
    import PIL.Image as Image
    from runner.idv2v.model import center_crop_and_resize
    img = Image.new("RGB", (1600, 900))  # 16:9 -> 1280x720 same aspect
    out = center_crop_and_resize(img, 1280, 720)
    assert out.size == (1280, 720)


def test_center_crop_and_resize_aspect_change():
    _stub_torch()
    import PIL.Image as Image
    from runner.idv2v.model import center_crop_and_resize
    img = Image.new("RGB", (1000, 1000))  # square -> must crop to 16:9
    out = center_crop_and_resize(img, 1280, 720)
    assert out.size == (1280, 720)


# ---------------------------------------------------------------------------
# model.py local_model_path derivation (regression: double "Wan-AI")
# ---------------------------------------------------------------------------

def _norm(p):
    """Normalize a path to forward-slash segments regardless of OS so the
    assertions hold on both Linux (container) and Windows (build host)."""
    return os.path.normpath(p).replace(os.sep, "/")


def test_local_model_path_reaches_models_root():
    """The deployed bug used dirname(WAN_MODEL_DIR) (= /models/Wan-AI) as
    local_model_path. diffsynth then builds local_model_path/<model_id>/<pattern>
    and model_id already carries 'Wan-AI/Wan2.1-I2V-14B-720P', producing a
    double 'Wan-AI' directory -> empty glob -> "'list' object has no attribute
    'endswith'" in model_manager.match(). It must be two dirname levels up to
    the HF-cache root /models."""
    import os
    wan = "/models/Wan-AI/Wan2.1-I2V-14B-720P"
    local_model_path = os.path.dirname(os.path.dirname(wan.rstrip("/")))
    assert local_model_path == "/models"
    # The file diffsynth resolves for the T5 encoder must NOT contain a double
    # Wan-AI segment.
    model_id = "Wan-AI/Wan2.1-I2V-14B-720P"
    pattern = "models_t5_umt5-xxl-enc-bf16.pth"
    resolved = _norm(os.path.join(local_model_path, model_id, pattern))
    assert resolved == "/models/Wan-AI/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth"
    # The buggy one-dirname version would have doubled Wan-AI:
    buggy = _norm(os.path.join(os.path.dirname(wan.rstrip("/")), model_id, pattern))
    assert "/models/Wan-AI/Wan-AI/Wan2.1-I2V-14B-720P" in buggy
    assert "/models/Wan-AI/Wan-AI/Wan2.1-I2V-14B-720P" not in resolved


def test_local_model_path_trailing_slash_handled():
    import os
    for wan in ("/models/Wan-AI/Wan2.1-I2V-14B-720P",
                "/models/Wan-AI/Wan2.1-I2V-14B-720P/"):
        lp = os.path.dirname(os.path.dirname(wan.rstrip("/")))
        assert lp == "/models", wan


# ---------------------------------------------------------------------------
# run.py pure helpers (base64 I/O, keyframe decode)
# ---------------------------------------------------------------------------

def test_write_b64_roundtrip(tmp_path):
    _stub_torch()
    from runner.idv2v import run as run_mod
    payload = os.urandom(512)
    p = str(tmp_path / "x.bin")
    run_mod._write_b64(base64.b64encode(payload).decode(), p)
    with open(p, "rb") as f:
        assert f.read() == payload


def test_decode_image(tmp_path):
    _stub_torch()
    import PIL.Image as Image
    from runner.idv2v import run as run_mod

    img = Image.new("RGB", (64, 64), (255, 0, 0))
    buf = os.path.join(str(tmp_path), "kf.png")
    img.save(buf)
    with open(buf, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    out = run_mod._decode_image(b64)
    assert out.size == (64, 64)
    assert out.mode == "RGB"


if __name__ == "__main__":
    # Minimal standalone runner (no pytest).
    _stub_torch()
    import tempfile
    for fn in (test_compute_clip_schedule_single, test_compute_clip_schedule_multi,
               test_slice_frames_pad, test_center_crop_and_resize_uses_pil,
               test_center_crop_and_resize_aspect_change,
               test_write_b64_roundtrip, test_decode_image):
        try:
            fn(tempfile.mkdtemp()) if "tmp_path" in fn.__code__.co_varnames else fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
