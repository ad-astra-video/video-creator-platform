"""Advertised model-spec metadata for the Livepeer video pipelines.

Sent inside the live-runner's registration/heartbeat ``metadata`` blob so the
Worker's ``GET /api/generate/models-specs`` can populate ``local_models`` (the
resolution / fps / duration options shown in the webapp's video settings picker)
for the Livepeer video path. This is what makes video generation *available* in
the web app: the runner is the authority on its own capabilities, and the Worker
*supplies* whatever the runner advertises instead of hardcoding an empty set.

The shape mirrors the Worker's ``VideoGenerationModelSpecItem`` contract so the
Worker can forward it straight through to the frontend:

    [ { pipeline, spec: {
          display_name,
          supported_resolutions_durations: { "<res>": { fps_to_durations: {"<fps>": [<dur>...]} } },
          a2v_supported_resolutions_durations: <same> | null } } ]
"""
from __future__ import annotations

from . import config

# Resolutions the LTX worker can render, in ascending order (ltx/gpu_profile).
RESOLUTION_DIMS = {
    "540p": (960, 544),
    "720p": (1280, 704),
    "1080p": (1920, 1088),
}

# --- Extend capability (run-time, GPU/Vram-bound) ---
#
# The runner's windowed extend holds the whole [context window + new frames] latent through
# every denoising step, so its VRAM footprint scales with frame count x resolution. The web app
# offers "seconds to add" choices, and those seconds are advertised HERE (the runner is the
# authority on its own ceiling) rather than hardcoded in the Worker/webapp. The active paid
# runner is a 32 GB card: a 4 s / 1080p extend (window 1 s + 4 s = 120 latent frames) succeeds
# while 8 s (216 frames) OOMs, so we budget conservatively at ~132 total latent frames at 1080p
# and scale by pixel area -- lower resolutions get proportionally more seconds.
EXTEND_CONTEXT_SECONDS = 1.0
EXTEND_FPS = 24
EXTEND_MIN_SECONDS = 2.0
EXTEND_MAX_SECONDS = 20.0
_REFERENCE_AREA = RESOLUTION_DIMS["1080p"][0] * RESOLUTION_DIMS["1080p"][1]
_REFERENCE_FRAMES_1080P = 132.0  # total latent frames (window + extend) safe at 1080p
_WINDOW_FRAMES = int(round(EXTEND_CONTEXT_SECONDS * EXTEND_FPS))


def build_extend_capability() -> dict:
    """Resolution -> max extend seconds this GPU can actually run."""
    table: dict[str, int] = {}
    for res, (w, h) in RESOLUTION_DIMS.items():
        area_scaled = _REFERENCE_FRAMES_1080P * (_REFERENCE_AREA / (w * h))
        max_extend_frames = area_scaled - _WINDOW_FRAMES
        secs = max(EXTEND_MIN_SECONDS, min(EXTEND_MAX_SECONDS, max_extend_frames / EXTEND_FPS))
        table[res] = int(secs)
    return {
        "context_window_seconds": EXTEND_CONTEXT_SECONDS,
        "min_duration_seconds": EXTEND_MIN_SECONDS,
        "max_duration_seconds": table,
    }

# Durations (seconds) offered per fps band, mirroring the LTX-2.3 catalog the
# desktop backend advertises for its own fast model.
_FULL_DURATIONS = [6, 8, 10, 12, 14, 16, 18, 20]
_SHORT_DURATIONS = [6, 8, 10]

_FPS_TO_DURATIONS = {
    "24": _FULL_DURATIONS,
    "25": _FULL_DURATIONS,
    "48": _SHORT_DURATIONS,
    "50": _SHORT_DURATIONS,
}


def _max_resolution_for_vram(vram_mb: int) -> str:
    """Highest resolution this GPU can safely generate (mirrors gpu_profile)."""
    gb = vram_mb / 1024.0
    if gb < 15:
        return "540p"
    if gb < 31:
        return "720p"
    return "1080p"


def build_model_specs() -> list[dict]:
    """Build the LTX-2.3 spec list limited to what this runner's GPU can do.

    Resolutions are capped at the GPU profile's max (540p/720p/1080p). ``fast``
    is the only advertised pipeline — the runner runs a single LTX-2.3 engine;
    ``pro`` would overstate a model we don't actually serve. A2V is advertised
    as unsupported (``None``) to match the LTX worker's ``not_supported`` route.
    """
    max_res = _max_resolution_for_vram(config.GPU_VRAM_MB)
    # Order resolutions ascending; stop once we reach the GPU's max.
    ordered = list(RESOLUTION_DIMS)
    resolutions = ordered[: ordered.index(max_res) + 1]

    supported = {
        res: {"fps_to_durations": dict(_FPS_TO_DURATIONS)}
        for res in resolutions
    }
    return [
        {
            "pipeline": "fast",
            "spec": {
                "display_name": "LTX-2.3 (Runner)",
                "supported_resolutions_durations": supported,
                "a2v_supported_resolutions_durations": None,
                # Resolution-aware extend ceiling, driven by this GPU's VRAM budget.
                "extend": build_extend_capability(),
            },
        }
    ]
