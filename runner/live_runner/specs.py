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
            },
        }
    ]
