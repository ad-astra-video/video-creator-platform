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

# Durations (seconds) offered per fps band. KEPT LEAN: this rides the orchestrator
# heartbeat metadata, which go-livepeer caps at 1024 bytes -- with a multi-worker box
# every worker adds a "{name}_up" flag, so the model_specs budget is tight. 24/48 fps
# keeps the picker useful while leaving headroom against the cap.
_DURATION_CANDIDATES = [4, 6, 8, 10, 12, 16]

# --- Video-create ("Create") duration budget -- VRAM + resolution aware ---
# The runner is the authority on how many "seconds to generate" it can actually run.
# During t2v/i2v generation the GPU holds the resident model weights PLUS activations
# that grow with frames x resolution, so the ceiling is derived from the GPU's VRAM and
# the per-resolution latent area -- NOT a hardcoded list (user-mandated 2026-08).
#
# Calibration anchor (user-validated): a 32 GB RTX 5090 (fp8 LTX-2.3, ~11 GB resident)
# generates up to 8 s at its top resolution (1080p) = 192 latent frames at 24 fps.
# Everything else scales from that reference by activation budget (VRAM minus the
# constant weights/overhead) and pixel area. A nominal 24 fps sets the seconds budget;
# both advertised fps bands share the same seconds list (48 fps is an output-rate
# option, not a doubling of the generation latent's frame count).
_MODEL_WEIGHTS_MB = 11 * 1024            # fp8 DiT resident during generation
_BASE_OVERHEAD_MB = 2 * 1024             # CUDA context / allocator pool / conditioning
_REFERENCE_VRAM_MB = 32 * 1024
_REFERENCE_RES = "1080p"
_REFERENCE_SECONDS = 8.0
_REFERENCE_FPS = 24
_MIN_GENERATION_SECONDS = 4.0
_MAX_GENERATION_SECONDS = 16.0           # largest duration we ever advertise


def _activation_budget_mb(vram_mb: int) -> float:
    """VRAM available for activations after the resident weights + fixed overhead."""
    return max(1.0, vram_mb - _MODEL_WEIGHTS_MB - _BASE_OVERHEAD_MB)


def _max_generation_seconds(vram_mb: int, res: str) -> float:
    """Longest video-create clip this GPU can run at *res* (VRAM/area estimate)."""
    area = RESOLUTION_DIMS[res][0] * RESOLUTION_DIMS[res][1]
    ref_w, ref_h = RESOLUTION_DIMS[_REFERENCE_RES]
    ref_area = ref_w * ref_h
    ref_frames = _REFERENCE_SECONDS * _REFERENCE_FPS
    budget_ratio = _activation_budget_mb(vram_mb) / _activation_budget_mb(_REFERENCE_VRAM_MB)
    max_frames = ref_frames * budget_ratio * (ref_area / area)
    secs = max_frames / _REFERENCE_FPS
    return min(_MAX_GENERATION_SECONDS, max(_MIN_GENERATION_SECONDS, secs))


def _durations_for(vram_mb: int, res: str) -> list[int]:
    """Advertised duration options at *res*, capped to the VRAM activation ceiling."""
    cap = _max_generation_seconds(vram_mb, res)
    return [d for d in _DURATION_CANDIDATES if d <= cap]


def _max_resolution_for_vram(vram_mb: int) -> str:
    """Highest resolution this GPU can safely generate (mirrors gpu_profile)."""
    gb = vram_mb / 1024.0
    if gb < 15:
        return "540p"
    if gb < 31:
        return "720p"
    return "1080p"


def build_model_specs(vram_mb: int) -> list[dict]:
    """Build the LTX-2.3 spec list limited to what this runner's GPU can do.

    ``vram_mb`` is the GPU's total VRAM in MiB, detected at runtime from the
    GPU-visible worker (the thin live-runner has no direct GPU/torch access) --
    NOT read from an env var. Resolutions are capped at the GPU profile's max
    (540p/720p/1080p) and durations at the VRAM + resolution activation ceiling.
    ``fast`` is the only advertised pipeline -- the runner runs a single LTX-2.3
    engine; ``pro`` would overstate a model we don't actually serve. A2V is
    advertised as unsupported (``None``) to match the LTX worker's
    ``not_supported`` route.
    """
    max_res = _max_resolution_for_vram(vram_mb)
    # Order resolutions ascending; stop once we reach the GPU's max.
    ordered = list(RESOLUTION_DIMS)
    resolutions = ordered[: ordered.index(max_res) + 1]

    # 24fps is the ONLY advertised fps band. 48fps is an output-rate nicety, not a
    # generation change (the worker treats it as output-rate, not a doubling of latent
    # frames), and every extra fps key costs precious bytes in the go-livepeer 1024-byte
    # heartbeat metadata cap -- which is already ~961 bytes with 2.3 alone. Dropping the
    # 48fps band keeps the picker useful while leaving room for the ltx-2.5 marker below.
    supported = {
        res: {
            "fps_to_durations": {
                "24": _durations_for(vram_mb, res),
            }
        }
        for res in resolutions
    }
    # LTX-2.5 companion entry. The 2.5 transformer rides the SAME 22B fp8-cast weights as
    # 2.3 and the SAME VRAM/activation budget, so it uses the IDENTICAL resolution / fps /
    # duration limits -- but we do NOT duplicate the matrix here: the whole model_specs blob
    # rides the orchestrator heartbeat metadata, which go-livepeer caps at 1024 bytes, and a
    # second full matrix would blow past it (1043 bytes vs the cap). Instead we advertise a
    # minimal 2.5 marker and the frontend (video-generation-model-specs.ts) resolves 2.5's
    # options by ALIASING this "fast" (2.3) limits block -- "mark available, use the 2.3
    # limits; don't send a separate set of limits" (user direction 2026-08-18). A2V + extend
    # stay unsupported for 2.5 (the worker routes it, but there's no 2.5 windowed-extend).
    return [
        {
            "pipeline": "fast",
            "spec": {
                "display_name": "LTX-2.3",
                "supported_resolutions_durations": supported,
                # A2V is advertised as unsupported via the ABSENCE of an a2v_* key (the frontend
                # falls back to supported_resolutions_durations); an explicit null key is omitted
                # to keep the heartbeat metadata under the 1024-byte cap.
                # Resolution-aware extend ceiling, driven by this GPU's VRAM budget.
                "extend": build_extend_capability(),
            },
        },
        {
            "pipeline": "ltx-2.5",
            "spec": {
                "display_name": "LTX-2.5",
                # Intentionally NO supported_resolutions_durations -- the frontend aliases the
                # "fast" (2.3) limits above (see video-generation-model-specs.ts). Keeping the
                # matrix here as a duplicate would push the heartbeat metadata over its cap.
            },
        },
        {
            # Bernini is a selectable T2V engine on the wan-worker (NOT an LTX
            # variant). Advertised as a minimal marker (display_name + native
            # 480p/16 hint) so it surfaces in the model picker WITHOUT a matrix;
            # the frontend (lib/bernini-delivery.ts) derives Bernini's full
            # resolution/fps/duration grid from its native 480p@16 + the RIFE /
            # FlashVSR post rails (never the LTX alias). Kept lean for the
            # go-livepeer 1024-byte heartbeat metadata cap.
            "pipeline": "bernini-1.3b",
            "spec": {
                "display_name": "Bernini 1.3B",
            },
        },
        {
            # Bernini 14B (Wan2.2-T2V-A14B), fp8-quantised on-box (69G at
            # /srv/video-creator/models/Bernini-R-Diffusers). Same minimal marker
            # pattern as 1.3B - no matrix (frontend derives grid from
            # lib/bernini-delivery.ts). Advertised so the picker can offer the
            # higher-fidelity "detailed" engine; 14B rejects >1 r2v reference.
            "pipeline": "bernini-14b",
            "spec": {
                "display_name": "Bernini 14B",
            },
        },
    ]
