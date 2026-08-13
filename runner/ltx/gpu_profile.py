"""GPU-aware runtime profile for the LTX-Desktop runner.

The runner must run on heterogeneous GPUs: RTX 4090 (24 GB, Ada/SM89),
RTX 5090 (32 GB, Blackwell/SM120), and the 96 GB RTX PRO 6000 (Blackwell) it
was originally built for. The generation mode, weight-offload strategy, FP8
quantization, and max supported resolution all depend on how much VRAM is
actually available.

This module mirrors the VRAM policy in LTX-Desktop's
``backend/runtime_config/runtime_policy.py`` and threads the resulting knobs
into the runner engine so the same image runs correctly on any of those GPUs.

VRAM tiers (CUDA):
  - < 15 GB            -> unsupported (local generation not viable)
  - 15-30 GB           -> streaming: weights stream from pinned host RAM
                          (4090 = 24 GB).               OffloadMode.CPU.
  - >= 31 GB           -> full-resident: FP8 transformer (~23 GB) held in
                          VRAM.                          OffloadMode.NONE.
                          (5090 = 32 GB, RTX PRO 6000 = 96 GB)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Literal

logger = logging.getLogger(__name__)

GenerationMode = Literal["streaming", "full", "unsupported"]

# Mirror LTX-Desktop runtime_policy thresholds.
STREAMING_MIN_GB = 15
FULL_RESIDENT_MIN_GB = 31

# streaming_prefetch_count -> ltx_pipelines OffloadMode (see
# LTX-Desktop/backend/services/ltx_pipeline_common.offload_mode_for_prefetch_count).
STREAMING_PREFETCH = 2

# Human-readable resolution constants.
RESOLUTION_DIMS = {
    "540p": (960, 544),
    "720p": (1280, 704),
    "1080p": (1920, 1088),
}

# Max resolution each mode can safely generate. Streaming (24 GB class) can't
# hold the activations for 1080p; full-resident (32 GB+) can.
MAX_RESOLUTION_FOR_MODE: dict[GenerationMode, str] = {
    "streaming": "720p",
    "full": "1080p",
    "unsupported": "540p",
}

# GPU name fragments we recognize, for stable profile reporting.
_NAME_BY_COMPUTE: dict[float, str] = {
    8.9: "Ada (RTX 40-series / 4090 class)",
    12.0: "Blackwell (RTX 50-series / 5090 / RTX PRO 6000 class)",
}


@dataclass
class GPUProfile:
    """Computed runtime profile for the selected GPU."""

    device_index: int
    gpu_name: str
    vram_gb: float
    compute_cap: tuple[int, int] | None

    mode: GenerationMode = "unsupported"
    streaming_prefetch_count: int | None = None
    offload_mode: str = "NONE"
    use_fp8: bool = False
    max_resolution: str = "540p"

    # Extra attributes for ops; kept as a plain dict so callers can attach
    # non-serializable values without breaking the dataclass.
    info: dict = field(default_factory=dict)

    @property
    def supports_generation(self) -> bool:
        return self.mode != "unsupported"

    def __post_init__(self) -> None:
        self.mode, self.streaming_prefetch_count, self.offload_mode, self.use_fp8 = (
            _decide_mode(self.vram_gb)
        )
        self.max_resolution = MAX_RESOLUTION_FOR_MODE[self.mode]


def _decide_mode(vram_gb: float) -> tuple[GenerationMode, int | None, str, bool]:
    """Pick generation mode / offload / fp8 from VRAM. Returns
    (mode, streaming_prefetch_count, offload_mode, use_fp8)."""
    if vram_gb < STREAMING_MIN_GB:
        return "unsupported", None, "NONE", False
    if vram_gb < FULL_RESIDENT_MIN_GB:
        # 24 GB class (4090): stream weights from pinned host RAM.
        return (
            "streaming",
            STREAMING_PREFETCH,
            "CPU",   # ltx_pipelines OffloadMode.CPU on CUDA
            True,    # fp8 still halves transformer bytes; saves host RAM too
        )
    # 32 GB+ (5090 / RTX PRO 6000): full-resident with fp8 (~23 GB transformer).
    return "full", None, "NONE", True


def _query_device(device_index: int) -> tuple[str, float, tuple[int, int] | None]:
    """Query a CUDA device's name, VRAM (GiB) and compute capability.

    Prefers torch for accuracy, falls back to nvidia-smi so the module also
    works before torch has enumerated CUDA.
    """
    # torch path (authoritative)
    try:
        import torch
        if torch.cuda.is_available():
            i = device_index
            props = torch.cuda.get_device_properties(i)
            name = torch.cuda.get_device_name(i)
            vram_gb = props.total_memory / (1024**3)
            cc = (props.major, props.minor)
            return name, vram_gb, cc
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("torch GPU query failed (%s); falling back to nvidia-smi", exc)

    # nvidia-smi fallback
    try:
        import subprocess
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
                "-i", str(device_index),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0 and r.stdout.strip():
            line = r.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in line.split(",")]
            name = parts[0]
            vram_gb = int(parts[1]) / 1024.0 if len(parts) > 1 else 0.0
            cc: tuple[int, int] | None = None
            if len(parts) > 2 and parts[2]:
                try:
                    mj, mn = parts[2].split(".")
                    cc = (int(mj), int(mn))
                except ValueError:
                    cc = None
            return name, vram_gb, cc
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("nvidia-smi GPU query failed: %s", exc)

    return "Unknown GPU", 0.0, None


def build_profile(
    device_index: int,
    vram_gb_override: float | None = None,
    gpu_name_override: str | None = None,
) -> GPUProfile:
    """Build the runtime profile for ``device_index``.

    ``GPU_VRAM_GB`` / ``GPU_NAME`` env overrides let an operator pin the values
    (useful under docker where the inner GPU index may differ from the host's).
    """
    if vram_gb_override is None:
        vram_gb_override = float(
            os.environ.get("GPU_VRAM_GB", "0") or 0
        ) or None
    if gpu_name_override is None:
        gpu_name_override = os.environ.get("GPU_NAME") or None

    if vram_gb_override and gpu_name_override:
        name, vram_gb, cc = gpu_name_override, float(vram_gb_override), None
    else:
        name, vram_gb, cc = _query_device(device_index)
        if vram_gb_override:
            vram_gb = float(vram_gb_override)
        if gpu_name_override:
            name = gpu_name_override

    profile = GPUProfile(
        device_index=device_index,
        gpu_name=name,
        vram_gb=vram_gb,
        compute_cap=cc,
    )
    profile.info = {
        "gpu_name": name,
        "vram_gb": round(vram_gb, 1),
        "compute_cap": f"{cc[0]}.{cc[1]}" if cc else None,
        "mode": profile.mode,
        "streaming_prefetch_count": profile.streaming_prefetch_count,
        "offload_mode": profile.offload_mode,
        "use_fp8": profile.use_fp8,
        "max_resolution": profile.max_resolution,
    }
    logger.info(
        "GPU[%d] %s | VRAM %.1f GiB | cc=%s | mode=%s | offload=%s | fp8=%s | max_res=%s",
        device_index, name, vram_gb,
        f"{cc[0]}.{cc[1]}" if cc else "n/a",
        profile.mode, profile.offload_mode, profile.use_fp8, profile.max_resolution,
    )
    return profile


def clamp_resolution(profile: GPUProfile, resolution: str) -> str:
    """Clamp a requested resolution down to what the GPU profile can handle.

    Returns the requested value if it's at/below the profile's max, otherwise
    the profile's max. Unknown requested resolutions pass through unchanged so
    callers keep their existing fallback behaviour.
    """
    if resolution not in RESOLUTION_DIMS:
        return resolution
    profile_rank = _rank(RESOLUTION_DIMS, profile.max_resolution)
    requested_rank = _rank(RESOLUTION_DIMS, resolution)
    if requested_rank > profile_rank:
        return profile.max_resolution
    return resolution


def _rank(mapping: dict, key: str) -> int:
    return list(mapping).index(key) if key in mapping else -1
