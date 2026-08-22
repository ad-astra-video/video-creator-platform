"""Video-processing worker (vp-worker) config.

Dedicated low-VRAM post-process container: RIFE motion-preserving fps-boost,
FlashVSR diffusion upscale, and general ffmpeg — plus standalone SAM3.
Consumed only by the vp-worker server; render workers never call in (the
live-runner orchestrates a combined /process stage after any render).
"""

from __future__ import annotations

import os

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8995"))
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(512 * 1024 * 1024)))

WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

GPU_DEVICE = os.environ.get("GPU_DEVICE", "cuda:0")

# ---- RIFE (motion-preserving fps-boost) ------------------------------------
RIFE_VERSION = os.environ.get("RIFE_VERSION", "4.26")
RIFE_ROOT = os.environ.get("RIFE_ROOT", "/models/rife")
# weights live at <RIFE_ROOT>/<something>/flownet.pkl after unzip; scanned at load.
RIFE_FPS_MODES = ("preserve_motion", "smooth")
# Target fps options the client may request (>= source; boost rail only).
RIFE_ALLOWED_TARGET_FPS = [int(x) for x in
                           os.environ.get("RIFE_ALLOWED_TARGET_FPS",
                                          "24,30,60").split(",")]

# ---- FlashVSR (diffusion upscale) -------------------------------------------
FLASHVSR_ENABLED = os.environ.get("FLASHVSR_ENABLED", "auto")
FLASHVSR_ROOT = os.environ.get("FLASHVSR_ROOT", "/models/flashvsr")
FLASHVSR_MODE = os.environ.get("FLASHVSR_MODE", "tiny")
FLASHVSR_SCALE = int(os.environ.get("FLASHVSR_SCALE", "4"))

# ---- SAM3 (standalone shared route) -----------------------------------------
SAM3_CKPT = os.environ.get("SAM3_CKPT", "/models/sam3")
SAM3_PROMPT = os.environ.get("SAM_PROMPT", "object to keep")


def worker_token() -> str:
    """Shared X-Worker-Token (matching every worker in the compose stack)."""
    return WORKER_TOKEN


def flashvsr_enabled() -> bool:
    v = FLASHVSR_ENABLED.strip().lower()
    if v in ("0", "false", "no", "off"):
        return False
    return os.path.isdir(FLASHVSR_ROOT)


def rife_enabled() -> bool:
    return os.path.isdir(RIFE_ROOT)


def _random_token() -> str:
    import secrets
    return secrets.token_hex(16)
