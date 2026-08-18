"""Configuration from environment variables for the gemma-worker LLM container.

This is a dedicated, small worker (sibling to ltx-worker / idv2v-worker) that
serves the always-on-idle Gemma 4 LLM backend via llama.cpp. It exposes the
standard root control surface (/health, /load, /evict) the live-runner's swap
policy drives, plus two inference routes: /video-creator/v1/prompt-enhance and
/video-creator/v1/chat (the future frontend-agent endpoint).

GPU story (drives the live-runner's evict decision, not this container):
  GEMMA_GPU_DEVICE = "" (BLANK)   -> shares the video GPU. The live-runner
                                      treats this worker as the idle-resident,
                                      EVICTABLE slot (loaded when the GPU is
                                      free, evicted for any render task).
  GEMMA_GPU_DEVICE = "1"/"cuda:1" -> a DEDICATED GPU. The live-runner pins it
                                      resident and NEVER evicts it; render tasks
                                      run elsewhere.
"""

from __future__ import annotations

import os

# Worker identity + auth (shared with the live-runner edge).
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

# ── Model (already downloaded; bind-mounted, never baked into the image) ─────
# Defaults point at the official Google Gemma 4 12B it QAT GGUF, placed under
# the shared /models/gemma mount. Override GEMMA_MODEL / GEMMA_MMPROJ for a
# different quant or the ggml-org any-to-any build.
GEMMA_MODEL = os.environ.get(
    "GEMMA_MODEL", "/models/gemma/gemma-4-12b-it-qat-q4_0.gguf"
)
GEMMA_MMPROJ = os.environ.get("GEMMA_MMPROJ", "")

# ── GPU ──────────────────────────────────────────────────────────────────────
# BLANK = shared video GPU (evictable idle slot); SET = dedicated (never evict).
GEMMA_GPU_DEVICE = os.environ.get("GEMMA_GPU_DEVICE", "")
# PHYSICAL GPU index this container is pinned to (host-visible index, NOT the
# remapped cuda:0). The live-runner threads this in from GEMMA_RESIDENT_GPU; it
# is the value gemma-worker reports as device_in_use so the scheduler knows which
# physical card it owns. (Because CUDA_VISIBLE_DEVICES pins the container to one
# card, the in-container index is always 0 regardless of host index.)
GEMMA_PHYSICAL_GPU = int(os.environ.get("GEMMA_RESIDENT_GPU", "0") or 0)
# llama.cpp layers to offload to the GPU. -1 = all (dedicated GPU, fastest);
# 0 = CPU-only (safe on a contended shared GPU). Overridable per deployment.
GEMMA_N_GPU_LAYERS = int(os.environ.get("GEMMA_N_GPU_LAYERS", "-1"))

# ── Concurrency ──────────────────────────────────────────────────────────────
# Max concurrent prompt executions the LLM backend admits. Actual GPU eval is
# serialized on the single shared context; this caps simultaneous in-flight
# prompt requests (queued beyond this).
GEMMA_MAX_PARALLEL = int(os.environ.get("GEMMA_MAX_PARALLEL", "3"))

# ── HTTP ─────────────────────────────────────────────────────────────────────
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8993"))  # distinct from ltx 8991 / idv2v 8992
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "3000000000"))
N_CTX = int(os.environ.get("GEMMA_N_CTX", "131072"))


def gemma_device_index() -> int:
    """CUDA index for llama.cpp ``main_gpu``, parsing ``cuda:N`` or ``N``."""
    d = (GEMMA_GPU_DEVICE or "0").strip()
    if d.startswith("cuda:"):
        d = d.split(":", 1)[1] or "0"
    try:
        return int(d)
    except ValueError:
        return 0


def is_dedicated_gpu() -> bool:
    """True when a specific GPU is named -> live-runner pins (never evicts)."""
    return bool(GEMMA_GPU_DEVICE and GEMMA_GPU_DEVICE.strip())


def _random_token() -> str:
    import random
    import string
    return "".join(random.choices(string.ascii_letters + string.digits, k=32))


def worker_token() -> str:
    """Return the worker auth token, auto-generating a stable one if blank."""
    global WORKER_TOKEN
    if WORKER_TOKEN:
        return WORKER_TOKEN
    if not WORKER_TOKEN:
        WORKER_TOKEN = os.environ.setdefault("WORKER_TOKEN", _random_token())
    return WORKER_TOKEN
