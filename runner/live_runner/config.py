"""Configuration from environment variables for the live-runner edge."""

import os
import random
import string

# Livepeer Orchestrator registration.
ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8935")
ORCHESTRATOR_SECRET = os.environ.get("ORCHESTRATOR_SECRET", "abcdef")
RUNNER_URL = os.environ.get("RUNNER_URL", "http://0.0.0.0:8991")
APP_ID = os.environ.get("APP_ID", "video-creator")
PRICE = float(os.environ.get("PRICE", "0"))
PRICE_UNIT = os.environ.get("PRICE_UNIT", "fixed")
PRICE_CURRENCY = (os.environ.get("PRICE_CURRENCY", "usd") or "usd").strip().lower()
GPU_NAME = os.environ.get("GPU_NAME", "RTX 5090")
# GPU VRAM is DETECTED at runtime from the GPU-visible workers (see server.py
# _poll_worker_gpu_info), NOT read from an env var (user-mandated 2026-08).
# This is a conservative fallback used only until runtime detection succeeds.
DEFAULT_VRAM_MB = 32768
HEARTBEAT_INTERVAL_S = float(os.environ.get("HEARTBEAT_INTERVAL_S", "5"))

# Worker endpoints (internal Docker network, by service name).
LTX_WORKER_URL = os.environ.get("LTX_WORKER_URL", "http://ltx-worker:8991")
IDV2V_WORKER_URL = os.environ.get("IDV2V_WORKER_URL", "http://idv2v-worker:8992")
GEMMA_WORKER_URL = os.environ.get("GEMMA_WORKER_URL", "http://gemma-worker:8993")
IMAGE_WORKER_URL = os.environ.get("IMAGE_WORKER_URL", "http://image-worker:8994")

# Plan B — GPU concurrency scheduler (one task -> one GPU, concurrent across GPUs).
# -----------------------------------------------------------------------------
# The box's GPU count (Docker gives every worker container access to ALL GPUs;
# the live-runner picks which single GPU each task runs on and sends it in the
# /load body). Default 3 (the .151 box is 3x RTX 5090).
GPU_COUNT = int(os.environ.get("GPU_COUNT", "3"))
# The GPU the gemma-worker loads on at startup and STAYS resident on. The
# scheduler holds this GPU out of the task pool (marks it resident for gemma).
GEMMA_RESIDENT_GPU = int(os.environ.get("GEMMA_RESIDENT_GPU", "0"))
# How long a task waits FIFO for a GPU to free up before timing out with 503.
SCHEDULER_QUEUE_TIMEOUT_S = float(os.environ.get("SCHEDULER_QUEUE_TIMEOUT_S", "600.0"))

# Gemma LLM backend residency (drives the swap policy + idle backfill).
#   LLM_GPU_DEVICE = "" (BLANK)  -> LLM shares the video GPU: idle-resident
#                                   EVICTABLE slot (evicted for any render task).
#   LLM_GPU_DEVICE = "1"/"cuda:1"-> LLM runs on a dedicated GPU: PINNED, NEVER evicted.
LLM_GPU_DEVICE = os.environ.get("LLM_GPU_DEVICE", "")
LLM_PINNED = bool(LLM_GPU_DEVICE and LLM_GPU_DEVICE.strip())
# Idle grace before the shared GPU backfills the LLM over a warm render worker.
GEMMA_IDLE_GRACE_S = float(os.environ.get("GEMMA_IDLE_GRACE_S", "20.0"))

# HTTP server.
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8991"))
_ROOT_PATH = "/video-creator/v1"

# Worker auth (shared with both workers).
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")


def _random_token() -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=32))


def worker_token() -> str:
    """Return the shared worker auth token, auto-generating a stable one if blank."""
    global WORKER_TOKEN
    if not WORKER_TOKEN:
        WORKER_TOKEN = os.environ["WORKER_TOKEN"] = _random_token()
    return WORKER_TOKEN


# Worker -> service-URL table (used by routing).
WORKERS = {
    "ltx-worker": LTX_WORKER_URL,
    "idv2v-worker": IDV2V_WORKER_URL,
    "gemma-worker": GEMMA_WORKER_URL,
    "image-worker": IMAGE_WORKER_URL,
}
