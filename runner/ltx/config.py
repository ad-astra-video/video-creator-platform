"""Configuration from environment variables."""
import os

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://localhost:8935")
ORCHESTRATOR_SECRET = os.environ.get("ORCHESTRATOR_SECRET", "abcdef")
RUNNER_URL = os.environ.get("RUNNER_URL", "http://0.0.0.0:8991")
IDV2V_WORKER_URL = os.environ.get("IDV2V_WORKER_URL", "http://idv2v-worker:8992")
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "/models/checkpoint")
TEXT_ENCODER_ROOT = os.environ.get("TEXT_ENCODER_ROOT", "/models/gemma")

# LTX-2.3 spatial upscaler (the 480/360p -> 720p step for restyle). The download
# scripts place it at <MODELS_DIR>/upscaler/, but some devices have it under
# /models/upsampler/ — so resolve both, plus /srv/video-creator/models. An
# explicit UPSCALER_PATH env always wins. If no candidate exists we fall back to
# "" rather than a bogus path: DistilledPipeline would try to load it eagerly and
# crash the whole worker on boot. The /upscale endpoint then surfaces a clear
# "upscaler not loaded" error instead.
_MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
_UPSCALER_FILENAME = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
_UPSCALER_ENV = os.environ.get("UPSCALER_PATH", "")
_UPSCALER_CANDIDATES = [
    _UPSCALER_ENV,
    f"{_MODELS_DIR}/upscaler/{_UPSCALER_FILENAME}",
    f"{_MODELS_DIR}/upsampler/{_UPSCALER_FILENAME}",
    f"/srv/video-creator/models/upscaler/{_UPSCALER_FILENAME}",
    f"/srv/video-creator/models/upsampler/{_UPSCALER_FILENAME}",
]


def _resolve_upscaler_path() -> str:
    for p in _UPSCALER_CANDIDATES:
        if p and os.path.exists(p):
            return p
    return ""


UPSCALER_PATH = _resolve_upscaler_path()
if UPSCALER_PATH:
    os.environ.setdefault("UPSCALER_PATH", UPSCALER_PATH)
GPU_DEVICE = int(os.environ.get("GPU_DEVICE", "0"))
PORT = int(os.environ.get("PORT", "8991"))
WARMUP = os.environ.get("WARMUP", "true").lower() == "true"
PRICE = float(os.environ.get("PRICE", "0.5"))
PRICE_UNIT = os.environ.get("PRICE_UNIT", "fixed")
HOST = os.environ.get("HOST", "0.0.0.0")

# ── Worker auth (shared with live-runner) ────────────────────────────────────
# Work persists to the env so the live-runner and worker agree. Auto-generated
# at first use when blank (see _worker_token()).
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")


def _random_token():
    import random
    import string
    return "".join(random.choices(string.ascii_letters + string.digits, k=32))


def worker_token() -> str:
    """Return the worker auth token, auto-generating a stable one if blank."""
    global WORKER_TOKEN
    if not WORKER_TOKEN:
        WORKER_TOKEN = os.environ["WORKER_TOKEN"] = _random_token()
    return WORKER_TOKEN


# Optional GPU profile overrides (see runner.gpu_profile) — useful under docker
# where the container's CUDA index may differ from the host's.
GPU_VRAM_GB = os.environ.get("GPU_VRAM_GB") or None   # e.g. "24" or "32"
GPU_NAME = os.environ.get("GPU_NAME") or None          # display name

# ── Prompt-enhancement backend ──────────────────────────────────────────────
# Enhancement runs the local Gemma encoder by default on GPU_DEVICE.
#   * ENHANCE_GPU_DEVICE  — run the local Gemma on a DIFFERENT GPU than the
#     video pipeline (e.g. "1"), so enhancement VRAM never contends with the
#     resident diffusion pipeline.
#   * ENHANCE_FORWARD_URL — bypass the local Gemma entirely and proxy
#     /prompt-enhance to a shared OpenAI-compatible chat-completions endpoint
#     (`<url>/v1/chat/completions`, e.g. one llama.cpp instance serving many
#     runners). When set, this wins over ENHANCE_GPU_DEVICE and the local Gemma
#     is never loaded.
#   * ENHANCE_FORWARD_MODEL / ENHANCE_FORWARD_API_KEY — optional model id and
#     Bearer key for the upstream endpoint.
#   * ENHANCE_FORWARD_TIMEOUT — upstream request timeout in seconds.
#   * ENHANCE_T2V_SYSTEM_PROMPT / ENHANCE_I2V_SYSTEM_PROMPT — optional override
#     of the default system prompt sent upstream when a request has none.
ENHANCE_GPU_DEVICE = os.environ.get("ENHANCE_GPU_DEVICE") or None
ENHANCE_FORWARD_URL = os.environ.get("ENHANCE_FORWARD_URL") or None
ENHANCE_FORWARD_MODEL = os.environ.get("ENHANCE_FORWARD_MODEL") or None
ENHANCE_FORWARD_API_KEY = os.environ.get("ENHANCE_FORWARD_API_KEY") or None
ENHANCE_FORWARD_TIMEOUT = float(os.environ.get("ENHANCE_FORWARD_TIMEOUT", "120"))
ENHANCE_T2V_SYSTEM_PROMPT = os.environ.get("ENHANCE_T2V_SYSTEM_PROMPT") or None
ENHANCE_I2V_SYSTEM_PROMPT = os.environ.get("ENHANCE_I2V_SYSTEM_PROMPT") or None

# ── LoRA cache (catalog download + disk budget + LRU eviction) ──────────────
LORA_CACHE_DIR = os.environ.get("LORA_CACHE_DIR", "/models/loras")
# Operator-controlled disk budget for downloaded LoRAs, in GiB.
LORA_CACHE_SIZE_GB = float(os.environ.get("LORA_CACHE_SIZE_GB", "2.0"))
# Source of the LoRA catalog (lora_catalog.json). Defaults to the raw file in
# the main LTX-Desktop repo so the runner needs no manually-shipped catalog;
# override with a URL or a local file path for a curated/offline catalog.
LORA_CATALOG_SOURCE = (
    os.environ.get("LORA_CATALOG_SOURCE")
    or "https://raw.githubusercontent.com/ad-astra-video/LTX-Desktop/main/"
    "backend/runtime_config/lora_catalog.json"
)
# (A set-but-empty value falls back to the default above.)
# Optional token for gated catalog repos (falls back to HF_TOKEN).
LORA_HF_TOKEN = os.environ.get("LORA_HF_TOKEN") or os.environ.get("HF_TOKEN") or None
