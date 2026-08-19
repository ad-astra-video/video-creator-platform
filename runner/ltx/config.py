"""Configuration from environment variables."""
import json
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


# ── LTX 2.5 provisioning (download via runner/ltx/download_ltx25.sh) ─────────
# LTX 2.5 is a NEW 22B audio-video model family (Lightricks/LTX-2.5), shipped in
# a ComfyUI-style kit (diffusion_models + gemma4-proj text encoder + video/audio
# VAEs + duration-head patch). The distilled-transformer variant is chosen by
# GPU at download time (NVFP4 on Blackwell sm100/sm120, INT8+ConvRot otherwise),
# driven by LTX25_VARIANT=int8|nvfp4 or auto-detect. These paths only describe
# WHERE the kit lives; the runner does NOT load LTX 2.5 yet (that inference
# forward-port is a separate tracked effort). Blocks that need a 2.5 artifact
# should resolve it here so the layout is single-sourced.
LTX25_MODEL_DIR = os.environ.get("LTX25_MODEL_DIR", "/models/ltx-2.5")
LTX25_VARIANT = os.environ.get("LTX25_VARIANT", "")  # "" = auto (NVFP4 if Blackwell)
LTX25_TRANSFORMER_NVFP4 = "ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"
LTX25_TRANSFORMER_INT8 = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
LTX25_TRANSFORMER_BF16 = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
LTX25_TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
LTX25_VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
LTX25_AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
LTX25_DURATION_HEAD = "ltx-2.5-duration-head-bf16.safetensors"


def ltx25_transformer_filename() -> str:
    """Return the 2.5 distilled-transformer filename for the configured/auto
    variant. Mirrors the downloader's GPU selection: NVFP4 needs Blackwell
    (sm100/sm103/sm120, compute capability >= 10.0). The BF16 variant is loaded
    with the fp8-cast CUDA policy (works on Ada + Blackwell with no ltx-kernels).
    An explicit LTX25_VARIANT always wins; supported: nvfp4|int8|bf16."""
    v = (LTX25_VARIANT or "auto").strip().lower()
    if v == "nvfp4":
        return LTX25_TRANSFORMER_NVFP4
    if v == "int8":
        return LTX25_TRANSFORMER_INT8
    if v == "bf16":
        return LTX25_TRANSFORMER_BF16
    # auto: ask nvidia-smi for the first GPU's compute capability (digits only)
    cc = None
    try:
        import subprocess
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            first = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
            cc = int("".join(ch for ch in first if ch.isdigit()))
    except Exception:
        cc = None
    if cc is not None and cc >= 100:
        return LTX25_TRANSFORMER_NVFP4
    return LTX25_TRANSFORMER_BF16


def ltx25_fp8cast() -> bool:
    """Whether to load the 2.5 transformer with the CUDA fp8-cast policy.

    The NVFP4 / comfy-int8-convrot files carry their own quant headers and are
    NOT consumable by the ltx-pipelines loader directly (it has fp8-cast,
    fp8-scaled-mm and nvfp4 prequant/cast policies only). The BF16 file is the
    one loadable here; fp8-cast shrinks its 22B weights to fit a 32 GB card
    without needing the ltx-kernels nvfp4 extension. An explicit
    LTX25_FP8CAST=0 forces plain (bf16) loading."""
    if os.environ.get("LTX25_FP8CAST", "").strip().lower() in ("0", "false", "no"):
        return False
    return (LTX25_VARIANT or "auto").strip().lower() == "bf16"


# The latent spatial upscaler is a 2.5-specific artifact under
# latent_upscale_models/ (distinct from the 2.3 upscaler in /upscaler).
LTX25_LATENT_UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"

# IC-LoRA Pixel Spatial Upscaler: REQUIRED for LTX-2.5 pixel-space 2x upscaling.
# Lives in its OWN separate GATED repo (a second agreement is needed on HF).
# Downloaded by download_ltx25.sh into <LTX25_MODEL_DIR>/loras/.
LTX25_IC_LORA_PIXEL_UPSCALER_REPO = "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler"
LTX25_IC_LORA_PIXEL_UPSCALER = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"


def ltx25_spatial_upscaler_path() -> str:
    """Resolve the LTX-2.5 LATENT spatial upscaler path, or "" if absent.

    The 2.5 distilled pipeline requires its OWN latent upscaler, downloaded by
    download_ltx25.sh into <LTX25_MODEL_DIR>/latent_upscale_models/. An explicit
    LTX25_LATENT_UPSCALER env wins; otherwise resolve under LTX25_MODEL_DIR and
    the /srv/video-creator/models layout (mirroring the 2.3 upscaler's candidate
    resolution). Returning "" surfaces a missing artifact as a clear runtime
    error from the 2.5 loader rather than a bogus-path load.
    """
    env = os.environ.get("LTX25_LATENT_UPSCALER", "")
    candidates = [
        env,
        f"{LTX25_MODEL_DIR}/latent_upscale_models/{LTX25_LATENT_UPSCALER}",
        f"/srv/video-creator/models/ltx-2.5/latent_upscale_models/{LTX25_LATENT_UPSCALER}",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return ""


if UPSCALER_PATH:
    os.environ.setdefault("UPSCALER_PATH", UPSCALER_PATH)
# GPU_DEVICE may be empty (auto-select via live-runner gpu-pick / nvidia-smi at
# warmup) — treat empty as 0 for the static bit, the server picks the real card.
_dev = (os.environ.get("GPU_DEVICE", "0") or "0").strip()
GPU_DEVICE = int(_dev) if _dev.isdigit() else 0
# The live-runner edge that owns the GPU scheduler. When unset, the worker
# auto-selects the idlest card locally (nvidia-smi); when set, the worker asks
# the live-runner's authoritative /gpu-pick endpoint for a free GPU at warmup.
LIVE_RUNNER_URL = os.environ.get("LIVE_RUNNER_URL", "").strip()
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
ENHANCE_GPU_DEVICE = os.environ.get("ENHANCE_GPU_DEVICE") or None
ENHANCE_FORWARD_URL = os.environ.get("ENHANCE_FORWARD_URL") or None
ENHANCE_FORWARD_MODEL = os.environ.get("ENHANCE_FORWARD_MODEL") or None
ENHANCE_FORWARD_API_KEY = os.environ.get("ENHANCE_FORWARD_API_KEY") or None
ENHANCE_FORWARD_TIMEOUT = float(os.environ.get("ENHANCE_FORWARD_TIMEOUT", "120"))

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

# ------------------------ Custom (user-supplied) LoRAs ------------------------
# Allowlist of hosts the runner will download a *custom* LoRA from (Option A:
# a client-supplied URL). Currently locked to Hugging Face only; a host matches
# if it equals an entry or is a subdomain of an entry. Override with a JSON list
# via LORA_ALLOWED_HOSTS.
_LORA_ALLOWED_HOSTS_RAW = os.environ.get("LORA_ALLOWED_HOSTS", '["huggingface.co"]')
try:
    _lora_allowed = json.loads(_LORA_ALLOWED_HOSTS_RAW)
    if not isinstance(_lora_allowed, list) or not _lora_allowed:
        raise ValueError("must be a non-empty JSON list")
    LORA_ALLOWED_HOSTS = [str(h).lower() for h in _lora_allowed]
except Exception as exc:  # pragma: no cover - config sanity guard
    raise ValueError(f"LORA_ALLOWED_HOSTS invalid: {exc}") from exc

# Max bytes a single custom LoRA download may be (streaming cap + Content-Length
# pre-check). Default 2 GiB.
LORA_MAX_CUSTOM_BYTES = int(float(os.environ.get("LORA_MAX_CUSTOM_BYTES", str(2 * 1024**3))))

# How long an orphaned custom-LoRA temp file may linger before a sweep removes
# it (crash/partial-download cleanup). 10 minutes.
LORA_CUSTOM_TTL_SECONDS = int(os.environ.get("LORA_CUSTOM_TTL_SECONDS", str(10 * 60)))
