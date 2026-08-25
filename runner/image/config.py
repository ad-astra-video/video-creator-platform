"""Configuration from environment variables (image-worker).

Serves Qwen-Image-Edit, Qwen-Image-Layered and Z-Image. Model roots point at
the /models/image/{edit,layered,zimage} layout produced by
runner/image/download_models.sh and consumed by the Docker image-worker
container (models bind-mounted at /models on the GPU host).

IMPORTANT: the QWEN_* / ZIMAGE_* names below are part of the inter-service
contract — other already-planned code reads them by these exact identifiers.
Do not rename them.
"""
import os

# ── Model roots (on the shared /models bind-mount) ──────────────────────────
# Qwen-Image-Edit-2511 (QwenImageEditPlusPipeline, multi-reference-image edits
# + fp8) is the current edit engine. Its own root keeps it separate from the
# legacy edit weights.
QWEN_EDIT_ROOT = os.environ.get("QWEN_EDIT_ROOT", "/models/image/edit-2511")
QWEN_LAYERED_ROOT = os.environ.get("QWEN_LAYERED_ROOT", "/models/image/layered")
ZIMAGE_ROOT = os.environ.get("ZIMAGE_ROOT", "/models/image/zimage")

# ── Inference knobs ──────────────────────────────────────────────────────────
# Weight precision for the Qwen image pipelines: fp8 (default) | bf16 | int8.
# fp8 loads the transformers via from_pretrained(..., torch_dtype=torch.float8_e4m3fn).
QWEN_DTYPE = os.environ.get("QWEN_DTYPE", "fp8")  # fp8|bf16|int8

# torchao weight-only quant applied after bf16 load: int8 | int4.
# int4 (~14GB for the 20B+7B stack) is what reliably fits a 32GB card.
QWEN_AO_QUANT = os.environ.get("QWEN_AO_QUANT", "int8").strip().lower()

# Enable enable_model_cpu_offload() so only the active submodule sits in VRAM.
QWEN_OFFLOAD = (os.environ.get("QWEN_OFFLOAD", "true").lower() in {"1", "true", "yes"})

# Qwen-Image-Layered weight precision, independent of QWEN_DTYPE (which drives
# the edit pipeline). 'fp8' (default) loads the pre-quantized FP8 E4M3FN
# transformer from T5B/Qwen-Image-Layered-FP8 directly — NO in-flight
# quantization. 'bf16'/'int8' fall back to the bf16-load path (int8 applies
# torchao quantization at load). The transformer file must match.
QWEN_LAYERED_DTYPE = os.environ.get("QWEN_LAYERED_DTYPE", "fp8").strip().lower()  # fp8|bf16|int8

# Default / requested number of decomposition layers (clamped to
# [2, QWEN_MAX_LAYERS] in the server /layer handler).
QWEN_LAYERS = int(os.environ.get("QWEN_LAYERS", "4"))
QWEN_MAX_LAYERS = int(os.environ.get("QWEN_MAX_LAYERS", "16"))

# ── HiDream-O1-Image (8B UiT) ───────────────────────────────────────────────
# HiDream-O1-Image runs in-place on the image-worker alongside
# Qwen/Z-Image/FLUX.2 klein via the vendored `hidream_models/` package. Model
# root is the full checkpoint dir (AutoProcessor + model safetensors).
#
# These are intentionally HARDCODED, not env-driven — the worker always runs
# the expected recipe. Workers should not be able to change steps/guidance at a
# fleet level; a per-request body override is the only knob (see _resolve_steps
# / the /image & /edit handlers).
HIDREAM_ROOT = "/models/image/hidream"
HIDREAM_DTYPE = "bf16"  # bf16|fp32
HIDREAM_MAX_SIDE = 2048  # Max output long-edge (px); clamp request/ref dims.
# Default steps / guidance for T2I when the request provides neither explicit
# steps nor a quality name (fast=20 / balanced=28 / high=50 live in
# HIDREAM_STEP_PRESETS in inference.py).
HIDREAM_STEPS = 50
HIDREAM_GUIDANCE = 5.0

# Default number of Qwen-Image-Layered denoise steps when the request doesn't
# specify num_inference_steps. Quality presets map to: Fast=25, Balanced=30,
# Detailed=50 (the client sends the chosen count; this is the fallback).
QWEN_STEPS = int(os.environ.get("QWEN_STEPS", "30"))

# Longest input-image side the /layer endpoint will try to decompose (px).
QWEN_LAYER_MAX_INPUT_SIDE = int(os.environ.get("QWEN_LAYER_MAX_INPUT_SIDE", "2048"))

# Side (px) of the square preview thumbnails included per layer in the
# /layer response.
QWEN_LAYER_PREVIEW_SIDE = int(os.environ.get("QWEN_LAYER_PREVIEW_SIDE", "256"))

# Approx cap for a full /layer response body (bytes). The server rejects an
# oversized projected layer response with HTTP 413. Default 300 MiB.
QWEN_LAYER_RESPONSE_CAP_BYTES = int(
    os.environ.get("QWEN_LAYER_RESPONSE_CAP_BYTES", str(300 * 1024 * 1024))
)

# ── Server binding ───────────────────────────────────────────────────────────
# Fallback CUDA device index used when a /load request omits `device`.
DEFAULT_DEVICE = int(os.environ.get("GPU_DEVICE", "0"))
PORT = int(os.environ.get("PORT", "8994"))
HOST = os.environ.get("HOST", "0.0.0.0")

APP_ID = "video-creator"

# ── FLUX.2 [klein] 4B image editing (style-frame) ────────────────────────────
# The 4B guidance+step-distilled image-edit model that styles the restyle first
# frame. Three components must be resident to edit: the 4B flow transformer
# (<KLEIN4B_MODEL>), the FLUX.2 autoencoder (<KLEIN4B_AE>), and a Qwen3 4B text
# embedder (<KLEIN4B_TEXT_ENC>). Runs on the same card as the Qwen/Z-Image
# pipelines; the engine evicts those before the editor allocates (see
# inference._evict_other -> 'klein').
#   KLEIN4B_ENABLED   "auto" (use only if <KLEIN4B_MODEL> exists) | "1" | "0"
#                     | "force" (error if absent).
#   KLEIN4B_MODEL / KLEIN4B_AE  bf16 safetensors on the shared /models/flux2 mount.
#   KLEIN4B_TEXT_ENC  HF id of the Qwen3 text embedder (default Qwen/Qwen3-4B, bf16).
#   KLEIN4B_GPU_DEVICE device for the editor (default "" = active engine device).
#   KLEIN4B_STEPS / KLEIN4B_GUIDANCE  distilled defaults (4 / 1.0).
#   KLEIN4B_MAX_SIDE  cap on the styled frame's long edge (default 1920).
#   KLEIN4B_REF_SIDE  cap on the reference image's long edge (default 1024).
KLEIN4B_ENABLED = os.environ.get("KLEIN4B_ENABLED", "auto")
KLEIN4B_MODEL = os.environ.get("KLEIN4B_MODEL", "/models/flux2/flux-2-klein-4b.safetensors")
KLEIN4B_AE = os.environ.get("KLEIN4B_AE", "/models/flux2/ae.safetensors")
KLEIN4B_TEXT_ENC = os.environ.get("KLEIN4B_TEXT_ENC", "Qwen/Qwen3-4B")
KLEIN4B_GPU_DEVICE = os.environ.get("KLEIN4B_GPU_DEVICE", "")
KLEIN4B_STEPS = int(os.environ.get("KLEIN4B_STEPS", "4"))
KLEIN4B_GUIDANCE = float(os.environ.get("KLEIN4B_GUIDANCE", "1.0"))
KLEIN4B_MAX_SIDE = int(os.environ.get("KLEIN4B_MAX_SIDE", "1920"))
KLEIN4B_REF_SIDE = int(os.environ.get("KLEIN4B_REF_SIDE", "1024"))


def klein4b_device() -> str:
    """GPU the FLUX.2 Klein editor runs on (defaults to the active engine device)."""
    if KLEIN4B_GPU_DEVICE:
        return KLEIN4B_GPU_DEVICE
    return f"cuda:{DEFAULT_DEVICE}"


def klein4b_steps() -> int:
    return KLEIN4B_STEPS


def klein4b_guidance() -> float:
    return KLEIN4B_GUIDANCE


def klein4b_enabled() -> bool:
    """Whether the FLUX.2 Klein style-frame editor may be engaged.

    "auto": engage when the flow weight is present on disk. "1"/"force": always
    (a missing weight then surfaces as a load error). "0": never.
    """
    mode = KLEIN4B_ENABLED.strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "force"):
        return True
    return os.path.isfile(KLEIN4B_MODEL)


# ── Worker auth (shared with live-runner) ────────────────────────────────────
# Identical auto-generate pattern to runner/ltx/config.py: persists to the env
# so the live-runner edge and this worker agree; generated stably at first use
# when blank.
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
