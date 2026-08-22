"""Configuration from environment variables for the ID-V2V worker."""

import os

# Worker identity + auth (shared with the live-runner edge).
# Auto-generated if blank at startup (see server.py).
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "")

# Model paths / knobs.
# IDV2V_SOURCE selects where the DiT+VACE weights come from:
#   "local"  -> MODEL_CHECKPOINT (idv2v.pth) then runtime int8/bf16 quant (legacy)
#   "hf-fp8" -> download pre-quantized per-channel FP8 from HF_REPO (no runtime int8)
IDV2V_SOURCE = os.environ.get("IDV2V_SOURCE", "local")
HF_REPO = os.environ.get("HF_REPO", "ad-astra-video/id-v2v-fp8")
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN", os.environ.get("HF_TOKEN", ""))
MODEL_CHECKPOINT = os.environ.get("MODEL_CHECKPOINT", "/models/idv2v.pth")

# Model variant for the hf-fp8 source. Each id-v2v "model" carries a weight
# folder plus a matching default denoise-step budget:
#   "fast"    -> local <MODEL_DIR>/fusionx  (FusionX I2V-14B LoRA fused + fp8;
#               ~8 steps). Folder name "fusionx" also names the HF subfolder it
#               syncs from when absent on disk.
#   "regular" -> HF_REPO root pre-quantized fp8 (original id-v2v, 30 steps).
# IDV2V_HF_SUBFOLDER, when set, overrides the folder used by every variant (e.g.
# a deploy that permanently pins one model). IDV2V_MODEL_VARIANT selects the
# default variant served when a restyle request omits `model`.
MODEL_VARIANTS = {
    "fast":    {"subfolder": "fusionx", "steps": 8},
    "regular": {"subfolder": "",        "steps": 30},
}
DEFAULT_MODEL_VARIANT = os.environ.get("IDV2V_MODEL_VARIANT", "fast")
IDV2V_HF_SUBFOLDER = os.environ.get("IDV2V_HF_SUBFOLDER", "")
# Base dir that holds the per-variant local weight folders (fast -> /models/fusionx).
IDV2V_MODEL_DIR = os.environ.get("IDV2V_MODEL_DIR", "/models")


def _norm_variant(variant: str) -> str:
    v = str(variant or "").strip().lower()
    return v if v in MODEL_VARIANTS else DEFAULT_MODEL_VARIANT


def subfolder_for(variant: str) -> str:
    """Weight folder (relative to IDV2V_MODEL_DIR) for `variant` ("" = HF repo root)."""
    if IDV2V_HF_SUBFOLDER:
        return IDV2V_HF_SUBFOLDER
    return MODEL_VARIANTS.get(_norm_variant(variant), {}).get("subfolder", "")


def steps_for(variant: str) -> int:
    """Default denoise-step budget for `variant` when a request omits steps."""
    return MODEL_VARIANTS.get(_norm_variant(variant), {}).get("steps", 30)

# HF-cache-style repo dir holding Wan2.1-I2V-14B-720P (T5/VAE/CLIP/tokenizer +
# DiT shards). diffsynth resolves files as <parent>/Wan-AI/Wan2.1-I2V-14B-720P/<pattern>,
# so local_model_path = dirname(WAN_MODEL_DIR) = /models.
WAN_MODEL_DIR = os.environ.get("WAN_MODEL_DIR", "/models/Wan-AI/Wan2.1-I2V-14B-720P")
SAM3_CKPT = os.environ.get("SAM3_CKPT", "/models/sam3")
SAM_PROMPT = os.environ.get("SAM_PROMPT", "person")
SKIP_SAM3 = os.environ.get("IDV2V_SKIP_SAM3", "0").lower() in {"1", "true", "yes"}

# GPU + runtime knobs.
GPU_DEVICE = os.environ.get("GPU_DEVICE", "cuda:0")
# The live-runner edge's base URL, used at startup to ask its authoritative
# scheduler (POST /video-creator/v1/gpu-pick) which physical GPU is free so this
# worker doesn't blindly default to GPU 0 (which the image worker may hold).
# Blank = no live-runner consult -> fall back to GPU_DEVICE / local select.
LIVE_RUNNER_URL = os.environ.get("LIVE_RUNNER_URL", "").strip()
GPU_NAME = os.environ.get("GPU_NAME", "RTX 5090")
GPU_VRAM_GB = float(os.environ.get("GPU_VRAM_GB", "32"))
IDV2V_QUANT = os.environ.get("IDV2V_QUANT", "int8")       # int8 | none | bf16
IDV2V_OFFLOAD = os.environ.get("IDV2V_OFFLOAD", "true").lower() in {"1", "true", "yes"}
IDV2V_VRAM_BUFFER = int(os.environ.get("IDV2V_VRAM_BUFFER", "10"))
# IDV2V_STAGED: staged RAM lifecycle — keep T5 resident only for the first text
# encode, then free it (11 GB) BEFORE loading the native-fp8 DiT+VACE. CLIP/VAE
# stay resident with the DiT (total ~22 GB -> fits the 28 GB available on .8),
# so the worker no longer needs to int8-quantize the T5/CLIP to fit.
IDV2V_STAGED = os.environ.get("IDV2V_STAGED", "true").lower() in {"1", "true", "yes"}

# Gemma 3 LLM support (prompt enhancement + automatic video captioning).
# Reuses the already-provisioned Lightricks/gemma-3-12b-it-qat-q4_0-unquantized
# checkpoint on the shared /models/gemma mount (the LTX runner drops it there).
# Gemma runs on the SAME GPU as the video model (GPU_DEVICE) — the two cannot
# coexist on one card (~19.5 GB id-v2v + ~24.5 GB Gemma > 32 GB), so the caller
# evicts the resident id-v2v model before Gemma loads (see gemma.py evict_cb).
#   GEMMA_ROOT       path to the Gemma 3 checkpoint (default /models/gemma)
#   GEMMA_GPU_DEVICE device Gemma loads on (default "" = GPU_DEVICE, i.e. the
#                    same card as the video model; set e.g. cuda:1 to override)
#   GEMMA_ATTN_IMPL  attention implementation for the LLM (default eager —
#                    portable, no flash_attn dependency in this image)
# NOTE: the embedded Gemma 3 is ALWAYS available as the local improve/enhance
# fallback (see config.gemma_enabled()) — there is intentionally NO env switch
# to disable it. It only ever loads lazily when the shared gemma-worker
# (GEMMA_FORWARD_URL) is unreachable.
GEMMA_ROOT = os.environ.get("GEMMA_ROOT", "/models/gemma")
GEMMA_GPU_DEVICE = os.environ.get("GEMMA_GPU_DEVICE", "")
GEMMA_ATTN_IMPL = os.environ.get("GEMMA_ATTN_IMPL", "eager")
# Forward prompt-enhance + auto-caption to the SHARED gemma-worker instead of
# loading the embedded Gemma 3. When set, the worker POSTs enhance/caption
# requests to <GEMMA_FORWARD_URL>/video-creator/v1/prompt-enhance (the shared
# llama.cpp Gemma 4 worker; on the compose network http://gemma-worker:8993)
# and does NOT load its own Gemma 3 (saves ~24.5 GB + the shared-GPU eviction
# choreography). Falls back to the embedded Gemma 3 if the target is
# unreachable (config.gemma_enabled() still governs that fallback).
GEMMA_FORWARD_URL = os.environ.get("GEMMA_FORWARD_URL", "").strip()


def gemma_device() -> str:
    """GPU the Gemma LLM loads on.

    Defaults to the SAME card as the video model (``GPU_DEVICE``). Because the
    id-v2v DiT/VACE and Gemma both need to share one GPU, the worker must evict
    the resident video model first (see gemma.py's evict hook) — callers that
    load Gemma are responsible for that eviction.
    """
    return GEMMA_GPU_DEVICE or GPU_DEVICE


def gemma_enabled() -> bool:
    """Whether the embedded Gemma LLM fallback is available for enhance/caption.

    ALWAYS enabled (hardcoded — no env override, by design). The embedded Gemma 3
    is the local fallback the worker uses whenever the shared gemma-worker
    (GEMMA_FORWARD_URL) is unreachable. It is loaded lazily only when that
    fallback is actually invoked; if the checkpoint (GEMMA_ROOT) is missing, the
    load raises and the caller degrades gracefully (non-fatal, original prompt
    preserved).
    """
    return True


def gemma_forward_base() -> str:
    """Base URL of the shared gemma-worker (trailing slash stripped); ``""`` = off.

    When non-empty, the worker prefers forwarding enhance/caption to this
    gemma-worker instead of loading its own embedded Gemma 3.
    """
    return GEMMA_FORWARD_URL.rstrip("/")


# --- Bernini (ByteDance) t2v/v2v/r2v engines — isolated venv/subprocess (SAM3 pattern). ---
# Engine discriminator (`config.resolve_model`): the existing diffsynth path is the
# ID-V2V FINE-TUNE -> "idv2v" (NOT vanilla Wan; fast/regular variants). Bernini =
# "bernini-1.3b" (Wan2.1-1.3B, multi-ref/r2v-capable, ~8 GB) / "bernini-14b"
# (Wan2.2-T2V-A14B two-expert MoE, ~28 GB bf16 / ~15 GB fp8). Weights are gated HF
# repos, provisioned to /models (bind-mounted in compose; never baked into the image).
BERNINI_ENABLED = os.environ.get("BERNINI_ENABLED", "auto")  # auto = any cp present
BERNINI_ROOT_13B = os.environ.get("BERNINI_ROOT_13B", "/models/Bernini-R-1.3B-Diffusers")
BERNINI_ROOT_14B = os.environ.get("BERNINI_ROOT_14B", "/models/Bernini-R-Diffusers")
BERNINI_VENV_PY = os.environ.get("BERNINI_VENV_PY", "/opt/bernini/venv/bin/python")
BERNINI_GPU_DEVICE = os.environ.get("BERNINI_GPU_DEVICE", "")  # "" = GPU_DEVICE
# Native render preset: long edge capped at BERNINI_MAX_IMAGE_SIZE, native fps 16.
# Delivery resolution/fps ABOVE native is reached via the RIFE/FlashVSR post rails
# (never by asking the model for an out-of-distribution size).
BERNINI_MAX_IMAGE_SIZE = int(os.environ.get("BERNINI_MAX_IMAGE_SIZE", "848"))
BERNINI_NATIVE_FPS = int(os.environ.get("BERNINI_NATIVE_FPS", "16"))
BERNINI_DURATIONS = [int(x) for x in os.environ.get("BERNINI_DURATIONS", "2,3,5").split(",")]
# Default denoise-step budget per engine (explicit num_inference_steps still wins;
# reference-tuned defaults; calibrated on-box in Task 9).
BERNINI_STEPS = {"bernini-1.3b": int(os.environ.get("BERNINI_13B_STEPS", "30")),
                 "bernini-14b": int(os.environ.get("BERNINI_14B_STEPS", "30"))}
BERNINI_MODELS = frozenset(["bernini-1.3b", "bernini-14b"])

def resolve_model(model) -> str:
    """Engine discriminator: 'idv2v' (existing diffsynth ID-V2V fine-tune) |
    'bernini-1.3b' | 'bernini-14b'. Aliases: 1.3b/13b/bernini13b -> 1.3B;
    bernini/14b -> 14B."""
    m = str(model or "").strip().lower()
    if m in {"bernini", "bernini-14b", "bernini_14b", "bernini-14", "bernini14b", "14b"}:
        return "bernini-14b"
    if m in {"bernini-1.3b", "bernini_1_3b", "1.3b", "13b", "bernini13b"}:
        return "bernini-1.3b"
    return "idv2v"

def bernini_root(model) -> str:
    """Weight root for a (Bernini) model id."""
    return BERNINI_ROOT_13B if resolve_model(model) == "bernini-1.3b" else BERNINI_ROOT_14B

def bernini_steps(model) -> int:
    """Default denoise-step budget for a Bernini model id."""
    return BERNINI_STEPS.get(resolve_model(model), 30)

def bernini_enabled() -> bool:
    """Whether the Bernini rail is available. 'auto' = enabled if any checkpoint exists."""
    v = BERNINI_ENABLED.strip().lower()
    if v in {"1", "true", "yes"}:
        return True
    if v in {"0", "false", "no"}:
        return False
    return os.path.isdir(BERNINI_ROOT_13B) or os.path.isdir(BERNINI_ROOT_14B)

# --- RIFE fps-boost (post rail; runs on the dedicated vp-worker) ---
# hzwer/ECCV2022-RIFE HDv3 (RIFE_HDv3/IFNet_HDv3/refine) + weights from the official
# HF mirror hzwer/RIFE (RIFEv4.26_0921.zip -> flownet.pkl). Arbitrary-timestep
# Model.inference_batch(I0,I1,timesteps,scale); anchors originals, fills gaps only.
RIFE_VERSION = os.environ.get("RIFE_VERSION", "4.26")
RIFE_ROOT = os.environ.get("RIFE_ROOT", "/models/rife")
RIFE_VENV_PY = os.environ.get("RIFE_VENV_PY", "/opt/rife/venv/bin/python")
RIFE_FPS_MODES = ("preserve_motion", "smooth")

# --- FlashVSR upscale (post rail; runs on the dedicated vp-worker) ---
# OpenImagingLab/FlashVSR (CVPR 2026) one-step 4x-tuned VSR; final scale encode via
# ffmpeg (final="raw" -> native 4x, NO ffmpeg downscale).
FLASHVSR_ENABLED = os.environ.get("FLASHVSR_ENABLED", "auto")
FLASHVSR_ROOT = os.environ.get("FLASHVSR_ROOT", "/models/flashvsr")
FLASHVSR_VENV_PY = os.environ.get("FLASHVSR_VENV_PY", "/opt/flashvsr/venv/bin/python")
FLASHVSR_SCALE = int(os.environ.get("FLASHVSR_SCALE", "4"))
FLASHVSR_MODE = os.environ.get("FLASHVSR_MODE", "tiny")
FLASHVSR_FINAL = os.environ.get("FLASHVSR_FINAL", "1080")

# HTTP server.
# HTTP server.
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8992"))  # distinct from LTX worker's 8991
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", "3000000000"))  # 3GB


def worker_token() -> str:
    """Return the worker auth token, auto-generating a stable one if blank."""
    global WORKER_TOKEN
    if WORKER_TOKEN:
        return WORKER_TOKEN
    if not WORKER_TOKEN:
        WORKER_TOKEN = os.environ.setdefault("WORKER_TOKEN", _random_token())
    return WORKER_TOKEN


def _random_token() -> str:
    import random
    import string
    return "".join(random.choices(string.ascii_letters + string.digits, k=32))
