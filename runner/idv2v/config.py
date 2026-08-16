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
#   GEMMA_ENABLED    "auto" (use only if checkpoint present) | "1" | "0"
#                    |"force" (error if absent)
GEMMA_ROOT = os.environ.get("GEMMA_ROOT", "/models/gemma")
GEMMA_GPU_DEVICE = os.environ.get("GEMMA_GPU_DEVICE", "")
GEMMA_ATTN_IMPL = os.environ.get("GEMMA_ATTN_IMPL", "eager")
GEMMA_ENABLED = os.environ.get("GEMMA_ENABLED", "auto")


def gemma_device() -> str:
    """GPU the Gemma LLM loads on.

    Defaults to the SAME card as the video model (``GPU_DEVICE``). Because the
    id-v2v DiT/VACE and Gemma both need to share one GPU, the worker must evict
    the resident video model first (see gemma.py's evict hook) — callers that
    load Gemma are responsible for that eviction.
    """
    return GEMMA_GPU_DEVICE or GPU_DEVICE


def gemma_enabled() -> bool:
    """Whether the Gemma LLM should be engaged for enhance/caption.

    "auto": engage when the checkpoint directory is present on disk.
    "1"/"force": always engage (a missing checkpoint then surfaces as an load
    error). "0": never.
    """
    mode = GEMMA_ENABLED.strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "force"):
        return True
    # auto
    try:
        return os.path.isdir(GEMMA_ROOT) and any(
            f.endswith(".safetensors") or f.endswith(".bin")
            for f in os.listdir(GEMMA_ROOT)
        )
    except OSError:
        return False


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
