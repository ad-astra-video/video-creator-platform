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

# TeaCache: skip redundant DiT-block computation across denoise steps by
# reusing the stored residual when the (rescaled) relative-L1 drift of the
# modulated time-embedding stays below the threshold. Cheap throughput win with
# minimal visual change when the threshold is kept low. The first and last
# denoise steps are ALWAYS computed (built into the algorithm).
#   IDV2V_TEACACHE        "1"/"0"      master switch (default ON)
#   IDV2V_TEACACHE_THRESH float        rel-L1 accumulation threshold. LOWER =
#                           more steps computed = less visual degradation, less
#                           speedup. Conservative default 0.10; 0.06-0.08 for
#                           near-invisible; 0.15-0.25 aggressive (flicker risk).
#   IDV2V_TEACACHE_MODEL  str          one of diffsynth's supported ids; "" =
#                           auto-select from height (I2V-14B-480P / -720P).
IDV2V_TEACACHE = os.environ.get("IDV2V_TEACACHE", "1").lower() in {"1", "true", "yes"}
IDV2V_TEACACHE_THRESH = float(os.environ.get("IDV2V_TEACACHE_THRESH", "0.10"))
IDV2V_TEACACHE_MODEL = os.environ.get("IDV2V_TEACACHE_MODEL", "")

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


# FLUX.2 [klein] 4B image editing (first-frame styling).
#
# BFL's earliest/lightest 4B image-editing model — a guidance- AND
# step-distilled rectified-flow transformer. Distillation fixes both knobs:
#   * num_steps = 4        (the 50-step "klein-base-4B" is for fine-tuning/control,
#                           not production editing — we use the distilled model)
#   * guidance  = 1.0      (no CFG; `denoise`, not `denoise_cfg`)
# Editing is single-reference conditioning: the first frame is encoded through
# the FLUX.2 AE into `ref_tokens` and passed to `denoise` as `img_cond_seq`;
# the prompt describes the DESIRED (styled) result.
#
# Three components must be resident to edit:
#   * the 4B flow transformer  (<KLEIN4B_MODEL>, ~8 GB bf16)
#   * the FLUX.2 autoencoder   (<KLEIN4B_AE>, from the FLUX.2-dev repo)
#   * a Qwen3 4B text embedder (<KLEIN4B_TEXT_ENC>, hidden states [9,18,27]) —
#     a SEPARATE LLM from Gemma. It cannot coexist on one card with the id-v2v
#     DiT/VACE (~19.5 GB) or a Gemma LLM, so the editor uses the same staged/
#     evict lifecycle (see flux_edit.py).
#
#   KLEIN4B_ENABLED   "auto" (use only if <KLEIN4B_MODEL> exists) | "1" | "0"
#                     | "force" (error if absent). Frames are styled with FLUX.2
#                     only when this is on AND the caller requests it.
#   KLEIN4B_MODEL     path to flux-2-klein-4b.safetensors (env KLEIN_4B_MODEL_PATH
#                     is what the BFL loader reads; we set it before load).
#   KLEIN4B_AE        path to ae.safetensors (env AE_MODEL_PATH).
#   KLEIN4B_TEXT_ENC  HF id of the Qwen3 text embedder (default Qwen/Qwen3-4B, bf16).
#   KLEIN4B_GPU_DEVICE device for the editor (default "" = video GPU_DEVICE; set
#                     e.g. cuda:1 for a card that doesn't contend with the video).
#   KLEIN4B_STEPS / KLEIN4B_GUIDANCE  distilled defaults (4 / 1.0) — overridable
#                     only for experimentation; BFL intends them fixed.
#   KLEIN4B_MAX_SIDE  cap on the styled frame's long edge. Default 1472 = the
#                     empirically-tested ceiling on a 31.4 GB card (flow 8 + AE +
#                     Qwen3 bf16 ~16 GB weights + encode/denoise activations): a
#                     1080p 1920 frame OOMs at the AE encode, 1472 fits. It anchors
#                     the 1080p video restyle, so keep it as high as the card allows.
KLEIN4B_ENABLED = os.environ.get("KLEIN4B_ENABLED", "auto")
KLEIN4B_MODEL = os.environ.get("KLEIN4B_MODEL", "/models/flux2/flux-2-klein-4b.safetensors")
KLEIN4B_AE = os.environ.get("KLEIN4B_AE", "/models/flux2/ae.safetensors")
KLEIN4B_TEXT_ENC = os.environ.get("KLEIN4B_TEXT_ENC", "Qwen/Qwen3-4B")
KLEIN4B_GPU_DEVICE = os.environ.get("KLEIN4B_GPU_DEVICE", "")
KLEIN4B_STEPS = int(os.environ.get("KLEIN4B_STEPS", "4"))
KLEIN4B_GUIDANCE = float(os.environ.get("KLEIN4B_GUIDANCE", "1.0"))
KLEIN4B_MAX_SIDE = int(os.environ.get("KLEIN4B_MAX_SIDE", "1472"))


def klein4b_device() -> str:
    """GPU the FLUX.2 Klein editor runs on (defaults to the video model's)."""
    return KLEIN4B_GPU_DEVICE or GPU_DEVICE


def klein4b_steps() -> int:
    return KLEIN4B_STEPS


def klein4b_guidance() -> float:
    return KLEIN4B_GUIDANCE


def klein4b_enabled() -> bool:
    """Whether the FLUX.2 Klein first-frame editor may be engaged.

    "auto": engage when the flow weight is present on disk. "1"/"force": always
    (a missing weight then surfaces as a load error). "0": never.
    """
    mode = KLEIN4B_ENABLED.strip().lower()
    if mode in ("0", "false", "no", "off"):
        return False
    if mode in ("1", "true", "yes", "force"):
        return True
    return os.path.isfile(KLEIN4B_MODEL)
