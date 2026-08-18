#!/bin/bash
# Download the LTX 2.5 model kit into a ComfyUI-style subtree under the shared
# models dir. Picks the distilled-transformer variant by GPU so you always fetch
# the format your card actually runs:
#
#   * NVFP4  (ltx-2.5-22b-distilled-transformer-nvfp4.safetensors) — Blackwell
#             (sm_100/sm_103/sm_120, i.e. compute capability >= 10.0 — e.g.
#             RTX 50-series like the RTX 5090, or the RTX PRO 6000 Blackwell).
#   * BF16 (ltx-2.5-22b-distilled-transformer-bf16.safetensors) — portable fallback,
#             loaded in-process with the CUDA fp8-cast policy (no ltx-kernels).
#
# Override the auto-detect with LTX25_VARIANT=int8|nvfp4|bf16.
#
# Produces (mirrors the ComfyUI repo layout, all under <MODEL_DIR>/ltx-2.5):
#   diffusion_models/ltx-2.5-22b-distilled-transformer-<variant>.safetensors
#   text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
#   vae/ltx-2.5-video-vae-bf16.safetensors
#   vae/ltx-2.5-audio-vae-bf16.safetensors
#   model_patches/ltx-2.5-duration-head-bf16.safetensors
#   latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
#   loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors
#
# NOTE: the LTX-2.5 distilled pipeline requires the 2.5-specific LATENT spatial
# upscaler (pulled below into latent_upscale_models/). The older LTX-2.3 spatial
# upscaler (ltx-2.3-spatial-upscaler-x2-1.1.safetensors) is a separate artifact
# that download_all_models.sh still pulls into <MODEL_DIR>/upscaler for the
# LTX-2.3 stack — keep that step too.
#
# The IC-LoRA Pixel Spatial Upscaler is ALSO REQUIRED for LTX-2.5 pixel-space
# 2x upscaling. It lives in its OWN SEPARATE GATED repo:
#   Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler
# — a SECOND agreement on huggingface.co (distinct from the LTX-2.5 repo) must
# be accepted before the token can fetch it. The same HF token is used for both.
#
# Usage (run on the GPU box as the user that owns the models dir):
#   export HUGGING_FACE_HUB_TOKEN=hf_...
#   MODEL_DIR=/home/brad/models bash ./runner/ltx/download_ltx25.sh
#
# Both Lightricks/LTX-2.5 and Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler
# are gated — accept BOTH agreements, and the HF token is required.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
TARGET="$MODEL_DIR/ltx-2.5"
mkdir -p "$TARGET/diffusion_models" "$TARGET/text_encoders" "$TARGET/vae" \
         "$TARGET/model_patches" "$TARGET/latent_upscale_models" "$TARGET/loras"

HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HUGGING_FACE_HUB_TOKEN not set — Lightricks/LTX-2.5 is gated." >&2
    echo "  export HUGGING_FACE_HUB_TOKEN=hf_...Then rerun." >&2
    exit 1
fi

# ---- variant selection ------------------------------------------------------
select_variant() {
    if [ -n "${LTX25_VARIANT:-}" ]; then
        echo "$LTX25_VARIANT"
        return 0
    fi
    # First GPU's compute capability, digits only (e.g. "120" = sm_120, "89" = sm_89).
    local cc
    cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null \
          | head -n1 | tr -d ' .')"
    if [ -n "$cc" ] && [ "$cc" -ge 100 ] 2>/dev/null; then
        # Blackwell -> native NVFP4
        echo "nvfp4"
    else
        # Non-Blackwell -> BF16 (loaded with the CUDA fp8-cast policy, no ltx-kernels).
        # The comfy-int8-convrot / nvfp4 files are ComfyUI formats the ltx-pipelines
        # loader cannot consume, so bf16 is the portable default.
        echo "bf16"
    fi
}

VARIANT="$(select_variant)"
case "$VARIANT" in
    nvfp4) TRANSFORMER="ltx-2.5-22b-distilled-transformer-nvfp4.safetensors" ;;
    int8)  TRANSFORMER="ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors" ;;
    bf16)  TRANSFORMER="ltx-2.5-22b-distilled-transformer-bf16.safetensors" ;;
    *) echo "ERROR: LTX25_VARIANT must be int8|nvfp4|bf16 (got: $VARIANT)" >&2; exit 1 ;;
esac

echo ">>> LTX 2.5 variant selected: $VARIANT ($TRANSFORMER)"
echo ">>> Downloading to $TARGET"

# ---- downloads --------------------------------------------------------------
echo ">>> [1/7] Transformer ($VARIANT)"
hf download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.5" "diffusion_models/$TRANSFORMER" \
    --local-dir "$TARGET"

echo ">>> [2/7] Gemma 4 12B text encoder w/ LTX-2.5 projection (bf16)"
hf download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.5" "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    --local-dir "$TARGET"

echo ">>> [3/7] Video VAE (bf16)"
hf download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.5" "vae/ltx-2.5-video-vae-bf16.safetensors" \
    --local-dir "$TARGET"

echo ">>> [4/7] Audio VAE (bf16)"
hf download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.5" "vae/ltx-2.5-audio-vae-bf16.safetensors" \
    --local-dir "$TARGET"

echo ">>> [5/7] Duration-head model patch (bf16)"
hf download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.5" "model_patches/ltx-2.5-duration-head-bf16.safetensors" \
    --local-dir "$TARGET"

echo ">>> [6/7] LTX 2.5 latent spatial upscaler x2 (bf16)"
hf download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.5" "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    --local-dir "$TARGET"

echo ">>> [7/7] IC-LoRA pixel spatial upscaler x2 (LoRA, separate gated repo)"
hf download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler" "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors" \
    --local-dir "$TARGET/loras"

# ---- verify ----------------------------------------------------------------
echo ">>> Verifying LTX 2.5 files"
missing=0
for f in \
    "diffusion_models/$TRANSFORMER" \
    "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" \
    "vae/ltx-2.5-video-vae-bf16.safetensors" \
    "vae/ltx-2.5-audio-vae-bf16.safetensors" \
    "model_patches/ltx-2.5-duration-head-bf16.safetensors" \
    "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors" \
    "loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors" \
; do
    if [ -s "$TARGET/$f" ]; then
        echo "  ok   $f ($(du -h "$TARGET/$f" | cut -f1))"
    else
        echo "  MISS $f" >&2
        missing=1
    fi
done
[ "$missing" -eq 0 ] || { echo "ERROR: LTX 2.5 download incomplete." >&2; exit 1; }

echo ">>> Done. LTX 2.5 ($VARIANT) is under $TARGET"
