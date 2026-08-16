#!/bin/bash
# Download ALL model weights for the video-creator 3-service compose into one
# /models tree, in the layout the workers + compose expect.
#
# This is the "download everything" script: it pulls both the LTX stack
# (ltx checkpoint + gemma text encoder + upscaler + processors) and the id-v2v
# stack (idv2v.pth + SAM3 + Wan) into a single MODELS_DIR bind-mounted at
# /models in docker-compose.video-creator.yml.
#
# Layout produced (mirrors the compose env defaults):
#   <MODELS_DIR>/checkpoint/ltx-2.3-22b-distilled-1.1.safetensors
#   <MODELS_DIR>/gemma/gemma-3-12b-it-qat-q4_0-unquantized/
#   <MODELS_DIR>/gemma/gemma-4-12b-it-qat-q4_0.gguf   (Gemma 4 LLM backend: gemma-worker)
#   <MODELS_DIR>/upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
#   <MODELS_DIR>/ltx-2.5/...                (LTX 2.5 kit, optional; runner/ltx/download_ltx25.sh -
#                                              NVFP4 on Blackwell, INT8+ConvRot otherwise.
#                                              Includes the 2.5 latent spatial upscaler under
#                                              latent_upscale_models/ needed by the distilled
#                                              pipeline)
#   <MODELS_DIR>/z-image/Z-Image-Turbo/
#   <MODELS_DIR>/processors/{dpt-hybrid-midas, yolox_l.torchscript.pt, dw-ll_ucoco_384_bs5.torchscript.pt}
#   <MODELS_DIR>/idv2v.pth                      (Eyeline-Labs/ID-V2V)
#   <MODELS_DIR>/sam3/                           (sam3-org/sam3)
#   <MODELS_DIR>/wan/                            (Wan-AI/Wan2.1-I2V-14B-720P)
#   <MODELS_DIR>/flux2/                          (FLUX.2 [klein] 4B first-frame styler:
#                                                   flux-2-klein-4b.safetensors + ae.safetensors; the
#                                                   Qwen3-4B bf16 text embedder goes to the HF cache)
#
# Usage (run on the GPU box as the user that owns the models dir):
#   export HUGGING_FACE_HUB_TOKEN=hf_...
#   MODELS_DIR=/home/brad/models ./runner/download_all_models.sh
#
# A token is required: LTX-2.3, gemma-3-12b-it-qat are gated on HF. Set
# HUGGING_FACE_HUB_TOKEN (or HF_TOKEN).

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
mkdir -p "$MODELS_DIR"

HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HUGGING_FACE_HUB_TOKEN not set — LTX-2.3 / gemma are gated models." >&2
    echo "  export HUGGING_FACE_HUB_TOKEN=hf_...Then rerun." >&2
    exit 1
fi

# Which LTX checkpoint variant to pull (1.1 is current per model_download_specs).
LTX_CP="${LTX_CP:-ltx-2.3-22b-distilled-1.1.safetensors}"
UPSCALER_CP="${UPSCALER_CP:-ltx-2.3-spatial-upscaler-x2-1.1.safetensors}"

echo ">>> Downloading models to $MODELS_DIR"

# ---------------------------------------------------------------- LTX stack --
echo ">>> [1/7] LTX main transformer: $LTX_CP (~46 GB)"
mkdir -p "$MODELS_DIR/checkpoint"
huggingface-cli download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.3" "$LTX_CP" \
    --local-dir "$MODELS_DIR/checkpoint"

echo ">>> [2/7] Gemma-3-12B text encoder (bfloat16, ~25 GB)"
mkdir -p "$MODELS_DIR/gemma"
huggingface-cli download --token "$HF_TOKEN" \
    "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized" \
    --local-dir "$MODELS_DIR/gemma/gemma-3-12b-it-qat-q4_0-unquantized"

echo ">>> [3/7] LTX 2x spatial upscaler ($UPSCALER_CP)"
mkdir -p "$MODELS_DIR/upscaler"
huggingface-cli download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.3" "$UPSCALER_CP" \
    --local-dir "$MODELS_DIR/upscaler"

echo ">>> [4/7] Z-Image-Turbo (text-to-image, ~31 GB)"
mkdir -p "$MODELS_DIR/z-image"
huggingface-cli download --token "$HF_TOKEN" \
    "Tongyi-MAI/Z-Image-Turbo" \
    --local-dir "$MODELS_DIR/z-image/Z-Image-Turbo"

echo ">>> [5/7] Processors (depth / yolox / dwpose)"
mkdir -p "$MODELS_DIR/processors/dpt-hybrid-midas"
huggingface-cli download --token "$HF_TOKEN" \
    "Intel/dpt-hybrid-midas" \
    --local-dir "$MODELS_DIR/processors/dpt-hybrid-midas"
huggingface-cli download --token "$HF_TOKEN" \
    "hr16/yolox-onnx" "yolox_l.torchscript.pt" \
    --local-dir "$MODELS_DIR/processors"
huggingface-cli download --token "$HF_TOKEN" \
    "hr16/DWPose-TorchScript-BatchSize5" "dw-ll_ucoco_384_bs5.torchscript.pt" \
    --local-dir "$MODELS_DIR/processors"

# --------------------------------------------------------------- id-v2v stack --
echo ">>> [6/7] ID-V2V weights (idv2v.pth + SAM3 + Wan)"
# Reuse the dedicated id-v2v downloader for the idv2v/wan/sam3 subtree.
MODEL_DIR="$MODELS_DIR" bash "$(dirname "$0")/idv2v/download_models.sh" || {
    echo "WARN: idv2v downloader exited nonzero (token may not cover Wan/SAM3)." >&2
}

echo ">>> [7/8] Gemma 4 LLM backend GGUF (gemma-worker)"
# Reuse the dedicated gemma downloader for the gemma-4 GGUF subtree.
MODEL_DIR="$MODELS_DIR" bash "$(dirname "$0")/gemma/download_models.sh" || {
    echo "WARN: gemma-4 downloader exited nonzero (token may not cover Gemma 4)." >&2
}

echo ">>> [8/9] LTX 2.5 model kit (runner/ltx/download_ltx25.sh)"
# Auto-picks NVFP4 (Blackwell) vs INT8+ConvRot (other GPUs). Optional explicit
# choice via LTX25_VARIANT=int8|nvfp4. Gated repo — needs the HF token.
MODEL_DIR="$MODELS_DIR" bash "$(dirname "$0")/ltx/download_ltx25.sh" || {
    echo "WARN: LTX 2.5 downloader exited nonzero (token may not cover LTX-2.5)." >&2
}

# ---------------------------------------------------------- image-worker stack --
echo ">>> [image] Qwen-Image-Edit / Qwen-Image-Layered / Z-Image (image-worker)"
MODEL_DIR="$MODELS_DIR" bash "$(dirname "$0")/image/download_models.sh" || {
    echo "WARN: image downloader exited nonzero (token may not cover gated Qwen-Image repos)." >&2
}

echo ">>> [9/9] Verifying"
du -sh "$MODELS_DIR" 2>/dev/null || true
echo ">>> Done. Now set MODELS_DIR=$MODELS_DIR before docker compose up."

echo
echo "Expect roughly: checkpoint/ 46G, gemma/ 25G, z-image/ 31G, idv2v.pth ~28G,"
echo "wan/ + sam3/ several GB. LTX 2.5 kit (optional): ~19 GB nvfp4 / ~22 GB int8 "
echo "on top of the LTX 2.3 stack above. Total on the order of 150-175 GB + LTX 2.5."
