#!/bin/bash
# Download ONLY the LTX model stack into the /models layout the ltx-worker
# expects. This is the LTX-side counterpart to ../download_all_models.sh (which
# pulls everything) and ../idv2v/download_models.sh (which pulls the id-v2v
# stack). Run this on the GPU box HOST before docker compose up — it is NOT
# needed (and not copied) inside the ltx-worker container.
#
# Layout produced (mirrors runner/ltx/config.py defaults):
#   <MODELS_DIR>/checkpoint/ltx-2.3-22b-distilled-1.1.safetensors
#   <MODELS_DIR>/gemma/gemma-3-12b-it-qat-q4_0-unquantized/
#   <MODELS_DIR>/upscaler/ltx-2.3-spatial-upscaler-x2-1.1.safetensors
#   <MODELS_DIR>/z-image/Z-Image-Turbo/
#   <MODELS_DIR>/processors/{dpt-hybrid-midas, yolox_l.torchscript.pt, dw-ll_ucoco_384_bs5.torchscript.pt}
#
# Usage:
#   export HUGGING_FACE_HUB_TOKEN=hf_...
#   MODELS_DIR=/home/brad/models ./runner/ltx/download_models.sh

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

echo ">>> Downloading LTX models to $MODELS_DIR"

echo ">>> [1/5] LTX main transformer: $LTX_CP (~46 GB)"
mkdir -p "$MODELS_DIR/checkpoint"
huggingface-cli download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.3" "$LTX_CP" \
    --local-dir "$MODELS_DIR/checkpoint"

echo ">>> [2/5] Gemma-3-12B text encoder (bfloat16, ~25 GB)"
mkdir -p "$MODELS_DIR/gemma"
huggingface-cli download --token "$HF_TOKEN" \
    "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized" \
    --local-dir "$MODELS_DIR/gemma/gemma-3-12b-it-qat-q4_0-unquantized"

echo ">>> [3/5] LTX 2x spatial upscaler ($UPSCALER_CP)"
mkdir -p "$MODELS_DIR/upscaler"
huggingface-cli download --token "$HF_TOKEN" \
    "Lightricks/LTX-2.3" "$UPSCALER_CP" \
    --local-dir "$MODELS_DIR/upscaler"

echo ">>> [4/5] Z-Image-Turbo (text-to-image, ~31 GB)"
mkdir -p "$MODELS_DIR/z-image"
huggingface-cli download --token "$HF_TOKEN" \
    "Tongyi-MAI/Z-Image-Turbo" \
    --local-dir "$MODELS_DIR/z-image/Z-Image-Turbo"

echo ">>> [5/5] Processors (depth / yolox / dwpose)"
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

echo ">>> Done. LTX models at $MODELS_DIR"
du -sh "$MODELS_DIR/checkpoint" "$MODELS_DIR/gemma" "$MODELS_DIR/z-image" 2>/dev/null || true
