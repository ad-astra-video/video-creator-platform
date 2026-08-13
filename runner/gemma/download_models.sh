#!/bin/bash
# Download the Gemma 4 12B it QAT GGUF (base + multimodal projector) for the
# gemma-worker into <MODEL_DIR>/gemma/, matching the compose GEMMA_MODEL /
# GEMMA_MMPROJ defaults.
#
# Note the filenames are LOWERCASE "4-12b" in this repo
# (gemma-4-12b-it-qat-q4_0.gguf, mmproj-gemma-4-12b-it-qat-q4_0.gguf) — the
# repo id keeps the capital B. The compose/config defaults point at these exact
# lowercase filenames.
#
# Usage:
#   export HUGGING_FACE_HUB_TOKEN=hf_...   # Gemma 4 may be gated
#   MODEL_DIR=/home/brad/models ./runner/gemma/download_models.sh

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
mkdir -p "$MODEL_DIR/gemma"

HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
if [ -z "$HF_TOKEN" ]; then
    echo "WARN: HUGGING_FACE_HUB_TOKEN not set — Gemma 4 may be gated." >&2
fi

echo ">>> Downloading Gemma 4 12B it QAT q4_0 GGUF (base model)"
huggingface-cli download --token "$HF_TOKEN" \
    google/gemma-4-12B-it-qat-q4_0-gguf \
    gemma-4-12b-it-qat-q4_0.gguf \
    --local-dir "$MODEL_DIR/gemma"

echo ">>> Downloading mmproj (image/audio input projector; optional — set GEMMA_MMPROJ to enable)"
huggingface-cli download --token "$HF_TOKEN" \
    google/gemma-4-12B-it-qat-q4_0-gguf \
    mmproj-gemma-4-12b-it-qat-q4_0.gguf \
    --local-dir "$MODEL_DIR/gemma" || true

echo ">>> Done. Files:"
ls -lh "$MODEL_DIR/gemma"
