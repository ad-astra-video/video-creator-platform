#!/bin/bash
# Download the image-worker model stack into the /models/image layout the
# image-worker expects. This is the image-side counterpart to
# ../download_all_models.sh (which pulls everything) and mirrors the style of
# ../ltx/download_models.sh.
#
# Uses the modern `hf download` CLI (NOT the deprecated huggingface-cli).
#
# Layout produced (mirrors runner/image/config.py defaults):
#   <MODELS_DIR>/image/edit-2511/                         -> Qwen-Image-Edit-2511 (QwenImageEditPlusPipeline, fp8) [CURRENT edit engine]
#   <MODELS_DIR>/image/edit/Qwen/qwen-image-edit/         -> Qwen-Image-Edit (legacy 2509; no longer the edit engine)
#   <MODELS_DIR>/image/layered/Qwen/qwen-image-layered/   -> Qwen-Image-Layered (fp8 shards)
#   <MODELS_DIR>/image/zimage/                            -> Z-Image (Turbo)
#   <MODELS_DIR>/image/hidream/                           -> HiDream-O1-Image (8B UiT)
#
# Usage (run on the GPU box HOST before docker compose up):
#   export HUGGING_FACE_HUB_TOKEN=hf_...
#   MODELS_DIR=/home/brad/models ./runner/image/download_models.sh

set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
mkdir -p "$MODELS_DIR"

HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-${HF_TOKEN:-}}"
# Most image models (Qwen-Image-Edit 2509, Qwen-Image-Layered, FLUX.2) are gated
# on HF — a token is required. EXCEPTION: Qwen-Image-Edit-2511 and its 1038lab
# fp8 transformer are NOT gated and download without a token (no --token there).
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HUGGING_FACE_HUB_TOKEN not set — most image models are gated." >&2
    echo "  export HUGGING_FACE_HUB_TOKEN=hf_...  Then rerun (2511 needs none)." >&2
    exit 1
fi

IMAGE_BASE="$MODELS_DIR/image"
# Which precision shards to pull. fp8 is the default (matches QWEN_DTYPE=fp8 in
# runner/image/config.py). Set IMAGE_DTYPE=bf16 to pull the bf16 shards instead.
IMAGE_DTYPE="${IMAGE_DTYPE:-fp8}"   # fp8 | bf16

echo ">>> Downloading image models to $IMAGE_BASE (dtype=$IMAGE_DTYPE)"

# [1/3] Qwen-Image-Edit (instruction-following image editing)
echo ">>> [1/3] Qwen-Image-Edit"
mkdir -p "$IMAGE_BASE/edit"
hf download --token "$HF_TOKEN" "Qwen/Qwen-Image-Edit" \
    --local-dir "$IMAGE_BASE/edit/Qwen/Qwen-Image-Edit"

# [2/3] Qwen-Image-Layered — full diffusers pipeline, transformer swapped for a
# PRE-QUANTIZED FP8 (E4M3FN) single-file checkpoint so the worker loads fp8
# directly and NEVER quantizes in flight. Layout = full pipeline at
# <MODELS_DIR>/image/layered (matches QWEN_LAYERED_ROOT; bind-mounted into the
# image-worker container as /models/image/layered).
echo ">>> [2/3] Qwen-Image-Layered (FP8 transformer via T5B)"
LAYERED_DIR="$IMAGE_BASE/layered"
mkdir -p "$LAYERED_DIR/transformer"
# Upstream scaffolding: text_encoder, vae, tokenizer, processor, scheduler,
# model_index.json, transformer/config.json. Exclude the upstream (bf16)
# transformer weight shards — we replace them with the fp8 single-file below.
hf download --token "$HF_TOKEN" "Qwen/Qwen-Image-Layered" \
    --exclude "transformer/diffusion_pytorch_model-*.safetensors" \
    --exclude "transformer/diffusion_pytorch_model.safetensors.index.json" \
    --local-dir "$LAYERED_DIR"
# T5B pre-quantized FP8 E4M3FN transformer: same 1934 keys as the bf16 shards,
# no 'transformer.' prefix; 843 FP8 linear weights + 1091 BF16 sensitive layers
# (norms/embeddings/biases). Drop-in for the transformer/ diffusers layout.
hf download --token "$HF_TOKEN" "T5B/Qwen-Image-Layered-FP8" \
    qwen_image_layered_fp8_e4m3fn.safetensors \
    --local-dir "$LAYERED_DIR/transformer"
mv -f "$LAYERED_DIR/transformer/qwen_image_layered_fp8_e4m3fn.safetensors" \
      "$LAYERED_DIR/transformer/diffusion_pytorch_model.safetensors"
echo ">>> [2/3] done — fp8 transformer at $LAYERED_DIR/transformer/diffusion_pytorch_model.safetensors"


# [3/3] Z-Image (Turbo) — text-to-image + img2img editing
echo ">>> [3/3] Z-Image-Turbo"
mkdir -p "$IMAGE_BASE/zimage"
hf download --token "$HF_TOKEN" "Tongyi-MAI/Z-Image-Turbo" \
    --local-dir "$IMAGE_BASE/zimage/Z-Image-Turbo"

# [4/4] FLUX.2 [klein] 4B — the /style-frame editor (styles the restyle first
# frame). Three components are needed to edit (matches runner/image/config.py
# KLEIN4B_* defaults):
#   <MODELS_DIR>/flux2/flux-2-klein-4b.safetensors   (flow transformer, 4B, ~8 GB bf16)
#   <MODELS_DIR>/flux2/ae.safetensors                (FLUX.2 autoencoder, shared w/ FLUX.2-dev)
#   Qwen/Qwen3-4B text embedder -> HF cache          (bf16, from_pretrained at runtime)
# KLEIN4B_ENABLED defaults to "auto", i.e. engages only when the flow weight is
# on disk — provisioning these turns the editor on.
echo ">>> [4/4] FLUX.2 [klein] 4B (flow + AE + Qwen3 bf16 text embedder)"
mkdir -p "$MODELS_DIR/flux2"
hf download --token "$HF_TOKEN" "black-forest-labs/FLUX.2-klein-4B" \
    flux-2-klein-4b.safetensors --local-dir "$MODELS_DIR/flux2" || true
hf download --token "$HF_TOKEN" "black-forest-labs/FLUX.2-dev" \
    ae.safetensors --local-dir "$MODELS_DIR/flux2" || true
hf download --token "$HF_TOKEN" "Qwen/Qwen3-4B" || true

# [6/6] Qwen-Image-Edit-2511 — the CURRENT edit engine (QwenImageEditPlusPipeline,
# multi-reference-image editing + fp8). Same recipe as Layered: pull the upstream
# (bf16) 2511 diffusers scaffolding (configs / scheduler / vae / text_encoder /
# tokenizer / processor) and swap its transformer weights for a PRE-QUANTIZED FP8
# (E4M3FN) single-file checkpoint so the worker keeps fp8 and never quantizes in
# flight. Transformer keys are bare diffusers QwenImageTransformer2DModel names
# (verified: 1933 tensors, all F8_E4M3, 20.4 GB) -> drop-in for transformer/.
echo ">>> [6/6] Qwen-Image-Edit-2511 (FP8 transformer via 1038lab)"
EDIT2511_DIR="$IMAGE_BASE/edit-2511"
mkdir -p "$EDIT2511_DIR/transformer"
# 2511 is NOT gated on HF (verified: gated:False, files served unauthenticated
# HTTP 200) — no --token, unlike the 2509/Layered/FLUX models above.
hf download "Qwen/Qwen-Image-Edit-2511" \
    --exclude "transformer/diffusion_pytorch_model-*.safetensors" \
    --exclude "transformer/diffusion_pytorch_model.safetensors.index.json" \
    --local-dir "$EDIT2511_DIR"
hf download "1038lab/Qwen-Image-Edit-2511-FP8" \
    Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors \
    --local-dir "$EDIT2511_DIR/transformer"
mv -f "$EDIT2511_DIR/transformer/Qwen-Image-Edit-2511-FP8_e4m3fn.safetensors" \
      "$EDIT2511_DIR/transformer/diffusion_pytorch_model.safetensors"

# [5/5] HiDream-O1-Image — the /image (text-to-image) + /edit (instruction
# editing) engine. An 8B pixel-level Unified Transformer run through the
# vendored runner/image/hidream_models/ pipeline (custom Qwen3VL UiT + repo
# flow-matching schedulers). NOT gated on HF (no token needed), so it downloads
# via plain `hf download`. Layout matches HIDREAM_ROOT=/models/image/hidream
# (volume-mounted into the image-worker from /srv/video-creator/models/image/hidream).
echo ">>> [5/5] HiDream-O1-Image (8B UiT)"
mkdir -p "$IMAGE_BASE/hidream"
hf download --token "$HF_TOKEN" "HiDream-ai/HiDream-O1-Image" \
    --local-dir "$IMAGE_BASE/hidream"

echo ">>> Done. Image models at $IMAGE_BASE"
du -sh "$IMAGE_BASE" 2>/dev/null || true
