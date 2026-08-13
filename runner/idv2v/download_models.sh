#!/bin/bash
# Download ID-V2V model weights and dependencies for the worker container.
# Runs on first boot inside the Docker container.
#
# Targets a single RTX 5090 (32 GB): the video model (Wan 2.1 I2V-14B DiT +
# VACE) is loaded int8-quantized with CPU offload in model.py, so the full
# footprint fits in 32 GB of VRAM.
#
# Adapted from the standalone id-v2v runner scripts/download_models.sh.
# MODEL_DIR defaults to /models (shared bind-mount with the live-runner host).

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
mkdir -p "$MODEL_DIR"

echo ">>> Downloading ID-V2V model files to $MODEL_DIR (int8 / offload target: RTX 5090)"

HF_TOKEN="${HUGGING_FACE_HUB_TOKEN:-}"
if [ -z "$HF_TOKEN" ]; then
    echo "WARN: HUGGING_FACE_HUB_TOKEN not set — some models may be gated"
fi

# ID-V2V finetuned checkpoint (DiT + VACE weights). Loaded + quantized to int8
# in-process (torchao int8_weight_only), so the on-disk bf16 checkpoint is fine.
# The hf-fp8 worker loads the pre-quantized per-channel FP8 DiT+VACE from
# HF_REPO via huggingface_hub.snapshot_download at runtime (into the HF cache,
# HF_HOME or ~/.cache/huggingface by default). Pre-fetch BOTH model variants
# here so first load is instant and the box carries both on disk:
#   (regular) repo root -> ad-astra-video/id-v2v-fp8          (~19 GB)
#   (fast)    /fusionx  -> FusionX I2V-14B LoRA fused + fp8  (~19 GB)
HF_REPO="${HF_REPO:-ad-astra-video/id-v2v-fp8}"
# Version-agnostic: older huggingface_hub has no snapshot_download(subfolder=),
# so the fast/fusionx variant is synced per-file into a local folder.
python3 - "$HF_REPO" "$HF_TOKEN" "$MODEL_DIR" <<'PY'
import sys, os
from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download
repo, token, model_dir = sys.argv[1], sys.argv[2] or None, sys.argv[3]
# regular: pre-quantized fp8 repo root -> HF cache (worker loads via snapshot_download)
print(">>> regular: snapshot_download", repo, "(repo root) ...")
snapshot_download(repo, token=token, allow_patterns=["*.safetensors", "*.json"])
# fast: fusionx subfolder -> local <MODEL_DIR>/fusionx
fus = os.path.join(model_dir, "fusionx")
os.makedirs(fus, exist_ok=True)
print(">>> fast: syncing", repo, "/fusionx ->", fus)
files = [f for f in list_repo_files(repo, token=token)
         if f.startswith("fusionx/") and (f.endswith(".safetensors") or f.endswith(".json"))]
for f in sorted(files):
    hf_hub_download(repo, f, token=token, local_dir=model_dir)
print(">>> done:", len(files), "fusionx files at", fus)
PY

echo ">>> Downloading idv2v.pth (DiT + VACE finetuned checkpoint, legacy local source)"
huggingface-cli download --token "$HF_TOKEN" \
    Eyeline-Labs/ID-V2V idv2v.pth \
    --local-dir "$MODEL_DIR" || true

# SAM3 segmentation model (foreground-on-gray preprocessing)
echo ">>> Downloading SAM3"
huggingface-cli download --token "$HF_TOKEN" sam3-org/sam3 \
    --local-dir "$MODEL_DIR/sam3" || true

# Wan 2.1 I2V-14B-720P base model (T5 + VAE + CLIP tokenizer used by pipeline).
echo ">>> Downloading Wan2.1 I2V-14B-720P (T5 + VAE + tokenizer + CLIP)"
huggingface-cli download --token "$HF_TOKEN" \
    Wan-AI/Wan2.1-I2V-14B-720P \
    --local-dir "$MODEL_DIR/wan" || true

# FLUX.2 [klein] 4B — the first-frame image-edit styler (see runner/idv2v/flux_edit.py).
# Three components must be resident to edit, matching KLEIN4B_* config defaults:
#   <MODEL_DIR>/flux2/flux-2-klein-4b.safetensors   (flow transformer, 4B, ~8 GB bf16)
#   <MODEL_DIR>/flux2/ae.safetensors                (FLUX.2 autoencoder, shared w/ FLUX.2-dev)
#   Qwen/Qwen3-4B-FP8 text embedder -> HF cache     (from_pretrained at runtime)
# KLEIN4B_ENABLED defaults to "auto", i.e. engages only when the flow weight is
# present on disk — so provisioning these is what turns the editor on.
echo ">>> Downloading FLUX.2 [klein] 4B (flow + AE + Qwen3 text embedder)"
mkdir -p "$MODEL_DIR/flux2"
huggingface-cli download --token "$HF_TOKEN" \
    black-forest-labs/FLUX.2-klein-4B flux-2-klein-4b.safetensors \
    --local-dir "$MODEL_DIR/flux2" || true
huggingface-cli download --token "$HF_TOKEN" \
    black-forest-labs/FLUX.2-dev ae.safetensors \
    --local-dir "$MODEL_DIR/flux2" || true
# Qwen3 4B FP8 text embedder (KLEIN4B_TEXT_ENC). Cached under HF_HOME so the
# worker's load_qwen3_embedder(variant="4B") resolves without a network hit.
huggingface-cli download --token "$HF_TOKEN" \
    Qwen/Qwen3-4B-FP8 || true

# Precompiled fine-grained-FP8 CUDA kernels (kernels-community/finegrained-fp8),
# used by the `kernels==0.15.2` pip package to execute Qwen3's fp8_linear at
# style-frame time. Cached under HF_HOME so first use doesn't need a runtime
# download/write into the cache (mirrors the Qwen3 pre-fetch above).
huggingface-cli download --token "$HF_TOKEN" \
    kernels-community/finegrained-fp8 || true

echo ">>> Model download complete"
ls -lh "$MODEL_DIR"
