# image-worker — serves /video-creator/v1/{edit,layer,image} (Qwen-Image-Edit,
# Qwen-Image-Layered, Z-Image) behind the live-runner. Heavy image: torch +
# diffusers. Does NOT register with the Orchestrator (that's the live-runner's
# job); it only serves the internal /load /evict /v1/* surface over the Docker
# network.
#
# CUDA 12.8 wheels cover Ada (4090, SM89) + Blackwell (5090 / RTX PRO 6000, SM120).
#
# Build (from the video-creator repo ROOT — the build context is one level
# above runner/, so `COPY runner/ ./runner/` works):
#   docker build -f runner/docker/image-worker.Dockerfile \
#       -t video-creator-image-worker .

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Core Python deps (aiohttp, Pillow, numpy, av, requests, huggingface_hub)
COPY runner/requirements.txt /tmp/runner-requirements.txt
RUN pip install --no-cache-dir -r /tmp/runner-requirements.txt

# PyTorch with CUDA 12.8 (Ada SM89 + Blackwell SM120). Pinned to the 2.11.0 build
# validated for the Qwen-Image-Layered fp8 stack on the .151 box.
# NOTE: torchvision IS required here — FLUX.2 [klein] (flux2.sampling/denoise)
# imports torchvision as part of its decode/processing stack, and the image-worker
# hosts Klein for both /style-frame and text-to-image. Install it from the same
# cu128 index so it matches torch 2.11.0.
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Diffusers + accelerate + transformers for the Qwen-Image-Edit / Layered / Z-Image
# pipelines. Pinned to the versions validated with the Qwen fp8 layered checkpoint.
# torchao provides the int8_weight_only() quant (used only for the edit engine when
# QWEN_DTYPE=int8); the layered engine loads a PRE-QUANTIZED FP8 checkpoint and
# does NOT quantize in flight (QWEN_LAYERED_DTYPE=fp8).
RUN pip install --no-cache-dir \
    "diffusers==0.39.0" "accelerate>=0.33.0" "transformers==5.15.0" "torchao==0.18.0"

# FLUX.2 [klein] 4B style-frame editor (black-forest-labs/flux2). Installed with
# --no-deps so the pinned torch==2.8.0 / transformers==4.56.1 in its pyproject
# CANNOT downgrade this image's authoritative torch 2.11.0 cu128 / diffusers /
# transformers. Its other deps (einops, safetensors, huggingface-hub, PIL) are
# added explicitly below. Pinned commit for reproducibility; installs the `flux`
# package (import `flux2`).
ARG FLUX2_REPO=https://github.com/black-forest-labs/flux2
ARG FLUX2_COMMIT=50fe5162777813d869182b139e83b10743caef15
RUN pip install --no-cache-dir --no-deps "git+${FLUX2_REPO}@${FLUX2_COMMIT}"

# extra deps the flux2/klein path needs that aren't in the base requirements.
RUN pip install --no-cache-dir "einops"

# scipy is required by the HiDream-O1-Image unipc flow solver
# (hidream_models/fm_solvers_unipc.py) used by the default 'full' model recipe.
RUN pip install --no-cache-dir "scipy"

RUN useradd -m runneruser

WORKDIR /app
COPY runner/ ./runner/

RUN mkdir -p /models && chown runneruser:runneruser /models
USER runneruser

# Worker liveness probe (open, no token).
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8994/video-creator/v1/health || exit 1

EXPOSE 8994
# Entry: the image-worker server. Uses python -m runner.image.server's main().
ENTRYPOINT ["python", "-m", "runner.image.server"]
