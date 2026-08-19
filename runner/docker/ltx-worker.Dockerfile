# LTX Worker — serves /video-creator/v1/* (generate, retake, extend, ic-lora)
# behind the live-runner. Heavy image: torch + LTX core/pipelines. Does NOT
# register with the Orchestrator (that's the live-runner's job); it only serves
# the internal /load /evict /v1/* surface over the Docker network.
#
# CUDA 12.8 wheels cover Ada (4090, SM89) + Blackwell (5090 / RTX PRO 6000, SM120).
#
# Build (from the video-creator repo root):
#   docker build -f docker/ltx-worker.Dockerfile -t video-creator-ltx-worker .

FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System deps (ffmpeg for video concat; curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git curl ca-certificates ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Core Python deps (aiohttp, Pillow, numpy, av, requests, huggingface_hub)
COPY runner/requirements.txt /tmp/runner-requirements.txt
RUN pip install --no-cache-dir -r /tmp/runner-requirements.txt

# PyTorch 2.13 (cu132) — aligns the LTX-2.5 diffusion-VAE decode with the stack
# upstream Lightricks/LTX-2 sanctions in packages/ltx-core's natten extra:
#   torch==2.13.0 (cu132) + natten 0.21.7+torch2130cu132
# (comment there: 'so DiffVAE does not hit ... on older PyTorch/NVIDIA stacks').
# An older / unpinned cu128 torch+triton crashes the raw swiglu kernel at decode
# with 'ValueError: Pointer argument ... cannot be accessed from Triton (cpu tensor?)'.
# Requires a host driver supporting CUDA 13.2 (check nvidia-smi on the GPU box).
# torchaudio has no wheel on the cu132 MAIN index for torch 2.13; upstream sources it from
# the cu132 TEST index (uv ``torchaudio = { index = "torch-test" }``), so add it as an
# extra source here. torch==2.13.0 stays pinned so main supplies torch/torchvision.
RUN pip install --no-cache-dir \
    torch==2.13.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu132 \
    --extra-index-url https://download.pytorch.org/whl/test/cu132/

# LTX core + pipelines from the LTX-2 repo (pinned to the ModelPaths-era rev that
# adds LTX-2.5 support; earlier rev 93777581 has no ModelPaths API and cannot load 2.5).
RUN pip install --no-cache-dir \
    "git+https://github.com/Lightricks/LTX-2.git@fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca#subdirectory=packages/ltx-core" \
    "git+https://github.com/Lightricks/LTX-2.git@fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca#subdirectory=packages/ltx-pipelines"

# NATTEN: the real 3D Neighborhood-Attention backend for the LTX-2.5 diffusion
# video VAE decoder. Without it ltx-core falls back to a Triton na3d kernel that
# crashes under the compiled decoder ("Pointer argument ... cannot be accessed
# from Triton (cpu tensor?)"). Pin to the wheel matching our torch build
# (2.13.0+cu132, py3.12) straight from the SHI-Labs NATTEN release (whl.natten.org
# is not PEP 503 for pip local-version resolution, so install the .whl directly).
RUN pip install --no-cache-dir \
    "https://github.com/SHI-Labs/NATTEN/releases/download/v0.21.7/natten-0.21.7%2Btorch2130cu132-cp312-cp312-linux_x86_64.whl"

# Diffusers for image generation (Z-Image-Turbo)
RUN pip install --no-cache-dir diffusers accelerate

RUN useradd -m runneruser

WORKDIR /app
COPY runner/ ./runner/

RUN mkdir -p /models && chown runneruser:runneruser /models
USER runneruser

# Worker liveness probe (open, no token).
HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8991/video-creator/v1/health || exit 1

EXPOSE 8991
# Entry: the LTX worker server. Uses python -m runner.ltx.server's main().
ENTRYPOINT ["python", "-m", "runner.ltx.server"]
