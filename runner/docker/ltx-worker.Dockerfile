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

# PyTorch with CUDA 12.8 (Ada SM89 + Blackwell SM120)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# LTX core + pipelines from the LTX-2 repo (same rev the desktop backend pins).
RUN pip install --no-cache-dir \
    "git+https://github.com/Lightricks/LTX-2.git@9377758131b1ffde4b7f766804590a6617bf2ab9#subdirectory=packages/ltx-core" \
    "git+https://github.com/Lightricks/LTX-2.git@9377758131b1ffde4b7f766804590a6617bf2ab9#subdirectory=packages/ltx-pipelines"

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
