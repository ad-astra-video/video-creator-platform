# vp-worker — dedicated low-VRAM video-processing container: RIFE fps-boost +
# FlashVSR upscale + ffmpeg + standalone SAM3. Orchestrated by the live-runner
# as a combined /process post-stage after ANY render worker; render workers
# never call it themselves.
#
# Light on VRAM: rails load lazily and stay warm independently (RIFE ~1GB,
# FlashVSR tiny ~5-6GB). No big DiT resident by default. Blackwell (sm_120).
#
# Build (from the video-creator-platform repo root):
#   docker build -f runner/docker/vp-worker.Dockerfile -t <reg>/video-creator:vp-worker

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates gnupg ffmpeg git curl build-essential \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV TORCH_CUDA_ARCH_LIST="12.0"

# torch cu128 (Blackwell sm_120 kernels), authoritative.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "torch>=2.7.0" "torchvision>=0.22.0" \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu128

# --- FlashVSR backend: the OpenImagingLab/FlashVSR diffsynth source tree.
# Provides ModelManager + FlashVSRTinyPipeline. Installed --no-deps so torch
# stays cu128 and we pin the few runtime deps ourselves (einops/omegaconf...).
ARG FLASHVSR_REPO=https://github.com/OpenImagingLab/FlashVSR
RUN git clone --depth 1 "${FLASHVSR_REPO}" /opt/flashvsr \
    && cd /opt/flashvsr \
    && pip install --no-cache-dir --no-deps --no-build-isolation /opt/flashvsr \
    && rm -rf /opt/flashvsr/.git

# Runtime deps for the rails (RIFE + FlashVSR + ffmpeg wrapper).
RUN pip install --no-cache-dir \
    numpy einops omegaconf huggingface_hub safetensors \
    opencv-python-headless Pillow tqdm imageio \
    imageio-ffmpeg aiohttp

# Worker source (runner/vp + the vendored flashvsr_utils + runner/idv2v for
# the shared SAM3 segment_single module).
RUN useradd -m runneruser
WORKDIR /app
COPY runner/ ./runner/
RUN mkdir -p /models && chown runneruser:runneruser /models
USER runneruser

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8995/health || exit 1

EXPOSE 8995
ENTRYPOINT ["python", "-m", "runner.vp"]
