# vp-worker — dedicated low-VRAM video-processing container: RIFE fps-boost +
# FlashVSR upscale + ffmpeg + standalone SAM3. Orchestrated by the live-runner
# as a combined /process post-stage after ANY render worker; render workers
# never call it themselves.
#
# Build (from the video-creator-platform repo root):
#   docker build -f runner/docker/vp-worker.Dockerfile -t <reg>/video-creator:vp-worker .
#
# FlashVSR notes (calibrated on the RTX 5090 / compute_120 / CUDA 13.0):
#   * diffsynth fork (OpenImagingLab/FlashVSR setup.py) supports sm_120 when
#     built with CUDA >= 12.8 — we set TORCH_CUDA_ARCH_LIST=12.0 so the
#     Block-Sparse-Attention CUDA kernel is compiled for Blackwell (sm120).
#   * FlashVSR hard-pins a cu124 toolchain in requirements.txt, so diffsynth is
#     pip-installed --no-deps and we pin the handful of runtime deps ourselves,
#     keeping torch cu128 for Blackwell (see verify below).
#   * the pipeline needs libc10/other torch .so at import: LD_LIBRARY_PATH must
#     include torch/lib.

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

# torch cu128 (Blackwell sm_120 kernels), authoritative. Keep cu128 for the box.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "torch>=2.7.0" "torchvision>=0.22.0" \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu128

# torch .so must be findable at import-time by compiled extensions (libc10.so).
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.11/site-packages/torch/lib"

# --- FlashVSR backend: the OpenImagingLab/FlashVSR diffsynth source tree. ---
# Provides ModelManager + FlashVSRTinyPipeline. Installed --no-deps so torch
# stays cu128; we pin the runtime deps ourselves below (requirements.txt pins a
# cu124 toolchain that conflicts with this image).
ARG FLASHVSR_REPO=https://github.com/OpenImagingLab/FlashVSR
RUN git clone --depth 1 "${FLASHVSR_REPO}" /opt/flashvsr \
    && cd /opt/flashvsr \
    && pip install --no-cache-dir --no-deps --no-build-isolation /opt/flashvsr \
    && rm -rf /opt/flashvsr/.git

# diffsynth runtime deps that MUST match its fork (transformers 4.46.2 imports
# PretrainedConfig from transformers.modeling_utils — newer 5.x moved it).
RUN pip install --no-cache-dir --upgrade pip \
    "transformers==4.46.2" "sentencepiece==0.2.0" "accelerate" "ftfy" "protobuf" \
    "huggingface_hub" "safetensors>=0.5.3" "modelscope" "einops" "omegaconf" \
    "opencv-python-headless" "Pillow" "tqdm" "imageio" "imageio-ffmpeg" \
    "matplotlib" "pandas" "peft" "pytorch-lightning" "torchsde" "datasets" "aiohttp"

# --- Block-Sparse-Attention CUDA kernel (FlashVSR's attention backend). ---
# Not on PyPI; built from mit-han-lab source. Must compile for sm120 (Blackwell,
# CUDA>=12.8) which it supports. MAX_JOBS=2 avoids the OOM seen at -j9.
RUN git clone --depth 1 https://github.com/mit-han-lab/Block-Sparse-Attention /opt/Block-Sparse-Attention \
    && pip install --no-cache-dir packaging ninja \
    && cd /opt/Block-Sparse-Attention \
    && MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 python setup.py install \
    && cd / && rm -rf /opt/Block-Sparse-Attention/build

# Aux posi-prompt tensor FlashVSRTinyPipeline.init_cross_kv needs (baked fallback
# so we don't depend on the /models volume or a CWD-relative path).
ARG FLASHVSR_PROMPT_URL=https://raw.githubusercontent.com/OpenImagingLab/FlashVSR/main/examples/WanVSR/prompt_tensor/posi_prompt.pth
RUN mkdir -p /opt/flashvsr_prompt \
    && curl -fSL -o /opt/flashvsr_prompt/posi_prompt.pth "${FLASHVSR_PROMPT_URL}" \
    || echo "WARN: could not fetch posi_prompt.pth at build (server will fall back to FLASHVSR_ROOT)"

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
