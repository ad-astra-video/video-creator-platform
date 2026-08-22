# wan-worker — the id-v2v worker renamed: serves /v1/restyle (identity-preserving
# video restylization) + the Bernini t2v/v2v/r2v rail behind the live-runner.
#
# Rename note: the ENGINE identifier stays `idv2v` (existing diffsynth Wan2.1
# I2V fine-tune); the CONTAINER/IMAGE name becomes `wan-worker`. The Python
# package stays `runner.idv2v`.
#
# Bernini runs in an ISOLATED venv (/opt/bernini/venv) + subprocess because its
# pinned transformers==4.57.3 conflicts with the image/idv2v stack needing >=5.6
# (same pattern as SAM3). bernini_cli.py (reused via /opt/bernini/src on
# PYTHONPATH) builds the Diffusers pipeline once and stays resident.
#
# Build (from the video-creator-platform repo root):
#   docker build -f runner/docker/wan-worker.Dockerfile -t <reg>/video-creator:wan-worker

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    python3.11 python3.11-venv python3.11-dev \
    git curl ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- Main worker venv (python 3.10, torch cu128 for Blackwell) ---
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir \
    "torch>=2.7.0" "torchvision>=0.22.0" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128

COPY runner/idv2v/requirements.txt /tmp/idv2v-requirements.txt
RUN pip install --no-cache-dir -r /tmp/idv2v-requirements.txt

# --- diffsynth (Wan pipeline fork) + idv2v package from the reference repo. ---
ARG IDV2V_REF_REPO=https://github.com/Eyeline-Labs/ID-V2V
ARG IDV2V_REF_COMMIT=33dd047
RUN pip install --no-cache-dir "setuptools<81" \
    && git clone --depth 1 "${IDV2V_REF_REPO}" /opt/idv2v-ref \
    && cd /opt/idv2v-ref \
    && (test "${IDV2V_REF_COMMIT}" = "HEAD" || git checkout "${IDV2V_REF_COMMIT}" || true) \
    && pip install --no-cache-dir --no-deps --no-build-isolation ./diffsynth_studio . \
    && rm -rf /opt/idv2v-ref/.git

ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/thu-ml/SageAttention.git"

# --- Bernini isolated venv (python 3.11, its own pinned transformers) ---
# Cloned at build time (like the idv2v reference repo above) so its large source
# tree isn't vendored into the platform repo. Installed on PYTHONPATH for the
# bernini_cli subprocess; its own venv keeps transformers at 4.57.3 (isolated
# from the worker's >=5.6).
ARG BERNINI_REF_REPO=https://github.com/ByteDance/Bernini
ARG BERNINI_REF_COMMIT=HEAD
RUN git clone --depth 1 "${BERNINI_REF_REPO}" /opt/bernini/src \
    && (test "${BERNINI_REF_COMMIT}" = "HEAD" || git -C /opt/bernini/src checkout "${BERNINI_REF_COMMIT}" || true) \
    && rm -rf /opt/bernini/src/.git \
    && python3.11 -m venv /opt/bernini/venv \
    && /opt/bernini/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/bernini/venv/bin/pip install --no-cache-dir \
         "torch>=2.7.0" "torchvision>=0.22.0" \
         --index-url https://pypi.org/simple \
         --extra-index-url https://download.pytorch.org/whl/cu128 \
    && /opt/bernini/venv/bin/pip install --no-cache-dir \
         "diffusers==0.35.2" "transformers==4.57.3" "accelerate==0.34.2" \
         einops imageio imageio-ffmpeg opencv-python-headless Pillow tqdm \
         sentencepiece "protobuf<5" safetensors \
    && (cd /opt/bernini/src && /opt/bernini/venv/bin/pip install --no-cache-dir --no-deps . ) || true

# Worker source
RUN useradd -m runneruser
WORKDIR /app
COPY runner/ ./runner/
RUN mkdir -p /models && chown runneruser:runneruser /models
USER runneruser

HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8992/health || exit 1

EXPOSE 8992
ENTRYPOINT ["python", "-m", "runner.idv2v"]
