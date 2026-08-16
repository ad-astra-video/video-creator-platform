# ID-V2V Worker — serves /v1/restyle (identity-preserving video restylization)
# behind the live-runner. Heavy image: torch + Wan2.1 I2V-14B DiT + VACE
# (diffsynth fork) + SAM3, int8-quantized with CPU offload, tuned for a 32GB
# RTX 5090. Does NOT register with the Orchestrator (the live-runner's job);
# it only serves the internal /health /load /evict /v1/restyle surface.
#
# Build (from the video-creator repo root):
#   docker build -f docker/idv2v-worker.Dockerfile -t video-creator-idv2v-worker
#
# Blackwell (sm_120) notes — RTX 5090:
#   * The reference repo (Eyeline-Labs/ID-V2V) pins torch==2.6 cu118 + a cu11
#     flash-attn wheel, both of which have NO sm_120 kernels -> on a 5090 that
#     throws "no kernel image is available for execution on the device".
#   * torch >= 2.7 cu128 is the first line with official sm_120 kernels.
#   * flash-attn is OPTIONAL: diffsynth's flash_attention() is a try/except gate
#     that falls back to native F.scaled_dot_product_attention (sm_120-capable)
#     when no flash lib is installed. We omit it -> zero-compile path.
#   * We install the reference source + deps with --no-deps to bypass the
#     stale hash-pinned torch2.6/cu11/flash-attn entries in its pyproject/uv.lock.

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Python 3.10 via deadsnakes — Ubuntu 24.04 base doesn't ship 3.10 in the main
# repos, but the diffsynth fork's flash_attn dep only ships cp310 wheels, so we
# match the reference build's Python. (flash-attn is optional, but keeping py3.10
# matches the reference and is harmless.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    git curl ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --- PyTorch for Blackwell (sm_120): torch >= 2.7 from the cu128 index. ---
# Installed here (before requirements) so torch/torchvision are authoritative
# and no later requirements line can downgrade them to a cu124/cu11 build.
RUN pip install --no-cache-dir \
    "torch>=2.7.0" "torchvision>=0.22.0" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128

# --- Worker + runtime deps (NO torch/torchvision below — see above). ---
COPY runner/idv2v/requirements.txt /tmp/idv2v-requirements.txt
RUN pip install --no-cache-dir -r /tmp/idv2v-requirements.txt

# FLUX.2 klein 4B moved to the image-worker (removed here).

# --- diffsynth (Wan pipeline fork) + idv2v package from the reference repo. ---
# Install with --no-deps so the stale hash-pinned torch2.6/cu11/flash-attn lines
# in the reference pyproject/uv.lock are bypassed. flash-attn is intentionally
# omitted (SDPA fallback).
ARG IDV2V_REF_REPO=https://github.com/Eyeline-Labs/ID-V2V
# Pin the reference commit the worker's model.py was ported against. The
# reference's own Dockerfile uses 33dd047; deviating to HEAD drifts the diffsynth
# API (e.g. model_manager.match() changing to pass a list of paths -> crash).
ARG IDV2V_REF_COMMIT=33dd047
RUN pip install --no-cache-dir "setuptools<81" \
    && git clone --depth 1 "${IDV2V_REF_REPO}" /opt/idv2v-ref \
    && cd /opt/idv2v-ref \
    && (test "${IDV2V_REF_COMMIT}" = "HEAD" || git checkout "${IDV2V_REF_COMMIT}" || true) \
    && pip install --no-cache-dir --no-deps --no-build-isolation ./diffsynth_studio . \
    && rm -rf /opt/idv2v-ref/.git

# SageAttention 2 (sm_120/Blackwell) — diffsynth's Wan attention already has a
# SAGE_ATTN_AVAILABLE branch (flash_attention() falls through flash3->flash2->SAGE
# ->sdpa); installing `sageattention` activates it with no pipeline changes.
# Source-built so it matches this image's torch 2.13+cu130 + Triton 3.7.
# SageAttention V2 compiles CUDA kernels from source (PyPI's `sageattention`
# is only V1/Triton — slower). Target sm_120 (RTX 5090/Blackwell) explicitly;
# nvcc comes from the cu13 devel base so it matches torch 2.13+cu130.
ENV TORCH_CUDA_ARCH_LIST="12.0"
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/thu-ml/SageAttention.git"

# Worker source
RUN useradd -m runneruser
WORKDIR /app
COPY runner/ ./runner/
RUN mkdir -p /models && chown runneruser:runneruser /models
USER runneruser

# Worker liveness probe (open, no token).
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8992/health || exit 1

EXPOSE 8992
ENTRYPOINT ["python", "-m", "runner.idv2v"]
