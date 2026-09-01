# wan-worker — serves /v1/restyle (identity-preserving video restylization, the
# id-v2v/diffsynth Wan2.1 I2V rail) + the Bernini t2v/v2v/r2v rail behind the
# live-runner.
#
# ONE torch + ONE SageAttention build, on ONE Python (3.12). The two stacks are
# merged by:
#   * moving BOTH venvs to Python 3.12 (noble's native interpreter — no deadsnakes
#     PPA, so no flaky launchpad dependency at build time);
#   * relaxing id-v2v's conservative `python_requires ==3.10.*` gate with
#     `--ignore-requires-python` (it runs fine on torch 2.13; the 3.10 pin is
#     just untested-metadata);
#   * creating the Bernini venv FROM the main venv with `--system-site-packages`,
#     so it INHERITS the main venv's torch + SageAttention (ONE compile of each)
#     and diffsynth. Bernini's OWN site-packages holds only what it pins that
#     must be isolated from the idv2v stack: transformers==4.57.3 (idv2v /
#     diffsynth need >=5.6) and diffusers==0.35.2, plus veomni/ftfy/decord. A
#     venv's own site-packages precedes the inherited system-site-packages on
#     sys.path, so the pinned versions shadow the main venv's for the Bernini
#     subprocess — isolation without a second torch or second sage build.
#
# `config.BERNINI_VENV_PY` still resolves to /opt/bernini/venv/bin/python (the
# manager's resident subprocess), which is now the inheriting Bernini venv —
# NO runner-code change needed.
#
# Bernini's MLLM (Qwen2.5-VL planner) runs on torch SDPA (mllm_attn_implementation
# ="sdpa"); the flash_attn package is only needed to satisfy
# modeling_qwen2_5_vl.py's import guard. Instead of building flash-attn, this
# image ships a tiny pure-python `flash_attn` shim (SDPA-backed) into the Bernini
# venv site-packages. SageAttention is built once (used by the idv2v engine).
#
# Build (from the video-creator-platform repo root):
#   docker build -f runner/docker/wan-worker.Dockerfile -t <reg>/video-creator:wan-worker

FROM nvidia/cuda:13.0.0-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Native Python 3.12 (noble) — no deadsnakes PPA required.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3.12-dev \
    git curl ffmpeg build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --- Main worker venv (python 3.12, torch cu128 for Blackwell) ---
# This is the SINGLE torch + SINGLE SageAttention build. The Bernini venv below
# inherits both via --system-site-packages.
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir \
    "torch>=2.7.0" "torchvision>=0.22.0" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cu128

COPY runner/idv2v/requirements.txt /tmp/idv2v-requirements.txt
RUN pip install --no-cache-dir -r /tmp/idv2v-requirements.txt

# --- diffsynth (Wan pipeline fork) + idv2v package from the reference repo. ---
# --no-deps bypasses the reference's stale hash-pinned torch2.6/cu11/flash-attn
# entries; --ignore-requires-python bypasses id-v2v's python_requires ==3.10.*
# gate so it installs on 3.12.
ARG IDV2V_REF_REPO=https://github.com/Eyeline-Labs/ID-V2V
ARG IDV2V_REF_COMMIT=33dd047
RUN pip install --no-cache-dir "setuptools<81" \
    && git clone --depth 1 "${IDV2V_REF_REPO}" /opt/idv2v-ref \
    && cd /opt/idv2v-ref \
    && (test "${IDV2V_REF_COMMIT}" = "HEAD" || git checkout "${IDV2V_REF_COMMIT}" || true) \
    && pip install --no-cache-dir --ignore-requires-python --no-deps --no-build-isolation ./diffsynth_studio . \
    && rm -rf /opt/idv2v-ref/.git

ENV TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS="6"
# ONE SageAttention build (Blackwell sm_120) — used by the id-v2v restyle engine.
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/thu-ml/SageAttention.git"

# --- Bernini venv (python 3.12) — INHERITS torch + SageAttention from main. ---
# Created from /opt/venv/bin/python (same 3.12) with --system-site-packages so it
# reuses the main venv's torch, SageAttention and diffsynth (no second builds).
# Its own site-packages carries ONLY the pinned/conflicting + missing runtime
# deps (transformers 4.57.3 shadows the main venv's >=5.6 for the Bernini
# subprocess; veomni/decord/ftfy are absent from the main venv).
ARG BERNINI_REF_REPO=https://github.com/ByteDance/Bernini
ARG BERNINI_REF_COMMIT=HEAD
RUN git clone --depth 1 "${BERNINI_REF_REPO}" /opt/bernini/src \
    && (test "${BERNINI_REF_COMMIT}" = "HEAD" || git -C /opt/bernini/src checkout "${BERNINI_REF_COMMIT}" || true) \
    && rm -rf /opt/bernini/src/.git \
    && /opt/venv/bin/python -m venv --system-site-packages /opt/bernini/venv \
    && /opt/bernini/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/bernini/venv/bin/pip install --no-cache-dir \
         "transformers==4.57.3" "diffusers==0.35.2" "accelerate==0.34.2" \
         "veomni==0.1.11" ftfy decord \
         einops imageio imageio-ffmpeg opencv-python-headless Pillow tqdm \
         sentencepiece "protobuf<5" safetensors

# Worker source
RUN useradd -m runneruser
WORKDIR /app
COPY runner/ ./runner/
# Ship the pure-python `flash_attn` shim into the Bernini venv so
# modeling_qwen2_5_vl.py's import guard is satisfied (SDPA-backed).
COPY runner/idv2v/attn_shim/flash_attn/ /opt/bernini/venv/lib/python3.12/site-packages/flash_attn/
RUN /opt/bernini/venv/bin/python /app/runner/idv2v/bernini_progress_patch.py
# Route Bernini's Wan-transformer attention through SageAttention (densely
# fused kernel) with FA2/SDPA fallback. SageAttention is already in the Bernini
# venv (inherited from the main venv); this only wires the dispatch.
RUN /opt/bernini/venv/bin/python /app/runner/idv2v/bernini_sage_patch.py
RUN mkdir -p /models && chown runneruser:runneruser /models
USER runneruser

HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8992/health || exit 1

EXPOSE 8992
ENTRYPOINT ["python", "-m", "runner.idv2v"]
