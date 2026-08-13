# gemma-worker — the always-on-idle LLM backend (Gemma 4 via llama.cpp).
# Serves prompt-enhance + chat/agent behind the live-runner edge, with the root
# /health /load /evict control surface the swap policy drives.
#
# Blackwell (sm_120) note — RTX 5090: llama-cpp-python's prebuilt CUDA wheels
# (PyPI = CPU-only; abetlen cu* indices are manylinux_2_35 prebuilts that, like
# SageAttention, generally do NOT ship sm_120 kernels). So GGML_CUDA is compiled
# here FROM SOURCE against the CUDA 13.0 devel base (nvcc), targeting sm_120 —
# the same pattern the idv2v-worker uses for SageAttention.
#
# The GGUF is NOT baked in: operators bind-mount their already-downloaded model
# (google/gemma-4-12B-it-qat-q4_0-gguf) at /models/gemma/*.gguf and set
# GEMMA_MODEL / GEMMA_MMPROJ accordingly.
#
# Build (from the video-creator repo root):
#   docker build -f docker/gemma-worker.Dockerfile -t video-creator-gemma-worker .

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# Python 3.11 via deadsnakes (Ubuntu 24.04 base doesn't ship 3.11 in main).
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common ca-certificates gnupg && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    git curl build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Build llama-cpp-python with CUDA from source, targeting sm_120 (RTX 5090).
# nvcc comes from the cu13 devel base so it matches the host driver family the
# stack already runs (torch cu128 / nvidia-cuda:13 base).
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120" \
    FORCE_CMAKE=1
# Named version; llama-cpp-python ships its own pin. numpy is a runtime dep too.
RUN pip install --no-cache-dir "llama-cpp-python>=0.3.30"
RUN pip install --no-cache-dir "aiohttp>=3.9" "numpy>=1.24"

# Runner source (whole tree; only runner.gemma + runner.ltx.enhance_forward
# are imported at runtime — both are aiohttp-only, no torch).
COPY runner ./runner/

RUN useradd -m runneruser && mkdir -p /models && chown runneruser:runneruser /models
USER runneruser

# Liveness probe on the worker's own open /health (no auth needed for probe).
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8993/health || exit 1

EXPOSE 8993
ENTRYPOINT ["python", "-m", "runner.gemma"]
