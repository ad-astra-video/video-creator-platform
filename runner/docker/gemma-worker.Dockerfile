# gemma-worker — the always-on-idle LLM backend (Gemma 4 via llama.cpp).
# Serves prompt-enhance + chat/agent behind the live-runner edge, with the root
# /health /load /evict control surface the swap policy drives.
#
# Single-modell-host architecture: the NATIVE llama.cpp `llama-server` binary
# (built from upstream ggml-org/llama.cpp) is the ONLY resident model instance.
# The aiohttp Python worker (`runner/gemma/server.py`) is the HTTP FRONT + control
# surface the live-runner swaps; every inference endpoint proxies to llama-server's
# OpenAI-compatible `/v1*`. pydantic-ai (and any OpenAI client) talks to that same
# endpoint directly.
#
# NOTE: llama-cpp-python is intentionally NOT installed — the worker never loads a
# Python `Llama` binding (two 12B residents can't co-exist on a 32 GB card, and the
# agent/reasoning path is the native server). Dropping it avoids the huge from-source
# CUDA compile and keeps the image lean.
#
# Blackwell (sm_120) note — RTX 5090: the native llama.cpp llama-server is compiled
# here FROM SOURCE against the CUDA 13.0 devel base (nvcc) because prebuilt `cu*`
# binaries generally do NOT ship sm_120 kernels. We target a cross-generation SASS
# set so the SAME image runs on Ampere (sm_80 / A100, RTX 3090), Ada (sm_89 /
# RTX 4090) and Blackwell (sm_120 / RTX 5090): CMAKE_CUDA_ARCHITECTURES=80;89;120.
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
    libvulkan1 \
    libx11-6 libxext6 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Runtime python deps for the aiohttp control/proxy worker + media helpers.
RUN pip install --no-cache-dir "aiohttp>=3.9" "numpy>=1.24" "av>=13.0" "Pillow>=10.0"

# Native llama.cpp `llama-server` (upstream ggml-org/llama.cpp). This is the
# OpenAI-compatible server that returns `reasoning_content` natively and does
# function/tool calling — the endpoint pydantic-ai agents point at for the Gemma
# backlog. Built from source for sm_120 (Blackwell) + sm_80/sm_89 (Ampere/Ada) so
# the image runs across the GPU fleet. The aiohttp worker runs it as a managed
# subprocess (spawn on /load, stop on /evict) so the GPU-swap policy still owns
# card residency.
# Bound the CUDA compile parallelism: an unbounded `-j` (all cores) spawns
# hundreds of nvcc jobs and OOM-kills the BuildKit daemon (rpc EOF) under
# Docker Desktop's limited WSL2 RAM. Override at build time with
# --build-arg BUILD_JOBS=<n>.
ARG BUILD_JOBS=4
# CUDA 13 link fix: the devel image's stub dir only ships an unversioned
# libcuda.so (SONAME libcuda.so.1). Linking llama-server then fails with
# "libcuda.so.1 ... not found" + undefined refs (cuGetErrorString, cuMemCreate,
# cuDeviceGet, ...). Provide the versioned name and put the stubs dir on the
# executable link path (the ld -rpath-link hint).
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    git clone --depth 1 --branch master https://github.com/ggml-org/llama.cpp /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=on -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="80;89;120" \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs -L/usr/local/cuda/lib64/stubs" && \
    cmake --build build --config Release -j${BUILD_JOBS} --target llama-server && \
    install -m 0755 build/bin/llama-server /usr/local/bin/llama-server && \
    # llama-server is a thin launcher that loads libllama-server-impl.so + the
    # ggml/mtmd .so sets. Copy EVERY build/bin shared-lib artifact (the *.so*
    # glob catches the versioned NAMEs as well: *.so alone only copies the
    # dangling dev symlink, not the real .so.0 payload, and an explicit
    # libllama*/libggml* list misses libmtmd). Missing any of these kills the
    # launcher at exec with "...: cannot open shared object file".
    cp -a build/bin/*.so* /usr/local/bin/ && \
    rm -rf /tmp/llama.cpp

# llama-server's launcher has NO $ORIGIN rpath and looks only on the loader's
# default search path — so export the dir the .so set was copied into, or it
# dies at exec with "libllama-server-impl.so: cannot open shared object file".
ENV LD_LIBRARY_PATH="/usr/local/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Vulkan loader bump. Ubuntu 24.04's libvulkan1 is 1.3.275, but the host
# NVIDIA 580 driver's Vulkan ICD advertises API 1.4.312; the 1.3 loader cannot
# negotiate with it ("Could not get 'vkCreateInstance' via
# 'vk_icdGetInstanceProcAddr' ... Found no drivers" -> VK_ERROR_INCOMPATIBLE_DRIVER,
# -9, and the ICD aborts with a core dump). We need a >= 1.4 loader.
#   - No noble PPA ships the loader (only the mesa drivers): verified via the
#     Launchpad API — kisak-mesa and oibaf each publish 0 libvulkan1 for noble.
#   - So pull Ubuntu's OWN newer-series loader: questing (25.10) ships
#     libvulkan1 = 1.4.321.0-1, whose ONLY dependency is libc6 (>= 2.38) —
#     satisfied by noble's 2.39, so it installs with NO libc6 upgrade / no
#     cross-release rebuild. dpkg -i correctly supersedes noble's 1.3.275 and
#     re-runs ldconfig.
# Kept as a late, separate RUN so the expensive llama.cpp sm_120 compile and the
# PyAV/Pillow layers above stay cache-hit on rebuild.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    curl -fsSL -o /tmp/libvulkan1.deb \
        http://archive.ubuntu.com/ubuntu/pool/main/v/vulkan-loader/libvulkan1_1.4.321.0-1_amd64.deb && \
    dpkg -i /tmp/libvulkan1.deb && \
    rm -f /tmp/libvulkan1.deb && rm -rf /var/lib/apt/lists/* && \
    echo "== installed libvulkan1 ==" && dpkg-query -W -f='${Package} ${Version}\n' libvulkan1

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
