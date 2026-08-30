# wan-worker OVERLAY — deploy-time image for the id-v2v worker renamed
# ("wan-worker" engine id stays `idv2v`, container keeps the same 8992 service).
#
# Derives from the already-built + pushed Hub `idv2v-worker` image instead of a
# from-scratch CUDA base rebuild: that base already carries torch cu128, the
# Wan/VACE diffsynth fork, SAM3, and the id-v2v engine. All we ADD here is:
#   1. the updated runner/ tree (server.py + bernini_cli.py + bernini.py +
#      bernini_io.py — the Bernini /t2v /v2v /r2v rail) + bernini_fa_patch.py,
#   2. an isolated, SELF-CONTAINED Bernini venv (validated end-to-end on .151,
#      RTX 5090 / sm120 / cc 12.0: a real t2v job rendered OK).
#
# Build (from the video-creator-platform repo root):
#   docker build -f runner/docker/wan-worker-overlay.Dockerfile \
#     -t adastravideo/video-creator:wan-worker .
#
# NOTE: This overlay must be re-built AFTER every change to runner/idv2v
# server/bernini code and re-pushed so the .151 box can `docker compose pull`.

FROM adastravideo/video-creator:idv2v-worker

# --- Bernini isolated venv (python3.10, fully self-contained) ---
# WHY self-contained: the base image keeps its own (different-version)
# torch/transformers/diffusers in /opt/venv, which is INVISIBLE to any nested
# venv (torch was missing -> the original "No module named 'torch'" failure)
# and WOULD CLASH if inherited. So install a coherent, validated sm120 stack
# directly into the venv (no --system-site-packages / no symlinks):
#   * torch==2.13.0 (+cu130, ships CUDA; supports Blackwell sm120)
#   * transformers==4.57.3 + diffusers==0.35.2  — BERNNINI'S OWN PINS, KEPT.
#     (4.57.3 lacks top-level AutoImageProcessor, but diffusers 0.35.2 only
#     imports it on loader paths Bernini never touches, so 4.57.3 is correct
#     and keeps the _pad_input/_upad_input/fa_peft_integration_check symbols
#     Bernini's modeling_qwen2_5_vl.py imports.)
#   * huggingface-hub<1.0 (transformers-compatible; base's 1.27 is too new)
#   * accelerate==0.34.2 (Wan2.2 low_cpu_mem_usage load), ftfy/scipy/ninja/
#     decord + the rest of Bernini's core requirements
#   * VeOmni (triton attention backend; its py3.11+ guard is bypassed via
#     --ignore-requires-python so it installs on the base's python3.10)
#   * the modeling patch below softens Bernini's hard fa2/fa3 import gate into
#     an SDPA fallback — no sm120-compatible flash-attn exists (FA3 unsupported
#     on consumer Blackwell; FA2 sm120 is a heavy source build).
ARG BERNINI_REF_REPO=https://github.com/ByteDance/Bernini
USER root
RUN git clone --depth 1 "${BERNINI_REF_REPO}" /opt/bernini/src \
    && rm -rf /opt/bernini/src/.git \
    && python3 -m venv /opt/bernini/venv \
    && /opt/bernini/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/bernini/venv/bin/pip install --no-cache-dir \
         "torch==2.13.0" torchvision \
    && /opt/bernini/venv/bin/pip install --no-cache-dir \
         "transformers==4.57.3" "diffusers==0.35.2" \
         "huggingface-hub>=0.34.0,<1.0" "accelerate==0.34.2" \
         ftfy scipy ninja decord einops imageio imageio-ffmpeg \
         opencv-python-headless Pillow tqdm sentencepiece \
         "protobuf<5" safetensors \
    && /opt/bernini/venv/bin/pip install --no-cache-dir --no-deps --ignore-requires-python \
         "git+https://github.com/ByteDance-Seed/VeOmni.git@v0.1.11"

WORKDIR /app
COPY runner/ ./runner/
# Soften Bernini's hard flash-attn(2/3) import gate -> SDPA fallback. Pure-text
# patch; asserts on structure so a Bernini clone change fails loudly.
RUN /opt/bernini/venv/bin/python /app/runner/idv2v/bernini_fa_patch.py
RUN /opt/bernini/venv/bin/python /app/runner/idv2v/bernini_progress_patch.py
USER runneruser

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8992/health || exit 1

EXPOSE 8992
ENTRYPOINT ["python", "-m", "runner.idv2v"]
