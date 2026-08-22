# wan-worker OVERLAY — deploy-time image for the id-v2v worker renamed
# ("wan-worker" engine id stays `idv2v`, container keeps the same 8992 service).
#
# Derives from the already-built + pushed Hub `idv2v-worker` image instead of a
# from-scratch CUDA base rebuild: that base already carries torch cu128, the
# Wan/VACE diffsynth fork, SAM3, and the id-v2v engine. All we ADD here is:
#   1. the updated runner/ tree (server.py + bernini_cli.py + bernini.py +
#      bernini_io.py — the Bernini /t2v /v2v /r2v rail),
#   2. an isolated Bernini venv that REUSES the base torch via
#      --system-site-packages (so we don't re-download ~3GB of torch wheels),
#      pinned to transformers 4.57.3 + ByteDance/Bernini source + runtime deps.
#
# Build (from the video-creator-platform repo root):
#   docker build -f runner/docker/wan-worker-overlay.Dockerfile \
#     -t adastravideo/video-creator:wan-worker .
#
# NOTE: This overlay must be re-built AFTER every change to runner/idv2v
# server/bernini code and re-pushed so the .151 box can `docker compose pull`.

FROM adastravideo/video-creator:idv2v-worker

# --- Bernini isolated venv (reuses base torch — no torch re-download) ---
# Idempotent: only builds if /opt/bernini/venv is missing. Because the base
# image already has torch that may be a NEWER 2.x than Bernini wants, we install
# --no-deps (keep base torch/diffusers) and only pin transformers (which IS
# separable and isolated here) + the small runtime deps. The Bernini source is
# cloned at build like the idv2v reference repo.
ARG BERNINI_REF_REPO=https://github.com/ByteDance/Bernini
USER root
RUN git clone --depth 1 "${BERNINI_REF_REPO}" /opt/bernini/src \
    && rm -rf /opt/bernini/src/.git \
    && python3 -m venv --system-site-packages /opt/bernini/venv \
    && /opt/bernini/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/bernini/venv/bin/pip install --no-cache-dir --no-deps \
         "transformers==4.57.3" einops imageio imageio-ffmpeg \
         opencv-python-headless Pillow tqdm sentencepiece "protobuf<5" safetensors

WORKDIR /app
COPY runner/ ./runner/
USER runneruser

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8992/health || exit 1

EXPOSE 8992
ENTRYPOINT ["python", "-m", "runner.idv2v"]
