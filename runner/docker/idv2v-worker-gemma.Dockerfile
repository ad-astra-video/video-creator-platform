# ID-V2V Worker — Gemma 3 LLM enhancement/captioning variant.
#
# Derived FROM the deployed fusionx image so we inherit the full heavy
# toolchain (torch 2.13+cu130 for sm_120, diffsynth fork, SageAttention 2) AND
# the downloaded FusionX fp8 weights reference — WITHOUT recompiling anything.
# We only overlay the worker source (runner/) that adds:
#   * Gemma 3 LLM (Lightricks/gemma-3-12b-it-qat-q4_0-unquantized on the shared
#     /models/gemma mount) for restyle prompt enhancement + auto video caption.
#   * config.gemma_enabled() / GEMMA_ROOT / GEMMA_GPU_DEVICE knobs.
# The Gemma model itself is NOT packaged in the image — it lives on the shared
# /models/gemma volume (provisioned once by the LTX runner).
#
# Build (from the video-creator repo root):
#   docker build -f docker/idv2v-worker-gemma.Dockerfile \
#       -t adastravideo/video-creator:idv2v-worker-gemma .
#
# The base image must already be present locally:
#   adastravideo/video-creator:idv2v-worker-fusionx

FROM adastravideo/video-creator:idv2v-worker-fusionx

LABEL org.opencontainers.image.description="id-v2v worker with Gemma 3 LLM (prompt enhance + auto video caption)"

# Re-overlay the worker source with the Gemma-LLM-enabled build. WORKDIR +
# ENTRYPOINT inherited from the base.
WORKDIR /app
COPY runner/ ./runner/
