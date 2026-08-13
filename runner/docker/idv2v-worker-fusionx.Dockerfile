# ID-V2V Worker — FusionX "fast" model variant.
#
# Derived FROM the existing deployed idv2v-worker-teacache image so we inherit
# the full heavy toolchain (torch 2.13+cu130 for sm_120, diffsynth fork pinned at
# 33dd047, SageAttention 2 source build for Blackwell) WITHOUT recompiling
# anything. We only overlay the updated worker source (runner/) that adds:
#   * HF_REPO subfolder selection for the fp8 weights ("fast" -> /fusionx,
#     "regular" -> repo root), selectable per restyle request via body["model"].
#   * Variant-default denoise-step budget (fast ~8, regular 30).
#
# Build (from the video-creator repo root):
#   docker build -f docker/idv2v-worker-fusionx.Dockerfile \
#       -t adastravideo/video-creator:idv2v-worker-fusionx .
#
# The base image must already be present locally (or pullable from the Hub):
#   adastravideo/video-creator:idv2v-worker-teacache

FROM adastravideo/video-creator:idv2v-worker-teacache

# telemetry-friendly image label
LABEL org.opencontainers.image.description="id-v2v worker with fast(FusionX)/regular model variants"

# Re-overlay the worker source (config / fp8_loader / model / run / server) with
# the variant-selectable build. WORKDIR + ENTRYPOINT inherited from the base.
WORKDIR /app
COPY runner/ ./runner/
