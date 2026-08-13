"""ID-V2V worker package for the Video-Creator runner.

Serves the identity-preserving video restylization model (Wan 2.1 I2V-14B DiT +
VACE-14B, int8-quantized, CPU-offloaded on a 32 GB 5090) as a swappable worker
container. Exposes /health, /load, /evict, and /v1/restyle for the live-runner
edge, which owns the shared-GPU swap policy across workers.
"""
