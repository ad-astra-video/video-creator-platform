"""image-worker: Qwen-Image-Edit / Qwen-Image-Layered / Z-Image inference worker.

Serves the /video-creator/v1/* image surface (edit / layer / image) plus the
token-gated worker control plane (/load /evict /health /info) behind the
live-runner edge, mirroring the structure of the runner/ltx worker.
"""
