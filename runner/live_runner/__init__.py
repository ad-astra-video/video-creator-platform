"""Live-runner: the single Livepeer-facing edge for the Video-Creator runner.

A thin aiohttp service that registers + heartbeats with the Livepeer
Orchestrator, owns the shared-GPU swap policy across the LTX and ID-V2V worker
containers, routes each /video-creator/v1/* request to the right worker, and
proxies request/response. It carries no heavy ML deps — just aiohttp + the
gateway SDK + aiohttp-client.

Container layout (three images, one shared 32 GB GPU):
    live-runner  <->  ltx-worker  /  idv2v-worker   (internal Docker network)
Only one worker has its model resident at a time; the live-runner enforces it
via POST /evict on the current resident then POST /load on the target.
"""
