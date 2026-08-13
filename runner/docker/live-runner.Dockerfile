# Live-Runner edge — registers + heartbeats to the Livepeer Orchestrator as
# app `video-creator`, owns the GPU swap policy, routes + proxies to workers.
# Intentionally THIN: no torch / no ltx / no diffsynth. It only talks HTTP.
#
# Build (from the video-creator repo root):
#   docker build -f docker/live-runner.Dockerfile -t video-creator-live-runner .

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Live-runner thin deps: aiohttp + the Livepeer gateway SDK (live-runner branch).
COPY runner/live_runner/requirements.txt /tmp/live-runner-requirements.txt
RUN pip install --no-cache-dir -r /tmp/live-runner-requirements.txt

# Runner source (whole runner tree; only live_runner is imported at runtime).
COPY runner ./runner/

RUN useradd -m runneruser
USER runneruser

# Health probe on the live-runner's own edge health (no auth needed for probe).
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8991/video-creator/v1/health || exit 1

EXPOSE 8991
ENTRYPOINT ["python", "-m", "runner.live_runner"]
