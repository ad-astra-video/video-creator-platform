# Deploy diff — Bernini rail + vp-worker (Task 9) — applied to .151 compose

Applies to `/srv/video-creator/docker/docker-compose.video-creator.yml`
AFTER `vp-worker` + `wan-worker` image builds are built + pushed to Hub.

## 1. live-runner environment — add the two worker URLs

In the `live-runner:` service `environment:` block, add:

```yaml
      - VP_WORKER_URL=http://vp-worker:8995
      # wan-worker = idv2v-worker renamed (same engine id, same 8992 port,
      # serves restyle + the new /t2v /v2v /r2v Bernini rail).
      - WAN_WORKER_URL=http://idv2v-worker:8992
```

## 2. New `vp-worker:` service (add before `networks:`)

```yaml
  vp-worker:
    image: adastravideo/video-creator:vp-worker
    container_name: video-creator-vp-worker
    depends_on:
      live-runner:
        condition: service_healthy
    runtime: nvidia
    environment:
      - WORKER_TOKEN=${WORKER_TOKEN:-}
      - PORT=8995
      - RIFE_ROOT=${RIFE_ROOT:-/models/rife/v426}
      - FLASHVSR_ROOT=${FLASHVSR_ROOT:-/models/flashvsr}
      - SAM3_CKPT=${SAM3_CKPT:-/models/sam3}
      - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    volumes:
      - /srv/video-creator/models:/models
    networks:
      - vc
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8995/video-creator/v1/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 60s
```

## 3. idv2v-worker — serve Bernini (wan-worker routes)

The idv2v-worker container already IS the wan-worker engine (same 8992 port).
New `/t2v /v2v /r2v` + `/bernini/evict` routes come from the rebuilt
`idv2v-worker` image (which includes runner/idv2v server.py + bernini_*).
Add Bernini env to the `idv2v-worker:` service:

```yaml
      - BERNINI_ROOT=${BERNINI_ROOT:-/models/Bernini-R-1.3B-Diffusers}
```

## Validation before/after

- `docker compose config | grep -E 'image:|source:' | grep -iE 'vp|models'`
- `docker compose ps` — expect 8 services up
- live-runner `/video-creator/v1/health` -> 200, capabilities include
  `bernini-t2v/v2v/r2v` + `process/fps-boost/upscale/ffmpeg`
- vp-worker `/video-creator/v1/health` -> 200
- idv2v-worker `/info` capabilities include `bernini-*`
