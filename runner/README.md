# Video-Creator Runner

The GPU inference backend for Video-Creator. This `runner/` folder holds the
self-contained inference services that **node operators launch** as Docker
images — there are no host-side deploy scripts (models and env vars are provided
at `docker run` time).

| Worker | Folder | Role |
|--------|--------|------|
| `live-runner` | `live_runner/` | Edge that registers + heartbeats to a Livepeer Orchestrator and routes to one resident worker |
| `ltx-worker` | `ltx/` | LTX-2.3 video generation (T2V / I2V / extend / retake) — documented below |
| `idv2v-worker` | `idv2v/` | Wan I2V-14B restyle / masked edit |
| `gemma-worker` | `gemma/` | Local prompt enhancement (Gemma) |

The Docker images are defined by the Dockerfiles in [`docker/`](docker/), kept
inside this runner folder. Build them from the **video-creator-platform repo
root** (one level above `runner/`), because the Dockerfiles `COPY runner/...`
into the image:

```bash
# from video-creator-platform/:
docker build -f runner/docker/ltx-worker.Dockerfile -t video-creator-ltx-worker .
```

Python deps ship via this folder's `requirements.txt` (the `ltx-worker`
Dockerfile copies `runner/requirements.txt`); the LTX-2 core/pipelines revision
is pinned inside the Dockerfile itself.

## Supported GPUs

The runner is VRAM-aware and runs out of the box on:

| GPU | VRAM | Arch | Compute | Mode | Offload | Max resolution |
|-----|------|------|---------|------|---------|----------------|
| RTX 4090 | 24 GB | Ada | 8.9 | streaming | CPU | 720p |
| RTX 5090 | 32 GB | Blackwell | 12.0 | full-resident | none (FP8) | 1080p |
| RTX PRO 6000 | 96 GB | Blackwell | 12.0 | full-resident | none (FP8) | 1080p |

The mode is picked automatically from detected VRAM (mirroring LTX-Desktop's
`backend/runtime_config/runtime_policy.py`):

- **streaming (< 31 GiB, e.g. 4090)**: model weights stream from pinned host
  RAM (`OffloadMode.CPU`), requests are clamped to 720p so 24 GB stays safe.
- **full-resident (>= 31 GiB, e.g. 5090 / RTX PRO 6000)**: the FP8-quantized
  (~23 GB) transformer is held in VRAM (`OffloadMode.NONE`), up to 1080p.

## Architecture

```
Video-Creator (Electron client)
  └── requests → Livepeer Orchestrator → live-runner → resident worker
                                                     ├── ltx-worker (LTX-2.3)
                                                     ├── idv2v-worker (Wan)
                                                     └── gemma-worker (enhance)
```

The runner:
1. Loads the LTX-2.3 distilled models into GPU VRAM
2. `live-runner` registers with a Livepeer Orchestrator via `register_runner()`
3. The LTX worker exposes HTTP endpoints for video generation (T2V, I2V, A2V)
4. Receives inference requests from the Orchestrator (proxied from the client / Livepeer)

## Quick Start (node operator)

```bash
# From the video-creator-platform repo root (the Dockerfiles COPY runner/ into the image):
docker build -f runner/docker/ltx-worker.Dockerfile -t video-creator-ltx-worker .

# Launch against an orchestrator. Models are bind-mounted at /models.
docker run --gpus all -d -p 8991:8991  # (worker; normally behind live-runner) \
  -e ORCHESTRATOR_URL=https://orchestrator:8935 \
  -e ORCHESTRATOR_SECRET=your-secret \
  -e RUNNER_URL=http://<host-routable-ip>:8991 \
  -e MODEL_CHECKPOINT=/models/checkpoint \
  -e TEXT_ENCODER_ROOT=/models/gemma \
  -e UPSCALER_PATH=/models/upsampler \
  --mount type=bind,source=/path/to/models,target=/models \
  video-creator-ltx-worker
```

Or with `docker compose`:

```bash
docker compose -f runner/docker/docker-compose.video-creator.yml up  # live-runner + ltx-worker + idv2v-worker + gemma-worker (+ optional orchestrator profile)
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| ORCHESTRATOR_URL | Yes | — | Livepeer Orchestrator URL |
| ORCHESTRATOR_SECRET | Yes | — | Orchestrator auth secret |
| RUNNER_URL | Yes | http://0.0.0.0:8991 | Public URL of this runner |
| MODEL_CHECKPOINT | Yes | — | Path to LTX distilled checkpoint |
| TEXT_ENCODER_ROOT | No* | /models/gemma | Path to Gemma text encoder (*only needed in local mode — not when `ENHANCE_FORWARD_URL` is set) |
| UPSCALER_PATH | No | — | Path to spatial upsampler |
| GPU_DEVICE | No | 0 | GPU device index |
| GPU_VRAM_GB | No | auto | Override detected VRAM (GiB), e.g. "24" or "32". Use when the container's CUDA index differs from the host's. |
| GPU_NAME | No | auto | Override GPU display name reported to the orchestrator |
| PORT | No | 8991 | HTTP server port |
| WARMUP | No | true | Run warmup generation on startup |
| PRICE | No | 0.5 | Price per generation (USD) |
| PRICE_UNIT | No | fixed | Pricing unit |
| ENHANCE_GPU_DEVICE | No | (GPU_DEVICE) | Run the local Gemma enhancement on a different CUDA index |
| ENHANCE_FORWARD_URL | No | — | Bypass local Gemma and proxy `/prompt-enhance` to a shared OpenAI-compatible endpoint |
| ENHANCE_FORWARD_MODEL | No | — | Model id sent to the forwarded endpoint |
| ENHANCE_FORWARD_API_KEY | No | — | Bearer API key for the forwarded endpoint |
| ENHANCE_FORWARD_TIMEOUT | No | 120 | Upstream request timeout (seconds) |
| ENHANCE_T2V_SYSTEM_PROMPT | No | built-in | Override the default t2v system prompt |
| ENHANCE_I2V_SYSTEM_PROMPT | No | built-in | Override the default i2v system prompt |
| LORA_CACHE_DIR | No | /models/loras | Directory where downloaded catalog LoRAs are cached |
| LORA_CACHE_SIZE_GB | No | 2.0 | Operator disk budget for the LoRA cache (GiB); LRU-evicts |
| LORA_CATALOG_SOURCE | No | main LTX-Desktop repo | URL or local path to `lora_catalog.json` |
| LORA_HF_TOKEN | No | (HF_TOKEN) | Hugging Face token for gated catalog LoRA repos |
| MAX_BODY_BYTES | No | 3000000000 | Max aiohttp request body size (matches go-livepeer's 3GB cap) |
| ZIMAGE_MODEL_DIR | No | /models/zimage | Directory for Z-Image-Turbo image-generation models |
| ZIMAGE_DTYPE | No | bf16 | bf16 (default repo) or fp8 (single-file Comfy checkpoint) |

## Endpoints

- `POST /video-creator/v1/t2v` — text-to-video generation (accepts optional `loras`)
- `POST /video-creator/v1/i2v` — image-to-video generation (accepts optional `loras`)
- `POST /video-creator/v1/a2v` — audio-to-video generation (not supported on runner)
- `POST /video-creator/v1/image` — image generation
- `POST /video-creator/v1/extend` — video extend
- `POST /video-creator/v1/retake` — video retake
- `POST /video-creator/v1/prompt-enhance` — prompt enhancement (local Gemma or forwarded)
- `POST /video-creator/v1/suggest-gap-prompt` — gap prompt suggestion
- `POST /video-creator/v1/extract-conditioning` — IC-LoRA conditioning extraction
- `POST /video-creator/v1/ic-lora-generate` — IC-LoRA guided generation
- `GET /video-creator/v1/health` — health check
- `GET /video-creator/v1/info` — runner info (GPU, model, capabilities)

## LoRA support (t2v / i2v)

The runner can apply **catalog LoRAs** to text-to-video and image-to-video
generation. The LTX-Desktop client forwards the user's selected LoRA(s); the
runner downloads and applies them:

- **Request field:** `loras: [{ "id": "<catalog-id>", "filename": "<optional variant file>", "scale": 1.0 }]`
- **Catalog-only:** an `id` must exist in the catalog. Unknown ids/files return
  HTTP 404 — the runner will never download an arbitrary repo from a client.
- **Catalog source:** by default the runner downloads `lora_catalog.json` from
  `LORA_CATALOG_SOURCE`. Point it at a different URL (or a local file path) for
  a curated/offline catalog.
- **Disk budget + LRU eviction:** downloaded weights live under `LORA_CACHE_DIR`
  and are bounded by `LORA_CACHE_SIZE_GB`.

## Prompt enhancement backends

`/prompt-enhance` can be served three ways, in order of precedence:

1. **Forwarded** (`ENHANCE_FORWARD_URL` set): proxy to a shared OpenAI-compatible
   chat-completions endpoint — one enhancement model serving many runners.
2. **Local on a different GPU** (`ENHANCE_GPU_DEVICE` set, no forward URL):
   Gemma loads on a separate CUDA index from the video pipeline.
3. **Local on the video GPU** (default): Gemma loads on `GPU_DEVICE`,
   evicting/reloading the video pipeline around the call.

## Notes for node operators

- **Models are NOT baked into the image.** Bind-mount `/models` (or your
  equivalent) with the checkpoint, Gemma encoder, and upsampler. The image only
  contains the code + dependencies.
- **Host RAM floor:** LTX reads the FP8 checkpoint via mmap; a box with less
  host RAM than the checkpoint size will fail with "Cannot allocate memory".
  The runner logs a clear warning at startup if its commit limit looks too low.
- **Gateway pin:** the runner pins `livepeer-python-gateway@2f29404` (the
  `ja/live-runner` revision that ships `register_runner`). Bump deliberately.
