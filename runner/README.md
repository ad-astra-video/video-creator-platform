# Video-Creator Runner

The GPU inference backend for Video-Creator. This `runner/` folder holds the
self-contained inference services that **node operators launch** as Docker
images — there are no host-side deploy scripts (models and env vars are provided
at `docker run` time).

| Worker | Folder | Role |
|--------|--------|------|
| `live-runner` | `live_runner/` | Edge that registers + heartbeats to a Livepeer Orchestrator, owns the multi-GPU allocation policy, and routes to the workers |
| `ltx-worker` | `ltx/` | LTX-2.3 video generation (T2V / I2V / extend / retake) — documented below |
| `idv2v-worker` | `idv2v/` | Wan I2V-14B restyle / masked edit |
| `image-worker` | `image/` | Z-Image / FLUX.2 (Klein) image generation |
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
  └── requests → Livepeer Orchestrator → live-runner (GPU scheduler)
                                          ├── ltx-worker (LTX-2.3)   → video GPU
                                          ├── idv2v-worker (Wan)     → video GPU
                                          ├── image-worker (Z-Image/Klein) → a free GPU
                                          └── gemma-worker (enhance) → resident GPU
```

The runner:
1. Loads the LTX-2.3 distilled models into GPU VRAM
2. `live-runner` registers with a Livepeer Orchestrator via `register_runner()`
3. The LTX worker exposes HTTP endpoints for video generation (T2V, I2V, A2V)
4. Receives inference requests from the Orchestrator (proxied from the client / Livepeer)

## GPU allocation (multi-GPU scheduling)

The **live-runner owns the GPU map** for every worker container. Each worker
container is launched with `--gpus all` (all physical GPUs visible,
`CUDA_VISIBLE_DEVICES` unset); the live-runner's scheduler decides which single
GPU each task runs on and tells the device-aware workers via the `device` field
in `/load`. The goal is to **keep every GPU busy and every warm model resident**
so nothing ever cold-reloads and two models never share one GPU's VRAM.

### Policy (in `live_runner/scheduler.py`)

When a generation task arrives for a worker, the scheduler places it in this
order:

1. **Reuse if already warm** — if that worker's model is already loaded on some
   GPU, route the task back to that same card (no reload).
2. **Use a free GPU and leave it warm** — otherwise, if any GPU currently has
   nothing resident, allocate the task there and keep the model warm on it.
   This is what spreads image and video generation across different GPUs.
3. **LRU-evict only when nothing is free** — only when EVERY GPU already holds a
   warm model, evict the **least-recently-used** warm model that the incoming
   task does NOT need, and hand its GPU over.

Eviction is **VRAM-safe**: the scheduler POSTs `/evict` to the evicted worker
(which frees the old model's VRAM) *before* the new model loads, so two models
never transiently co-reside — this is the fix for the image-then-video OOM,
which was a placement bug (both landing on GPU 0), not a VRAM-capacity limit.

Two subtle behaviors matter for operators:

- **Video workers are pinned.** `ltx-worker` and `idv2v-worker` build their
  engine once at startup on a hardcoded `GPU_DEVICE` and ignore `device` in
  `/load`, so the scheduler always routes them to `VIDEO_GPU` (a dedicated
  "video card"). When a video task needs that card, the scheduler **evicts**
  whatever else is warm there first.
- **Image avoids the video card when another GPU is free.** `image-worker` is
  device-aware, so it lands on a *different* card unless that's the only free
  one — meaning a video gen right after an image gen never collides. Because
  models stay warm resident, the map is reconciled from each worker's `/info`
  (`device_in_use`) at every heartbeat, so a crashed/restarted worker
  self-heals. A dead worker's slot is freed automatically; gemma's pinned card
  is never touched by eviction.

### Verification

`GET /video-creator/v1/scheduler/status` (with the `X-Worker-Token` header)
returns the live GPU map, e.g.:

```json
{"gpu_count":3,"gemma_resident_gpu":2,"gpus":[
  {"gpu_id":0,"worker":"image-worker","state":"busy","resident":true},
  {"gpu_id":1,"worker":null,"state":"idle","resident":false},
  {"gpu_id":2,"worker":"gemma-worker","state":"busy","resident":true}]}
```

`resident:true` means the model is warm in VRAM on that card.

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
| LORA_CACHE_DIR | No | /models/loras | Directory where downloaded catalog LoRAs are cached |
| LORA_CACHE_SIZE_GB | No | 2.0 | Operator disk budget for the LoRA cache (GiB); LRU-evicts |
| LORA_CATALOG_SOURCE | No | main LTX-Desktop repo | URL or local path to `lora_catalog.json` |
| LORA_HF_TOKEN | No | (HF_TOKEN) | Hugging Face token for gated catalog LoRA repos |
| MAX_BODY_BYTES | No | 3000000000 | Max aiohttp request body size (matches go-livepeer's 3GB cap) |
| ZIMAGE_MODEL_DIR | No | /models/zimage | Directory for Z-Image-Turbo image-generation models |
| ZIMAGE_DTYPE | No | bf16 | bf16 (default repo) or fp8 (single-file Comfy checkpoint) |
| GPU_COUNT | No | 3 | Number of physical GPUs the live-runner's scheduler manages |
| VIDEO_GPU | No | 0 | Dedicated card for the pinned video workers (`ltx-worker`, `idv2v-worker`); their `/load` ignores `device`, so the scheduler always routes them here and evicts anything else warm on it first |
| GEMMA_RESIDENT_GPU | No | 0 | GPU where the Gemma prompt-enhancer is pinned resident (held out of the task pool; the box sets this to `2`) |
| GEMMA_IDLE_GRACE_S | No | 20 | How long Gemma stays resident after its last use before it may be evicted |
| SCHEDULER_QUEUE_TIMEOUT_S | No | 600 | How long a task waits for a free GPU before the scheduler gives up (→ HTTP 503, retriable) when nothing is evictable |
| LLM_GPU_DEVICE | No | (GPU_DEVICE) | CUDA index for the local Gemma worker (box sets `2`) |

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

## LTX 2.5 (optional) model requirements

LTX 2.5 (`model: "ltx-2.5"`) is a NEW 22B audio-video model family. It runs
through the same `ltx_pipelines` `DistilledPipeline` as LTX-2.3, but needs its
OWN model kit downloaded by `runner/ltx/download_ltx25.sh` (a ComfyUI-style
subtree under `<models>/ltx-2.5/`), plus a pinned `ltx_pipelines` rev that ships
the `ModelPaths` API. **All of the following are REQUIRED for 2.5 generation:**

| File (under `<models>/ltx-2.5/`) | Source repo (all gated) | Purpose |
|----------------------------------|-------------------------|---------|
| `diffusion_models/ltx-2.5-22b-distilled-transformer-{nvfp4,int8-convrot}.safetensors` | `Lightricks/LTX-2.5` | Distilled transformer (NVFP4 on Blackwell, INT8+ConvRot otherwise) |
| `text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors` | `Lightricks/LTX-2.5` | Gemma 4 12B text encoder w/ 2.5 projection (bf16) |
| `vae/ltx-2.5-video-vae-bf16.safetensors` | `Lightricks/LTX-2.5` | Video VAE |
| `vae/ltx-2.5-audio-vae-bf16.safetensors` | `Lightricks/LTX-2.5` | Audio VAE |
| `model_patches/ltx-2.5-duration-head-bf16.safetensors` | `Lightricks/LTX-2.5` | Duration-head model patch |
| `latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | `Lightricks/LTX-2.5` | LATENT spatial upscaler x2 (2.5-specific) |
| **`loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors`** | **`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`** | **IC-LoRA Pixel Spatial Upscaler x2 — REQUIRED for pixel-space upscaling** |

Important notes for operators:

- **The IC-LoRA Pixel Spatial Upscaler lives in its OWN SEPARATE gated repo**
  (`Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler`). You must accept a
  **second agreement** on huggingface.co (in addition to the `Lightricks/LTX-2.5`
  repo) before your HF token can download it. Without it, LTX-2.5 pixel-space
  upscaling is unavailable; `download_ltx25.sh` fails at step [7/7] with
  "Access denied ... requires approval".
- **Both repos are gated** — the same `HUGGING_FACE_HUB_TOKEN` (with both
  agreements accepted) is used by `download_ltx25.sh`.
- **Latent vs pixel upscaler are NOT interchangeable**: the *latent* spatial
  upscaler (`latent_upscale_models/`) and the *IC-LoRA pixel* upscaler (`loras/`)
  are separate 2.5 artifacts for different upscale passes. Keep both.
- **To provision the kit on a GPU box:**
  ```bash
  export HUGGING_FACE_HUB_TOKEN=hf_...
  MODEL_DIR=/path/to/models bash runner/ltx/download_ltx25.sh   # LTX25_VARIANT=nvfp4|int8
  ```

The ltx-worker currently serves LTX 2.3 (`model` default). The 2.5 code path is
selected by `model: "ltx-2.5"` once the runner is built against an
`ltx_pipelines` revision with the `ModelPaths` API (`Lightricks/LTX-2` >=
`fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca`) and the kit above is present. See
`ltx/config.py` `LTX25_*` for the single-sourced artifact names, and
`ltx/download_ltx25.sh` for the downloader.

## Notes for node operators

- **Models are NOT baked into the image.** Bind-mount `/models` (or your
  equivalent) with the checkpoint, Gemma encoder, and upsampler. The image only
  contains the code + dependencies.
- **Host RAM floor:** LTX reads the FP8 checkpoint via mmap; a box with less
  host RAM than the checkpoint size will fail with "Cannot allocate memory".
  The runner logs a clear warning at startup if its commit limit looks too low.
- **Gateway pin:** the runner pins `livepeer-python-gateway@2f29404` (the
  `ja/live-runner` revision that ships `register_runner`). Bump deliberately.
