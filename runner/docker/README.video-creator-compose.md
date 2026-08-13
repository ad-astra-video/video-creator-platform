# Video-Creator 3-service runner compose

Builds and runs `live-runner` + `ltx-worker` + `idv2v-worker` on **one shared
GPU** (RTX 5090 / RTX PRO 6000, 32 GB).

```
live-runner ──(registers/heartbeats)──> Livepeer Orchestrator   app=video-creator
    │  owns swap policy
    ├─(routing/proxy)──> ltx-worker    :8991   generate/retake/extend/ic-lora
    └─(routing/proxy)──> idv2v-worker  :8992   /v1/restyle
```

- Only ONE worker model is resident on the GPU at a time. The live-runner evicts
  the current worker (`/evict`) before loading the requested one (`/load`).
- Workers do NOT register with the Orchestrator — they only serve the internal
  `/health /load /evict /v1/*` surface over the Docker network.
- Health-gating: workers start only after `live-runner` is healthy
  (`depends_on: condition: service_healthy`).
- Worker auth: all three share `WORKER_TOKEN` (sent as `X-Worker-Token`).
  Requests without it are rejected with 403 (covered by
  `runner/tests/test_worker_auth.py`).

## Build & run

From the video-creator repo root:

```bash
export WORKER_TOKEN="$(openssl rand -hex 16)"     # or set a fixed one
export MODELS_DIR=/home/brad/models                # host models bind-mount
docker compose -f docker/docker-compose.video-creator.yml up --build
```

To also bring up a **local** go-livepeer orchestrator (offchain, for testing
discovery/heartbeats):

```bash
docker compose -f docker/docker-compose.video-creator.yml \
  --profile orchestrator up --build
```

For production point `ORCHESTRATOR_URL` / `ORCHESTRATOR_SECRET` at your real
orchestrator (the `orchestrator` profile service is then unused).

## Notes on MODELS_DIR and the .env gotcha

`MODELS_DIR` controls where the stack bind-mounts your model tree. The deploy
compose uses `- ${MODELS_DIR:-/models}:/models` — the **host** side defaults to
`/models`, which is almost never where your models actually live (e.g. they are
often under `/srv/video-creator/models`). If that default wins, workers mount an
empty/wrong tree and crash at warmup with:

    FileNotFoundError: No files matching pattern 'tokenizer.model' found under /models/gemma

(the ltx-worker loads its `gemma/` text encoder from `TEXT_ENCODER_ROOT`,
default `/models/gemma`, so a bad mount surfaces there first).

### Why a plain `.env` may silently not apply

Docker Compose only reads `.env` from the **project directory**, and the project
directory is resolved in this order (see the official *Interpolation* docs):

1. `--project-directory <dir>` if set
2. **the directory of the first compose file passed with `-f`**
3. otherwise your shell's current working directory

So running:

```bash
cd /srv/video-creator
docker compose -f docker/docker-compose.video-creator.yml up
```

with `.env` at `/srv/video-creator/.env` does **NOT** pick it up: because `-f`
points into `docker/`, the project directory becomes `/srv/video-creator/docker/`
and Compose looks for `.env` at `/srv/video-creator/docker/.env` — which doesn't
exist. `MODELS_DIR` then falls back to `/models`, and you get the crash above.

### Reliable launch (pick one)

**Preferred — pass `--env-file` explicitly** (no directory guessing):

```bash
export WORKER_TOKEN="$(openssl rand -hex 16)"
docker compose --env-file /srv/video-creator/.env \
  -f docker/docker-compose.video-creator.yml up -d
```

**Or** keep `.env` next to the compose file so the default lookup finds it:

```bash
cp /srv/video-creator/.env /srv/video-creator/docker/.env
export WORKER_TOKEN="$(openssl rand -hex 16)"
cd /srv/video-creator/docker
docker compose -f docker-compose.video-creator.yml up -d
```

**Or** make the `.env` load an explicit first-class citizen via an `env_file:`
directive on the services, so `.env` is read regardless of project-directory
resolution:

```yaml
services:
  ltx-worker:
    # ...image/volumes/etc...
    env_file:
      - .env        # variables here flow INTO the container's environment
```

> **Important distinction:** `env_file:` injects the listed variables into the
> container's environment at runtime. It does **not** feed Compose's
> `${MODELS_DIR}` interpolation in the YAML itself — interpolation is only fed by
> the shell, `--env-file`, or the project-directory `.env`. So combining
> `env_file:` with `${MODELS_DIR:-/models}` in `volumes:` still needs Compose to
> see `MODELS_DIR` via `--env-file` (or the project-directory `.env`) for the
> volume source to resolve. If you want `env_file:` to be your only mechanism,
> the volume line must use the literal host path instead of `${MODELS_DIR}`:
>
> ```yaml
> volumes:
>   - /srv/video-creator/models:/models   # literal, no interpolation needed
> ```

**Or** point the project directory back at your cwd:

```bash
export WORKER_TOKEN="$(openssl rand -hex 16)"
docker compose --project-directory /srv/video-creator \
  -f docker/docker-compose.video-creator.yml up -d
```

### Verify before recreating

Always confirm `MODELS_DIR` resolved to the real tree before `up`:

```bash
docker compose --env-file /srv/video-creator/.env \
  -f docker/docker-compose.video-creator.yml config \
  | grep -A3 -i "ltx-worker" | grep -i source
```

This must print `source: /srv/video-creator/models` (or wherever your models
live), not `/models`. Then bring it up; the ltx-worker should clear warmup and
report `Inference engine ready on cuda:0` instead of the `tokenizer.model`
FileNotFoundError.

**Simplest of all:** hardcode the host path in the compose default
(`- ${MODELS_DIR:-/srv/video-creator/models}:/models`). It is deterministic on a
single box and removes the `.env` dependency entirely.

## Images

| Service        | Dockerfile                        | Entry                    |
|----------------|-----------------------------------|--------------------------|
| live-runner    | `docker/live-runner.Dockerfile`   | `python -m runner.live_runner` |
| ltx-worker     | `docker/ltx-worker.Dockerfile`    | `python -m runner.ltx.server`  |
| idv2v-worker   | `docker/idv2v-worker.Dockerfile`  | `python -m runner.idv2v`       |

The idv2v-worker installs the Wan/VACE `diffsynth` fork + SAM3 from the
Eyeline-Labs/ID-V2V reference repo (cloned at build), then int8-quantizes
DiT+VACE with CPU offload for the 32 GB card (`IDV2V_QUANT=int8`,
`IDV2V_OFFLOAD=true`).

## Model files (bind-mount at /models)

- `ltx-worker`: `checkpoint/`, `gemma/`, optional `loras/`
- `idv2v-worker`: `idv2v.pth`, `wan/`, `sam3/` (or set `IDV2V_SKIP_SAM3=1`)
