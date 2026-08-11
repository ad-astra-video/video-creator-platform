# Video Creator web API contract (Worker reimplementation spec)

Source: `video-creator/frontend/generated/backend-openapi.ts` (openapi-typescript from the Python
backend's OpenAPI doc) — **read-only reference**. This is the route table the Cloudflare Worker must
reimplement in TypeScript (Phase 2.1), and the source of truth for what the UI calls.

> Progress note (Decisions): the current desktop polls `GET /api/generation/progress` and the
> `/download/progress` endpoints over REST every ~3s. Under the locked design, job/live progress
> moves to a **WebSocket between the browser and the orchestrator** (Worker-relayed); those GET
> endpoints are retained only as a fallback, not as the primary channel.

## Endpoints (55 operations, 54 paths)

| Method | Path | Purpose | Disposition |
|---|---|---|---|
| GET | `/health` | liveness | keep-remote |
| GET | `/api/auth/huggingface/callback` | HF OAuth callback | worker OAuth |
| POST | `/api/auth/huggingface/login` | start HF OAuth | worker OAuth |
| POST | `/api/auth/huggingface/logout` | HF logout | worker OAuth |
| GET | `/api/auth/huggingface/status` | HF auth status | worker OAuth |
| POST | `/api/enhance-prompt` | prompt enhancement job | worker-dispatch |
| POST | `/api/extend` | timeline extend job | worker-dispatch |
| POST | `/api/generate` | t2v generation job | worker-dispatch |
| POST | `/api/generate-image` | image generation job | worker-dispatch |
| POST | `/api/generate/cancel` | cancel in-flight job | worker-dispatch |
| GET | `/api/generate/models-specs` | available t2v model specs | keep (Worker/catalog) |
| GET | `/api/generation/progress` | job progress (currently REST-polled 3s) | superseded-by-WS; keep GET fallback |
| GET | `/api/gpu-info` | local GPU info | DROP (no local GPU) |
| GET | `/api/gpu-info/mps` | macOS MPS info | DROP (no local runtime) |
| POST | `/api/ic-lora/extract-conditioning` | IC-LoRA conditioning pre-step | worker/runner |
| POST | `/api/ic-lora/generate` | IC-LoRA generation job | worker-dispatch |
| GET | `/api/ic-loras` | IC-LoRA catalog list | keep (Worker/catalog) |
| POST | `/api/ic-loras/download` | start IC-LoRA model download | worker/runner |
| GET | `/api/ic-loras/download/progress` | download progress | superseded-by-WS; GET fallback |
| GET | `/api/loras` | LoRA catalog list | keep (Worker/catalog) |
| POST | `/api/loras/download` | start LoRA download | worker/runner |
| GET | `/api/loras/download/progress` | download progress | superseded-by-WS; GET fallback |
| GET | `/api/models` | installed/available model list | keep (Worker/catalog) |
| POST | `/api/models/active-ltx-model` | set active base model | worker/runner |
| POST | `/api/models/check-access` | check model access | keep |
| DELETE | `/api/models/delete` | remove a model | worker/runner |
| POST | `/api/models/describe` | model metadata | keep |
| POST | `/api/models/download` | start model download | worker/runner |
| GET | `/api/models/download/active` | active downloads | superseded-by-WS; GET fallback |
| GET | `/api/models/download/progress` | download progress | superseded-by-WS; GET fallback |
| GET | `/api/models/img-gen-recommendation` | recommended img-gen model | keep |
| GET | `/api/models/ltx-ic-lora-recommendation` | recommended IC-LoRA model | keep |
| GET | `/api/models/ltx-recommendation` | recommended t2v model | keep |
| GET | `/api/models/ltx-versions` | available LTX versions | keep |
| GET | `/api/models/text-encoder-recommendation` | recommended text encoder | keep |
| GET | `/api/platform/balance` | credits balance | keep (existing Worker handler) |
| POST | `/api/platform/checkout` | Stripe checkout | keep (existing Worker handler) |
| POST | `/api/platform/link-email` | link recovery email | keep (existing Worker handler) |
| POST | `/api/platform/recover/confirm` | confirm recovery / rotate key | keep (existing Worker handler) |
| POST | `/api/platform/recover/request` | request recovery code | keep (existing Worker handler) |
| GET | `/api/platform/status` | platform status | keep (existing Worker handler) |
| GET | `/api/providers` | provider list | rework: from orchestrator runner discovery |
| POST | `/api/providers/discover` | run discovery | rework: orchestrator discovery |
| POST | `/api/providers/exclude` | exclude provider | rework |
| POST | `/api/providers/select` | select provider | rework |
| POST | `/api/restyle` | video restyle job | worker-dispatch |
| POST | `/api/restyle/extract-first-frame` | first-frame extract pre-step | worker/runner |
| POST | `/api/restyle/segment-subject` | subject segmentation pre-step (SAM3) | worker/runner |
| POST | `/api/restyle/style-frame` | style-frame selection pre-step | worker/runner |
| POST | `/api/retake` | retake job | worker-dispatch |
| GET | `/api/runtime-policy` | runtime capabilities/policy | keep/rework to runner caps |
| GET,POST | `/api/settings` | app settings get/set | keep (Worker/D1, per-user) |
| POST | `/api/suggest-gap-prompt` | suggest prompt for gap | keep/worker |
| POST | `/api/system/shutdown` | shut down local backend | DROP (no local process) |

## Disposition tally

- **DROP (no local GPU)**: 1
- **DROP (no local process)**: 1
- **DROP (no local runtime)**: 1
- **keep**: 7
- **keep (Worker/D1, per-user)**: 1
- **keep (Worker/catalog)**: 4
- **keep (existing Worker handler)**: 6
- **keep-remote**: 1
- **keep/rework to runner caps**: 1
- **keep/worker**: 1
- **rework**: 2
- **rework: from orchestrator runner discovery**: 1
- **rework: orchestrator discovery**: 1
- **superseded-by-WS; GET fallback**: 4
- **superseded-by-WS; keep GET fallback**: 1
- **worker OAuth**: 4
- **worker-dispatch**: 8
- **worker/runner**: 9

## Notes

- **worker-dispatch** routes: the Worker validates the per-user key, checks+decrements the credit
  ledger, creates a D1 job record, dispatches to a runner via the orchestrator, returns a `job_id`,
  and streams progress over WS. The (large, GPU-bound) request body is forwarded to the runner; the
  heavy model/pipeline logic stays on the runner.
- **DROP**: `gpu-info*`, `system/shutdown` — no local runtime, no local GPU. Provider selection
  (`/api/providers/*`) is reworked to discover remote runners through the orchestrator.
- **keep (existing Worker handler)**: `/api/platform/*` already live in `video-creator-platform/src`.
- **worker OAuth**: `/api/auth/huggingface/*` — Worker-served redirect/popup flow.
