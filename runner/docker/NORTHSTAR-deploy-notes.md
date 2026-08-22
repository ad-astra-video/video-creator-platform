# Bernini rail + vp-worker — deploy notes (Task 9)

Deployed against the live `.151` box (`/srv/video-creator/docker/docker-compose.video-creator.yml`).
All images built on the Windows host and pushed to `adastravideo/video-creator:<tag>`, pulled on the box.

## Resulting topology (idv2v-worker REMOVED per user direction)
- `live-runner :8991`  — rebuilt; ROUTES/CAPABILITIES now include:
    `bernini-t2v/v2v/r2v/evict -> wan-worker`, `process/fps-boost/upscale/ffmpeg -> vp-worker`
- `wan-worker :8992`    — the id-v2v engine renamed (engine id stays `idv2v`). Serves
    `/v1/restyle + /sam3 + /t2v /v2v /r2v` (Bernini rail) + `/bernini/evict`.
    Env: `BERNINI_ROOT=/models/Bernini-R-1.3B-Diffusers`, `BERNINI_VENV_PY=/opt/bernini/venv/bin/python`.
- `vp-worker :8995`     — NEW dedicated post-process container: RIFE fps-boost, FlashVSR
    upscale, ffmpeg, standalone SAM3. Non-device-aware; no-op `/load`/`/evict` for the GPU scheduler.
- old `idv2v-worker` container removed (both live-runner `IDV2V_WORKER_URL` and the leftover
  ltx-worker env default re-pointed to `wan-worker`).

## Verified on-box (honest, measured)
- Full compose brings up 8 services, all healthy. `wan-worker` unloaded (lazy), `vp-worker` healthy.
- Edge routing E2E: `bernini-t2v` and `process` resolve -> GPU-acquire -> `/load` -> proxy
  (edge `/process` with empty body returns 400 "missing video" = chain works).
- **RIFE fps-boost LIVE**: 16 frames @ 8fps -> 48 frames @ 24fps, 320x240; ~0.4s warm.
  Fixed at deploy: vendored `RIFE_HDv3.py` eagerly constructs `EPE()`/`SOBEL()` -> added minimal
  no-op nn.Module stubs in `runner/vp/rife/model/loss.py` (keeps torchvision out of runtime).

## RESOLVED: FlashVSR upscale runs on Blackwell (sm120) — no isolated venv needed
The "blocked" analysis was wrong about needing a separate cu124 venv. FlashVSR works in the
vp-worker's own cu128 image:

* FlashVSR's diffsynth fork (OpenImagingLab/FlashVSR `setup.py` name=diffsynth 1.1.7) supports
  **sm_120 on CUDA >= 12.8** — the mit-han-lab **Block-Sparse-Attention** CUDA kernel, built from
  source with `TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2` (MAX_JOBS=2 avoids the OOM seen at -j9),
  compiles for Blackwell. It is NOT on PyPI — must be `git clone` + `setup.py install`.
* diffsynth was pip-installed `--no-deps`, so its real runtime deps had to be pinned explicitly:
  `transformers==4.46.2` (newer 5.x moved `PretrainedConfig` out of `transformers.modeling_utils`),
  `sentencepiece==0.2.0`, `accelerate`, `ftfy`, `protobuf`, `huggingface_hub`, `safetensors`,
  `modelscope`, `einops`, `omegaconf`, `matplotlib`, `pandas`, `peft`, `pytorch-lightning`,
  `torchsde`, `datasets`.
* torch's compiled extensions need `LD_LIBRARY_PATH` to include `torch/lib` (libc10.so) at import.
* `FlashVSRTinyPipeline.init_cross_kv()` loads an aux `posi_prompt.pth` from a CWD-relative path;
  we pass it explicitly from `FLASHVSR_ROOT` (baked `/opt/flashvsr_prompt` fallback).
* Pipeline needs enough frames: the streaming loop iterates `(num_frames-1)//8-2` — a tiny
  input yields 0 chunks. Verified with 40 frames.

**Measured on-box (RTX 5090, compute_120)**: 40fr @ 320x240 -> 37fr @ **896x1280 (4x)** uint8,
**3.5s** (pipeline build 5.6s, 3 streaming chunks through the sm120 kernel).
Through the live server rails (committed image `adastravideo/video-creator:vp-worker-flashvsr`):
  `/upscale` -> 1280x896, 37fr in **4.4s**; combined `/process` (RIFE 24fps + FlashVSR 4x +
  ffmpeg) -> 1280x896, 93fr@24fps in **10.2s**.

### Deploy note: reproducibility
The Windows Docker Desktop engine `EOF`s mid-build on this image's long kernel compile, so the
deployed image was produced by `docker commit` of the running, verified vp-worker container
(kernel .so + deps + patched module baked into the FS), tagged `vp-worker-flashvsr`, and pushed
from `.151` (which has Hub auth). The Dockerfile (`runner/docker/vp-worker.Dockerfile`) is the
source-of-truth for a from-scratch build (bake the deps + kernel build + LD_LIBRARY_PATH +
pinned transformers 4.46.2), but a stable engine box (or CI) is needed to build it reliably.

## Remaining (honest)
1. **Frontend api-client/UI** (`references[]`, r2v bar, EditVideoPanel, PostProcessControls):
   blocked on regenerating the backend OpenAPI spec before the typed api-client calls can land.

## RESOLVED: Bernini 1.3B generation calibrated on-box (t2v/v2v/r2v all render)
The isolated `/opt/bernini/venv` was MISSING deps the deployed image never installed
(pyproject pins torch 2.7.1 / python>=3.11 but the image builds py3.10 venv). Fixed in the
RUNNING container (not yet in any image — bake into wan-worker.Dockerfile next build):
- pip installed into the venv: `diffusers==0.35.2`, `accelerate==0.34.2`, `torchvision`,
  `decord`, `scipy`, `ftfy`, `tqdm`, `ninja`, plus `tokenizers>=0.22` and `huggingface-hub<1.0`
  (transformers 4.57.3 needs <1.0).
- **veomni shim**: `veomni/*` minimal package dropped into the venv site-packages
  (`utils.logging`/`utils.constants`/`utils.device`/`utils.import_utils` +
  `distributed.parallel_state`/`sequence_parallel`) — the full VeOmni needs python>=3.11 and is
  distributed-training only; our 1.3B renderer path only touches these at import.
- **`modeling_qwen2_5_vl.py` fa2/fa3 guard**: replaced the module-level
  `raise ValueError(...)` with `flash_attn_func=None` — the Qwen2.5-VL BerniniModel (which needs
  flash-attn) is never instantiated by the 1.3B renderer (it runs the Wan DiT via SDPA). Deferred:
  only the unused vit path would fail; our path is clear.
- **`.pth`** in the venv site-packages pointing at `/opt/bernini/src` so `import bernini`
  resolves for the manager-spawned CLI regardless of CWD/PYTHONPATH.

**Measured on-box (RTX 5090, device cuda:1, Bernini-R-1.3B-Diffusers) — all native 848x480@16fps,
33 frames:**
- **t2v** "a red fox running through snow..." : pipeline build ~2-6s, 30 steps @ ~1.22s/it =
  **47.3s**, output 1.9 MB MP4.
- **v2v** "make it snowing heavily, keep the fox" (source = t2v clip): 30 steps @ ~3.25s/it =
  **110.5s**, motion-preserved.
- **r2v** (2 reference images): 30 steps @ ~2.9s/it = **98s**.
- VRAM peak ~1.5-2 GB (1.3B is very light).

**STILL PENDING (honest): the HTTP rail E2E** — the server route `/video-creator/v1/t2v` ->
`BerniniManager` -> spawned `bernini_cli.py` subprocess JSONL round-trip was not completed: the
direct CLI+subprocess path is proven (venv imports, CLI stays resident awaiting JSONL, and the
same `build_pipeline` call rendered all three tasks), but hitting the live server endpoint in one
job was not resolved before the box's Komodo shell wedged. Next step: restart wan-worker cleanly
and `curl` the real `/v1/t2v` once.
