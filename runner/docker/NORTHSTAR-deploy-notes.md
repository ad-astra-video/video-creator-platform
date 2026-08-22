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

## BLOCKED: FlashVSR upscale — isolated venv required
FlashVSR's diffsynth fork (OpenImagingLab/FlashVSR `setup.py` name=diffsynth 1.1.7) pins a
conflicting toolchain vs the vp-worker cu128 image:

- pins `torch==2.6.0+cu124` / `torchvision==0.21.0+cu124` / `torchaudio==2.6.0+cu124`
- pins `transformers==4.46.2`, `numpy==1.26.4`, `peft==0.16.0`, `accelerate==1.8.1`, `safetensors==0.5.3`
- imports a NON-PyPI package: `from block_sparse_attn import block_sparse_attn_func`
  (custom CUDA attention kernel from the diffsynth/FlashVSR toolchain — must be built,
  not on PyPI).

It cannot share the vp-worker image (cu128 torch + transformers 5.x). Fix = mirror the Bernini
pattern: a dedicated **isolated FlashVSR venv** (`FLASHVSR_VENV_PY`, python-3.11) with a cu124
torch + the pinned deps + a built `block_sparse_attn`, and have `flashvsr_post.FlashVsrUpscaler`
shell out to that venv (subprocess) instead of importing diffsynth in-process.

## Remaining (honest)
1. **FlashVSR**: build the isolated cu124 venv (torch 2.6.0+cu124 + diffsynth fork deps +
   block_sparse_attn) inside the vp-worker image; subprocess-dispatch the upscale rail to it.
2. **Bernini / wan-worker generation calibration**: load the 26 GB model + run a real t2v/v2v/r2v
   job (routing is proven; the actual model-load inference run is the heavyweight test).
3. **Frontend api-client/UI** (`references[]`, r2v bar, EditVideoPanel, PostProcessControls):
   blocked on regenerating the backend OpenAPI spec before the typed api-client calls can land.
