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
All baked into `runner/docker/vp-worker.Dockerfile` (rebuild + push + pull to deploy).

## Remaining (honest)
1. **Bernini / wan-worker generation calibration**: load the 26 GB model + run a real t2v/v2v/r2v
   job (routing is proven; the actual model-load inference run is the heavyweight test).
2. **Frontend api-client/UI** (`references[]`, r2v bar, EditVideoPanel, PostProcessControls):
   blocked on regenerating the backend OpenAPI spec before the typed api-client calls can land.
