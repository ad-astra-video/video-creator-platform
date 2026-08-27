"""Inference pipeline for the ID-V2V worker — decode inputs, run model, encode output.

Drives the ID-V2V model loaded by `model.ModelManager` (int8-quantized Wan2.1
DiT + VACE with CPU offload, suitable for a 32 GB RTX 5090).

Unlike the standalone id-v2v runner, the model is passed in explicitly (a
ModelManager instance owned by the worker) so it can be evicted/kept-warm by the
live-runner swap policy.

Ported from c:\\dev\\id-v2v\\runner\\src\\idv2v_runner\\run.py.
"""

import asyncio
import base64
import io
import logging
import os
import tempfile
import time

import numpy as np

from . import config

logger = logging.getLogger("video_creator.runner.idv2v.run")

DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


async def process_job(model, body: dict, job_id: str | None = None) -> dict:
    """Process a restylization job request.

    Args:
        model: a ready `ModelManager` instance (owned/evicted by the worker).
        body: parsed JSON containing source_video, stylized_first_frame, prompt,
              and parameters.

    Returns:
        dict matching the IdV2VResponse schema.
    """
    if not model.is_ready:
        raise RuntimeError("Model is not loaded yet — retry in a moment")

    start = time.time()

    prompt = body.get("prompt", "")
    # No default frame cap: when the caller does not request a specific budget,
    # restyle reproduces the FULL source video length, chunked into per-clip
    # windows (num_frames_per_clip, default 81 = ~3s) so longer inputs stay
    # VRAM-manageable. Defaulting to 81 here truncated every restyle to one
    # ~3s clip (the "only processes 3 seconds" regression).
    _raw_max = body.get("max_frames")
    max_frames = int(_raw_max) if _raw_max not in (None, "") else None
    # Output fps: an explicit 24/25/30 wins; otherwise (or on a non-standard
    # value) the output is encoded at the SOURCE video's fps so the returned
    # restyle plays at the same rate/duration as the input.
    _raw_fps = body.get("fps")
    _explicit_fps = int(_raw_fps) if _raw_fps not in (None, "") else None
    if _explicit_fps is not None and _explicit_fps not in (24, 25, 30):
        _explicit_fps = None
    # A 0/omitted inference_steps resolves to the loaded model variant's default
    # budget (fast=8, regular=30). An explicit >0 value is honored as-is.
    inference_steps = int(body.get("inference_steps") or config.steps_for(model.variant))
    cfg_scale = float(body.get("cfg_scale", 5.0))
    vace_scale = float(body.get("vace_scale", 1.0))
    width = int(body.get("width", 1280))
    height = int(body.get("height", 720))
    # Cap generation resolution so the fp8 DiT + activations fit the 31 GB
    # GPU (720p/81fr does not fit this box). This worker emits its output at
    # whatever resolution it generates; the LTX spatial upscaler has been
    # removed from the restyle path. Default max side 640 (360p for a 720p
    # source).
    gen_max_side = int(os.environ.get("IDV2V_GEN_MAX_SIDE", "640"))
    _sc = gen_max_side / max(width, height) if max(width, height) > gen_max_side else 1.0
    width = max(256, int(round(width * _sc) // 16 * 16))
    height = max(256, int(round(height * _sc) // 16 * 16))
    num_frames_per_clip = int(body.get("num_frames_per_clip", 81))
    # Server-side cap on per-clip frame count: 720p inflates the fp8 DiT
    # activation tensors ~4x over 360p, and 81fr/720p OOMs the 32 GB GPU.
    # Capping per-clip frames (e.g. 41) keeps each clip's temporal activations
    # small enough to fit, at the cost of splitting into more clips.
    _max_per_clip = int(os.environ.get("IDV2V_MAX_FRAMES_PER_CLIP", "81"))
    num_frames_per_clip = min(num_frames_per_clip, _max_per_clip)
    logger.info("restyle num_frames_per_clip=%d (max_allowed=%d)", num_frames_per_clip, _max_per_clip)
    seed = int(body.get("seed", 123))
    keyframes = body.get("keyframes", [])   # [{"frame": N, "image": "<b64>"}, ...]

    logger.info(
        "Processing restyle job: prompt=%r, max_frames=%s, steps=%d, cfg=%.1f",
        prompt, max_frames, inference_steps, cfg_scale,
    )

    source_b64 = body.get("source_video", "")
    stylized_b64 = body.get("stylized_first_frame", "")

    if not source_b64 or not stylized_b64:
        raise ValueError("source_video and stylized_first_frame are required")

    keyframe_specs = []
    for kf in keyframes:
        idx = kf.get("frame")
        img = kf.get("image")
        if not isinstance(idx, int) or idx < 1:
            raise ValueError(f"keyframe 'frame' must be int >= 1, got {idx!r}")
        if not img:
            raise ValueError(f"keyframe {idx} is missing 'image'")
        keyframe_specs.append((idx, img))

    set_progress(job_id, 0.01, "preprocessing", "decoding source + conditioning")
    def _prog(progress, stage, message):
        set_progress(job_id, progress, stage, message)

    # Optional Gemma LLM stage: automatically caption the source video when the
    # prompt is blank/placeholder, and/or run prompt enhancement on the final
    # prompt when the caller set enhance_prompt. Runs on the dedicated
    # GEMMA_GPU_DEVICE (never evicts the resident id-v2v model). Disabled when
    # the checkpoint isn't provisioned (config.gemma_enabled()).
    # The Gemma LLM stage (prompt enhance + auto video caption) runs BEFORE the
    # video model loads, because they share ONE GPU and cannot coexist (see
    # server.handle_restyle). process_job receives the already-enhanced prompt
    # (body["prompt"]) and the LLM metadata (body["_gemma_meta"]) to surface in
    # the response for the UI.
    gem_meta = body.get("_gemma_meta") or {
        "caption": None, "enhanced_prompt": None, "enhanced": False,
    }
    set_progress(job_id, 0.04, "preprocessing", "decoding source + conditioning")

    set_generation_active(True)
    try:
        result = await asyncio.to_thread(
            _run_pipeline,
            model, source_b64, stylized_b64, prompt,
            max_frames, inference_steps, cfg_scale, vace_scale,
            num_frames_per_clip, seed, keyframe_specs, width, height,
            _explicit_fps, _prog,
        )
    finally:
        set_generation_active(False)

    elapsed = time.time() - start
    logger.info("Restyle job complete in %.1fs", elapsed)

    return {
        "output_video": result["b64"],
        "frames_generated": result["frames"],
        "resolution": f"{width}x{height}",
        "processing_time_sec": round(elapsed, 2),
        # Gemma LLM artifacts for the UI to save/display for reference.
        "used_prompt": prompt,
        "video_caption": gem_meta.get("caption") if gem_meta.get("caption") else None,
        "enhanced_prompt": gem_meta.get("enhanced_prompt") if gem_meta.get("enhanced_prompt") else None,
    }


def _run_pipeline(model, source_b64, stylized_b64, prompt, max_frames,
                  inference_steps, cfg_scale, vace_scale, num_frames_per_clip,
                  seed, keyframe_specs, width, height,
                  fps=None, progress_cb=None) -> dict:
    """Synchronous pipeline body (runs in a worker thread)."""
    tmpdir = tempfile.mkdtemp(prefix="idv2v_")

    source_path = os.path.join(tmpdir, "source.mp4")
    stylized_path = os.path.join(tmpdir, "stylized_first_frame.png")
    _write_b64(source_b64, source_path)
    _write_b64(stylized_b64, stylized_path)

    cond_frames = _segment_foreground(source_path, stylized_path, tmpdir, width, height)

    decoded_kf = []
    for idx, img_b64 in keyframe_specs:
        decoded_kf.append((idx, _decode_image(img_b64)))

    input_image = _load_anchor(stylized_path, width, height)

    frames = model.infer(
        prompt=prompt,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        input_image=input_image,
        condition_videos=[cond_frames],
        keyframes=decoded_kf,
        width=width, height=height,
        num_frames=num_frames_per_clip,
        max_frames=max_frames,
        num_inference_steps=inference_steps,
        cfg_scale=cfg_scale,
        vace_scale=vace_scale,
        seed=seed,
        progress_cb=progress_cb,
    )

    # Output fps: explicit 24/25/30 wins; otherwise match the source video's fps
    # (falling back to 24 if the container doesn't carry a sane rate).
    src_fps = _read_source_fps(source_path)
    out_fps = fps if fps is not None else (src_fps or 24.0)
    logger.info("restyle output fps=%s (requested=%s source=%s)", out_fps, fps, src_fps)

    b64 = _encode_frames_mp4(frames, out_fps)
    return {"b64": b64, "frames": len(frames)}


import threading

# True while the diffusion pipeline is actively generating (between the Gemma
# stage and job completion). The worker refuses to evict the video model (for a
# shared-GPU Gemma) while a generation is live, since the running job still owns
# the model + GPU.
_GENERATION_ACTIVE = False
_GENERATION_LOCK = threading.Lock()


def set_generation_active(active: bool) -> None:
    global _GENERATION_ACTIVE
    with _GENERATION_LOCK:
        _GENERATION_ACTIVE = bool(active)


def generation_active() -> bool:
    with _GENERATION_LOCK:
        return _GENERATION_ACTIVE


# Per-job progress store (job_id -> {"progress": float, "stage": str, "message": str, "done": bool}).
PROGRESS_LOCK = threading.Lock()
_PROGRESS: dict = {}


def set_progress(job_id, progress, stage="generating", message="",
                  step=None, total=None):
    if not job_id:
        return
    rec = {
        "progress": round(max(0.0, min(1.0, float(progress))), 4),
        "stage": stage, "message": message, "done": False,
    }
    if step is not None:
        rec["step"] = int(step)
    if total is not None:
        rec["total"] = int(total)
    with PROGRESS_LOCK:
        _PROGRESS[job_id] = rec


def get_progress(job_id):
    with PROGRESS_LOCK:
        return _PROGRESS.get(job_id)


def clear_progress(job_id):
    with PROGRESS_LOCK:
        _PROGRESS.pop(job_id, None)


def _write_b64(b64str: str, path: str):
    data = base64.b64decode(b64str)
    with open(path, "wb") as f:
        f.write(data)


def _decode_image(b64str: str):
    """Decode a base64 image into a PIL RGB image."""
    from PIL import Image
    data = base64.b64decode(b64str)
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_anchor(path: str, width: int, height: int):
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return img.resize((width, height))


def _read_video_frames_cv2(source_path, width, height):
    """Read a video with OpenCV into center-cropped/resized PIL RGB frames."""
    from .model import center_crop_and_resize
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(source_path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return [center_crop_and_resize(f, width, height) for f in frames]


def _gemma_stage(enhancer, source_b64, prompt, enhance_prompt, seed):
    """Run the Gemma LLM stage before restyle.

    Auto-captions the source video when the prompt is blank/placeholder, then
    optionally enhances the resulting prompt. Synchronous (call via
    asyncio.to_thread); uses the dedicated GEMMA_GPU_DEVICE so the resident
    id-v2v model is untouched.

    Returns ``(final_prompt, meta)`` where ``meta`` carries the artifacts the
    UI wants back for reference:
      * ``caption``         — the Gemma auto-caption of the source video
                              (None when the caller supplied their own prompt)
      * ``enhanced_prompt`` — the Gemma-enhanced prompt (None if not enhanced)
      * ``enhanced``        — whether enhancement actually ran
    The returned ``final_prompt`` (the caption, or its enhanced rewrite, or the
    caller's prompt) is what actually feeds the restyle generation task.
    """
    want_caption = (not prompt.strip()
                    or prompt.strip().lower() == "restyle this video")
    enhancer.ensure_loaded()
    final = prompt
    meta = {"caption": None, "enhanced_prompt": None, "enhanced": False}
    if want_caption:
        frames = _sample_source_frames(source_b64)
        if frames:
            logger.info("Gemma captioning %d sampled frame(s)", len(frames))
            caption_seed = None if seed is None else (seed + 1)
            captioned = enhancer.caption_video(frames, seed=caption_seed)
            if captioned:
                final = captioned
                meta["caption"] = captioned
                logger.info("Gemma auto-caption result: %r", final)
            else:
                logger.info("Gemma caption empty — keeping original prompt")
    if enhance_prompt and final.strip():
        enhanced = enhancer.enhance_restyle(final, seed)
        if enhanced:
            meta["enhanced_prompt"] = enhanced
            meta["enhanced"] = True
            final = enhanced
            logger.info("Gemma enhanced prompt: %r", final)
    return final, meta


def _sample_source_frames(source_b64, max_frames: int = 4):
    """Decode the source video and return `max_frames` evenly-spaced PIL RGB
    frames for Gemma auto-captioning. Returns [] on decode failure.
    """
    import base64
    import shutil
    import tempfile

    import cv2
    from PIL import Image

    tmpdir = tempfile.mkdtemp(prefix="gemma_frames_")
    try:
        path = os.path.join(tmpdir, "src.mp4")
        with open(path, "wb") as f:
            f.write(base64.b64decode(source_b64))
        cap = cv2.VideoCapture(path)
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 0:
                return []
            if max_frames <= 1:
                idxs = [0]
            else:
                idxs = sorted({
                    int(round(i * (total - 1) / (max_frames - 1)))
                    for i in range(max_frames)
                })
            idxs = set(idxs)
            frames = []
            pos = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pos += 1
                if pos in idxs:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
                    idxs.discard(pos)
                    if not idxs:
                        break
            return frames
        finally:
            cap.release()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _downscale_source_for_sam3(source_path, dst_path, max_side):
    """Write a max_side-capped copy of the source video to cut SAM3's memory.

    SAM3 (video segmentation at 720p/full-res) OOMs the .8 box (~24 GB) at
    720p/81 frames (subprocess peaked ~22 GB). Reducing the input resolution
    drops that roughly linearly with pixel count. The resulting mask video is
    still upscaled to (width,height) by _read_video_frames_cv2 downstream, so
    the final condition resolution is unchanged. Returns (dst_path, scaled?)
    (False if scale factor was 1, i.e. already small enough)."""
    import cv2
    cap = cv2.VideoCapture(source_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        cap.release()
        return source_path, False
    nw, nh = max(2, int(w * scale)), max(2, int(h * scale))
    nw -= nw % 2; nh -= nh % 2
    out = cv2.VideoWriter(dst_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (nw, nh))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.write(cv2.resize(frame, (nw, nh)))
    out.release()
    cap.release()
    logger.info("SAM3 source downscaled %dx%d -> %dx%d (max_side=%d)",
                w, h, nw, nh, max_side)
    return dst_path, True


def _segment_foreground(source_path, stylized_path, tmpdir, width, height):
    """SAM3 foreground-on-gray segmentation (VACE condition video).

    Reproduces the reference `scripts/preprocess.sh` by invoking the two
    upstream CLI modules:
      1. python -m idv2v.preprocess.sam3        (segmentation + mask cleanup)
      2. python -m idv2v.preprocess.orig_pixel  (foreground-on-gray pixels)

    Set IDV2V_SKIP_SAM3=1 to bypass and use raw source frames (relighting path).
    """
    if config.SKIP_SAM3:
        logger.info("IDV2V_SKIP_SAM3=1 — using raw source frames as condition")
        return _read_video_frames_cv2(source_path, width, height)

    try:
        import idv2v  # noqa: F401  (ensures the reference package is installed)
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 foreground segmentation requires the Eyeline ID-V2V reference "
            f"package, but `import idv2v` failed: {exc}. Install the reference repo "
            "(diffsynth_studio + src) into the environment, or set IDV2V_SKIP_SAM3=1 "
            "to serve the raw-frame (relighting) path instead."
        ) from exc

    preproc_dir = os.path.join(tmpdir, "preprocessing")
    os.makedirs(preproc_dir, exist_ok=True)
    cond_video = os.path.join(preproc_dir, "orig_pixel.mp4")

    # Downscale the source for SAM3 to keep the .8 box (only ~24 GB RAM) from
    # OOMing on full-res video segmentation. Not needed when already small.
    sam3_max_side = int(os.environ.get("IDV2V_SAM3_MAX_SIDE", "768"))
    sam_source, _ = _downscale_source_for_sam3(source_path,
                                               os.path.join(preproc_dir, "source_sam3.mp4"),
                                               sam3_max_side)

    gpu_id = "0"
    model_dev = os.environ.get("GPU_DEVICE", config.GPU_DEVICE)
    if model_dev.startswith("cuda:"):
        gpu_id = model_dev.split(":", 1)[1] or "0"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_id)

    import subprocess
    import sys

    step1 = subprocess.run(
        [sys.executable, "-m", "idv2v.preprocess.sam3",
         "--video_path", sam_source,
         "--sam_prompt", os.environ.get("SAM_PROMPT", config.SAM_PROMPT),
         "--output_dir", preproc_dir,
         "--model_path", config.SAM3_CKPT,
         "--joint_mask_post_proc"],
        capture_output=True, text=True, env=env,
    )
    if step1.returncode != 0:
        raise RuntimeError("SAM3 segmentation failed:\n" + (step1.stderr or step1.stdout))

    step2 = subprocess.run(
        [sys.executable, "-m", "idv2v.preprocess.orig_pixel",
         "--video_path", sam_source,
         "--mask_folder", preproc_dir,
         "--mask_image_file_name", "sam3Mask_id_all.png",
         "--result_save_path", cond_video],
        capture_output=True, text=True, env=env,
    )
    if step2.returncode != 0:
        raise RuntimeError("orig_pixel (foreground-on-gray) failed:\n" + (step2.stderr or step2.stdout))

    if not os.path.isfile(cond_video):
        raise RuntimeError(f"orig_pixel did not produce {cond_video}")

    logger.info("SAM3 condition written to %s", cond_video)
    return _read_video_frames_cv2(cond_video, width, height)


def _read_source_fps(path):
    """Read a video's nominal frame rate from its container; None when absent/invalid."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or None
        cap.release()
        if fps and 1.0 <= fps <= 240.0:
            return round(float(fps), 3)
    except Exception:
        pass
    return None


def _encode_frames_mp4(frames, fps=24.0) -> str:
    """Encode a list of PIL frames to an MP4 (H.264) at the given fps, and return
    base64. fps defaults to 24 for callers that don't care; the restyle path passes
    the source-matched (or user-selected) rate explicitly."""
    import imageio.v2 as imageio

    arr = np.stack([np.asarray(f.convert("RGB")) for f in frames], axis=0)
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        imageio.mimwrite(path, arr, format="FFMPEG", fps=fps, codec="libx264", quality=8)
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
