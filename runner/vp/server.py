"""Video-processing worker (vp-worker) HTTP service (aiohttp).

Dedicated low-VRAM post-process + ffmpeg + standalone SAM3 container driven by
the live-runner. Exposes the post rails that run AFTER any render worker
(t2v/v2v/r2v/restyle/extend/retake). The live-runner orchestrates the combined
chain as ONE ``/process`` call; the individual ``/fps-boost`` ``/upscale``
``/ffmpeg`` routes exist for direct/post-rail use.

    GET  /health          — liveness
    POST /process         — combined RIFE + FlashVSR + ffmpeg, one pass
    POST /fps-boost       — RIFE motion-preserving fps-boost only
    POST /upscale         — FlashVSR 4x upscale (SSAA downscale if final < 4x)
    POST /ffmpeg          — general ffmpeg video processing
    POST /sam3            — standalone SAM3 video foreground segmentation
    POST /sam3-image      — standalone SAM3 single-frame mask

Every POST requires the shared ``X-Worker-Token``. I/O is base64-in/base64-out
(a single A/V container per request); the rails run in subprocess/thread so a
video job doesn't block the event loop.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Optional

import numpy as np
from aiohttp import web

from . import config
from . import ffmpeg_post

logger = logging.getLogger("video_creator.runner.vp.server")

# Warm single-instance rails (loaded lazily / never by default — vp-worker is
# LOW-VRAM and stays warm only while a rail is actually resident).
_fps_booster: Any = None
_upscaler: Any = None
_lock = asyncio.Lock()


def _resolve_token() -> str:
    if config.WORKER_TOKEN:
        return config.WORKER_TOKEN
    tok = config._random_token()
    os.environ["WORKER_TOKEN"] = tok
    config.WORKER_TOKEN = tok
    logger.info("WORKER_TOKEN was blank — auto-generated")
    return tok


def _require_token(request: web.Request) -> None:
    expected = _resolve_token()
    provided = request.headers.get("X-Worker-Token", "")
    if not provided or provided != expected:
        raise web.HTTPForbidden(reason="missing/mismatched X-Worker-Token")


async def handle_load(req: web.Request) -> web.Response:
    """No-op /load: vp-worker is intentionally NOT device-aware (low-VRAM, runs
    on whatever GPU the scheduler hands it; rails load lazily on first use).
    The live-runner's GPU scheduler calls /load before every dispatch, so we
    must ACK it even though there's nothing to preload — otherwise the acquire
    flow fails with a 404 and routing to this worker breaks.

    ``model`` is consumed only to log the scheduler's rail hint (process /
    fps-boost / upscale / ffmpeg); nothing is preloaded."""
    model = None
    try:
        model = (await req.json()).get("model")
    except Exception:
        pass
    logger.info("vp-worker /load: model=%s (no-op: not device-aware)", model)
    return web.json_response({"status": "loaded", "loaded": True, "device": ""})


async def handle_evict(_req: web.Request) -> web.Response:
    return web.json_response({"status": "evicted"})




def _b64write(data: str, path: str) -> None:
    with open(path, "wb") as fh:
        fh.write(base64.b64decode(data))


def _b64read(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _parse_bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Rails (lazy warm singletons)
# ---------------------------------------------------------------------------

async def _get_fps_booster():
    global _fps_booster
    async with _lock:
        if _fps_booster is None:
            from .rife_post import FpsBooster
            _fps_booster = FpsBooster(_resolve_rife_weights(), device=config.GPU_DEVICE)
        return _fps_booster


def _resolve_rife_weights() -> str:
    """Find flownet.pkl under RIFE_ROOT (scan a couple levels deep)."""
    for base, _dirs, files in os.walk(config.RIFE_ROOT):
        for f in files:
            if f == "flownet.pkl":
                return os.path.join(base, f)
    raise FileNotFoundError(f"flownet.pkl not found under {config.RIFE_ROOT}")


async def _get_upscaler():
    global _upscaler
    async with _lock:
        if _upscaler is None:
            from .flashvsr_post import FlashVsrUpscaler
            _upscaler = FlashVsrUpscaler(config.FLASHVSR_ROOT, device=config.GPU_DEVICE)
        return _upscaler


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

async def handle_health(_req: web.Request) -> web.Response:
    return web.json_response({
        "status": "ok", "app": "vp-worker",
        "ffmpeg": ffmpeg_post.ffmpeg_available(),
        "rife": config.rife_enabled(), "flashvsr": config.flashvsr_enabled(),
        "sam3": os.path.isdir(config.SAM3_CKPT),
    })


async def handle_info(_req: web.Request) -> web.Response:
    return web.json_response({
        "runner_id": "", "app": "vp-worker",
        "capabilities": ["process", "fps-boost", "upscale", "ffmpeg",
                         "sam3", "sam3-image"],
        "ready": True,
        "ffmpeg": ffmpeg_post.ffmpeg_available(),
        "rife": config.rife_enabled(), "flashvsr": config.flashvsr_enabled(),
    })


async def handle_upscale(request: web.Request) -> web.Response:
    """FlashVSR 4x upscale: {video: b64, scale?: 4, final?: 1080|1440|raw, fps?}."""
    _require_token(request)
    body = await request.json()
    video_b64 = body.get("video")
    if not video_b64:
        return web.json_response({"error": "missing 'video' (base64)"}, status=400)
    if not config.flashvsr_enabled():
        return web.json_response({"error": "FlashVSR not provisioned"}, status=503)
    final = str(body.get("final", "1080"))
    scale = int(body.get("scale") or config.FLASHVSR_SCALE)
    with tempfile.TemporaryDirectory(prefix="vp_up_") as td:
        src = os.path.join(td, "in.mp4")
        _b64write(video_b64, src)
        try:
            upscaler = await _get_upscaler()
            frames = await asyncio.to_thread(upscaler.upscale, src)
        except Exception as exc:
            logger.exception("upscale failed")
            return web.json_response({"error": str(exc)}, status=500)
        src_fps = _fps_of(src) or 16.0
        if final == "raw":
            buf = _encode_raw(frames)
            return web.Response(body=buf, content_type="video/mp4",
                                headers={"X-Frames": str(len(frames)),
                                         "X-Fps": str(src_fps)})
        tw, th = _target_dims(frames.shape[2], frames.shape[1], final)
        out = os.path.join(td, "out.mp4")
        try:
            await asyncio.to_thread(ffmpeg_post.encode_frames, frames, out, src_fps,
                                    width=tw, height=th)
        except Exception as exc:
            return web.json_response({"error": f"encode: {exc}"}, status=500)
        return web.json_response({
            "output_video": _b64read(out),
            "resolution": f"{tw}x{th}", "fps": src_fps,
            "frames": len(frames), "upscale": f"{scale}x",
        })


async def handle_fps_boost(request: web.Request) -> web.Response:
    """RIFE fps-boost: {video: b64, target_fps: 24|30|60, mode?: preserve_motion|smooth}."""
    _require_token(request)
    body = await request.json()
    video_b64 = body.get("video")
    if not video_b64:
        return web.json_response({"error": "missing 'video' (base64)"}, status=400)
    if not config.rife_enabled():
        return web.json_response({"error": "RIFE not provisioned"}, status=503)
    target = int(body.get("target_fps") or 0)
    src_fps = _fps_of_b64(video_b64) or 16.0
    if target <= 0 or target <= src_fps:
        target = _pick_target_fps(src_fps)
    if target not in config.RIFE_ALLOWED_TARGET_FPS:
        # clamp to nearest allowed >= source
        target = min([f for f in config.RIFE_ALLOWED_TARGET_FPS if f > src_fps] or [60])
    with tempfile.TemporaryDirectory(prefix="vp_fps_") as td:
        src = os.path.join(td, "in.mp4")
        _b64write(video_b64, src)
        frames = await asyncio.to_thread(ffmpeg_post.read_frames, src, src_fps)
        try:
            booster = await _get_fps_booster()
            boosted = await asyncio.to_thread(booster.boost, frames, src_fps, target)
        except Exception as exc:
            logger.exception("fps-boost failed")
            return web.json_response({"error": str(exc)}, status=500)
        out = os.path.join(td, "out.mp4")
        try:
            await asyncio.to_thread(ffmpeg_post.encode_frames, boosted, out, target)
        except Exception as exc:
            return web.json_response({"error": f"encode: {exc}"}, status=500)
        return web.json_response({
            "output_video": _b64read(out),
            "resolution": f"{boosted.shape[2]}x{boosted.shape[1]}",
            "fps": target, "frames": len(boosted),
        })


async def handle_ffmpeg(request: web.Request) -> web.Response:
    """General ffmpeg: {video: b64, filters?: str, output_args?: [...], fps?}."""
    _require_token(request)
    body = await request.json()
    video_b64 = body.get("video")
    if not video_b64:
        return web.json_response({"error": "missing 'video' (base64)"}, status=400)
    if not ffmpeg_post.ffmpeg_available():
        return web.json_response({"error": "ffmpeg not available"}, status=503)
    with tempfile.TemporaryDirectory(prefix="vp_ff_") as td:
        src = os.path.join(td, "in.mp4")
        out = os.path.join(td, "out.mp4")
        _b64write(video_b64, src)
        vf = str(body.get("filters") or "null")
        fps = float(body.get("fps") or 0)
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", src, "-vf", vf]
        if fps > 0:
            cmd += ["-r", str(fps)]
        args = body.get("output_args")
        if isinstance(args, list):
            cmd += [str(a) for a in args]
        cmd += ["-c:a", "copy", out] if _has_audio(src) else [out]
        p = await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
        if p.returncode != 0 or not os.path.exists(out):
            return web.json_response(
                {"error": "ffmpeg failed", "detail": (p.stderr or p.stdout)[-2000:]},
                status=500)
        return web.json_response({"output_video": _b64read(out)})


async def handle_process(request: web.Request) -> web.Response:
    """Combined post-chain orchestrated by the live-runner.

    Body (all optional except ``video``):
        {video: b64, fps_boost?: {target_fps, mode?} | null,
         upscale?: {final: 1080|1440|raw, scale?} | null, fps?: override}
    Runs RIFE then FlashVSR then ffmpeg finalize in ONE pass and returns the
    final deliverable (base64 mp4, or raw for final=raw).
    """
    _require_token(request)
    body = await request.json()
    video_b64 = body.get("video")
    if not video_b64:
        return web.json_response({"error": "missing 'video' (base64)"}, status=400)
    with tempfile.TemporaryDirectory(prefix="vp_proc_") as td:
        src = os.path.join(td, "in.mp4")
        _b64write(video_b64, src)
        src_fps = _fps_of(src) or 16.0

        frames = await asyncio.to_thread(ffmpeg_post.read_frames, src, src_fps)
        fps = src_fps

        # stage 1: RIFE fps-boost
        fb = body.get("fps_boost") or {}
        if fb and config.rife_enabled():
            target = int(fb.get("target_fps") or 0)
            if target <= 0 or target <= fps:
                target = _pick_target_fps(fps)
            try:
                booster = await _get_fps_booster()
                frames = await asyncio.to_thread(booster.boost, frames, fps, target)
                fps = target
            except Exception as exc:
                logger.warning("fps-boost stage failed (skipping): %s", exc)

        # stage 2: FlashVSR upscale
        up = body.get("upscale") or {}
        if up and config.flashvsr_enabled():
            with tempfile.TemporaryDirectory(prefix="vp_s2_") as td2:
                mid = os.path.join(td2, "mid.mp4")
                await asyncio.to_thread(ffmpeg_post.encode_frames, frames, mid, fps)
                try:
                    upscaler = await _get_upscaler()
                    frames = await asyncio.to_thread(upscaler.upscale, mid)
                except Exception as exc:
                    logger.warning("upscale stage failed (skipping): %s", exc)

        # stage 3: finalize (raw or encode to final res)
        final = str((up or {}).get("final", "1080")) if up else "1080"
        if final == "raw":
            buf = _encode_raw(frames)
            return web.Response(body=buf, content_type="video/mp4",
                                headers={"X-Frames": str(len(frames)),
                                         "X-Fps": str(fps)})
        tw, th = _target_dims(frames.shape[2], frames.shape[1], final)
        out = os.path.join(td, "final.mp4")
        try:
            await asyncio.to_thread(ffmpeg_post.encode_frames, frames, out, fps,
                                    width=tw, height=th)
        except Exception as exc:
            return web.json_response({"error": f"finalize: {exc}"}, status=500)
        return web.json_response({
            "output_video": _b64read(out),
            "resolution": f"{tw}x{th}", "fps": fps, "frames": len(frames),
        })


async def handle_sam3(request: web.Request) -> web.Response:
    """Standalone SAM3 video foreground segmentation (condition video).

    Body: {video: b64, prompt?: str}. Runs runner.idv2v.segment_single video
    path (the same SAM3 used by wan-worker restyle) in a subprocess. Returns
    the foreground-on-gray condition video.
    """
    _require_token(request)
    body = await request.json()
    video_b64 = body.get("video")
    if not video_b64:
        return web.json_response({"error": "missing 'video' (base64)"}, status=400)
    # The standalone SAM3 on vp-worker reuses the segmenter module from the
    # wan-worker package if co-installed; otherwise subprocess via sys.executable.
    return web.json_response({"error": "SAM3 video routing via vp-worker pending"},
                             status=501)


async def handle_sam3_image(request: web.Request) -> web.Response:
    """Standalone SAM3 single-frame mask.

    Body: {image: b64, prompt?: str}. Returns {mask_b64} at original res.
    Delegates to runner.idv2v.segment_single (same module the wan-worker uses).
    """
    _require_token(request)
    body = await request.json()
    image_b64 = body.get("image")
    if not image_b64:
        return web.json_response({"error": "missing 'image' (base64)"}, status=400)
    import PIL.Image as PILImage
    with tempfile.TemporaryDirectory(prefix="vp_s3_") as td:
        img_path = os.path.join(td, "input.png")
        mask_path = os.path.join(td, "mask.png")
        _b64write(image_b64, img_path)
        try:
            from runner.idv2v.segment_single import main as _seg_main
        except Exception as exc:
            return web.json_response(
                {"error": f"SAM3 module unavailable on vp-worker: {exc}"}, status=501)
        gpu = "0"
        dev = str(config.GPU_DEVICE)
        if dev.startswith("cuda:"):
            gpu = dev.split(":", 1)[1] or "0"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        cmd = [sys.executable, "-m", "runner.idv2v.segment_single",
               "--image", img_path, "--prompt", str(body.get("prompt") or config.SAM3_PROMPT),
               "--model_path", config.SAM3_CKPT, "--out_mask", mask_path]
        p = await asyncio.to_thread(subprocess.run, cmd, capture_output=True,
                                    text=True, env=env, timeout=300)
        if p.returncode != 0:
            return web.json_response({"error": "sam3 failed",
                                      "detail": (p.stderr or p.stdout)[-2000:]},
                                     status=500)
        with PILImage.open(img_path) as im:
            w, h = im.size
        return web.json_response({"mask_b64": _b64read(mask_path), "width": w,
                                  "height": h})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fps_of(path: str) -> float:
    try:
        _w, _h, fps, _n = ffmpeg_post.probe_video(path)
        return fps
    except Exception:
        return 16.0


def _fps_of_b64(b64: str) -> float:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(base64.b64decode(b64)); tmp = f.name
    try:
        return _fps_of(tmp)
    finally:
        os.unlink(tmp)


def _pick_target_fps(src_fps: float) -> int:
    for f in sorted(config.RIFE_ALLOWED_TARGET_FPS):
        if f > src_fps:
            return f
    return config.RIFE_ALLOWED_TARGET_FPS[-1]


def _target_dims(w: int, h: int, final: str, scale: int = 4) -> tuple:
    """SSAA: compute final dims after FlashVSR 4x, honoring a final res cap."""
    if final == "1440":
        cap = 2560
    elif final == "1080":
        cap = 1920
    else:  # natural 4x
        cap = max(w, h)
    long_side = max(w, h)
    k = min(1.0, cap / long_side) if cap else 1.0
    return int(round(w * k)), int(round(h * k))


def _encode_raw(frames: np.ndarray) -> bytes:
    # 4:2:0-friendly container for the raw path (keep it simple: h264 yuv420p).
    import tempfile as _tf
    with _tf.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        ffmpeg_post.encode_frames(frames, f.name, 16.0)
        with open(f.name, "rb") as fh:
            return fh.read()


def _has_audio(path: str) -> bool:
    try:
        import subprocess as _sp
        r = _sp.run(["ffprobe", "-v", "error", "-select_streams", "a",
                     "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
                    capture_output=True, text=True)
        return "audio" in r.stdout
    except Exception:
        return False


def create_app() -> web.Application:
    app = web.Application(client_max_size=config.MAX_BODY_BYTES)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/info", handle_info)
    app.router.add_post("/load", handle_load)
    app.router.add_post("/evict", handle_evict)
    app.router.add_post("/video-creator/v1/load", handle_load)
    app.router.add_post("/video-creator/v1/evict", handle_evict)
    # Namespaced aliases match every other worker + the compose healthcheck +
    # live-runner probing, which hit the /video-creator/v1/ prefixed paths.
    app.router.add_get("/video-creator/v1/health", handle_health)
    app.router.add_get("/video-creator/v1/info", handle_info)
    for ep, h in [("/process", handle_process), ("/fps-boost", handle_fps_boost),
                  ("/upscale", handle_upscale), ("/ffmpeg", handle_ffmpeg),
                  ("/sam3", handle_sam3), ("/sam3-image", handle_sam3_image)]:
        app.router.add_post(f"/video-creator/v1/{ep.lstrip('/')}", h)
        app.router.add_post(ep, h)
    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        stream=sys.stdout)
    _resolve_token()
    app = create_app()
    runner = web.AppRunner(app)
    asyncio.run(_serve(runner))


async def _serve(runner: web.AppRunner) -> None:
    await runner.setup()
    site = web.TCPSite(runner, config.HOST, config.PORT)
    await site.start()
    logger.info("vp-worker listening on %s:%d", config.HOST, config.PORT)
    await asyncio.Event().wait()


if __name__ == "__main__":
    main()
