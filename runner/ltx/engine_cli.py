"""LTX model subprocess entrypoint (the GPU child).

Spawned by the ltx-worker aiohttp server (running in a SEPARATE process) via
``runner.common.engineproc.EngineProc`` on /load (or lazily on the first
generation):

    python -m runner.ltx.engine_cli --device N

It builds the real ``VideoCreatorInferenceEngine`` on GPU N and serves the
JSONL command channel implemented by ``engineproc.run_child_loop``: one request
line in, one result line out, with interim progress lines forwarded to the
server's SSE / progress callback.

The whole point of the subprocess split is that killing THIS process (which the
server's /evict does) destroys its CUDA primary context — PyTorch has no
in-process API to do that (``empty_cache()`` only returns the caching-allocator
pool while the process stays attached to the GPU with a ~0.5-0.7 GB driver
floor). The parent aiohttp server stays alive and CUDA-free for lazy reload.

Wire format (handled by engineproc, not here):
    -> {"op": <name>, "args": {...}}
    <- {"type": "ready", "ok": true}                            (startup)
    <- {"type": "progress", ...}                                (optional, Nx)
    <- {"ok": true, "result": ...} | {"ok": false, "error": ...} (final)

All request args / results are JSON-safe. The only non-JSON parameter the
server sends is the LoRA list ``list[tuple[path, scale]]``, which the server
proxy serializes as ``[[path, scale], ...]`` (tuples are not JSON) and the
handlers here re-hydrate into tuples before calling the engine.
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Callable, Dict

from runner.common import engineproc
from runner.ltx import config

logger = logging.getLogger("runner.ltx.engine_cli")


# ---------------------------------------------------------------------------
# Argument (de)serialization helpers shared by the child handlers.
# ---------------------------------------------------------------------------
def _deser_loras(raw: Any) -> "list[tuple[str, float]] | None":
    """Re-hydrate a serialized LoRA list into ``[(path, scale), ...]``."""
    if raw is None:
        return None
    return [(str(p), float(s)) for p, s in raw]


def build_engine(device_idx: int):
    """Construct the REAL VideoCreatorInferenceEngine for CUDA device N.

    Mirrors the pre-subprocess ``server.on_startup`` construction: same config
    sources (MODEL_CHECKPOINT / TEXT_ENCODER_ROOT / UPSCALER_PATH), same
    VRAM-aware GPU profile, optional separate prompt-enhance GPU. This runs
    ONLY inside the GPU child, so it may touch torch.cuda freely — the parent
    never calls it.
    """
    import torch
    from runner.ltx.gpu_profile import build_profile
    from runner.ltx.inference import VideoCreatorInferenceEngine

    profile = build_profile(device_idx, config.GPU_VRAM_GB, config.GPU_NAME)
    device = torch.device(f"cuda:{device_idx}")
    enhance_device = (
        torch.device(f"cuda:{config.ENHANCE_GPU_DEVICE}")
        if config.ENHANCE_GPU_DEVICE else device
    )
    logger.info(
        "Building VideoCreatorInferenceEngine on %s (checkpoint=%s, mode=%s)",
        device, config.MODEL_CHECKPOINT, profile.mode,
    )
    return VideoCreatorInferenceEngine(
        config.MODEL_CHECKPOINT,
        config.TEXT_ENCODER_ROOT,
        config.UPSCALER_PATH,
        device,
        profile=profile,
        enhance_device=enhance_device,
    )


# ---------------------------------------------------------------------------
# Handlers: {op: callable(engine, args, progress_cb) -> JSON-safe}.
# Each mirrors the engine method the server used to call in-process. Methods
# that write straight to an output file return None (the server reads the file
# back itself); ones that return a path / string return it directly.
# ---------------------------------------------------------------------------
def _make_handlers() -> Dict[str, Callable[..., Any]]:

    def t2v(eng, a, _pc):
        eng.generate_t2v(
            a["prompt"], a["seed"], a["width"], a["height"], a["num_frames"],
            a["fps"], a["output_path"], _deser_loras(a.get("loras")),
            a.get("model", ""),
        )
        return None

    def i2v(eng, a, _pc):
        eng.generate_i2v(
            a["prompt"], a["image_base64"], a["seed"], a["width"], a["height"],
            a["num_frames"], a["fps"], a["output_path"],
            _deser_loras(a.get("loras")), a.get("model", ""),
        )
        return None

    def _extend_cb(progress_cb):
        """Adapt engine extend progress_cb(stage, message, progress=None) to the
        loop's dict-shaped progress_cb(obj)."""
        def _cb(stage: str, message: str, progress) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb({
                    "stage": stage, "message": message, "progress": progress,
                })
            except Exception:  # noqa: BLE001 - never break generation on progress
                pass
        return _cb

    def extend(eng, a, progress_cb):
        eng.generate_extend(
            prompt=a["prompt"], video_base64=a["video_base64"],
            extend_frames=a["extend_frames"], mode=a["mode"], seed=a["seed"],
            fps=a["fps"], output_path=a["output_path"],
            context_seconds=a.get("context_seconds", 1.0),
            model=a.get("model", ""), progress_cb=_extend_cb(progress_cb),
        )
        return None

    def retake(eng, a, _pc):
        return eng.generate_retake(
            prompt=a["prompt"], video_base64=a["video_base64"],
            start_time=a["start_time"], end_time=a["end_time"], seed=a["seed"],
            fps=a["fps"], regenerate_video=a.get("regenerate_video", True),
            regenerate_audio=a.get("regenerate_audio", True),
            output_path=a["output_path"],
        )

    def image(eng, a, _pc):
        return eng.generate_image(
            prompt=a["prompt"], width=a["width"], height=a["height"],
            num_steps=a.get("num_steps", 9), seed=a.get("seed", 42),
            guidance_scale=a.get("guidance_scale"),
        )

    def edit(eng, a, _pc):
        return eng.edit_image(
            prompt=a["prompt"], image_path=a["image_path"],
            mask_path=a.get("mask_path"), keep_subject=a.get("keep_subject", False),
            sam3_url=a.get("sam3_url"), sam3_prompt=a.get("sam3_prompt", "person"),
            keep_mask_b64=a.get("keep_mask_b64"),
            worker_token=a.get("worker_token", ""),
            strength=a.get("strength", 0.6), num_steps=a.get("num_steps", 9),
            seed=a.get("seed", 42), guidance_scale=a.get("guidance_scale"),
        )

    def ic_lora_full_video(eng, a, _pc):
        eng.generate_ic_lora_full_video(
            prompt=a["prompt"], control_video_path=a["control_video_path"],
            seed=a["seed"], width=a["width"], height=a["height"],
            num_frames=a["num_frames"], fps=a["fps"],
            output_path=a["output_path"],
            conditioning_strength=a.get("conditioning_strength", 1.0),
            lora_path=a.get("lora_path", ""),
            lora_strength=a.get("lora_strength", 1.0),
            skip_stage_2=a.get("skip_stage_2", False),
            resolution_factor=a.get("resolution_factor", 2.0),
        )
        return None

    def enhance_prompt(eng, a, _pc):
        return eng.enhance_prompt(
            a["prompt"],
            image_base64=a.get("image_base64"),
            seed=a.get("seed"),
            system_prompt=a.get("system_prompt"),
        )

    def clamp_resolution(eng, a, _pc):
        return eng.clamp_resolution(a["resolution"])

    def warmup(eng, a, _pc):
        eng.warmup(a["output_path"])
        return None

    return {
        "generate_t2v": t2v,
        "generate_i2v": i2v,
        "generate_extend": extend,
        "generate_retake": retake,
        "generate_image": image,
        "edit_image": edit,
        "generate_ic_lora_full_video": ic_lora_full_video,
        "enhance_prompt": enhance_prompt,
        "clamp_resolution": clamp_resolution,
        "warmup": warmup,
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description="LTX model subprocess (GPU child of runner.ltx.server).")
    ap.add_argument("--device", type=int, required=True,
                    help="CUDA device index to pin the model to (e.g. 1)")
    args = ap.parse_args(argv)

    def _build():
        return build_engine(args.device)

    handlers = _make_handlers()
    # run_child_loop imports torch, pins cuda:set_device BEFORE build_engine (so
    # no context leaks onto an unassigned card), emits the ready handshake, then
    # serves the JSONL loop. Blocking; returns the process exit code.
    return engineproc.run_child_loop(f"cuda:{args.device}", _build, handlers)


if __name__ == "__main__":
    sys.exit(main())
