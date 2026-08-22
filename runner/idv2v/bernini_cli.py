"""Bernini renderer subprocess worker (wan-worker /t2v /v2v /r2v backend).

Runs INSIDE the isolated Bernini venv (``/opt/bernini/venv``) as a persistent
stdin/stdout worker so the model stays resident across requests (warm/swap/
evict managed by :mod:`runner.idv2v.bernini`). One JSON object per line on
stdin, one JSON result per line on stdout; the pipeline is built once at
startup.

The full ByteDance ``bernini`` source tree must be importable (PYTHONPATH or
site-packages). The model directory is passed on the command line:

    python bernini_cli.py --model-dir /models/Bernini-R-1.3B-Diffusers \
                          --device cuda:1 --guidance t2v

Request field reference (all optional except ``prompt``/``output``):

    {
      "prompt":  "a cat skating",            # required
      "output":  "/out/clip.mp4",            # required
      "image":   "/in/first.png",            # single image (i2i / first-frame)
      "images":  ["/in/a.png", "/in/b.png"], # reference images (r2v / rv2v)
      "video":   ["/in/src.mp4"],            # source video (v2v)
      "task_name": "t2v",                    # or v2v / r2v (drives prompt enh + guidance)
      "system_prompt": "",                   # defaults auto-selected by task
      "neg_prompt":  "",                     # defaults to the Bernini standard neg
      "num_frames": 81, "max_image_size": 848,
      "height": 0, "width": 0,               # 0/0 => follow source media / native
      "num_inference_steps": 40, "fps": 16, "seed": 42,
      "guidance_mode": null,                 # default derived from task_name
      "omega_vid": 1.25, "omega_img": 4.5, "omega_txt": 4.0, "omega_tgt": 0.5,
      "omega_scale": 0.8, "flow_shift": 5.0, "eta": 0.5, "momentum": 0,
      "planning_step": 25, "vit_txt_cfg": 1.2, "vit_img_cfg": 1.0,
      "vit_denoising_step": 5
    }

Result line:

    {"ok": true, "output": "/out/clip.mp4", "frames": N}         on success
    {"ok": false, "error": "..."}                               on failure

The rendered clip is written in the native resolution/fps of the model
(Bernini native is 480p/16 @ max 848px). Delivery ABOVE native goes through
the vp-worker post rails, driven by the live-runner, not here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Optional

import torch

from bernini.cli import (
    build_pipeline,
    generation_kwargs,
    resolve_system_prompt,
)
from bernini.pipeline import BerniniPipeline

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bernini_cli")

# Standard Wan2.2 negative prompt (mirrors bernini.cli.DEFAULT_NEG_PROMPT).
DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# Default guidance mode derived from the requested task type.
TASK_GUIDANCE = {"t2v": "t2v", "v2v": "v2v", "r2v": "rv2v"}


def _arg_defaults() -> argparse.Namespace:
    """Base argparse Namespace mirroring infer_single_gpu.py common args.

    Values that the worker always fixes, plus safe defaults; per-job fields
    override at call time. Only the fields the pipeline actually consumes are
    populated.
    """
    ns = argparse.Namespace()
    ns.config = None           # set at startup from --model-dir
    ns.high_noise_ckpt = None
    ns.low_noise_ckpt = None
    ns.use_unipc = True
    ns.use_src_tgt_id = True
    ns.interpolate_src_id = True
    ns.max_trained_src_id = 5
    ns.use_pe = False
    ns.pe_model = None
    ns.system_prompt = ""
    ns.neg_prompt = DEFAULT_NEG_PROMPT
    # generation defaults (per-job fields override these in the call)
    ns.num_frames = 81
    ns.max_image_size = 848
    ns.height = 480
    ns.width = 848
    ns.num_inference_steps = 40
    ns.guidance_mode = "rv2v"
    ns.omega_vid = 1.25
    ns.omega_img = 4.5
    ns.omega_txt = 4.0
    ns.omega_tgt = 0.5
    ns.omega_scale = 0.8
    ns.planning_step = 25
    ns.vit_txt_cfg = 1.2
    ns.vit_img_cfg = 1.0
    ns.vit_denoising_step = 5
    ns.flow_shift = 5.0
    ns.seed = 42
    ns.fps = 16
    ns.eta = 0.5
    ns.norm_threshold = [50.0, 50.0, 50.0]
    ns.momentum = 0
    ns.task_type = "v2v"
    return ns


def _apply_job(ns: argparse.Namespace, job: dict) -> None:
    """Fold per-job generation params into the namespace for generation_kwargs."""
    for key in (
        "num_frames", "max_image_size", "height", "width",
        "num_inference_steps", "guidance_mode", "omega_vid", "omega_img",
        "omega_txt", "omega_tgt", "omega_scale", "planning_step",
        "vit_txt_cfg", "vit_img_cfg", "vit_denoising_step", "flow_shift",
        "seed", "fps", "eta", "norm_threshold", "momentum",
    ):
        if key in job:
            setattr(ns, key, job[key])
    if job.get("neg_prompt"):
        ns.neg_prompt = job["neg_prompt"]
    if job.get("system_prompt"):
        ns.system_prompt = job["system_prompt"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Bernini renderer worker")
    ap.add_argument("--model-dir", required=True, help="Bernini Diffusers dir")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--guidance", default="rv2v", choices=("t2v", "v2v", "rv2v"))
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    ns = _arg_defaults()
    ns.config = args.model_dir

    logger.info("Building Bernini pipeline from %s (device=%s)", args.model_dir, args.device)
    pipeline = build_pipeline(ns, device)
    logger.info("Pipeline built: %s", type(pipeline).__name__)

    def handle(job: dict) -> dict:
        if not job.get("prompt"):
            return {"ok": False, "error": "missing 'prompt'"}
        out = job.get("output")
        if not out:
            return {"ok": False, "error": "missing 'output'"}
        _apply_job(ns, job)
        task_name = job.get("task_name") or ns.task_type
        if not job.get("guidance_mode"):
            ns.guidance_mode = TASK_GUIDANCE.get(task_name, ns.guidance_mode)
        task = {
            "prompt": job["prompt"],
            "task_type": task_name,
            "video": job.get("video"),
            "image": job.get("image"),
            "images": job.get("images"),
            "output": out,
        }
        try:
            sys_prompt = resolve_system_prompt(task, ns)
            if isinstance(pipeline, BerniniPipeline):
                pipeline(
                    task_name,
                    task["prompt"],
                    video=task["video"],
                    image=task["image"],
                    images=task["images"],
                    output_path=out,
                    system_prompt=sys_prompt,
                    **generation_kwargs(ns),
                )
            else:
                pipeline(
                    task["prompt"],
                    video=task["video"],
                    image=task["image"],
                    images=task["images"],
                    output_path=out,
                    system_prompt=sys_prompt,
                    **generation_kwargs(ns),
                )
            return {"ok": True, "output": out,
                    "frames": ns.num_frames, "task": task_name}
        except Exception as exc:  # noqa: BLE001 - report to the manager
            logger.exception("job failed")
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    logger.info("Worker ready; awaiting JSONL jobs on stdin")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except json.JSONDecodeError as exc:
            sys.stdout.write(json.dumps({"ok": False, "error": f"bad json: {exc}"}) + "\n")
            sys.stdout.flush()
            continue
        result = handle(job)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
