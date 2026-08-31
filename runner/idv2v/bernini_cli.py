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
      "omega_vid": 1.25, "omega_img": 6.5, "omega_txt": 4.0, "omega_tgt": 0.5,
      "omega_scale": 1.0, "flow_shift": 5.0, "eta": 0.5, "momentum": 0,
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
import os
import sys
from typing import Optional

import torch

# --- Bernini source must be importable. The manager spawns us with the runner
# --- dir as sys.path[0] (which contains runner's own `bernini.py` manager, NOT
# --- the ByteDance package), so force the real source dir onto the path first.
_BERNINI_SRC = os.environ.get("BERNINI_SRC", "/opt/bernini/src")
if _BERNINI_SRC and sys.path[:1] != [_BERNINI_SRC]:
    sys.path.insert(0, _BERNINI_SRC)

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


def _progress_emitter():
    """Return a (fraction, step, total) callback that writes one JSON progress
    line per denoise step to stdout (the manager logs/skips these and relays
    the terminal result line). Mirrors idv2v's per-step progress_cb."""
    def _cb(fraction, step, total):
        try:
            sys.stdout.write(json.dumps({
                "type": "progress",
                "fraction": round(float(fraction), 4),
                "step": int(step),
                "total": int(total),
            }) + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - never let progress break a job
            pass
    return _cb

# Standard Wan2.2 negative prompt (mirrors bernini.cli.DEFAULT_NEG_PROMPT).
DEFAULT_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# Default guidance mode derived from the requested task type.
TASK_GUIDANCE = {"t2v": "t2v", "v2v": "v2v", "r2v": "rv2v"}

# The 14b fp8 checkpoint drives the WIT-CFG sampler (`sample_bernini_wvitcfg`),
# whose accepted guidance-mode set differs from the plain 1.3b sampler
# (`sample_one_step`): the WIT-CFG sampler accepts `vae_txt_vit`,
# `vae_txt_vit_wapg`, `rv2v_wapg`, `r2v_wapg`, `v2v_apg` — NOT the plain
# `t2v`/`v2v`/`rv2v`. These mirror the upstream `BERNINI_V2_TASK_DEFAULTS`
# mapping (gradio_demo.py), which uses `vae_txt_vit_wapg` for every task
# except the dedicated `rv2v` (video + ref-images) task.
FP8_TASK_GUIDANCE = {
    "t2v": "vae_txt_vit_wapg",
    "v2v": "vae_txt_vit_wapg",
    "r2v": "vae_txt_vit_wapg",
}


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
    ns.num_inference_steps = 40  # full-quality native default
    ns.guidance_mode = "rv2v"
    ns.omega_vid = 1.25
    ns.omega_img = 6.5
    ns.omega_txt = 4.0
    ns.omega_tgt = 0.5
    ns.omega_scale = 1.0
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
    ap.add_argument("--turbo-lora-dir", default="/models/bernini-turbo-lora",
                    help="dir with rzgar high/low noise LoRA safetensors (optional)")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    ns = _arg_defaults()
    ns.config = args.model_dir

    is_fp8 = os.path.exists(os.path.join(args.model_dir, "quantization_config.json"))
    logger.info("Building Bernini pipeline from %s (device=%s, fp8=%s)",
                args.model_dir, args.device, is_fp8)
    if is_fp8:
        import bernini_fp8  # runner-local fp8-aware loader (14b/quantized checkpoints)
        pipeline = bernini_fp8.build_fp8_pipeline(args.model_dir, device)
    else:
        pipeline = build_pipeline(ns, device)
    logger.info("Pipeline built: %s", type(pipeline).__name__)

    # 14B fp8 memory fit: the combined renderer (diff_dec high + diff_dec_low) is
    # ~28.6 GiB fp8, too big to hold BOTH legs plus WIT-CFG activations (~7.4 GiB)
    # on the 32 GB card. The sampler already supports a local_device_moves mode
    # that keeps ONE leg resident on GPU at a time and streams the other on demand
    # (high leg during high-timestep phase, low leg after the boundary swap) --
    # but it only engages when self.transformer is on CPU at sample start. Our fp8
    # build leaves the renderer wherever the pipeline placed it (GPU), which keeps
    # BOTH legs resident -> OOM. Offload both legs to CPU so the sampler streams
    # them per phase (peak ~14.3 GiB leg + activations, fits). No sampler change.
    if is_fp8:
        try:
            pm = pipeline.model
            off = 0
            for attr in ("diff_dec", "diff_dec_low"):
                m = getattr(pm, attr, None)
                if m is None:
                    continue
                for leg in ("transformer", "transformer_2"):
                    t = getattr(m, leg, None)
                    if t is not None:
                        t.to("cpu")
                        off += 1
            torch.cuda.empty_cache()
            logger.info("fp8 renderer legs CPU-resident (%d legs), sampler streams "
                        "one leg to GPU per phase", off)
        except Exception:  # noqa: BLE001 - never block startup on the offload
            logger.warning("Could not offload fp8 renderer legs to CPU", exc_info=True)

    # Optional rzgar 4-step LoRA for the turbo toggle. Load once at startup;
    # per-job `turbo` flips it on/off via TurboLora.apply()/restore().
    tl = None
    if is_fp8 and args.turbo_lora_dir and os.path.isdir(args.turbo_lora_dir):
        high = os.path.join(args.turbo_lora_dir, "Bernini-R_LightX2V_high_noise.safetensors")
        low = os.path.join(args.turbo_lora_dir, "Bernini-R_LightX2V_low_noise.safetensors")
        if os.path.exists(high) and os.path.exists(low):
            import bernini_lora
            tl = bernini_lora.TurboLora(pipeline.model, high, low, device=args.device)
            logger.info("rzgar 4-step LoRA loaded for turbo toggle (linears=%d patches=%d)",
                        len(tl.linear), len(tl.patch))

    def handle(job: dict) -> dict:
        if not job.get("prompt"):
            return {"ok": False, "error": "missing 'prompt'"}
        try:
            import torch as _torch
            m = pipeline.model
            def _dev(o):
                if o is None:
                    return "None"
                try:
                    return str(next(o.parameters()).device)
                except StopIteration:
                    return "no-params"
            _alloc = _torch.cuda.memory_allocated() / 1e9
            _ldm = _dev(m.diff_dec.transformer) == "cpu"
            print(f"DIAG alloc_GB={_alloc:.2f} "
                  f"diff_dec.tr={_dev(m.diff_dec.transformer)} "
                  f"diff_dec.tr2={_dev(m.diff_dec.transformer_2)} "
                  f"low.tr={_dev(m.diff_dec_low.transformer)} "
                  f"low.tr2={_dev(m.diff_dec_low.transformer_2)} "
                  f"ldm={_ldm}", flush=True)
        except Exception as _e:  # noqa: BLE001 - diagnostic never blocks a job
            print(f"DIAG_FAIL {type(_e).__name__}: {_e}", flush=True)
        if not job.get("prompt"):
            return {"ok": False, "error": "missing 'prompt'"}
        out = job.get("output")
        if not out:
            return {"ok": False, "error": "missing 'output'"}
        _apply_job(ns, job)
        # Turbo (4-step rzgar LoRA + DPM++2M-SDE sgm_uniform) re-enabled
        # 2026-08-30: the green was NOT the LoRA/sampler/fp8 — it was the
        # scale_shift_table uint8-plain-copy bug in stream_fill (bernini_fp8.py,
        # fixed). Native 40-step UniPC stays the default; a job with
        # "turbo": true goes 4-step via TurboLora + DPM++2M-SDE below.
        turbo = bool(job.get("turbo"))
        if tl is not None:
            if turbo and not tl.active:
                tl.apply()
                logger.info("turbo LoRA APPLIED")
            elif (not turbo) and tl.active:
                tl.restore()
                logger.info("turbo LoRA restored")
        if turbo and "num_inference_steps" not in job:
            ns.num_inference_steps = 4
        # Sampler: UniPC everywhere. The 4-step turbo LoRA is recipe-tuned for
        # DPM++2M-SDE + sgm_uniform, but that path renders garbage post-fix
        # ("not all green, but not a real video"). UniPC is our validated
        # native sampler (20-step produces valid output since the
        # scale_shift_table fix), so run the 4-step turbo on UniPC too. If
        # UniPC+4step is also garbage, the fault is the fp8 x LoRA merge, not
        # the sampler. (DPM++2M-SDE swap removed 2026-08-30.)
        pm = getattr(pipeline, "model", None)
        if pm is not None and hasattr(pm, "use_unipc") and not pm.use_unipc:
            pm.use_unipc = True
            logger.info("sampler: UniPC (turbo=%s)", turbo)
        task_name = job.get("task_name") or ns.task_type
        if not job.get("guidance_mode"):
            mapping = FP8_TASK_GUIDANCE if is_fp8 else TASK_GUIDANCE
            ns.guidance_mode = mapping.get(task_name, ns.guidance_mode)
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
                common = dict(
                    video=task["video"],
                    image=task["image"],
                    images=task["images"],
                    output_path=out,
                    system_prompt=sys_prompt,
                    **generation_kwargs(ns))
                try:
                    # Per-step progress_cb is threaded in by the build-time
                    # source patch; if the patch didn't apply upstream, fall
                    # back to a call without it (never fail a generation just
                    # because per-step progress is unavailable).
                    pipeline(task_name, task["prompt"], **common,
                             progress_cb=_progress_emitter())
                except TypeError:
                    pipeline(task_name, task["prompt"], **common)
            else:
                common = dict(
                    video=task["video"],
                    image=task["image"],
                    images=task["images"],
                    output_path=out,
                    system_prompt=sys_prompt,
                    **generation_kwargs(ns))
                try:
                    pipeline(task["prompt"], **common,
                             progress_cb=_progress_emitter())
                except TypeError:
                    pipeline(task["prompt"], **common)
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
