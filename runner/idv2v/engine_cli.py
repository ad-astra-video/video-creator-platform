"""id-v2v diffsynth ModelManager subprocess (child entrypoint).

Runs the heavy ID-V2V model in a SEPARATE process so its CUDA primary context
can be torn down on /evict. PyTorch has no in-process API to destroy a CUDA
context (``empty_cache()`` only returns the caching-allocator pool to the
driver, leaving the process attached to the GPU with a ~0.5-0.7 GB
driver-reserved floor for its whole lifetime), so the worker's aiohttp server
hosts a *proxy* (:mod:`runner.idv2v.engine`) and the real ``ModelManager``
lives here. Killing this process on /evict destroys the context entirely, so
the server can stay alive for a lazy reload while the GPU is fully released.

Spawned by the server as::

    python -m runner.idv2v.engine_cli --device cuda:N

Wire format (engineproc JSONL): one JSON object per line on stdin/stdout with
interim ``{"type": "progress", ...}`` lines on stdout. See
``runner.common.engineproc`` for the transport contract and ``run_child_loop``
for the blocking dispatch loop used here.

GPU ops hosted by this child (everything that needs torch.cuda):

    * ``load``  — ``set_variant()`` + ``load()`` the ModelManager (async, run
                  on a fresh event loop inside the blocking child loop).
    * ``infer_frames`` — the restyle GPU step: decode the already-conditioned
                  frame inputs (condition video, anchor frame, keyframes — all
                  base64 PNG) and run ``ModelManager.infer()``, returning the
                  generated frames as base64 PNG for the parent to encode.
    * ``status`` — report ready/device/variant (used by the parent proxy to
                  refresh cached state).

The SAM3 conditioning subprocess, the Gemma prompt stage, and the final MP4
encode all stay in the parent (they never touch torch.cuda); only the model +
denoise runs here.

For tests, ``IDV2V_ENGINE_BUILDER`` may name a module exposing ``build_engine``
used instead of the real ModelManager (fake-torch stubbing, no GPU needed).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import logging
import os
import sys
import tempfile

logger = logging.getLogger("video_creator.runner.idv2v.engine_cli")


def _default_build_engine(device: str):
    """Construct the real ModelManager on the assigned CUDA card.

    Runs inside the child process only — the parent server never calls this.
    """
    from runner.idv2v import config
    from runner.idv2v.model import ModelManager

    # Pin the child's config so the manager targets the assigned card (the
    # /load body device is authoritative, and RUN inside this process, so it
    # cannot leak onto a different GPU).
    config.GPU_DEVICE = device
    return ModelManager(device=device)


def _load_build_engine():
    """Build-engine factory: the real one, or a test stub via env override."""
    name = os.environ.get("IDV2V_ENGINE_BUILDER", "").strip()
    if not name:
        return _default_build_engine
    import importlib
    mod = importlib.import_module(name)
    return mod.build_engine


# ---------------------------------------------------------------------------
# JSON-safe serialization helpers (parent<->child over the engineproc pipe).
# ---------------------------------------------------------------------------


def _b64_to_pil(b64str: str):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64str))).convert("RGB")


def _pil_to_b64(img) -> str:
    from PIL import Image
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# GPU-op handlers (child side)
# ---------------------------------------------------------------------------


def _handle_status(engine, args, progress_cb) -> dict:
    """Report ready/device/variant — used by the parent proxy to refresh state."""
    return {
        "ready": bool(getattr(engine, "is_ready", False)),
        "device": str(getattr(engine, "device", "") or ""),
        "variant": getattr(engine, "variant", "") or "",
    }


async def _do_load(engine, args: dict) -> dict:
    """set_variant + load (async body of the ``load`` op)."""
    variant = args.get("variant")
    if variant and hasattr(engine, "set_variant"):
        engine.set_variant(variant)
    if not bool(getattr(engine, "is_ready", False)):
        await engine.load()
    return {
        "ready": bool(getattr(engine, "is_ready", False)),
        "device": str(getattr(engine, "device", "") or ""),
        "variant": getattr(engine, "variant", "") or "",
    }


def _handle_load(engine, args, progress_cb) -> dict:
    # The blocking child loop is sync; model.load() is async, so run it on a
    # fresh event loop here. One-shot (the loop otherwise sits idle), matches
    # /load semantics.
    return asyncio.run(_do_load(engine, args))


def _handle_infer_frames(engine, args, progress_cb) -> dict:
    """Run the restyle GPU step: conditioned frames in, generated frames out.

    Decodes the parent-supplied condition inputs, calls ``ModelManager.infer``
    (the only GPU-bound part of the restyle job), and returns the generated
    frames as base64 PNGs for the parent to encode to MP4. Per-clip denoise
    progress is bridged onto ``progress_cb`` so the parent's SSE rail stays
    live through the pipe.
    """
    def _prog(progress, stage="generating", message=None, step=None, total=None):
        try:
            progress_cb({
                "type": "progress", "progress": progress, "stage": stage,
                "message": message, "step": step, "total": total,
            })
        except Exception:  # noqa: BLE001 - never let progress break a job
            pass

    max_frames = args.get("max_frames")
    if max_frames in (None, ""):
        max_frames = None
    else:
        max_frames = int(max_frames)

    frames = engine.infer(
        prompt=args["prompt"],
        negative_prompt=args["negative_prompt"],
        input_image=_b64_to_pil(args["input_image"]),
        condition_videos=[[_b64_to_pil(b) for b in args["condition_frames"]]],
        keyframes=[(int(idx), _b64_to_pil(b))
                   for idx, b in (args.get("keyframes") or [])],
        width=int(args["width"]),
        height=int(args["height"]),
        num_frames=int(args["num_frames"]),
        max_frames=max_frames,
        num_inference_steps=int(args["num_inference_steps"]),
        cfg_scale=float(args["cfg_scale"]),
        vace_scale=float(args["vace_scale"]),
        seed=int(args["seed"]),
        progress_cb=_prog,
    )
    # Persist each generated frame as a PNG scratch file and hand the parent a
    # small list of paths (mirrors bernini_cli's file-path wire) — the parent
    # reads them back to encode the MP4, so the child->parent pipe never carries
    # the frames' base64.
    out_dir = tempfile.mkdtemp(prefix="idv2vfr-", dir=tempfile.gettempdir())
    paths = []
    for i, f in enumerate(frames):
        p = os.path.join(out_dir, "f%05d.png" % i)
        f.convert("RGB").save(p, format="PNG")
        paths.append(p)
    return {"frame_dir": out_dir, "frames": paths, "count": len(paths)}


HANDLERS = {
    "status": _handle_status,
    "load": _handle_load,
    "infer_frames": _handle_infer_frames,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="id-v2v diffsynth engine subprocess")
    ap.add_argument("--device", required=True,
                    help="cuda:N target GPU (authoritative from /load)")
    args = ap.parse_args(argv)

    build_engine = _load_build_engine()
    from runner.common import engineproc

    return engineproc.run_child_loop(
        args.device,
        lambda: build_engine(args.device),
        HANDLERS,
        ready_msg=f"idv2v engine ready (device={args.device})",
    )


if __name__ == "__main__":
    raise SystemExit(main())
