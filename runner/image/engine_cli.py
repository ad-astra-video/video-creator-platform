"""Child entrypoint for the image-worker's MODEL subprocess.

Runs the real GPU engine (Qwen-Image-Edit / Qwen-Image-Layered / Z-Image /
HiDream-O1-Image + the FLUX.2 klein singleton) inside a DEDICATED python
process, spawned by the parent aiohttp server via::

    python -m runner.image.engine_cli --device N

This process owns the CUDA primary context. The parent proxies every engine
call to it over JSONL (``runner.common.engineproc``), and /evict TERMINATES this
process — process exit is the only clean way to destroy a CUDA primary context
(PyTorch has no in-process teardown API for it).

Constraints honoured here
    * ``engineproc.run_child_loop`` pins ``torch.cuda.set_device(device)`` BEFORE
      the engine builds, so no context leaks onto an unassigned card.
    * The FLUX.2 klein singleton (``flux_edit``) lives in THIS process too, so
      stopping this process on /evict frees klein's VRAM alongside the Qwen /
      Z-Image / HiDream pipelines.
    * Only JSON goes to stdout (the command channel); engine logs go to stderr.

The parent NEVER runs this code and never imports it (so it stays CUDA-free).
"""

from __future__ import annotations

import argparse
import sys


def _build_engine(device: int):
    """Mirror how ``server._engine_for`` used to construct an engine: a bare
    ``ImageInferenceEngine`` bound to ``device`` via ``current_device`` (which
    every generation method reads to pick the active GPU)."""
    from runner.image.inference import ImageInferenceEngine

    engine = ImageInferenceEngine()
    engine.current_device = int(device)
    return engine


def _progress2(progress_cb):
    """Adapt the engine's ``progress_cb(step, total)`` (a 2-arg simple callable)
    to the harness's ``progress_cb(dict)`` so each step is emitted as a
    ``{"type":"progress", "step":.., "total_steps":..}`` line to the parent."""
    if progress_cb is None:
        return None

    def _cb(step: int, total: int):
        try:
            progress_cb({"step": int(step), "total_steps": int(total)})
        except Exception:  # noqa: BLE001 - progress must never break inference
            pass

    return _cb


def _img_result(out: object) -> dict:
    """Persist a PIL result to a scratch PNG and return its path.

    The parent aiohttp server reads the file back (and re-encodes it for the
    HTTP response). The JSONL wire carries just a small path instead of the
    multi-MB base64 — mirrors bernini_cli's file-path contract and keeps the
    child->parent pipe small and deterministic."""
    import os
    import tempfile
    fd, path = tempfile.mkstemp(prefix="imgr-", suffix=".png")
    os.close(fd)
    out.convert("RGB").save(path, format="PNG")
    return {"image_path": path}


def _b64_string_to_path(v: str) -> str:
    """Decode a PNG-base64 string to a scratch file and return its path."""
    import base64 as _b64
    import os
    import tempfile
    fd, path = tempfile.mkstemp(prefix="imgl-", suffix=".png")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(_b64.b64decode(v))
        return path
    except Exception:  # noqa: BLE001 - never hand a broken path to the parent
        try:
            os.remove(path)
        except OSError:
            pass
        return ""


def _replace_b64_with_paths(obj):
    """Walk a JSON-safe dict/list; write every non-empty ``*_b64`` string to a
    scratch file and replace it with the file path. The parent restores the
    same fields to base64 by reading the files back (small wire, mirroring
    bernini_cli's file-path contract). Non-``*_b64`` values pass through."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, str) and k.endswith("_b64") and v:
                obj[k] = _b64_string_to_path(v)
            else:
                obj[k] = _replace_b64_with_paths(v)
    elif isinstance(obj, list):
        return [_replace_b64_with_paths(x) for x in obj]
    return obj


def _handlers(device: int) -> dict:
    """Build the ``{op: fn(engine, args, progress_cb) -> json-safe}`` map.

    All image/mask inputs arrive as base64 strings (or lists of base64 strings
    for multi-reference edits) straight from the HTTP body; the engine methods
    already decode them via ``inference._decoded_pil``, so we pass them through
    unchanged. PIL outputs are encoded to base64 via ``_pil_to_b64``.
    """
    from runner.image import flux_edit

    def _edit(e, args, progress_cb):
        kw = dict(args.get("kw") or {})
        img = e.edit_image(
            args["image"],
            args["prompt"],
            engine=args.get("engine", "qwen-edit"),
            mask=args.get("mask"),
            keep_subject=bool(args.get("keep_subject", False)),
            strength=float(args.get("strength", 0.6)),
            padding_mask_crop=int(args.get("padding_mask_crop", 0) or 0),
            mask_composite=bool(args.get("mask_composite", True)),
            progress_cb=_progress2(progress_cb),
            **kw,
        )
        return _img_result(img)

    def _hidream_edit(e, args, progress_cb):
        kw = dict(args.get("kw") or {})
        img = e.hidream_edit(
            args["image"],
            args["prompt"],
            seed=args.get("seed"),
            keep_original_aspect=bool(args.get("keep_original_aspect", True)),
            num_inference_steps=args.get("num_inference_steps"),
            quality=args.get("quality"),
            progress_cb=_progress2(progress_cb),
            **kw,
        )
        return _img_result(img)

    def _hidream_image(e, args, progress_cb):
        kw = dict(args.get("kw") or {})
        img = e.hidream_image(
            args["prompt"],
            width=args.get("width", 1024),
            height=args.get("height", 1024),
            seed=args.get("seed"),
            num_inference_steps=args.get("num_inference_steps"),
            guidance_scale=args.get("guidance_scale"),
            quality=args.get("quality"),
            progress_cb=_progress2(progress_cb),
            **kw,
        )
        return _img_result(img)

    def _plain_image(e, args, progress_cb):
        img = e.plain_image(args["prompt"], **dict(args.get("kw") or {}))
        return _img_result(img)

    def _klein_image(e, args, progress_cb):
        img = e.klein_image(
            args["prompt"],
            seed=args.get("seed", 123),
            width=args.get("width", 1024),
            height=args.get("height", 1024),
            num_inference_steps=args.get("num_inference_steps"),
            **dict(args.get("kw") or {}),
        )
        return _img_result(img)

    def _style_frame(e, args, progress_cb):
        img = e.style_frame(
            args["image"],
            args["prompt"],
            seed=args.get("seed", 123),
            width=args.get("width"),
            height=args.get("height"),
            num_inference_steps=args.get("num_inference_steps"),
        )
        out = _img_result(img)
        out["width"] = img.size[0]
        out["height"] = img.size[1]
        return out

    def _layered_decompose(e, args, progress_cb):
        # The engine returns the /layer contract as a JSON-safe dict already
        # (b64 layers / composite + dims). Write each base64 image to a scratch
        # file and hand the parent a small path-only dict to read back.
        return _replace_b64_with_paths(e.layered_decompose(
            args["image"],
            layers=args.get("layers"),
            resolution=args.get("resolution"),
            preview_only=bool(args.get("preview_only", False)),
            num_inference_steps=args.get("num_inference_steps"),
            progress_cb=_progress2(progress_cb),
        ))

    def _klein_resident_device(e, args, progress_cb):
        ed = flux_edit.get_editor()
        if ed.is_ready:
            dev = str(ed.device)
            if dev.startswith("cuda:"):
                return int(dev.split(":", 1)[1])
        return None

    def _klein_evict(e, args, progress_cb):
        flux_edit.evict_editor()
        return None

    def _klein_ready(e, args, progress_cb):
        return bool(flux_edit.get_editor().is_ready)

    return {
        "edit_image": _edit,
        "hidream_edit": _hidream_edit,
        "hidream_image": _hidream_image,
        "plain_image": _plain_image,
        "klein_image": _klein_image,
        "style_frame": _style_frame,
        "layered_decompose": _layered_decompose,
        "klein_resident_device": _klein_resident_device,
        "klein_evict": _klein_evict,
        "klein_ready": _klein_ready,
    }


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="runner.image.engine_cli",
        description="Image-worker GPU model subprocess (child of the aiohttp server).",
    )
    parser.add_argument("--device", type=int, required=True,
                        help="CUDA device index this child pins + owns.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    from runner.common import engineproc  # lazily import the shared harness

    args = _parse_args(argv)
    return engineproc.run_child_loop(
        device=f"cuda:{args.device}",
        build_engine=lambda: _build_engine(args.device),
        handlers=_handlers(args.device),
        ready_msg=f"image worker model subprocess ready on cuda:{args.device}",
    )


if __name__ == "__main__":
    sys.exit(main())
