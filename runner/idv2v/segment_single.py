"""Single-image SAM3 foreground/object mask (text prompt).

Runs Segment Anything 3 (transformers Sam3VideoModel/Processor) on ONE image and
writes a binary union mask of every detected object for the given text prompt
(e.g. "person"). This mask is the object the restyle UI wants to KEEP unchanged;
the image edit then inverts it so everything else is regenerated.

Isolated as a subprocess so SAM3 never competes with the resident id-v2v model
for GPU/RAM on the .8 box, mirroring how the restyle pipeline already shells out
to ``idv2v.preprocess.sam3``.

Usage: python -m runner.idv2v.segment_single --image in.png --prompt person \\
        --model_path /models/sam3 --out_mask out.png [--gpu 0]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image


def _union_mask(frames: list, width: int, height: int) -> np.ndarray:
    """Combine per-instance masks from ``pack_per_frame_instances`` into one binary mask."""
    union = np.zeros((height, width), dtype=np.uint8)
    for inst in frames:
        m = inst.get("mask")
        if m is None:
            continue
        m = np.asarray(m)
        m = (m > 0).astype(np.uint8)
        if m.shape != (height, width):
            m = np.asarray(
                Image.fromarray(m * 255, mode="L").resize((width, height), Image.NEAREST)
            )
        union[m > 0] = 255
    return union


def main() -> int:
    p = argparse.ArgumentParser(description="Single-image SAM3 foreground mask")
    p.add_argument("--image", required=True, help="Path to the input image (PNG/JPEG)")
    p.add_argument("--prompt", default="person", help="Text prompt for segmentation")
    p.add_argument("--model_path", default="/models/sam3", help="HF repo id or local SAM3 path")
    p.add_argument("--out_mask", required=True, help="Path to write the binary mask PNG")
    p.add_argument("--gpu", default="0", help="CUDA device index")
    a = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)

    import torch
    from idv2v.preprocess import sam3 as _ref  # lazy: only present in the worker image

    image = Image.open(a.image).convert("RGB")
    width, height = image.size

    model, processor, device = _ref.init_sam3_video(model_path=a.model_path, dtype=torch.bfloat16)
    try:
        with torch.inference_mode():
            outputs = _ref.run_sam3_video(
                model, processor, device, [image], a.prompt, dtype=torch.bfloat16,
            )
        packed = _ref.pack_per_frame_instances(outputs)
        instances = packed.get(0, [])
        union = _union_mask(instances, width, height)
        Image.fromarray(union, mode="L").save(a.out_mask)
        return 0
    finally:
        try:
            model = processor = None  # noqa: F841
            import gc
            gc.collect()
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
