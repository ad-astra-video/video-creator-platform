"""Inference-only stub for RIFE's training losses.

Upstream ComfyUI-VFI ships a full `model/loss.py` (EPE / Ternary / SOBEL /
VGGPerceptualLoss) that exists solely for TRAINING RIFE (`Model.update()`).
It imports torchvision (models.vgg19) at module load. This vp-worker only runs
INFERENCE (`Model.inference` / `Model.inference_batch`), which never touches
any loss name — `RIFE_HDv3.py` does `from ..model.loss import *` at module top,
and the referenced names are only used inside `update()`. We therefore ship this
stub so the runtime image does not need torchvision.

Swap back the upstream file only if you add a training path.
"""
