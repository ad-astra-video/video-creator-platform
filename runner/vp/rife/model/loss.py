"""Inference-only stubs for RIFE's training losses.

Upstream ComfyUI-VFI ships a full `model/loss.py` (EPE / Ternary / SOBEL /
VGGPerceptualLoss) that exists solely for TRAINING RIFE (`Model.update()`). It
imports torchvision (models.vgg19) at module load. This vp-worker only runs
INFERENCE (`Model.inference` / `Model.inference_batch`): those code paths never
call a loss function, but `RIFE_HDv3.Model.__init__` DOES eagerly construct
``self.epe = EPE()`` and ``self.sobel = SOBEL()``, so those two names must
exist as lambdas and construct without error.

We therefore ship minimal no-op `torch.nn.Module` subclasses so the runtime
image doesn't need torchvision and the server boots. They are never invoked
(inference has no training step).
"""

import torch.nn as nn


class EPE(nn.Module):
    """No-op placeholder for RIFE's end-point-error training loss."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError("EPE is training-only; not used in inference")


class SOBEL(nn.Module):
    """No-op placeholder for RIFE's sobel smoothness training loss."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError("SOBEL is training-only; not used in inference")


# Ternary / VGGPerceptualLoss are only referenced inside `Model.update()`, which
# we never run — leaving them undefined is fine (the `import *` in RIFE_HDv3.py
# binds whatever exists; the missing names are only used in training paths).
