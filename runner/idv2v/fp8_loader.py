"""Diffsynth-free, memory-bounded loader for the per-channel FP8 ID-V2V checkpoint.

WHY (not diffsynth's loader):
  * diffsynth.load_state_dict() materialises the WHOLE checkpoint in system RAM as
    bf16 before assigning it anywhere — the 73 GB local .pth, or the ~28 GB bf16
    dequant for the FP8 HF repo. On the .8 box (30 GB RAM, ~28 GB available) that
    single peak OOMs for BOTH the int8 and fp8 paths.
  * This loader never builds a full state_dict. It streams shard-by-shard and fills
    each nn.Parameter in place, so peak RAM = model resident in its *stored*
    precision (fp8) + ONE shard (~1.9 GB) + ONE dequantised layer (~52 MB).

NATIVE FP8 (no whole-model bf16 anywhere):
  * Quantisable layers are swapped for FP8Linear/FP8ConvNd, whose weight stays
    fp8-E4M3 resident in system RAM (~19.5 GB for DiT+VACE, half of bf16). Each
    layer dequantises *itself* to bf16 on the compute device in forward, then runs
    a plain matmul. Weights never leave fp8, so no 28 GB bf16 copy ever exists.

CHECKPOINT FORMAT (id-v2v-fp8 is a ComfyUI-style export):
  * EVERY parameter is stored keyspace ``<name>.weight`` (+ ``<name>.weight_scale``
    when fp8-quantized) under a ``dit.`` / ``vace.`` stem.
  * Module-weight leaves (the FP8Linear/FP8ConvNd) match the wrapper weight path.
  * Plain ``nn.Parameter`` leaves (modulation, head.modulation, ...) are stored as
    ``<name>.weight`` while the model param is ``<name>`` -> resolved by stripping
    the ``.weight`` suffix.
  * Plain params that were fp8-quantized (modulation) are DEQUANTIZED in place
    (fp8*scale -> the param's bf16 dtype). Plain bf16 leaves (norms, biases) copy
    directly.
  * The WanModel/VaceWanModel are constructed under diffsynth's
    ``init_weights_on_device()`` (meta device); ``materialize_meta()`` rewrites the
    small meta leaves to real CPU tensors so ``copy_`` can write them.

NOT PROVIDED HERE (your choice):
  * The transformer forward pass (Wan DiT attention/RoPE/cross-attn) + VACE forward.
    diffsynth's WanModel/VaceWanModel are plain torch.nn.Modules and CAN be used
    with this loader outside diffsynth's pipeline. This file has ZERO diffsynth
    import — it only needs the target nn.Module(s).
"""

from __future__ import annotations

import gc
import logging
import os
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
from safetensors import safe_open

logger = logging.getLogger("video_creator.runner.idv2v.fp8_loader")


# ---------------------------------------------------------------------------
# Quantised layers: fp8 weight + per-channel scale resident; dequant on compute
# device per call. scale stored [Cout,1,...] so it broadcasts over the full
# weight shape (the 4D-conv pitfall).
# ---------------------------------------------------------------------------


class FP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, has_bias: bool = False,
                 device=None):
        super().__init__()
        dev = device or "cpu"
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float8_e4m3fn, device=dev),
            requires_grad=False,
        )
        self.scale = nn.Parameter(
            torch.ones(out_features, 1, dtype=torch.bfloat16, device=dev), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=dev), requires_grad=False)
            if has_bias
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.scale.to(x.dtype)  # per-layer dequant [out,in]
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class FP8ConvNd(nn.Module):
    """Conv wrapper for patch-embed / proj convs (1D/2D/3D). fp8 [Cout,Cin,*ks], scale [Cout,1,..]."""

    def __init__(self, weight_shape, has_bias: bool, stride, padding, device=None):
        super().__init__()
        dev = device or "cpu"
        self.weight = nn.Parameter(
            torch.empty(weight_shape, dtype=torch.float8_e4m3fn, device=dev), requires_grad=False
        )
        self.scale = nn.Parameter(
            torch.ones(weight_shape[0], *([1] * (len(weight_shape) - 1)), dtype=torch.bfloat16, device=dev),
            requires_grad=False,
        )
        self.bias = (
            nn.Parameter(torch.empty(weight_shape[0], dtype=torch.bfloat16, device=dev), requires_grad=False)
            if has_bias
            else None
        )
        self.stride, self.padding = stride, padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.to(x.dtype) * self.scale.to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        ndim = w.dim() - 2
        if ndim == 1:
            return F.conv1d(x, w, b, self.stride, self.padding)
        if ndim == 2:
            return F.conv2d(x, w, b, self.stride, self.padding)
        return F.conv3d(x, w, b, self.stride, self.padding)


def _quantisable_layer(module: nn.Module) -> Optional[str]:
    if isinstance(module, nn.Linear):
        return "FP8Linear"
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "FP8ConvNd"
    return None


def replace_quantisable_layers(model: nn.Module, device=None) -> int:
    """Swap quantisable leaves for FP8 wrappers in place; returns count replaced."""
    n = 0
    for name, child in list(model.named_children()):
        cls = _quantisable_layer(child)
        if cls is not None:
            has_bias = getattr(child, "bias", None) is not None
            if cls == "FP8Linear":
                new = FP8Linear(child.in_features, child.out_features, has_bias, device)
            else:
                new = FP8ConvNd(tuple(child.weight.shape), has_bias, child.stride, child.padding, device)
            setattr(model, name, new)
            n += 1
        else:
            n += replace_quantisable_layers(child, device)
    return n


def materialize_meta(module: nn.Module, device=None, dtype=None) -> int:
    """Rewrite any ``meta``-device parameter to a real empty tensor on `device`
    (CPU default) at `dtype` (default: the param's own dtype).

    ``init_weights_on_device()`` builds WanModel/VaceWanModel params on ``meta``
    at their *default* dtype, so LayerNorm/nn.Parameter leaves come out fp32 and
    ``copy_`` cannot write into a meta tensor. We replace ONLY the meta params
    (the small non-quantised leaves) in place via ``module._parameters`` — the
    bulk fp8 weights are already real inside FP8Linear/FP8ConvNd and untouched.
    Pass dtype=torch.bfloat16 so the model runs all-bf16 and diffsynth's
    fp32-weight-LayerNorm-vs-bf16-activation mismatch disappears.
    """
    n = 0
    for sub in module.modules():
        for name, p in list(sub._parameters.items()):
            if p is not None and p.device.type == "meta":
                sub._parameters[name] = nn.Parameter(
                    torch.empty(p.shape, dtype=dtype or p.dtype, device=device or "cpu"),
                    requires_grad=p.requires_grad,
                )
                n += 1
    return n


# ---------------------------------------------------------------------------
# Streaming fill. One tensor at a time; peak = fp8 model + one shard + one layer.
# ---------------------------------------------------------------------------


def params_lookup(mod: nn.Module, name: str):
    for n, pp in mod.named_parameters():
        if n == name:
            return pp
    return None


def load_fp8_into_models(
    dit: nn.Module,
    shards: List[str],
    vace: Optional[nn.Module] = None,
    *,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, int]:
    """Fill post-surgery DiT (+ VACE) directly from the HF FP8 shards, memory-bounded.

    Returns counts: {"fp8": n, "plain": n, "dequant": n, "replaced": n}.
    """
    counts = {"fp8": 0, "plain": 0, "dequant": 0, "replaced": 0}

    def route(k: str):
        """Return (module, target_name) for a checkpoint key."""
        if vace is not None and k.startswith("vace."):
            return vace, k[len("vace."):]
        if k.startswith("dit."):
            return dit, k[len("dit."):]
        return dit, k

    def resolve(mod: nn.Module, target: str, params: Dict[str, nn.Parameter]):
        """Return the param for a (stem-stripped) ckpt key, mapping ``<n>.weight``
        onto a bare ``<n>`` param when there is no ``<n>.weight`` param."""
        if target in params:
            return target, params[target]
        if target.endswith(".weight"):
            base = target[: -len(".weight")]
            if base in params:
                return base, params[base]
        return None, None

    for mod in (dit, vace):
        if mod is None:
            continue
        counts["replaced"] += len(
            [m for m in mod.modules() if isinstance(m, (FP8Linear, FP8ConvNd))]
        )

    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for k in keys:
                if k.endswith(".comfy_quant"):
                    continue
                mod, target = route(k)
                params = {name: pp for name, pp in mod.named_parameters()}
                pname, p = resolve(mod, target, params)
                if p is None:
                    continue
                is_fp8 = k.endswith(".weight") and (k[: -len(".weight")] + ".weight_scale") in keys
                if is_fp8:
                    w = f.get_tensor(k)
                    s = f.get_tensor(k[: -len(".weight")] + ".weight_scale")
                    if p.dtype == torch.float8_e4m3fn:
                        # FP8Linear/FP8ConvNd weight: fp8 weight + separate per-channel scale
                        p.data.copy_(w.to(p.dtype).to(device))
                        sc = params_lookup(mod, target[: -len(".weight")] + ".scale")
                        if sc is not None:
                            sc.data.copy_(s.to(sc.dtype).to(device))
                        counts["fp8"] += 1
                    else:
                        # plain nn.Parameter that was fp8-quantized in the export:
                        # dequant fp8*scale -> the param's (bf16) dtype.
                        p.data.copy_((w.to(p.dtype) * s.to(p.dtype)).to(device))
                        counts["dequant"] += 1
                    del w, s
                else:
                    p.data.copy_(f.get_tensor(k).to(p.dtype).to(device))
                    counts["plain"] += 1
        gc.collect()
        logger.info("shard %s done: %s", os.path.basename(shard), counts)

    return counts


def snapshot_fp8_checkpoint(repo: str, token: Optional[str], subfolder: str = "",
                           local_dir: str = "/models", allow_patterns=None) -> List[str]:
    """Return the sorted .safetensors shards for a model variant.

    * ``subfolder == ""``  -> the REGULAR model: download (or use the exist-cache)
      HF repo root via ``snapshot_download`` and list its shards (existing path).
    * ``subfolder != ""``  -> the FAST model: read the local folder
      ``<local_dir>/<subfolder>`` (e.g. /models/fusionx). If that folder is
      missing/incomplete, sync it from the HF repo's ``<subfolder>`` via
      per-file ``hf_hub_download`` (avoids ``snapshot_download(subfolder=...)``,
      which older huggingface_hub versions reject).

    ``local_dir`` defaults to config.IDV2V_MODEL_DIR.
    """
    from huggingface_hub import snapshot_download, list_repo_files, hf_hub_download

    allow = allow_patterns or ["*.safetensors", "*.json"]
    if subfolder:
        shard_dir = os.path.join(local_dir, subfolder)
        present = os.path.isdir(shard_dir) and any(
            f.endswith(".safetensors") for f in os.listdir(shard_dir))
        if not present:
            logger.info("Syncing fast-model folder %s from HF %s/%s ...",
                        shard_dir, repo, subfolder)
            os.makedirs(shard_dir, exist_ok=True)
            files = [f for f in list_repo_files(repo, token=token or None)
                     if f.startswith(subfolder + "/")
                     and (f.endswith(".safetensors") or f.endswith(".json"))]
            for f in sorted(files):
                hf_hub_download(repo, f, token=token or None, local_dir=local_dir)
        return sorted(
            os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
            if f.endswith(".safetensors"))
    local = snapshot_download(repo, token=token or None, allow_patterns=allow)
    return sorted(
        os.path.join(local, f) for f in os.listdir(local)
        if f.endswith(".safetensors"))
