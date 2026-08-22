"""FlashVSR diffusion upscale rail for the vp-worker.

Upscales a source video (or frame dir) with FlashVSR v1.1 tiny — a one-step
streaming diffusion VSR (Wan2.1-DMD backbone) that runs at 4x with a tiny
conditional decoder. Weights: `JunhaoZhuang/FlashVSR-v1.1` (public) under
``FLASHVSR_ROOT``.

Interface mirrors the repo's example (examples/WanVSR/infer_flashvsr_v1.1_tiny.py)
so the vendored ``Causal_LQ4x_Proj`` / ``build_tcdecoder`` pairing stays correct.
Because FlashVSR needs the diffsynth package (>= ...) which also conflicts with
the fatter worker venvs, it runs INSIDE the vp-worker's own isolated venv
(``FLASHVSR_VENV_PY``); the module here is imported directly by
``runner.vp.server`` which lives in that same venv.

Pipeline (all on GPU, one-shot DMD stream + decoupled TCDecoder):
    video -> LQ proj (Causal_LQ4x_Proj) -> FlashVSRTinyPipeline(dmd) -> TCDecoder -> frames

I/O is a HIGH-RES numpy frame stack; the server owns the streamed transport.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np
import torch

from .flashvsr_utils.utils import Causal_LQ4x_Proj
from .flashvsr_utils.TCDecoder import build_tcdecoder

logger = logging.getLogger("video_creator.runner.vp.flashvsr_post")


class UpscaleError(RuntimeError):
    pass


def _tensor2frames(frames: torch.Tensor) -> np.ndarray:
    """T C H W bf16 [-1,1] -> [F,H,W,3] uint8 numpy."""
    from einops import rearrange
    fr = rearrange(frames, "C T H W -> T H W C")
    fr = ((fr.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
    return fr


def _prepare_video_tensor(path: str, scale: int = 4, multiple: int = 128,
                          dtype: torch.dtype = torch.bfloat16,
                          device: str = "cuda"):
    """Read a video, upscale-then-center-crop to a 128-multiple target.

    Returns (LQ_tensor[1,C,F,H,W], tH, tW, F, src_fps). Mirrors the repo's
    prepare_input_tensor video branch (with the 8n+1 frame padding).
    """
    import imageio
    from PIL import Image

    rdr = imageio.get_reader(path)
    first = Image.fromarray(rdr.get_data(0)).convert("RGB")
    w0, h0 = first.size
    try:
        meta = rdr.get_meta_data()
        fps = int(round(meta.get("fps", 30)))
    except Exception:
        fps = 30

    def count_frames(r):
        try:
            nf = r.get_meta_data().get("nframes")
            if isinstance(nf, int) and nf > 0:
                return nf
        except Exception:
            pass
        try:
            return r.count_frames()
        except Exception:
            n = 0
            try:
                while True:
                    r.get_data(n); n += 1
            except Exception:
                return n

    total = count_frames(rdr)
    if total <= 0:
        rdr.close()
        raise UpscaleError(f"cannot read frames from {path}")

    sW = int(round(w0 * scale)); sH = int(round(h0 * scale))
    tW = (sW // multiple) * multiple; tH = (sH // multiple) * multiple
    if tW == 0 or tH == 0:
        rdr.close()
        raise UpscaleError(f"scaled size too small ({sW}x{sH}) for multiple={multiple}")

    def largest_8n1(n):
        return 0 if n < 1 else ((n - 1) // 8) * 8 + 1

    idx = list(range(total)) + [total - 1] * 4
    F = largest_8n1(len(idx))
    idx = idx[:F]

    frames = []
    for i in idx:
        img = Image.fromarray(rdr.get_data(i)).convert("RGB")
        if tW > sW or tH > sH:
            rdr.close()
            raise UpscaleError("target crop exceeds scaled size")
        up = img.resize((sW, sH), Image.BICUBIC)
        l = (sW - tW) // 2; t = (sH - tH) // 2
        img_out = up.crop((l, t, l + tW, t + tH))
        tt = torch.from_numpy(np.asarray(img_out, np.uint8)).to(
            device=device, dtype=torch.float32)
        tt = tt.permute(2, 0, 1) / 255.0 * 2.0 - 1.0
        frames.append(tt.to(dtype))
    rdr.close()
    vid = torch.stack(frames, 0).permute(1, 0, 2, 3).unsqueeze(0)
    return vid, tH, tW, F, fps


class FlashVsrUpscaler:
    """Warm one-step FlashVSR v1.1 tiny upscaler."""

    def __init__(self, root: str, device: Optional[str] = None,
                 scale: int = 4, topk_ratio: float = 1.2):
        if not root or not os.path.isdir(root):
            raise UpscaleError(f"FlashVSR weights not found: {root!r}")
        self.root = root.rstrip("/")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.topk_ratio = topk_ratio
        self._pipe: Any = None

        torch.set_grad_enabled(False)
        self._build()

    def _build(self) -> None:
        from diffsynth import ModelManager, FlashVSRTinyPipeline

        dmd = os.path.join(self.root, "diffusion_pytorch_model_streaming_dmd.safetensors")
        if not os.path.exists(dmd):
            raise UpscaleError(f"FlashVSR DMD weights missing: {dmd!r}")
        mm = ModelManager(torch_dtype=self.dtype, device="cpu")
        mm.load_models([dmd])
        pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=self.device)
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(
            in_dim=3, out_dim=1536, layer_num=1).to(self.device, dtype=self.dtype)
        lq_path = os.path.join(self.root, "LQ_proj_in.ckpt")
        if os.path.exists(lq_path):
            pipe.denoising_model().LQ_proj_in.load_state_dict(
                torch.load(lq_path, map_location="cpu"), strict=True)
        pipe.denoising_model().LQ_proj_in.to(self.device)
        pipe.TCDecoder = build_tcdecoder(
            new_channels=[512, 256, 128, 128],
            new_latent_channels=16 + 768)
        tc_path = os.path.join(self.root, "TCDecoder.ckpt")
        if os.path.exists(tc_path):
            pipe.TCDecoder.load_state_dict(
                torch.load(tc_path, map_location="cpu"), strict=False)
        pipe.to(self.device)
        pipe.enable_vram_management(num_persistent_param_in_dit=None)
        pipe.init_cross_kv()
        pipe.load_models_to_device(["dit", "vae"])
        self._pipe = pipe
        logger.info("FlashVSR v1.1 tiny upscaler ready (device=%s, scale=%s)",
                    self.device, self.scale)

    def upscale(self, video_path: str, height: Optional[int] = None,
                width: Optional[int] = None, seed: int = 0,
                color_fix: bool = True, local_range: int = 11) -> np.ndarray:
        """Upscale a video file; returns [F,H,W,3] uint8 frames."""
        if self._pipe is None:
            raise UpscaleError("FlashVSR not loaded")
        LQ, tH, tW, F, fps = _prepare_video_tensor(
            video_path, scale=self.scale, dtype=self.dtype, device=self.device)
        # SSAA: when the caller wants a final dimension smaller than the raw
        # 4x output, FlashVSR still renders full 4x and the server downscales.
        pipe = self._pipe
        sparse_ratio = self.topk_ratio * 768 * 1280 / (tH * tW)
        out = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0,
            num_inference_steps=1, seed=seed,
            LQ_video=LQ, num_frames=F, height=tH, width=tW,
            is_full_block=False, if_buffer=True,
            topk_ratio=sparse_ratio, kv_ratio=3.0,
            local_range=local_range, color_fix=color_fix,
        )
        torch.cuda.empty_cache()
        return _tensor2frames(out)

    def unload(self) -> None:
        self._pipe = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
