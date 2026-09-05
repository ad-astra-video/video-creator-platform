"""Image inference engine for the image-worker.

Serves three capabilities:
  * Qwen-Image-Edit        — instruction-following image editing.
  * Qwen-Image-Layered     — semantic layer decomposition (foreground / midground
                             planes / background) for downstream compositing.
  * Z-Image (Turbo)        — fast text-to-image + whole-frame img2img edits.

All torch / diffusers imports are LAZY (inside the functions that need them), so
``import runner.image.inference`` succeeds on a host without torch or diffusers
installed — the pure-Python aiohttp route layer stays importable and testable
standalone. No CUDA work happens at import time.
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import threading
from typing import Any

from PIL import Image

from runner.image import config as _cfg

logger = logging.getLogger(__name__)


# ── FP8 linear (native Blackwell, no torchao) ────────────────────────────────
# TESTED 2026-08 on this stack (py3.12 / torch 2.11+cu128 / SM120): torchao
# 0.18.0 ITSELF imports fine (wheel is cp310-abi3 = cross-version, so the tag
# is not a python blocker), but its native fp8 CUDA extensions (_C_mxfp8,
# _C_cutlass_90a) fail to load under cu12.8 — built against CUDA 13
# (libcudart.so.13 mismatch). Without those kernels torchao's fp8 runs via
# SOFTWARE emulation (no fp8 tensor cores -> no advantage). So the accelerated
# fp8 path here is the native torch._scaled_mm shim below: we wrap the
# pre-quantized fp8 Linear weights in a small module that quantizes activations
# per-token and does the scaled matmul against the fp8 weight. Weights stay
# genuinely fp8 (the user requirement); only the activation is quantized
# dynamically (standard fp8 w8a8). Revisit only if a torchao fp8 kernel is ever
# shown to load AND beat _scaled_mm on this exact stack.

_FP8_MAX = 448.0


# ── Pure-Python image helpers (no torch/diffusers needed) ────────────────────

def _decoded_pil(image: Any) -> Image.Image:
    """Coerce an input into a PIL.Image: an Image, raw bytes, or a base64/data URI
    str/bytes. All remote inputs arrive as base64, so this is the common path."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(image))).convert("RGB")
    if isinstance(image, str):
        raw = image
        if raw.startswith("data:"):
            raw = raw.split(",", 1)[-1]
        return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    raise TypeError(f"cannot decode image of type {type(image)!r}")


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 string (used for every response field)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _fit(img: Image.Image, side: int) -> Image.Image:
    """Resize an image to fit within a ``side``x``side`` box (LANCZOS), keeping
    aspect ratio. Never upscales beyond the source."""
    w, h = img.size
    scale = min(side / w, side / h, 1.0)
    if scale >= 1.0:
        return img
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def _edit_step_cb(progress_cb, total: int):
    """Build a diffusers ``callback_on_step_end`` callable for the Qwen edit pipes.

    diffusers calls it with (pipe, step, timestep, callback_kwargs); we forward a
    human 1-based step + total to ``progress_cb`` and pass kwargs through so
    sampling runs normally."""
    def _cb(pipe, step, timestep, cb_kwargs):
        try:
            progress_cb(min(step + 1, total), total)
        except Exception:
            logger.debug("edit progress_cb raised", exc_info=True)
        return cb_kwargs
    return _cb


def _to_pil(out: Any) -> Image.Image:
    """Convert a diffusers pipeline return value to a single PIL.Image.

    Tolerates the common shapes: a bare PIL.Image, a list/tuple of images, or an
    object exposing ``.images`` (diffusers PipelinesOutput)."""
    if isinstance(out, Image.Image):
        return out
    if isinstance(out, (list, tuple)):
        if out and isinstance(out[0], Image.Image):
            return out[0]
        raise TypeError("empty / non-image pipeline output sequence")
    images = getattr(out, "images", None)
    if images is not None:
        if isinstance(images, Image.Image):
            return images
        if isinstance(images, (list, tuple)) and images:
            return images[0]
    raise TypeError(f"cannot convert pipeline output {type(out)!r} to PIL.Image")


def _composite_masked(edited: Image.Image, original: Image.Image,
                       mask: Image.Image) -> Image.Image:
    """Paste the ORIGINAL pixels back over every image area the mask marks black.

    White mask = the region the model repainted (kept from ``edited``); black =
    everything to preserve. Resizes the mask to the output size if needed and
    composites so the untouched frame is pixel-identical to the source.
    """
    # Normalize sizes (edited may come back at a padded/cropped resolution).
    w = edited.size[0]
    h = edited.size[1]
    base = original.resize((w, h), Image.LANCZOS).convert("RGB")
    m = mask.convert("L").resize((w, h), Image.NEAREST)
    return Image.composite(edited.convert("RGB"), base, m)


def _naive_layers(src: Image.Image, n: int) -> list[Image.Image]:
    """Deterministic fallback decomposition: split the (already clamped) image into
    ``n`` horizontal bands, each returned as a full-frame RGBA layer whose own band
    is opaque and the rest transparent.

    Used when the Qwen layered pipeline isn't loaded or its output can't be
    introspected, so the /layer endpoint still returns a valid contract."""
    w, h = src.size
    src_rgba = src.convert("RGBA")
    band = max(1, h // n)
    layers = []
    for i in range(n):
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        top = i * band
        bottom = h if i == n - 1 else min(h, (i + 1) * band)
        crop = src_rgba.crop((0, top, w, bottom))
        layer.paste(crop, (0, top))
        layers.append(layer)
    return layers


def _extract_layers(out: Any, n: int) -> list[Image.Image] | None:
    """Best-effort extraction of ``n`` per-layer RGBA images from a layered-pipeline
    return. Returns None (caller falls back) on any shape we can't parse.

    diffusers' QwenImageLayeredPipeline returns a *nested* ``out.images``: one inner
    list per batch item, each holding that batch's layer frames (the raw composite
    frame is stripped by the pipeline before decode, so every inner element is a
    single decomposed layer). E.g. for 4 layers / batch=1::

        out.images == [ [layer_0, layer_1, layer_2, layer_3] ]

    We take the first batch's inner list and keep the first ``n`` PIL frames.
    """
    try:
        images = getattr(out, "images", None)
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(images, (list, tuple)):
            # Unwrap one level of batching: if the outer sequence holds lists/tuples
            # (diffusers nested output), flatten to the first batch's frames.
            if images and isinstance(images[0], (list, tuple)):
                images = list(images[0])
            images = [im for im in images if isinstance(im, Image.Image)]
        else:
            return None
        if len(images) < n:
            return None
        return [im.convert("RGBA") for im in images[:n]]
    except Exception:  # pragma: no cover - defensive
        return None


def _composite(layer_imgs: list[Image.Image]) -> Image.Image:
    """Composite layers back onto a common background (bottom layer first)."""
    base = layer_imgs[0] if layer_imgs else Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    for layer in layer_imgs[1:]:
        if layer.size == base.size:
            base = Image.alpha_composite(base, layer.convert("RGBA"))
    return base


# Quality preset -> HiDream-O1-Image denoise steps. The 'full' recipe is
# autorad 50 steps; 'dev' is 28. The client threads a bare quality name and the
# worker translates (/image /edit), mirroring the other engines.
HIDREAM_STEP_PRESETS = {"fast": 20, "balanced": 28, "high": 50}


def _resolve_steps(num_inference_steps, quality, engine: str) -> int:
    """Resolve denoise steps: explicit num_inference_steps > quality preset."""
    if num_inference_steps is not None:
        try:
            return max(1, int(num_inference_steps))
        except (TypeError, ValueError):
            pass
    q = str(quality or "").strip().lower()
    presets = HIDREAM_STEP_PRESETS if engine == "hidream" else {}
    if q in presets:
        return presets[q]
    return _cfg.HIDREAM_STEPS


def _hidream_dims(width, height) -> tuple[int, int]:
    """Clamp HiDream-O1-Image output dims to the model's max, preserving any
    aspect ratio the caller requested (0/None width/height -> default 1024)."""
    try:
        w = int(width or 1024)
        h = int(height or 1024)
    except (TypeError, ValueError):
        w = h = 1024
    w = max(256, min(w, _cfg.HIDREAM_MAX_SIDE))
    h = max(256, min(h, _cfg.HIDREAM_MAX_SIDE))
    return w, h


class ImageInferenceEngine:
    """Wraps Qwen-Image-Edit, Qwen-Image-Layered and Z-Image pipelines.

    Pipelines are built lazily on first use (see the ``_*_pipe`` builders) and
    evicted via ``free()`` / ``_evict_other()`` so only one model occupies VRAM.
    The server sets ``current_device`` from the /load body; everything else
    defaults to ``DEFAULT_DEVICE``.
    """

    def __init__(self, profile: Any = None) -> None:
        self._profile = profile
        self._qwen_edit: Any = None          # QwenImageEditPipeline
        self._qwen_edit_inpaint: Any = None  # QwenImageEditInpaintPipeline (shares _qwen_edit components)
        self._qwen_layered: Any = None       # QwenImageLayeredPipeline
        self._zimage: Any = None             # ZImagePipeline (+ ZImageImg2Img/Inpaint)
        self._hidream: Any = None            # HiDream-O1-Image UiT (model, processor)
        # Serializes lazy builds and evictions (a load must never race an evict).
        self._model_lock = threading.RLock()
        # The active CUDA device index, set by the server from the /load body.
        self.current_device: int | None = None
        self.ready = False

    # ------------------------------------------------------------------
    # Device / dtype resolution (torch imported lazily)
    # ------------------------------------------------------------------
    def _active_device(self) -> int:
        """Resolve the CUDA device index to use, falling back to DEFAULT_DEVICE."""
        idx = self.current_device
        if idx is None:
            idx = _cfg.DEFAULT_DEVICE
        return int(idx)

    def _torch_device(self):
        idx = self._active_device()
        import torch
        if torch.cuda.is_available():
            return torch.device(f"cuda:{idx}")
        return torch.device("cpu")

    def _resolve_dtype(self, torch):
        """Pick the module load dtype from QWEN_DTYPE (fp8|bf16|int8)."""
        d = _cfg.QWEN_DTYPE.strip().lower()
        if d == "fp8":
            # Load the transformer weights already downcast to fp8. Fall back to
            # bf16 if this torch build lacks float8_e4m3fn.
            return getattr(torch, "float8_e4m3fn", torch.bfloat16)
        if d == "int8":
            # int8 is applied post-load via quantization (torchao) rather than a
            # torch dtype; load bf16 here and let apply_int8_() handle it below.
            return torch.bfloat16
        return torch.bfloat16  # bf16 (and anything unexpected)

    def _maybe_offload(self, pipe: Any) -> None:
        """Apply enable_model_cpu_offload() when QWEN_OFFLOAD and CUDA are present;
        otherwise move the whole pipeline to the active device."""
        import torch
        if _cfg.QWEN_OFFLOAD and torch.cuda.is_available():
            pipe.enable_model_cpu_offload(gpu_id=self._active_device())
        else:
            pipe.to(self._torch_device())

    # ------------------------------------------------------------------
    # Lazy pipeline builders
    # ------------------------------------------------------------------
    def _qwen_edit_pipe(self):
        """Build (once) and return the Qwen-Image-Edit-2511 pipeline.

        Uses ``QwenImageEditPlusPipeline`` (the 2509/2511 class that accepts a
        LIST of reference images for multi-image editing). When QWEN_DTYPE=fp8
        the transformer is a PRE-QUANTIZED fp8 checkpoint loaded per-component
        (text_encoder/VAE bf16, transformer fp8) and routed through the native
        torch._scaled_mm ``_swap_fp8_linears`` path, exactly like the layered
        engine — so fp8 weights stay fp8 instead of being silently widened.
        """
        with self._model_lock:
            if self._qwen_edit is None:
                import torch
                from diffusers import QwenImageEditPlusPipeline
                dtype = _cfg.QWEN_DTYPE.strip().lower()
                root = _cfg.QWEN_EDIT_ROOT
                logger.info("Loading Qwen-Image-Edit-2511 from %s (dtype=%s, offload=%s)",
                            root, dtype, _cfg.QWEN_OFFLOAD)
                if dtype == "fp8":
                    # Per-component fp8 load: transformer from the pre-quantized
                    # fp8 single-file (kept fp8), text_encoder/VAE bf16. Mirrors
                    # _qwen_layered_pipe's fp8 branch (incl. the native
                    # torch._scaled_mm swap; torchao's fp8 CUDA kernels can't
                    # load under cu12.8, so torchao fp8 = emulation only).
                    from transformers import (
                        AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration,
                    )
                    from diffusers import (
                        AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
                        QwenImageTransformer2DModel,
                    )
                    _sched = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{root}/scheduler")
                    _vae = AutoencoderKLQwenImage.from_pretrained(f"{root}/vae", torch_dtype=torch.bfloat16)
                    _te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        f"{root}/text_encoder", torch_dtype=torch.bfloat16)
                    _tok = AutoTokenizer.from_pretrained(f"{root}/tokenizer")
                    _proc = AutoProcessor.from_pretrained(f"{root}/processor")
                    _tr = QwenImageTransformer2DModel.from_pretrained(
                        f"{root}/transformer", torch_dtype=torch.float8_e4m3fn)
                    ImageInferenceEngine._fix_nonlinear_dtypes(_tr)
                    ImageInferenceEngine._swap_fp8_linears(_tr)
                    self._qwen_edit = QwenImageEditPlusPipeline(
                        scheduler=_sched, vae=_vae, text_encoder=_te,
                        tokenizer=_tok, processor=_proc, transformer=_tr)
                else:
                    self._qwen_edit = QwenImageEditPlusPipeline.from_pretrained(
                        root, torch_dtype=self._resolve_dtype(torch))
                    if dtype == "int8":
                        self._apply_int8(self._qwen_edit)
                self._maybe_offload(self._qwen_edit)
                self.ready = True
            return self._qwen_edit

    def _qwen_layered_pipe(self):
        """Build (once) and return the QwenImageLayeredPipeline.

        Honors QWEN_LAYERED_DTYPE independently of QWEN_DTYPE. When 'fp8'
        (default), the transformer is the PRE-QUANTIZED FP8 (E4M3FN) single-file
        checkpoint from T5B/Qwen-Image-Layered-FP8. It is loaded with fp8 only on
        the transformer (text_encoder + vae stay bf16 via PipelineQuantizationConfig)
        so the pre-quantized fp8 fits the card AND every matmul is dtype-consistent.
        'bf16'/'int8' fall back to the bf16-load path (int8 then applies torchao).
        """
        with self._model_lock:
            if self._qwen_layered is None:
                import torch
                from diffusers import QwenImageLayeredPipeline
                ldt = _cfg.QWEN_LAYERED_DTYPE
                logger.info("Loading Qwen-Image-Layered from %s (layered_dtype=%s, offload=%s)",
                            _cfg.QWEN_LAYERED_ROOT, ldt, _cfg.QWEN_OFFLOAD)
                if ldt == "fp8":
                    # Qwen-Image-Layered-FP8 is PRE-QUANTIZED (mixed F8_E4M3 + BF16
                    # on disk). Pitfalls that break naive loads:
                    #  * torch_dtype=torch.bfloat16 WIDENS the fp8 weights to bf16
                    #    (~2x VRAM) -> OOM on 24-32GB cards.
                    #  * torch_dtype=None on the WHOLE pipeline keeps fp8 but leaves
                    #    the text_encoder/VAE as fp32 -> dtype mismatch on their convs.
                    #  * torch_dtype=float8_e4m3fn on the WHOLE pipeline cannot
                    #    deserialize (set_default_dtype(fp8) unsupported for the
                    #    transformers text_encoder).
                    # Correct load: per-component. Transformer loads at its NATIVE
                    # pre-quantized dtype (fp8 deserialized as torchao Float8Tensor,
                    # so fp8xbf16 scaled matmul dispatches correctly); text_encoder +
                    # vae load at bf16 (they aren't fp8 and must be dtype-consistent).
                    from transformers import (
                        AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration,
                    )
                    from diffusers import (
                        AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler,
                        QwenImageTransformer2DModel,
                    )
                    root = _cfg.QWEN_LAYERED_ROOT
                    _sched = FlowMatchEulerDiscreteScheduler.from_pretrained(f"{root}/scheduler")
                    _vae = AutoencoderKLQwenImage.from_pretrained(f"{root}/vae", torch_dtype=torch.bfloat16)
                    _te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        f"{root}/text_encoder", torch_dtype=torch.bfloat16)
                    _tok = AutoTokenizer.from_pretrained(f"{root}/tokenizer")
                    _proc = AutoProcessor.from_pretrained(f"{root}/processor")
                    _tr = QwenImageTransformer2DModel.from_pretrained(
                        f"{root}/transformer", torch_dtype=torch.float8_e4m3fn)
                    # Keep Linear weights fp8 (the user's requirement); restore
                    # norms/biases to bf16; swap Linears for _Fp8Linear (native
                    # torch._scaled_mm). torchao's fp8 CUDA ext can't load under
                    # cu12.8 (built vs CUDA 13) -> emulation only, so native
                    # _scaled_mm stays the accelerated fp8 path here.
                    ImageInferenceEngine._fix_nonlinear_dtypes(_tr)
                    ImageInferenceEngine._swap_fp8_linears(_tr)
                    self._qwen_layered = QwenImageLayeredPipeline(
                        scheduler=_sched, vae=_vae, text_encoder=_te,
                        tokenizer=_tok, processor=_proc, transformer=_tr)
                else:
                    self._qwen_layered = QwenImageLayeredPipeline.from_pretrained(
                        _cfg.QWEN_LAYERED_ROOT,
                        torch_dtype=self._resolve_dtype(torch),
                    )
                    if _cfg.QWEN_DTYPE.strip().lower() == "int8":
                        self._apply_int8(self._qwen_layered)
                self._maybe_offload(self._qwen_layered)
                self.ready = True
            return self._qwen_layered

    def _qwen_edit_inpaint_pipe(self):
        """Return a QwenImageEditInpaintPipeline built AROUND the already-loaded
        QwenImageEditPipeline's components (vae/text_encoder/tokenizer/processor/
        transformer) so the masked inpainting path adds NO extra model RAM — the
        edit model is a ~52 GB bf16 load, so a second from_pretrained() would
        OOM the host. The scheduler is not shared (inpaint uses its own), but all
        heavy modules are. Enable CPU offload once on the wrapper (hooks already
        live on the shared sub-modules from the edit pipe)."""
        with self._model_lock:
            if self._qwen_edit_inpaint is None:
                import torch
                from diffusers import QwenImageEditInpaintPipeline
                edit = self._qwen_edit_pipe()
                self._qwen_edit_inpaint = QwenImageEditInpaintPipeline(
                    scheduler=edit.scheduler,
                    vae=edit.vae,
                    text_encoder=edit.text_encoder,
                    tokenizer=edit.tokenizer,
                    processor=edit.processor,
                    transformer=edit.transformer,
                )
                if _cfg.QWEN_OFFLOAD and torch.cuda.is_available():
                    self._qwen_edit_inpaint.enable_model_cpu_offload(gpu_id=self._active_device())
                else:
                    self._qwen_edit_inpaint.to(self._torch_device())
                logger.info("Built QwenImageEditInpaintPipeline sharing editor components")
            return self._qwen_edit_inpaint

    def _zimage_pipe(self):
        """Build (once) and return the ZImagePipeline (text-to-image).

        Z-Image loads at bf16 (NOT the fp8 from QWEN_DTYPE): the generic
        diffusers ``from_pretrained`` re-runs transformers' fp8 path on the
        whole pipeline and fails with ``TypeError: couldn't find storage object
        Float8_e4m3fnStorage`` (fp8 only works for the Qwen transformer via the
        dedicated per-component branches above). Z-Image Turbo (~6B) fits bf16
        on the same card."""
        with self._model_lock:
            if self._zimage is None:
                import torch
                from diffusers import ZImagePipeline
                logger.info("Loading Z-Image from %s (dtype=bf16, offload=%s)",
                            _cfg.ZIMAGE_ROOT, _cfg.QWEN_OFFLOAD)
                self._zimage = ZImagePipeline.from_pretrained(
                    _cfg.ZIMAGE_ROOT,
                    torch_dtype=torch.bfloat16,
                )
                self._maybe_offload(self._zimage)
                self.ready = True
            return self._zimage

    def _hidream_pipe(self):
        """Build (once) and return the HiDream-O1-Image UiT (model, processor).

        HiDream-O1-Image is an 8B pixel-level Unified Transformer run through the
        vendored `hidream_models/` pipeline (custom Qwen3VL UiT + repo schedulers).
        Unlike the Qwen/Z-Image pipelines it is a plain transformers model, NOT a
        diffusers pipeline: load AutoProcessor + Qwen3VLForConditionalGeneration
        (bf16) and cache the (model, processor) pair. Rendered pixels are returned
        directly by generate_image() — there is no VAE/decoder. Uses the eager
        4D-mask attention path by default (HIDREAM_USE_FLASH_ATTN defaults off)."""
        with self._model_lock:
            if self._hidream is None:
                import torch
                from .hidream_models.qwen3_vl_transformers import (
                    Qwen3VLForConditionalGeneration,
                )
                from transformers import AutoProcessor
                root = _cfg.HIDREAM_ROOT
                logger.info("Loading HiDream-O1-Image from %s (dtype=%s, flash=%s)",
                            root, _cfg.HIDREAM_DTYPE, "on")
                dtype = torch.float32 if _cfg.HIDREAM_DTYPE == "fp32" else torch.bfloat16
                processor = AutoProcessor.from_pretrained(root)
                # Load on CPU then move to the target GPU: device_map triggers
                # accelerate's meta-device lazy path in transformers 5.15,
                # which leaves non-persistent buffers (e.g. rope inv_freq) on
                # meta -> "Cannot copy out of meta tensor". A plain load + .to()
                # materializes everything.
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    root, torch_dtype=dtype,
                ).to(f"cuda:{self._active_device()}").eval()
                self._hidream = {"model": model, "processor": processor,
                                 "device": f"cuda:{self._active_device()}"}
                self.ready = True
            return self._hidream

    def hidream_image(self, prompt, width=1024, height=1024, seed=None,
                      num_inference_steps=None, guidance_scale=None, quality=None,
                      progress_cb=None, **kw) -> Image.Image:
        """Text-to-image via HiDream-O1-Image -> a single PIL.Image.

        ``quality`` (fast/balanced/high) maps to step counts when
        num_inference_steps isn't given. ``num_inference_steps``/guidance
        default to the 'full' recipe (50 / 5.0) unless overridden. Produces
        native-resolution output up to HIDREAM_MAX_SIDE."""
        self._evict_other("hidream")
        hid = self._hidream_pipe()
        model, processor = hid["model"], hid["processor"]
        from .hidream_models import pipeline as _hp

        steps = _resolve_steps(num_inference_steps, quality, "hidream")
        guid = float(guidance_scale) if guidance_scale else _cfg.HIDREAM_GUIDANCE
        width, height = _hidream_dims(width, height)

        def _cb(step, total, decode=None):
            if progress_cb:
                try:
                    progress_cb(step + 1, total)
                except Exception:
                    pass

        return _hp.generate_image(
            model=model, processor=processor, prompt=prompt,
            ref_image_paths=None, height=height, width=width,
            num_inference_steps=int(steps), guidance_scale=guid,
            shift=3.0, timesteps_list=None, scheduler_name="default",
            seed=int(seed) if seed is not None else 32,
            callback=_cb if progress_cb else None,
        )

    def hidream_edit(self, image, prompt, seed=None, keep_original_aspect=True,
                     num_inference_steps=None, quality=None, progress_cb=None,
                     **kw) -> Image.Image:
        """Instruction-based image editing via HiDream-O1-Image (single ref image).

        ``image`` is the source (base64 str, bytes, or PIL). The edit runs the
        repo pipeline with exactly one reference image; ``keep_original_aspect``
        (default True) preserves the source's aspect ratio at the model's native
        resolution. The full (undistilled) model is recommended for editing."""
        self._evict_other("hidream")
        hid = self._hidream_pipe()
        model, processor = hid["model"], hid["processor"]
        from .hidream_models import pipeline as _hp

        # Multi-reference edit: image may be a single source or a list; write
        # every source to a temp PNG and pass all of them as ref_image_paths.
        images = image if isinstance(image, (list, tuple)) else [image]
        srcs = [_decoded_pil(i) for i in images]
        import tempfile, os
        paths = []
        for src in srcs:
            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            src.save(path, format="PNG")
            paths.append(path)
        try:
            steps = _resolve_steps(num_inference_steps, quality, "hidream")
            def _cb(step, total, decode=None):
                if progress_cb:
                    try:
                        progress_cb(step + 1, total)
                    except Exception:
                        pass
            return _hp.generate_image(
                model=model, processor=processor, prompt=prompt,
                ref_image_paths=paths, height=2048, width=2048,
                num_inference_steps=int(steps), guidance_scale=_cfg.HIDREAM_GUIDANCE,
                shift=3.0, timesteps_list=None, scheduler_name="default",
                seed=int(seed) if seed is not None else 32,
                keep_original_aspect=bool(keep_original_aspect),
                callback=_cb if progress_cb else None,
            )
        finally:
            for path in paths:
                try:
                    os.remove(path)
                except OSError:
                    pass

    @staticmethod
    def _apply_int8(pipe: Any) -> None:
        """Best-effort torchao int8 weight-only quantization when QWEN_DTYPE=int8.
        Silently skips if torchao isn't installed so the worker still runs."""
        # A diffusers *pipeline* object is not an nn.Module (it has no
        # named_children), so quantize_() must be applied to each quantizable
        # sub-model (transformer, text_encoder, vae) individually.
        try:
            import torch
            import torchao
            from torchao.quantization import quantize_, Int4WeightOnlyConfig, Int8WeightOnlyConfig
            # torchao 0.18 config-based API. int4 (~14GB for the 20B+7B stack)
            # reliably fits a 32 GB card; int8 (~30GB) is marginal there.
            cfg = Int4WeightOnlyConfig() if _cfg.QWEN_AO_QUANT == "int4" else Int8WeightOnlyConfig()
            quantized = 0
            for name in ("transformer", "text_encoder", "vae", "unet"):
                m = getattr(pipe, name, None)
                if m is not None and isinstance(m, torch.nn.Module):
                    quantize_(m, cfg)
                    quantized += 1
            if quantized == 0:
                logger.warning("torchao: no quantizable sub-model found; using bf16")
        except Exception as exc:  # pragma: no cover - env-dependent
            logger.warning("torchao weight-only quant unavailable (%s); using bf16 load", exc)


    @staticmethod
    def _fix_nonlinear_dtypes(mod: Any) -> None:
        """After loading a transformer at torch_dtype=fp8, keep Linear weights fp8
        but restore every NON-Linear parameter (norms, scales, biases) to bf16.
        fp8 is only valid inside the matmul; LayerNorm-style multiplies and
        additions need a normal dtype."""
        import torch
        for name, child in mod.named_children():
            if not isinstance(child, torch.nn.Linear):
                for pname, p in list(child.named_parameters(recurse=False)):
                    if p.dtype == torch.float8_e4m3fn:
                        setattr(child, pname, torch.nn.Parameter(p.to(torch.bfloat16)))
                ImageInferenceEngine._fix_nonlinear_dtypes(child)

    @staticmethod
    def _swap_fp8_linears(mod: Any) -> None:
        """Replace every nn.Linear in the transformer with _Fp8Linear (native
        torch._scaled_mm); the fp8 weight is kept as-is (still fp8)."""
        import torch

        class _Fp8Linear(torch.nn.Module):
            """fp8 dynamic-activation linear via torch._scaled_mm; fp8 weight kept
            as a buffer so accelerate's CPU-offload hook moves it (+bias) on/off
            GPU. scale_b=1.0 (weights already fp8 -> literal); activations
            quantized per-token with scale_a = amax / _FP8_MAX."""

            def __init__(self, weight, bias):
                super().__init__()
                if bias is None:
                    bias = torch.zeros(
                        weight.shape[0], dtype=torch.bfloat16, device=weight.device)
                self.register_buffer("weight", weight.detach().contiguous())  # (out,in)
                self.register_buffer("bias", bias.detach().to(torch.bfloat16))

            def forward(self, x):  # noqa: D401
                orig = x.shape
                x2 = x.reshape(-1, x.shape[-1]).float()
                dev = x.device
                w = self.weight.to(dev)
                sx = (x2.abs().amax(dim=1, keepdim=True).clamp_(min=1e-6)
                      / _FP8_MAX).to(torch.float32)
                xq = (x2 / sx).to(torch.float8_e4m3fn)
                sb = torch.ones(1, w.shape[0], device=dev, dtype=torch.float32)
                y = torch._scaled_mm(
                    xq, w.t(), scale_a=sx, scale_b=sb,
                    out_dtype=torch.bfloat16)
                out = y[0] if isinstance(y, tuple) else y
                out = out + self.bias.to(dev).unsqueeze(0)
                return out.reshape(*orig[:-1], -1).to(x.dtype)

        for name, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear):
                setattr(mod, name, _Fp8Linear(child.weight, child.bias))
            else:
                ImageInferenceEngine._swap_fp8_linears(child)

    # ------------------------------------------------------------------
    # Intra-container eviction
    # ------------------------------------------------------------------
    def _evict_other(self, keeper: str) -> None:
        """Drop every pipeline except the one named by ``keeper`` ('edit' | 'layered' |
        'zimage' | 'klein'), freeing their GPU memory. Used before a build so only one
        image model occupies VRAM at a time on constrained cards."""
        import torch
        if keeper != "klein":
            # Any non-klein build evicts the FLUX.2 klein editor first (it shares the card).
            try:
                from . import flux_edit
                flux_edit.evict_editor()
            except Exception:
                pass
        with self._model_lock:
            if keeper != "edit":
                self._qwen_edit = None
                self._qwen_edit_inpaint = None
            if keeper != "layered":
                self._qwen_layered = None
            if keeper != "zimage":
                self._zimage = None
            if keeper != "hidream":
                self._hidream = None
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

    def free(self) -> None:
        """Release ALL resident pipelines + cached GPU memory.

        Called by the worker's /evict endpoint so another worker can take the GPU.
        Each pipeline reloads lazily on the next request via the ``_*_pipe`` builders."""
        import torch
        try:
            from . import flux_edit
            flux_edit.evict_editor()
        except Exception:
            pass
        with self._model_lock:
            # Drain any in-flight kernels on this engine's GPU FIRST so their
            # allocations are returned before we drop the pipelines and flush
            # the caching allocator. Without the synchronize, a still-running
            # stream can hold VRAM even after empty_cache (context not cleared).
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize(self.current_device)
                except Exception:
                    torch.cuda.synchronize()
            self._qwen_edit = None
            self._qwen_edit_inpaint = None
            self._qwen_layered = None
            self._zimage = None
            self._hidream = None
            self.ready = False
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()

    @property
    def profile(self):
        return self._profile

    # ------------------------------------------------------------------
    # Generation methods
    # ------------------------------------------------------------------
    def edit_image(self, image, prompt, engine="qwen-edit", mask=None,
                   keep_subject=False, strength=0.6, padding_mask_crop=0,
                   mask_composite=True, progress_cb=None, **kw) -> Image.Image:
        """Instruction / img2img image edit -> a single PIL.Image.

        engine='qwen-edit' -> Qwen-Image-Edit. When ``mask`` is supplied the
        edit runs through QwenImageEditInpaintPipeline (white mask = repaint,
        black = preserve) with ``strength`` controlling how aggressively the
        masked region is regenerated and ``padding_mask_crop`` giving the model
        surrounding context around a small masked object. The inpaint result is
        hard-composited back over ORIGINAL source pixels outside the mask when
        ``mask_composite`` (default) so untouched regions are pixel-identical.

        engine='zimage' -> Z-Image whole-frame img2img edit (masked edits use
        Z-Image inpaint)."""
        # Multi-reference-image edit (Qwen-Image-Edit-2511 / HiDream): ``image``
        # may be a single base64 str/bytes/PIL OR a list of them (the frontend
        # attaches 1..n references). Engines that are single-conditioning only
        # (zimage, mask path) collapse to the first image; qwen-edit / hidream
        # forward the whole list as their reference images.
        is_multi = isinstance(image, (list, tuple))
        src = _decoded_pil(image[0] if is_multi else image)
        mask_img = _decoded_pil(mask) if mask is not None else None
        engine = str(engine).lower()

        if engine == "zimage":
            self._evict_other("zimage")
            pipe = self._zimage_pipe()
            call_kw = dict(prompt=prompt, strength=strength)
            self._zimage_call_kw(kw)
            if mask_img is not None:
                # Use the Z-Image inpaint pipeline for a masked edit.
                from diffusers import ZImageInpaintPipeline  # lazy
                inpaint = ZImageInpaintPipeline.from_pretrained(
                    _cfg.ZIMAGE_ROOT, **self._pipe_kwargs()
                )
                self._maybe_offload(inpaint)
                out = inpaint(image=src, mask_image=mask_img, **call_kw, **kw)
            else:
                from diffusers import ZImageImg2ImgPipeline  # lazy
                i2i = ZImageImg2ImgPipeline.from_pretrained(
                    _cfg.ZIMAGE_ROOT, **self._pipe_kwargs()
                )
                self._maybe_offload(i2i)
                out = i2i(image=src, **call_kw, **kw)
            return _to_pil(out)

        # default: Qwen-Image-Edit
        self._evict_other("edit")
        pipe = self._qwen_edit_pipe()
        # diffusers 0.39.0 QwenImageEditPipeline takes a seeded `generator`, not a bare
        # `seed` (passing `seed` TypeErrors at __call__) — reuse the same seed->generator
        # translation the Z-Image paths use. `kw` is a fresh dict, safe to mutate.
        self._zimage_call_kw(kw)
        if mask_img is not None:
            # Region-masked edit via QwenImageEditInpaintPipeline (white = repaint,
            # black = preserve). The wrapper shares the editor's loaded components so
            # this adds no extra model RAM. `strength` controls regeneration
            # aggressiveness; `padding_mask_crop` gives context around a small object.
            inpaint = self._qwen_edit_inpaint_pipe()
            in_kw = dict(prompt=prompt, strength=float(strength))
            if _cfg.QWEN_STEPS and "num_inference_steps" not in kw:
                in_kw["num_inference_steps"] = _cfg.QWEN_STEPS
            for k in ("num_inference_steps", "guidance_scale"):
                if kw.get(k) is not None:
                    in_kw[k] = kw[k]
            if int(padding_mask_crop or 0) > 0:
                in_kw["padding_mask_crop"] = int(padding_mask_crop)
            if progress_cb is not None:
                in_kw["callback_on_step_end"] = _edit_step_cb(
                    progress_cb, int(in_kw.get("num_inference_steps", _cfg.QWEN_STEPS)))
            out = inpaint(image=src, mask_image=mask_img, **in_kw,
                          generator=kw.get("generator"))
            edited = _to_pil(out)
            if mask_composite:
                # Hard-composite the ORIGINAL source back over the mask's black
                # (preserve) region so everything outside the mask is pixel-identical.
                edited = _composite_masked(edited, src, mask_img)
            return edited
        call_kw = dict(prompt=prompt)
        if progress_cb is not None:
            call_kw["callback_on_step_end"] = _edit_step_cb(
                progress_cb, int(kw.get("num_inference_steps", _cfg.QWEN_STEPS)))
        ref_images = ([_decoded_pil(i) for i in image]
                      if is_multi else [src])
        # 2511 multi-image conditioning: pass the reference list to the pipe.
        # A single image may be passed as a bare PIL for exact parity.
        out = pipe(image=(ref_images[0] if len(ref_images) == 1 else ref_images),
                   **call_kw, **kw)
        return _to_pil(out)

    def plain_image(self, prompt, **kw) -> Image.Image:
        """Text-to-image generation via Z-Image (Turbo) -> a single PIL.Image."""
        self._evict_other("zimage")
        self._zimage_call_kw(kw)
        pipe = self._zimage_pipe()
        out = pipe(prompt=prompt, **kw)
        return _to_pil(out)

    def klein_image(self, prompt, seed=123, width=1024, height=1024,
                    num_inference_steps=None, **kw) -> Image.Image:
        """Text-to-image generation via FLUX.2 [klein] 4B -> a single PIL.Image.

        Klein is a step/guidance-distilled model (BFL: 4 steps, guidance 1.0, no
        CFG), so ``num_inference_steps`` (and guidance) are FEELERS — the editor
        clamps/ignores guidance and exposes a modest step override. Dispatched
        when the /image handler receives engine='klein'.
        """
        self._evict_other("klein")
        from . import flux_edit
        editor = flux_edit.get_editor()
        # Device-aware: bind klein to THIS engine's assigned GPU (the scheduler's
        # X-Worker-Device) so its residency is visible to the live-runner's /info
        # map and can be evicted per-card instead of parking silently on cuda:0.
        editor.relocate(self._active_device())
        editor.ensure_loaded()
        steps = num_inference_steps if num_inference_steps is not None else None
        width = int(width or 1024)
        height = int(height or 1024)
        return editor.generate(
            prompt, int(seed), width=width, height=height, steps=steps,
        )

    def layered_decompose(self, image, layers=None, resolution=None,
                          preview_only=False, num_inference_steps=None,
                          progress_cb=None) -> dict:
        """Decompose an image into semantically-ordered layers.

        Returns the /layer contract::

            {
              "layers": [ {index, rgba_b64, preview_b64, alpha_b64, label}, ... ],
              "composite": "<b64 png>",
              "width": int, "height": int, "layers_requested": int,
            }

        Labels: layer 0 -> 'foreground', last -> 'background', middle -> 'midground'.
        In ``preview_only`` mode only the small ``preview_b64`` thumbnails are
        populated (rgba_b64/alpha_b64 are empty strings) so the response stays tiny."""
        src = _decoded_pil(image)
        max_side = _cfg.QWEN_LAYER_MAX_INPUT_SIDE
        w, h = src.size
        scale = max_side / max(w, h) if max(w, h) > max_side else 1.0
        if scale < 1.0:
            src = src.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
            w, h = src.size

        n = layers if layers is not None else _cfg.QWEN_LAYERS
        n = max(2, min(int(n), _cfg.QWEN_MAX_LAYERS))

        # Qwen-Image-Layered's VAE first-conv expects a 4-channel RGBA input
        # (the layers are spawned FROM the source's alpha-bearing image). The
        # generic decoder returns RGB, so promote to RGBA for the layered call
        # (unused channels are opaque white, matching the official example).
        layered_src = src.convert("RGBA")

        layer_imgs: list[Image.Image] | None = None
        steps = int(num_inference_steps) if num_inference_steps else _cfg.QWEN_STEPS
        if not preview_only:
            def _on_step(step: int, total: int, **_: Any) -> None:
                if progress_cb:
                    try:
                        progress_cb(step, total)
                    except Exception:
                        logger.debug("progress_cb raised", exc_info=True)
            try:
                self._evict_other("layered")
                pipe = self._qwen_layered_pipe()
                call_kw = dict(image=layered_src, layers=n, num_inference_steps=steps)
                if resolution:
                    call_kw["resolution"] = resolution
                if progress_cb:
                    def _step_cb(pglob, step, timestep, cb_kw):
                        _on_step(step + 1, steps)
                        return cb_kw
                    call_kw["callback_on_step_end"] = _step_cb
                out = pipe(**call_kw)
                layer_imgs = _extract_layers(out, n)
                if layer_imgs is None:
                    logger.warning("layered pipeline output not introspectable; falling back")
            except Exception as exc:
                logger.warning("Qwen layered decomposition failed (%s); using naive fallback", exc)
        if layer_imgs is None:
            layer_imgs = _naive_layers(src, n)

        preview_side = _cfg.QWEN_LAYER_PREVIEW_SIDE
        out_layers = []
        for i, img in enumerate(layer_imgs[:n]):
            label = "foreground" if i == 0 else ("background" if i == n - 1 else "midground")
            out_layers.append({
                "index": i,
                "rgba_b64": "" if preview_only else _pil_to_b64(img.convert("RGBA")),
                "preview_b64": _pil_to_b64(_fit(img, preview_side)),
                "alpha_b64": "" if preview_only else _pil_to_b64(
                    img.split()[-1].convert("RGBA")
                ),
                "label": label,
            })

        return {
            "layers": out_layers,
            "composite": _pil_to_b64(_composite(layer_imgs)),
            "width": w,
            "height": h,
            "layers_requested": n,
        }

    def style_frame(self, image, prompt, seed=123, width=None, height=None,
                    num_inference_steps=None) -> Image.Image:
        """Style ``image`` (a restyle first frame) with FLUX.2 klein 4B.

        Returns the styled PIL RGB image. Mirrors the id-v2v worker's /style-frame
        contract (single-reference edit: encode the frame through the FLUX.2 AE,
        denoise with ``prompt`` describing the desired styled result). The klein
        editor evicts the Qwen/Z-Image pipelines first; callers are responsible for
        the prompt/composition-hold suffix. ``num_inference_steps`` (Fast 4 /
        Balanced 8 / High 12) lets quality presets nudge the distilled editor.
        """
        src = _decoded_pil(image)
        self._evict_other("klein")
        from . import flux_edit
        editor = flux_edit.get_editor()
        # Device-aware: bind klein to THIS engine's assigned GPU (the scheduler's
        # X-Worker-Device) so its residency is visible to the live-runner's /info
        # map and can be evicted per-card instead of parking silently on cuda:0.
        editor.relocate(self._active_device())
        editor.ensure_loaded()
        try:
            styled = editor.edit(src, prompt, int(seed), width=width, height=height,
                                 steps=num_inference_steps)
            return styled
        finally:
            # Keep klein resident until the next non-klein build evicts it, so a
            # burst of style-frame requests reuses the loaded editor (like other
            # pipelines). free() drops it.
            pass

    def _zimage_call_kw(self, kw: dict) -> dict:
        """Translate a ``seed`` kwarg into the ``generator`` Z-Image expects, and
        default CFG guidance to 0.0.

        Z-Image-Turbo is a GUIDANCE-DISTILLED (guidance-free) model. diffusers'
        ``ZImagePipeline`` defaults ``guidance_scale`` to 5.0 and applies CFG
        (``pred = pos + g * (pos - neg)``) at that value unless overridden —
        running a distilled turbo with CFG at 5.0 yields over-saturated, washed-out,
        incorrect output (verified empirically: gs=0.0 vs gs=5.0 differ by ~110/765
        mean-abs pixel). The desktop reference forces guidance_scale=0.0 ("Turbo is
        guidance-free"). We honour an EXPLICIT client guidance_scale (setdefault),
        but a request that omits it gets the correct turbo default 0.0 instead of 5.0.

        ZImagePipeline / ZImageImg2ImgPipeline / ZImageInpaintPipeline take a
        ``generator`` (a seeded torch.Generator) for reproducibility and DO NOT
        accept a bare ``seed`` key — passing one raises TypeError. When a seed is
        present, pop it and build a generator seeded with it; otherwise leave the
        kwargs untouched (random draw). Return the (possibly mutated) kwargs dict.
        """
        if "seed" not in kw:
            return kw
        import torch  # lazy
        seed = int(kw.pop("seed"))
        idx = self.current_device
        device = f"cuda:{idx}" if idx is not None else "cpu"
        kw["generator"] = torch.Generator(device=device).manual_seed(seed)
        # Turbo is guidance-distilled: never let the diffusers default CFG (5.0) apply.
        # Honour an explicit client guidance_scale; default 0.0 otherwise.
        kw.setdefault("guidance_scale", 0.0)
        return kw

    def _pipe_kwargs(self) -> dict:
        """Shared kwargs for from_pretrained pipeline loads (dtype resolution)."""
        import torch  # lazy
        return {"torch_dtype": self._resolve_dtype(torch)}
