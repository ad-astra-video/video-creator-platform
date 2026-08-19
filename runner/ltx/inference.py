"""Video-Creator GPU inference engine (LTX-2.3 + ID-V2V model families).

All generation endpoints accept base64-encoded input and return base64-encoded
output files (mp4 for video, png for image). This matches the contract expected
by the LTX-Desktop remote inference client.
"""

from __future__ import annotations

import base64
import gc
import io
import logging
import os
import random
import tempfile
import threading
from typing import TYPE_CHECKING, Any, cast

import torch
from PIL import Image

if TYPE_CHECKING:
    from runner.ltx.gpu_profile import GPUProfile

logger = logging.getLogger(__name__)


def _memlog(tag: str) -> None:
    import torch as _t
    if not _t.cuda.is_available():
        return
    parts = [f"g{i}={_t.cuda.memory_allocated(i)//1048576}M/{_t.cuda.memory_reserved(i)//1048576}M"
             for i in range(_t.cuda.device_count())]
    logger.info("MEMLOG %-24s %s", tag, " ".join(parts))


def _reflect_pad_to_target(image: "Image.Image", width: int, height: int) -> "Image.Image":
    """Resize (contain) + reflect-pad a start image to the exact target dims.

    Mirrors ltx_pipelines' ResizeMode.REFLECT_PAD conditioning behaviour: preserve
    the source aspect ratio (fit within), then reflect-pad the shorter axis to the
    exact (width, height). Because the result is already exactly the target size,
    the pipeline's downstream ``resize_and_center_crop`` becomes a no-op, so the
    image is never centre-cropped and its full framing survives on frame 0. Falls
    back to edge padding when a padding strip would be too wide for a reflect pad
    (same guard ltx_pipelines uses).
    """
    import numpy as np

    src_w, src_h = image.size
    if height >= src_h and width >= src_w:
        new_w, new_h = src_w, src_h
    else:
        scale = min(height / src_h, width / src_w)
        new_w = round(src_w * scale)
        new_h = round(src_h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
    pad_bottom = height - new_h
    pad_right = width - new_w
    if pad_bottom > 0 or pad_right > 0:
        mode = "reflect" if pad_bottom < new_h and pad_right < new_w else "edge"
        arr = np.pad(
            np.asarray(image),
            ((0, pad_bottom), (0, pad_right), (0, 0)),
            mode=mode,
        )
        image = Image.fromarray(arr)
    return image


# Sentinel: audio latent encoder not yet built (None means build attempted+failed).
_AUDIO_COND_MISSING = object()


def _gemma_generate(
    text_encoder: Any,
    messages: list[dict[str, object]],
    image: "torch.Tensor | None",
    seed: int,
    max_new_tokens: int = 512,
) -> str:
    """Generate an enhanced prompt with the Gemma text encoder.

    Mirrors the desktop app's ``LtxPromptEnhancerPipeline._generate`` exactly:
    applies the chat template, feeds (optionally image-conditioned) messages to
    the model, and decodes the completion. Enhancement is a quick, exploratory
    rewrite, so ``do_sample=True`` with a freshly-drawn seed keeps a redo from
    reproducing the same (potentially garbled) output.
    """
    encoder = cast(Any, text_encoder)
    assert encoder.processor is not None

    text = encoder.processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = encoder.processor(text=text, images=image, return_tensors="pt").to(
        encoder.model.device
    )

    fork_devices = [encoder.model.device] if encoder.model.device.type == "cuda" else []
    with torch.inference_mode(), torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        outputs = encoder.model.generate(
            **model_inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7
        )
        generated_ids = outputs[0][len(model_inputs.input_ids[0]):]
        return cast(str, encoder.processor.tokenizer.decode(generated_ids, skip_special_tokens=True))


class VideoCreatorInferenceEngine:
    """Wraps DistilledPipeline for standalone runner inference.

    Supports: t2v, i2v, extend, retake, image generation.
    IC-LoRA and A2V require additional model dependencies not yet packaged.
    """

    def __init__(
        self,
        checkpoint: str,
        gemma_root: str,
        upsampler_path: str,
        device: torch.device,
        profile: "GPUProfile | None" = None,
        enhance_device: "torch.device | None" = None,
    ) -> None:
        self._checkpoint = checkpoint
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._device = device
        self._dtype = torch.bfloat16
        # Prompt enhancement may run on a DIFFERENT GPU than the video pipeline,
        # so it never competes for the resident diffusion pipeline's VRAM.
        self._enhance_device = enhance_device if enhance_device is not None else device
        self._profile = profile
        self._pipeline: "DistilledPipeline | None" = None
        # LTX-2.5 additive pipeline (activated by model='ltx-2.5'). Kept as a SEPARATE
        # instance from self._pipeline so the LTX-2.3 default is never disturbed; it is
        # loaded lazily only when a 2.5 request arrives.
        self._pipeline25 = None
        # DistilledPipeline API generation of the installed ltx_pipelines: 'old'
        # (distilled_checkpoint_path / gemma_root constructor, LTX-2.3 monolith only) or
        # 'new' (ModelPaths constructor, LTX-2.5-capable). Resolved lazily on first load.
        self._pipe_api: str | None = None
        self._loaded_loras25: list[tuple[str, float]] | None = None
        self._zimage_pipe = None  # ZImagePipeline, lazily built (text-to-image)
        self._zimg2img_pipe = None  # ZImageImg2ImgPipeline, lazily built (whole-frame edit)
        self._zinpaint_pipe = None  # ZImageInpaintPipeline, lazily built (masked edit)
        self._loras: list[tuple[str, float]] = []
        # The LoRA set currently baked into the loaded pipeline (None until load).
        # Used to detect when a per-request LoRA change requires a reload.
        self._loaded_loras: list[tuple[str, float]] | None = None
        # Exactly one of video/_zimage_pipe may be resident at a time on
        # constrained cards; this lock serializes swaps so a load never
        # happens concurrently with an eviction/other load.
        self._model_lock = threading.RLock()
        # Lazily-built audio latent encoder for faithful extend reproduction (LTX-Desktop
        # regenerates audio during extension). Sentinel (_AUDIO_COND_MISSING) until first
        # use; None once a build attempt failed (extend then degrades to video-only).
        self._audio_cond: Any = _AUDIO_COND_MISSING
        # Checkpoint the AudioConditioner was built from (the 2.5 kit has its OWN audio-vae
        # path distinct from the 2.3 monolith), so the cache invalidates on a model switch.
        self._audio_cond_cp: str | None = None

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------

    def _api_generation(self, pipe_cls: type) -> str:
        """Return the DistilledPipeline API generation of the installed ltx_pipelines:
        ``'new'`` (ModelPaths-based, LTX-2.5-capable) or ``'old'``.

        The upstream ltx-pipelines revision that added LTX-2.5 support replaced the old
        ``distilled_checkpoint_path``/``gemma_root`` constructor with a single
        ``model_paths: ModelPaths`` and changed ``__call__`` to return a 4-tuple. ``'new'``
        is required to load the 2.5 split kit; ``'old'`` only knows the LTX-2.3 monolith.
        Detected by inspecting the constructor signature (no torch import needed) and cached
        per engine so it is resolved at most once.
        """
        if self._pipe_api is not None:
            return self._pipe_api
        import inspect
        try:
            params = inspect.signature(pipe_cls.__init__).parameters
            api = "new" if "model_paths" in params else "old"
        except (TypeError, ValueError):
            api = "old"
        self._pipe_api = api
        return api

    def _load_pipeline(self, loras: list[tuple[str, float]] | None = None) -> None:
        """Load (or reload) the DistilledPipeline."""
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.types import OffloadMode

        loras = loras or self._loras
        lora_entries = [
            LoraPathStrengthAndSDOps(path=path, strength=scale, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            for path, scale in loras
        ]

        # FP8 quantization on CUDA devices (works on Ada and Blackwell).
        quantization = None
        if self._profile is not None and not self._profile.use_fp8:
            logger.info("GPU profile disables FP8 quantization")
        elif self._device.type == "cuda":
            try:
                from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
                quantization = build_fp8_cast_policy(self._checkpoint)
            except Exception:
                logger.warning("FP8 quantization not available, running without it")

        # Offload strategy from the GPU profile: `streaming` -> OffloadMode.CPU
        # (weights stream from host RAM for 24 GB class cards), `full` ->
        # OffloadMode.NONE (fp8 transformer resident for 32 GB+ cards).
        if self._profile is not None and self._profile.offload_mode == "CPU":
            offload_mode = OffloadMode.CPU
        else:
            offload_mode = OffloadMode.NONE

        logger.info(
            "Loading DistilledPipeline from %s (device=%s, gemma=%s, upsampler=%r, "
            "offload=%s, fp8=%s)",
            self._checkpoint, self._device, self._gemma_root, self._upsampler_path,
            self._profile.offload_mode if self._profile else "NONE",
            quantization is not None,
        )

        # LTX-2.3 uses the fat ("monolith") checkpoint. The new ModelPaths API expresses it
        # via ModelPaths.from_monolith; the old API via distilled_checkpoint_path/gemma_root.
        if self._api_generation(DistilledPipeline) == "new":
            from ltx_pipelines.utils.model_paths import ModelPaths
            self._pipeline = DistilledPipeline(
                model_paths=ModelPaths.from_monolith(
                    self._checkpoint, gemma_root=self._gemma_root,
                ),
                spatial_upsampler_path=self._upsampler_path,
                loras=lora_entries,
                device=self._device,
                quantization=quantization,
                offload_mode=offload_mode,
            )
        else:
            self._pipeline = DistilledPipeline(
                distilled_checkpoint_path=self._checkpoint,
                gemma_root=self._gemma_root,
                spatial_upsampler_path=self._upsampler_path,
                loras=lora_entries,
                device=self._device,
                quantization=quantization,
                offload_mode=offload_mode,
            )
        logger.info("DistilledPipeline loaded (api=%s, offload=%s, fp8=%s)",
                    self._pipe_api, offload_mode, quantization is not None)
        _memlog("pipeline23 constructed")

    def _load_pipeline25(self, loras: list[tuple[str, float]] | None = None) -> None:
        """Load the LTX-2.5 DistilledPipeline into ``self._pipeline25`` (model='ltx-2.5').

        Builds a SEPARATE pipeline instance from the ComfyUI-style 2.5 split kit downloaded by
        ``runner/ltx/download_ltx25.sh`` (diffusion_models + gemma4-proj text encoder +
        video/audio VAEs + duration-head patch + the 2.5 latent spatial upscaler), using the
        new ltx-pipelines ``ModelPaths.from_split`` interface. The LTX-2.3 default pipeline
        (``self._pipeline``) is left untouched.

        REQUIRES the upstream ltx-pipelines revision that ships the ModelPaths API
        (Lightricks/LTX-2 >= fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca). The currently pinned
        rev 9377758131b1ffde4b7f766804590a6617bf2ab9 does NOT expose ModelPaths and cannot
        load 2.5 at all — this raises a clear, actionable error there instead of failing deep
        inside a forward pass.
        """
        from runner.ltx import config as _cfg
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode

        if self._api_generation(DistilledPipeline) != "new":
            raise RuntimeError(
                "LTX-2.5 requires the ModelPaths-based ltx-pipelines API. Apply the pin bump: "
                "Lightricks/LTX-2 -> fd4ded7f2d88d3da713abcdd4ad41ecc4a9314ca for BOTH "
                "packages/ltx-core and packages/ltx-pipelines, then rebuild the ltx-worker "
                "image (the API change is not backward compatible with rev 9377758131...)."
            )

        loras = loras or self._loras
        lora_entries = [
            LoraPathStrengthAndSDOps(path=path, strength=scale, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            for path, scale in loras
        ]

        if self._profile is not None and self._profile.offload_mode == "CPU":
            offload_mode = OffloadMode.CPU
        else:
            offload_mode = OffloadMode.NONE

        # ComfyUI-style 2.5 kit layout under LTX25_MODEL_DIR (single-sourced from config).
        model_dir = _cfg.LTX25_MODEL_DIR
        transformer = os.path.join(model_dir, "diffusion_models", _cfg.ltx25_transformer_filename())
        text_encoder = os.path.join(model_dir, "text_encoders", _cfg.LTX25_TEXT_ENCODER)
        video_vae = os.path.join(model_dir, "vae", _cfg.LTX25_VIDEO_VAE)
        audio_vae = os.path.join(model_dir, "vae", _cfg.LTX25_AUDIO_VAE)
        duration_head = os.path.join(model_dir, "model_patches", _cfg.LTX25_DURATION_HEAD)
        upscaler25 = _cfg.ltx25_spatial_upscaler_path()

        missing = [
            p for p in (transformer, text_encoder, video_vae, audio_vae, duration_head, upscaler25)
            if not os.path.exists(p)
        ]
        if missing:
            raise RuntimeError(
                "LTX-2.5 kit is incomplete; missing: " + ", ".join(missing)
                + ". Run runner/ltx/download_ltx25.sh on the GPU box (HUGGING_FACE_HUB_TOKEN "
                "required) and confirm the latent spatial upscaler is downloaded, then restart."
            )

        logger.info(
            "Loading LTX-2.5 DistilledPipeline (api=new, transformer=%s, upscaler=%r, offload=%s)",
            transformer, upscaler25, offload_mode,
        )
        # The 2.5 transformer variant/load policy is driven by LTX25_VARIANT via
        # ltx25_transformer_filename()/ltx25_fp8cast(). For the BF16 variant we apply
        # the CUDA fp8-cast policy (bf16 -> fp8 on load) so the 22B weights fit a 32 GB
        # card without the ltx-kernels nvfp4 extension. NVFP4/comfy-int8-convrot files
        # are NOT loadable by this loader (no matching quantization policy), so they are
        # only usable with explicit downstream scripts; we default to none for them.
        quantization = None
        if _cfg.ltx25_fp8cast():
            try:
                from ltx_core.quantization.fp8_cast import build_policy as build_fp8_cast_policy
                quantization = build_fp8_cast_policy(transformer)
                logger.info("LTX-2.5: applying CUDA fp8-cast quantization policy for %s",
                            _cfg.ltx25_transformer_filename())
            except Exception:
                logger.warning("LTX-2.5: fp8-cast policy unavailable, loading BF16 without it")
        self._pipeline25 = DistilledPipeline(
            model_paths=ModelPaths.from_split(
                transformer_path=transformer,
                text_encoder_path=text_encoder,
                video_vae_path=video_vae,
                audio_vae_path=audio_vae,
                duration_head_path=duration_head,
            ),
            spatial_upsampler_path=upscaler25,
            loras=lora_entries,
            device=self._device,
            quantization=quantization,
            offload_mode=offload_mode,
        )
        logger.info("LTX-2.5 DistilledPipeline loaded (fp8cast=%s)",
                    quantization is not None)
        _memlog("pipeline25 constructed")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _free_vram(self) -> None:
        """Release cached GPU memory so a fresh model load has room."""
        if self._device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def _evict_zimage(self) -> None:
        """Drop the image/edit pipelines from GPU. Called before the video model loads."""
        if self._zimage_pipe is not None or self._zimg2img_pipe is not None or self._zinpaint_pipe is not None:
            logger.info("Evicting Z-Image pipelines from GPU before loading video model")
            self._zimage_pipe = None
            self._zimg2img_pipe = None
            self._zinpaint_pipe = None
            self._free_vram()

    def _evict_video(self) -> None:
        """Drop the video pipelines (2.3 + 2.5) from GPU before an image model loads."""
        if self._pipeline is not None or self._pipeline25 is not None:
            logger.info("Evicting video pipelines (2.3 + 2.5) from GPU before loading image model")
            self._pipeline = None
            self._pipeline25 = None
            self._free_vram()

    def can_enhance_locally(self) -> bool:
        """Return True only if the local prompt-enhance Gemma can actually load.

        The enhance model is usable only when ``ltx_core.text_encoders.gemma``
        exposes the loader ops this pin's ``enhance_prompt`` imports
        (``GEMMA_LLM_KEY_OPS``). At some ltx-core pins those names were renamed;
        importing them here fails, which means ``enhance_prompt`` would fail at
        startup too. The server uses this to skip the startup warmup entirely
        (no pointless eviction, no stray CUDA context) when the module can't
        import, while still reporting the reason.

        Does a real import so the check reflects exactly what ``enhance_prompt``
        will hit, but never allocates CUDA.
        """
        try:
            from ltx_core.text_encoders.gemma import (  # noqa: F401
                GEMMA_LLM_KEY_OPS,
                GEMMA_MODEL_OPS,
                GemmaTextEncoderConfigurator,
                module_ops_from_gemma_root,
            )
            return True
        except ImportError as exc:
            logger.warning(
                "Local prompt-enhance Gemma unavailable (module import failed: %s) "
                "— skipping enhance startup warmup",
                exc,
            )
            return False

    def free(self) -> None:
        """Release ALL resident pipelines + cached GPU memory.

        Called by the worker's /evict endpoint so another worker can take the
        GPU. The video and image pipelines reload lazily on the next generate
        call (via _ensure_pipeline / _ensure_zimage), so no warm-up is needed
        here beyond the engine's existing lazy-load behavior.
        """
        with self._model_lock:
            if (
                self._pipeline is not None
                or self._pipeline25 is not None
                or self._zimage_pipe is not None
                or self._zimg2img_pipe is not None
                or self._zinpaint_pipe is not None
            ):
                logger.info("Freeing full engine (video + image pipelines) from GPU")
                self._pipeline = None
                self._pipeline25 = None
                self._zimage_pipe = None
                self._zimg2img_pipe = None
                self._zinpaint_pipe = None
                self._free_vram()

    @property
    def device_index(self) -> int:
        """CUDA index of the GPU this engine currently targets."""
        try:
            return int(self._device.index) if self._device.type == "cuda" else -1
        except Exception:
            return -1

    def set_device(self, gpu_idx: int) -> None:
        """Relocate the whole engine onto another CUDA GPU.

        Frees every resident pipeline and retargets ``self._device``; the model
        reloads on the new GPU on the next generation call (lazily, via
        ``_ensure_pipeline`` / ``_ensure_zimage``). Used when the scheduler hands
        this worker a different card (warm-resident all-GPUs placement). The GPU
        profile / offload mode is assumed identical across the box's cards (all
        5090s), so it is reused as-is — only the CUDA index changes.
        """
        with self._model_lock:
            old = self._device
            self.free()
            self._device = torch.device(f"cuda:{int(gpu_idx)}")
            # If prompt-enhance wasn't pinned to a separate GPU, follow the move.
            if self._enhance_device is not None and self._enhance_device == old:
                self._enhance_device = self._device
            # Audio-conditioner / lora-version caches are device-bound; reset so
            # the next use rebuilds for the new card.
            self._audio_cond = _AUDIO_COND_MISSING
            self._loaded_loras = None
            self._loaded_loras25 = None
            logger.info("Engine relocated -> %s", self._device)

    def set_loras(self, loras: list[tuple[str, float]] | None) -> None:
        """Set the desired LoRA set for the next generation.

        _ensure_pipeline triggers a pipeline reload when this differs from the
        set currently baked into the loaded DistilledPipeline."""
        with self._model_lock:
            self._loras = list(loras) if loras else []

    def _ensure_pipeline(self, model: str = "") -> Any:
        """Load and return the video pipeline for a request, first evicting the image
        pipeline so only one model occupies VRAM (bf16 Z-Image + video DiT cannot coexist
        on 32 GB).

        ``model=='ltx-2.5'`` selects the additive LTX-2.5 pipeline (``self._pipeline25``,
        loaded lazily on first 2.5 request); anything else uses the LTX-2.3 default
        (``self._pipeline``). Each reloads when the requested LoRA set changed: LoRAs are
        baked into the pipeline at construction, so a different set needs a reload (the
        server's generation lock serializes this)."""
        with self._model_lock:
            if model == "ltx-2.5":
                if self._pipeline25 is None or self._loras != self._loaded_loras25:
                    if self._pipeline25 is not None:
                        logger.info("Requested LoRA set changed — reloading LTX-2.5 pipeline")
                    self._evict_zimage()
                    self._load_pipeline25()
                    self._loaded_loras25 = list(self._loras)
                return self._pipeline25
            if self._pipeline is None or self._loras != self._loaded_loras:
                if self._pipeline is not None:
                    logger.info("Requested LoRA set changed — reloading pipeline")
                self._evict_zimage()
                self._load_pipeline()
                self._loaded_loras = list(self._loras)
            return self._pipeline

    def _pad_latent_frames(self, latent: torch.Tensor, pad_frames: int, at: str) -> torch.Tensor:
        """Zero-pad a latent on its temporal axis (dim 2): front for ``start``, back for
        ``end``. Mirrors LTX-Desktop's LTXRetakePipeline._pad_latent_frames."""
        if pad_frames <= 0:
            return latent
        pad_shape = list(latent.shape)
        pad_shape[2] = pad_frames
        pad = torch.zeros(pad_shape, device=latent.device, dtype=latent.dtype)
        return torch.cat([pad, latent] if at == "start" else [latent, pad], dim=2)

    def _extend_audio_conditioner(self, pipe=None) -> Any:
        """Lazily build the audio latent encoder (AudioConditioner) needed to reproduce
        LTX-Desktop's extend (encode source audio -> pad -> regenerate). Cached per
        checkpoint; None on failure so extend degrades to video-only rather than crashing.

        The 2.5 kit ships its OWN audio VAE, distinct from the 2.3 monolith's bundled
        audio path -- so which checkpoint we build from is pipeline-dependent and the
        cache invalidates when the model switches (2.3 <-> 2.5). If the 2.5 audio-vae
        path isn't consumable by AudioConditioner at this rev, the except below degrades
        to video-only extend, which is the safe fallback."""

        cp = self._checkpoint
        if pipe is self._pipeline25:
            from runner.ltx import config as _cfg
            cp = os.path.join(_cfg.LTX25_MODEL_DIR, "vae", _cfg.LTX25_AUDIO_VAE)
        if self._audio_cond_cp != cp:
            self._audio_cond = _AUDIO_COND_MISSING
            self._audio_cond_cp = cp
        if self._audio_cond is not _AUDIO_COND_MISSING:
            return self._audio_cond
        try:
            from ltx_pipelines.utils.blocks import AudioConditioner
            self._audio_cond = AudioConditioner(
                cp,
                dtype=torch.bfloat16,
                device=self._device,
            )
            logger.info("Built AudioConditioner for extend audio regeneration (%s)", cp)
        except Exception:
            logger.warning("AudioConditioner unavailable; extend will regenerate video only",
                           exc_info=True)
            self._audio_cond = None
        return self._audio_cond

    # ------------------------------------------------------------------
    # GPU-profile helpers
    # ------------------------------------------------------------------

    @property
    def profile(self):
        return self._profile

    def max_resolution(self) -> str:
        if self._profile is not None:
            return self._profile.max_resolution
        return "1080p"

    def clamp_resolution(self, resolution: str) -> str:
        """Clamp a requested resolution down to what the GPU can handle."""
        if self._profile is None:
            return resolution
        from runner.ltx.gpu_profile import clamp_resolution as _clamp
        return _clamp(self._profile, resolution)

    @staticmethod
    def default_tiling_config():
        # New rev of ltx-pipelines turned TilingConfig into a Union
        # (TileSizeConfig | TileCountConfig) and moved .default() onto TileSizeConfig;
        # the old rev had TilingConfig.default(). Detect by which object exposes .default().
        try:
            from ltx_core.model.video_vae import TileSizeConfig
            if hasattr(TileSizeConfig, "default"):
                return TileSizeConfig.default()
        except Exception:
            pass
        from ltx_core.model.video_vae import TilingConfig
        return TilingConfig.default()

    def tiling_config_for(self, pipe) -> Any:
        """Pick a decode tiling config for the given pipeline.

        LTX-2.3 uses the classic conv video VAE and works with ``TileSizeConfig.default()``
        (temporal overlap 24). LTX-2.5 uses a diffusion video VAE whose decoder requires a
        larger minimum temporal overlap (40 frames); the ltx-pipelines pipeline
        auto-recommends DiffVAE-appropriate tiling when handed the ``AUTO_TILING`` sentinel,
        so we pass that for the 2.5 pipeline and let it resolve (the pipeline returns the
        resolved config, which chunking uses). Returns ``AUTO_TILING`` for the 2.5 pipeline,
        ``TileSizeConfig.default()`` otherwise.
        """
        if pipe is self._pipeline25:
            try:
                from ltx_core.model.video_vae import AUTO_TILING
                return AUTO_TILING
            except Exception:
                pass
        return self.default_tiling_config()

    @staticmethod
    def video_chunks_number(num_frames: int, tiling_config) -> int:
        """Number of video chunks for a given frame count + tiling config."""
        from ltx_core.model.video_vae import get_video_chunks_number
        return int(get_video_chunks_number(num_frames, tiling_config))

    def _resolve_decode_tiling(self, pipe, tiling, shape) -> Any:
        """Return a CONCRETE TileSizeConfig for a decode call.

        The top-level ``pipe(...)`` entry auto-resolves the ``AUTO_TILING`` sentinel
        into a DiffVAE-appropriate layout internally, but our extend path drives
        ``pipe.video_decoder(...)`` directly, which needs a real ``TileSizeConfig``
        (it calls ``to_splitters()`` on it). When ``tiling`` is the AUTO_TILING
        sentinel (LTX-2.5 diffusion video VAE), ask the decoder for its recommended
        config for this exact shape; otherwise pass through the given concrete config.
        """
        if getattr(tiling, "__class__", None) and tiling.__class__.__name__ == "AutoTiling":
            try:
                return pipe.video_decoder.recommended_tiling_config(
                    height=int(shape.height),
                    width=int(shape.width),
                    num_frames=int(shape.frames),
                )
            except Exception as exc:
                logger.warning(
                    "DiffVAE recommended tiling unavailable (%s); falling back to TileSizeConfig.default()", exc
                )
                return self.default_tiling_config()
        return tiling

    def snap_frames_to_grid(self, num_frames: int, pipe=None) -> int:
        """Round ``num_frames`` down to the VAE temporal grid (k*time+1).

        The ModelPaths-era ltx-pipelines enforces ``(frames-1) % scale.time == 0`` for
        generation (LTX-2.3: time=8; LTX-2.5: time=2). Raw ``duration * fps`` values like
        120 are off-grid and are rejected deep in VAE validation, so we snap here based on
        the loaded pipeline's decoder checkpoint.
        """
        pipe = pipe or self._pipeline
        try:
            from ltx_pipelines.utils.helpers import snap_frames_to_grid, tiling_scale_factors_for_vae
            cp = pipe.video_decoder.checkpoint_path
            scale = tiling_scale_factors_for_vae(cp)
            snapped = int(snap_frames_to_grid(num_frames, scale))
            if snapped != num_frames:
                logger.info("Snapping video frames %d -> %d (VAE grid %s)", num_frames, snapped, scale)
            return snapped
        except Exception:
            return num_frames

    @staticmethod
    def encode_video_output(
        video: torch.Tensor,
        fps: int,
        output_path: str,
        video_chunks_number_value: int,
    ) -> None:
        from ltx_pipelines.utils.media_io import encode_video
        encode_video(
            video=video,
            fps=fps,
            audio=None,
            output_path=output_path,
            video_chunks_number=video_chunks_number_value,
        )

    @staticmethod
    def read_video_frames(video_path: str):
        """Read a video file into a torch tensor [T, H, W, C]."""
        import av
        frames = []
        container = av.open(video_path)
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb = frame.to_rgb().to_ndarray(format="rgb24").astype("float32") / 255.0
            frames.append(rgb)
        container.close()
        return torch.tensor(frames).permute(0, 3, 1, 2)  # [T, C, H, W]

    # ------------------------------------------------------------------
    # T2V / I2V generation
    # ------------------------------------------------------------------

    def _call_pipeline(
        self,
        pipe,
        *,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images,
        tiling_config,
    ) -> tuple[Any, Any, int, Any]:
        """Invoke ``DistilledPipeline.__call__`` transparently across the old vs new API.

        The new (LTX-2.5-era) ``__call__`` returns a 4-tuple ``(video, audio, num_frames,
        tiling_config)`` and requires ``vae_dtype``; the old API returns a 2-tuple and takes
        neither. Returns ``(video, audio, num_frames, tiling_config)`` uniformly — for the old
        API the requested values are echoed back so chunking/encoding logic is identical.
        ``api`` is resolved lazily (pipeline loads have already pinned ``self._pipe_api``).
        """
        api = self._pipe_api or self._api_generation(pipe.__class__)
        kwargs = dict(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
        )
        if api == "new":
            video, audio, out_frames, out_tiling = pipe(
                **kwargs, vae_dtype=self._dtype, enhance_prompt=False,
            )
        else:
            video, audio = pipe(**kwargs)
            out_frames, out_tiling = num_frames, tiling_config
        return video, audio, out_frames, out_tiling

    @torch.inference_mode()
    def generate_t2v(
        self,
        prompt: str,
        seed: int,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        output_path: str,
        loras: list[tuple[str, float]] | None = None,
        model: str = "",
    ) -> None:
        """Text-to-video generation (optionally applying the given catalog LoRAs).

        ``model=='ltx-2.5'`` routes to the additive LTX-2.5 pipeline; anything else uses the
        default LTX-2.3 pipeline.
        """
        if loras is not None:
            self.set_loras(loras)
        pipe = self._ensure_pipeline(model=model)
        num_frames = self.snap_frames_to_grid(num_frames, pipe)
        tiling_config = self.tiling_config_for(pipe)

        video, audio, out_frames, out_tiling = self._call_pipeline(
            pipe,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(fps),
            images=[],
            tiling_config=tiling_config,
        )
        chunks = self.video_chunks_number(out_frames, out_tiling)

        self.encode_video_output(
            video=video,
            fps=fps,
            output_path=output_path,
            video_chunks_number_value=chunks,
        )

    @torch.inference_mode()
    def generate_i2v(
        self,
        prompt: str,
        image_base64: str,
        seed: int,
        width: int,
        height: int,
        num_frames: int,
        fps: int,
        output_path: str,
        loras: list[tuple[str, float]] | None = None,
        model: str = "",
    ) -> None:
        """Image-to-video generation (optionally applying the given catalog LoRAs).

        ``model=='ltx-2.5'`` routes to the additive LTX-2.5 pipeline; anything else uses the
        default LTX-2.3 pipeline.
        """
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        if loras is not None:
            self.set_loras(loras)
        pipe = self._ensure_pipeline(model=model)
        num_frames = self.snap_frames_to_grid(num_frames, pipe)

        # Decode base64, reflect-pad to the exact generation dims (no centre-crop).
        img_bytes = base64.b64decode(image_base64)
        with Image.open(io.BytesIO(img_bytes)) as _img:
            _img = _img.convert("RGB")
            start_img = _reflect_pad_to_target(_img, width, height)
        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            start_img.save(tmp_img.name, format="PNG")
            tmp_img.close()

            tiling_config = self.tiling_config_for(pipe)

            video, audio, out_frames, out_tiling = self._call_pipeline(
                pipe,
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=float(fps),
                images=[_LtxImageInput(tmp_img.name, frame_idx=0, strength=1.0)],
                tiling_config=tiling_config,
            )
            chunks = self.video_chunks_number(out_frames, out_tiling)

            self.encode_video_output(
                video=video,
                fps=fps,
                output_path=output_path,
                video_chunks_number_value=chunks,
            )
        finally:
            if os.path.exists(tmp_img.name):
                os.unlink(tmp_img.name)

    # ------------------------------------------------------------------
    # Extend: append or prepend frames to existing video
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def upscale_video(
        self,
        video_path: str,
        output_path: str,
        target_width: int,
        target_height: int,
        fps: int,
    ) -> None:
        """Upscale a finished video to (target_height, target_width) with the
        LTX-2.3 spatial upsampler, then decode and resize.

        Flow: read frames -> VideoEncoder -> latent -> VideoUpsampler (2x) ->
        VideoDecoder -> bilinear resize -> mp4. This is the 480->720p step in the
        restyle chain (ID-V2V generates at a box-fitting resolution; this restores
        full output resolution). Uses the already-loaded DistilledPipeline's
        upsampler/decoder; the encoder is built from the same checkpoint so the
        per-channel latent stats line up.
        """
        from ltx_pipelines.utils.allocator_trim_strategy import AllocatorTrimStrategy
        from ltx_pipelines.utils.blocks import gpu_model

        self._ensure_pipeline()
        pipe = self._pipeline
        trim = AllocatorTrimStrategy.TRIM

        if getattr(pipe, "upsampler", None) is None:
            raise RuntimeError(
                "LTX spatial upscaler not loaded. Point UPSCALER_PATH at "
                "ltx-2.3-spatial-upscaler-x2-1.1.safetensors (downloaded under "
                "/models/upscaler or /models/upsampler) and restart the ltx worker."
            )

        # 1) Read + normalize frames to [-1,1]; pad temporal dim to 1+8k.
        v = self.read_video_frames(video_path)              # [T, C, H, W] float 0..1
        t = int(v.shape[0])
        pad_n = (8 * ((t - 1) // 8) + 1) - t
        if pad_n > 0:
            v = torch.cat([v, v[-1:].repeat(pad_n, 1, 1, 1)], dim=0)
        x = (v * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W] in [-1,1]
        x = x.to(device=self._device, dtype=self._dtype)
        logger.info("Upscale input: %s", tuple(x.shape))

        # 2) Encode frames -> latent (reuse the upsampler's encoder builder so the
        #    normalization stats match those the upsampler expects).
        encoder_builder = pipe.upsampler._encoder_builder
        with gpu_model(
            encoder_builder.build(device=self._device, dtype=self._dtype).eval(),
            alloc_trim_strategy=trim,
        ) as encoder:
            latent = encoder(x)                              # [1,128,F',H',W'] normalized
        logger.info("Upscale latent: %s", tuple(latent.shape))

        # 3) Spatial 2x upsample in latent space.
        up_latent = pipe.upsampler(latent)                   # [1,128,F',2H',2W']
        logger.info("Upscale up_latent: %s", tuple(up_latent.shape))

        # 4) Decode upscaled latent -> channel-last frame chunks, resize each to
        #    the exact target, and stream straight to the mp4 encoder.
        lat_frames = up_latent.shape[2]

        def _resized_chunks():
            for c in pipe.video_decoder(up_latent, tiling_config=None):
                c = c.float()
                if c.ndim == 5:
                    c = c[0]                                  # (1,F,H,W,C) -> (F,H,W,C)
                logger.info("Upscale: decode chunk %s (decoded frame count ~%d)",
                            tuple(c.shape), c.shape[0])
                c = c.permute(0, 3, 1, 2)                     # [F,C,H,W] in [0,1]
                if c.shape[2] != target_height or c.shape[3] != target_width:
                    c = torch.nn.functional.interpolate(
                        c, size=(target_height, target_width),
                        mode="bilinear", align_corners=False,
                    )
                c = c.permute(0, 2, 3, 1).clamp(0.0, 1.0)     # [F,H,W,C]
                yield c.to("cpu")

        self.encode_video_output(
            video=_resized_chunks(),
            fps=fps,
            output_path=output_path,
            video_chunks_number_value=max(1, (lat_frames + 7) // 8),
        )
        logger.info("Upscaled video saved to %s (%dx%d)",
                    output_path, target_width, target_height)

    @torch.no_grad()
    def generate_extend(
        self,
        prompt: str,
        video_base64: str,
        extend_frames: int,
        mode: str,  # "start" or "end"
        seed: int,
        fps: float,
        output_path: str,
        context_seconds: float = 1.0,
        model: str = "",
        progress_cb: Any | None = None,
    ) -> None:
        """Extend a video - windowed reproduction of LTX-Desktop's extend.

        ``model=='ltx-2.5'`` extends with the additive LTX-2.5 pipeline (its OWN
        diffusion video VAE + audio VAE); anything else uses the default LTX-2.3
        pipeline.

        The desktop algorithm (encode WHOLE source video+audio to a latent, zero-pad the
        time axis, TemporalRegionMask over only the new region + seam feather, prompt-gated
        diffusion, regenerated audio) is the correct one for following the prompt - but the
        whole-latent scope makes the latent (and VRAM) grow with total clip length, which
        OOMs a 32 GB card at 1080p.

        This windowed variant keeps the exact same latent mechanics but scopes the source to
        the last ``context_seconds`` (default 1 s) of the video - a real VIDEO latent, not a
        single frame - then diffuses only [window + new frames] and splices the freshly
        generated frames onto the UNCHANGED source via ffmpeg concat. A 1 s video window still
        carries temporal motion context while the text prompt remains the governing
        conditioning for the new frames. Memory per pass is bounded by ~(window+extend) frames,
        never the clip length.
        """
        self._ensure_pipeline()
        import shutil
        import subprocess as _sp
        from ltx_pipelines.utils.media_io import get_videostream_metadata as _gvm

        def _ffmpeg(*args: str) -> None:
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
            r = _sp.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {cmd}\n{r.stderr[-2000:]}")

        def _has_audio(path: str) -> bool:
            r = _sp.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
                 "-of", "csv=p=0", path],
                capture_output=True, text=True,
            )
            for line in r.stdout.splitlines():
                if line.strip() == "audio":
                    return True
            return False

        def _concat(a: str, b: str, out: str) -> None:
            """Concatenate two mp4s, handling sources that lack an audio track.

            The latent-extend segment always carries (regenerated) audio, but the source
            cut (prefix/rest) may have no audio at all -- a hardcoded ``[0:a:0]`` then fails.
            Probe both inputs and use an audio concat only when BOTH have audio.
            """
            with_audio = _has_audio(a) and _has_audio(b)
            if with_audio:
                spec = "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]"
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                       "-i", a, "-i", b, "-filter_complex", spec,
                       "-map", "[v]", "-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
            else:
                spec = "[0:v:0][1:v:0]concat=n=2:v=1:a=0[v]"
                cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                       "-i", a, "-i", b, "-filter_complex", spec, "-map", "[v]"]
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", out]
            r = _sp.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed: {cmd}\n{r.stderr[-2000:]}")

        workdir = tempfile.mkdtemp(prefix="vcext_")
        src = os.path.join(workdir, "src.mp4")
        prefix = os.path.join(workdir, "prefix.mp4")
        window = os.path.join(workdir, "window.mp4")
        segment = os.path.join(workdir, "segment.mp4")
        try:
            with open(src, "wb") as _f:
                _f.write(base64.b64decode(video_base64))

            meta = _gvm(src)
            total = max(1, meta.frames)
            ctx_frames = max(1, min(int(round(context_seconds * meta.fps)), total))
            fr = str(int(round(fps or meta.fps)))

            def _emit(stage: str, message: str) -> None:
                if progress_cb is not None:
                    try:
                        progress_cb(stage, message, None)
                    except Exception:
                        pass

            _emit("encoding", "Encoding source window...")
            if mode == "end":
                remain = total - ctx_frames
                if remain >= 1:
                    _ffmpeg("-i", src, "-frames:v", str(remain),
                            "-t", f"{remain / meta.fps:.6f}",
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", fr,
                            "-c:a", "aac", "-b:a", "128k", prefix)
                # Last `context_seconds` of REAL source as the conditioning window.
                _ffmpeg("-sseof", f"-{context_seconds}", "-i", src,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", fr,
                        "-c:a", "aac", "-b:a", "128k", window)
                # Faithful latent extend over the 1 s window -> [window + new frames].
                self._extend_file(window, prompt, extend_frames, mode, seed, fps, segment, model=model, progress_cb=progress_cb)
                if remain >= 1:
                    _concat(prefix, segment, output_path)
                else:
                    _sp.run(["cp", segment, output_path], check=True)
            else:  # "start": window = head; new frames prepended; concat(segment, remainder)
                _ffmpeg("-i", src, "-frames:v", str(ctx_frames),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", fr,
                        "-c:a", "aac", "-b:a", "128k", window)
                remain = total - ctx_frames
                rest = os.path.join(workdir, "rest.mp4")
                if remain >= 1:
                    _ffmpeg("-ss", f"{context_seconds}", "-i", src, "-frames:v", str(remain),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", fr,
                            "-c:a", "aac", "-b:a", "128k", rest)
                self._extend_file(window, prompt, extend_frames, mode, seed, fps, segment, model=model, progress_cb=progress_cb)
                if remain >= 1:
                    _concat(segment, rest, output_path)
                else:
                    _sp.run(["cp", segment, output_path], check=True)
            _emit("finalizing", "Finalizing output...")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    @torch.no_grad()
    def _extend_file(
        self,
        video_path: str,
        prompt: str,
        extend_frames: int,
        mode: str,
        seed: int,
        fps: float,
        output_path: str,
        model: str = "",
        progress_cb: Any | None = None,
    ) -> None:
        """Faithful LTX-Desktop extend on an already-extracted source FILE (a 1 s window).

        ``model=='ltx-2.5'`` extends with the LTX-2.5 pipeline (AUTO_TILING decode +
        its OWN audio VAE, because the 2.5 diffusion video VAE needs the larger 40-frame
        overlap and a different audio latent space); anything else uses LTX-2.3.

        Encodes the whole ``video_path`` (video + audio) to a latent, zero-pads the latent on
        the time axis by ``extend_frames`` (front for ``start``, back for ``end``), lays a
        ``TemporalRegionMask`` over ONLY the new region (source frozen, 0.5 s seam feather),
        and runs prompt-gated diffusion over the whole latent so the TEXT PROMPT drives the
        newly generated frames. Audio is regenerated too. This is the exact algorithm from
        LTX-Desktop's ``LTXRetakePipeline.extend`` (services/retake_pipeline/ltx_retake_pipeline.py),
        driven through the already-loaded ``DistilledPipeline`` components.
        """
        pipe = self._ensure_pipeline(model=model)
        dtype = torch.bfloat16
        device = self._device
        is25 = pipe is self._pipeline25

        def _emit(stage: str, message: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(stage, message, None)
                except Exception:
                    pass

        from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.video_vae import (
            DimensionSizeConfig,
            TileSizeConfig,
            get_video_chunks_number,
        )
        from ltx_core.types import AudioLatentShape, VideoLatentShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES as _distilled_sigmas
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import audio_latent_from_file, video_latent_from_file
        from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
        from ltx_pipelines.utils.types import ModalitySpec

        # Decode tiling is pipeline-dependent: the 2.5 DIFFUSION video VAE requires the
        # larger 40-frame temporal overlap (AUTO_TILING resolves it), while 2.3 keeps the
        # classic conv-VAE default (24). AUTO_TILING is only passed for the 2.5 pipeline.
        tiling = self.tiling_config_for(pipe)
        # Source-encoding tiling: 2.3 uses 24-frame/16-overlap tiles; the 2.5 diffusion VAE
        # encoder needs a larger temporal overlap, so bump overlap for 2.5. The tile MUST
        # stay strictly larger than the overlap (DimensionSizeConfig enforces
        # overlap < tile), so a 40-frame overlap needs a >40 tile.
        enc_tile, enc_overlap = (48, 40) if is25 else (24, 16)
        encoding_tiling = TileSizeConfig(
            frames=DimensionSizeConfig(tile_size=enc_tile, overlap=enc_overlap),
            height=DimensionSizeConfig(tile_size=256, overlap=64),
            width=DimensionSizeConfig(tile_size=256, overlap=64),
        )

        # --- Encode the whole window (video + audio) to latents (tiled) ---
        output_shape = get_videostream_metadata(video_path)
        initial_video_latent = pipe.image_conditioner(
            lambda enc: video_latent_from_file(
                video_encoder=enc,
                file_path=video_path,
                output_shape=output_shape,
                dtype=dtype,
                device=device,
                tiling_config=encoding_tiling,
            )
        )
        audio_cond = self._extend_audio_conditioner(pipe)
        initial_audio_latent = (
            audio_cond(
                lambda enc: audio_latent_from_file(
                    audio_encoder=enc,
                    file_path=video_path,
                    output_shape=output_shape,
                    dtype=dtype,
                    device=device,
                )
            )
            if audio_cond is not None
            else None
        )

        # --- Resolve target shape + the temporal region to regenerate ---
        target_shape = output_shape._replace(frames=output_shape.frames + extend_frames)
        # Keep the extended target on the loaded VAE's temporal grid (2.3: time=8, 2.5:
        # time=2) so the decode's grid validation passes; padding is derived below from
        # the snapped target vs source latent-frame counts.
        target_shape = target_shape._replace(frames=self.snap_frames_to_grid(target_shape.frames, pipe))
        pad_video_frames = (
            VideoLatentShape.from_pixel_shape(target_shape).frames
            - VideoLatentShape.from_pixel_shape(output_shape).frames
        )
        if initial_video_latent is not None:
            initial_video_latent = self._pad_latent_frames(initial_video_latent, pad_video_frames, mode)
        if initial_audio_latent is not None:
            pad_audio_frames = (
                AudioLatentShape.from_video_pixel_shape(target_shape).frames
                - AudioLatentShape.from_video_pixel_shape(output_shape).frames
            )
            initial_audio_latent = self._pad_latent_frames(initial_audio_latent, pad_audio_frames, mode)
        # Seam feather into the kept window so the frozen -> generated boundary blends.
        mask_delta_frames = round(0.5 * output_shape.fps)
        if mode == "start":
            region_start = 0.0
            region_end = min(target_shape.frames, extend_frames + mask_delta_frames) / output_shape.fps
        else:
            region_start = max(0, output_shape.frames - mask_delta_frames) / output_shape.fps
            region_end = target_shape.frames / output_shape.fps

        # --- Text conditioning: the prompt drives the newly generated region ---
        generator = torch.Generator(device=device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        (ctx_p,) = pipe.prompt_encoder([prompt], enhance_first_prompt=False)
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding

        video_modality_spec = ModalitySpec(
            context=v_context_p,
            conditionings=[TemporalRegionMask(start_time=region_start, end_time=region_end, fps=output_shape.fps)],
            initial_latent=initial_video_latent,
            frozen=False,
        )
        audio_modality_spec = None
        if a_context_p is not None:
            audio_modality_spec = ModalitySpec(
                context=a_context_p,
                conditionings=[TemporalRegionMask(start_time=region_start, end_time=region_end, fps=output_shape.fps)]
                if initial_audio_latent is not None else [],
                initial_latent=initial_audio_latent,
                frozen=initial_audio_latent is not None,
            )

        _emit("generating", f"Extending {extend_frames} new frames...")
        sigmas = torch.tensor(_distilled_sigmas).to(dtype=torch.float32, device=device)
        denoiser = SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
        video_state, audio_state = pipe.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=target_shape.width,
            height=target_shape.height,
            frames=target_shape.frames,
            fps=target_shape.fps,
            video=video_modality_spec,
            audio=audio_modality_spec,
        )

        _emit("decoding", "Decoding output...")
        decoded_audio = pipe.audio_decoder(audio_state.latent)
        # Resolve AUTO_TILING (2.5 diff-VAE sentinel) to a CONCRETE TileSizeConfig:
        # the direct .video_decoder() call can't auto-resolve it the way pipe() does.
        decode_tiling = self._resolve_decode_tiling(pipe, tiling, target_shape)
        decoded_video = pipe.video_decoder(video_state.latent, decode_tiling, generator)
        # Chunk-count hint only -- the DECODE above uses `tiling` (AUTO_TILING for the 2.5
        # diffusion VAE). get_video_chunks_number needs a CONCRETE config, so fall back to a
        # plain default if handed the AUTO_TILING sentinel (it still drives how encode_video
        # streams frames; accurate count is cosmetically the mp4 chunk size).
        try:
            video_chunks = get_video_chunks_number(target_shape.frames, tiling)
        except Exception:
            video_chunks = get_video_chunks_number(target_shape.frames, TileSizeConfig.default())
        encode_video(
            video=decoded_video,
            fps=int(fps),
            audio=decoded_audio,
            output_path=output_path,
            video_chunks_number=video_chunks,
        )
    # ------------------------------------------------------------------
    # Retake: regenerate a segment of a video with new prompt
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate_retake(
        self,
        prompt: str,
        video_base64: str,
        start_time: float,
        end_time: float,
        seed: int,
        fps: float,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        output_path: str | None = None,
    ) -> str:
        """Retake a segment of video with a new prompt.

        Strategy: extract frames from [start_time, end_time], use the first frame
        as conditioning, generate new segment, then splice back into source.
        """
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        self._ensure_pipeline()

        # Save source video
        src_bytes = base64.b64decode(video_base64)
        src_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        src_file.write(src_bytes)
        src_file.close()

        if output_path is None:
            output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        import av
        container = av.open(src_file.name)
        video_stream = container.streams.video[0]
        total_frames = int(video_stream.frames) or None

        # Calculate frame range for the segment
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        segment_frames = end_frame - start_frame

        # Extract the conditioning frame (first frame of segment)
        frames = []
        for i, frame in enumerate(container.decode(video_stream)):
            if i == start_frame:
                frames.append(frame.to_ndarray(format="rgb24"))
                break
        container.close()

        if not frames:
            raise ValueError(f"Frame {start_frame} not found in source video")

        from PIL import Image
        cond_img = Image.fromarray(frames[0])
        cond_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        cond_img.save(cond_path)

        h, w, _ = frames[0].shape
        w = round(w / 64) * 64
        h = round(h / 64) * 64

        segment_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name

        try:
            if regenerate_video:
                tiling_config = self.default_tiling_config()
                chunks = self.video_chunks_number(segment_frames, tiling_config)

                video, audio, _out_frames, _out_tiling = self._call_pipeline(
                    self._pipeline,
                    prompt=prompt,
                    seed=seed,
                    height=h,
                    width=w,
                    num_frames=segment_frames,
                    frame_rate=float(fps),
                    images=[_LtxImageInput(cond_path, frame_idx=0, strength=1.0)],
                    tiling_config=tiling_config,
                )

                self.encode_video_output(
                    video=video,
                    fps=int(fps),
                    output_path=segment_path,
                    video_chunks_number_value=chunks,
                )

                # Reassemble: before + segment + after
                parts = []
                if start_frame > 0:
                    before = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    self._trim_video(src_file.name, 0, start_frame / fps, before)
                    parts.append(before)

                parts.append(segment_path)

                if end_frame < (total_frames or end_frame):
                    after = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                    self._trim_video(src_file.name, start_time, end_time, after)
                    parts.append(after)

                if len(parts) > 1:
                    self._concat_videos(parts, output_path)
                else:
                    import shutil
                    shutil.copy2(parts[0], output_path)
            else:
                # Copy source as-is (regenerate_audio only, which we don't support remotely)
                import shutil
                shutil.copy2(src_file.name, output_path)
        finally:
            for p in [src_file.name, cond_path, segment_path]:
                if os.path.exists(p):
                    os.unlink(p)
            # Clean up trim temp files
            for root, dirs, files in os.walk(os.path.dirname(output_path) if os.path.dirname(output_path) else "."):
                pass  # cleanup handled by unique suffixes

        return output_path

    # ------------------------------------------------------------------
    # Image generation (text-to-image) — FP8 Z-Image-Turbo
    # ------------------------------------------------------------------

    def _build_zimage_pipeline(self):
        """Build a Z-Image-Turbo pipeline from the default diffusers repo.

        Small components (text_encoder, VAE, tokenizer, scheduler) and the
        transformer all come from the official `Tongyi-MAI/Z-Image-Turbo`
        diffusers folder on disk. The transformer is loaded with
        ``from_pretrained`` at its native bf16 precision by default (the default
        repo model). Set ``ZIMAGE_DTYPE=fp8`` to instead load the Comfy-Org
        single-file FP8 checkpoint via ``from_single_file``.

        Components:
          text_encoder  Qwen3Model (~7.5 GB bf16)
          tokenizer     Qwen2Tokenizer
          vae           AutoencoderKL (~160 MB)
          scheduler     FlowMatchEulerDiscreteScheduler
          transformer   ZImageTransformer2DModel (3 shards, ~24.6 GB bf16 default repo)
        """
        from diffusers import ZImagePipeline, ZImageImg2ImgPipeline, ZImageInpaintPipeline
        from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        from diffusers.models import AutoencoderKL
        from transformers import AutoTokenizer, AutoModel

        zdir = os.environ.get("ZIMAGE_MODEL_DIR", "/models/zimage")
        dtype_name = os.environ.get("ZIMAGE_DTYPE", "bf16").strip().lower()
        dtype = torch.bfloat16

        # Load small components from the official diffusers folder explicitly.
        text_encoder = AutoModel.from_pretrained(
            os.path.join(zdir, "text_encoder"), torch_dtype=dtype
        )
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(zdir, "tokenizer"))
        vae = AutoencoderKL.from_pretrained(
            os.path.join(zdir, "vae"), torch_dtype=dtype
        )
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(zdir, "scheduler")
        )

        # Load the transformer. Default = the default repo (bf16, from_pretrained,
        # uses the shards the config/index already reference). ZIMAGE_DTYPE=fp8
        # switches to the single-file FP8 Comfy-Org checkpoint.
        if dtype_name == "fp8":
            transformer_path = os.environ.get(
                "ZIMAGE_TRANSFORMER",
                os.path.join(zdir, "z_image_turbo_fp8_e4m3fn.safetensors"),
            )
            logger.info("Z-Image transformer: FP8 single-file %s", transformer_path)
            transformer = ZImageTransformer2DModel.from_single_file(
                transformer_path, torch_dtype=dtype
            ).to(self._device)
        else:
            logger.info("Z-Image transformer: default repo bf16 from_pretrained")
            transformer = ZImageTransformer2DModel.from_pretrained(
                os.path.join(zdir, "transformer"), torch_dtype=dtype
            ).to(self._device)

        components = dict(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
        )
        t2i = ZImagePipeline(**components)
        t2i.to(self._device)
        img2img = ZImageImg2ImgPipeline(**components)
        img2img.to(self._device)
        inpaint = ZImageInpaintPipeline(**components)
        inpaint.to(self._device)
        self._zimg2img_pipe = img2img
        self._zinpaint_pipe = inpaint
        return t2i

    @torch.inference_mode()
    def generate_image(
        self,
        prompt: str,
        width: int,
        height: int,
        num_steps: int = 9,
        seed: int = 42,
        guidance_scale: float | None = None,
    ) -> str:
        """Generate an image from text prompt (FP8 Z-Image-Turbo).

        Returns path to generated PNG file.
        """
        from PIL import Image

        tmp_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_out.close()

        try:
            if self._zimage_pipe is None:
                with self._model_lock:
                    # Evict the video model first so the bf16 image model has room.
                    self._evict_video()
                    logger.info("Loading Z-Image-Turbo pipeline ...")
                    self._zimage_pipe = self._build_zimage_pipeline()
                    logger.info("Z-Image-Turbo pipeline ready (%s)", os.environ.get("ZIMAGE_DTYPE", "bf16"))

            pipe_kwargs: dict = {
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_steps,
                "generator": torch.Generator(device=self._device).manual_seed(seed),
            }
            # Only pass guidance_scale when the client explicitly sent it; otherwise
            # let the diffusers ZImagePipeline default apply (pipeline is authoritative).
            if guidance_scale is not None:
                pipe_kwargs["guidance_scale"] = guidance_scale
            result = self._zimage_pipe(**pipe_kwargs)
            img = result.images[0]
            img.save(tmp_out.name)
            return tmp_out.name

        except Exception as exc:
            logger.warning("Image generation failed (%s), returning placeholder", exc)
            placeholder = Image.new("RGB", (width, height), color=(30, 30, 40))
            placeholder.save(tmp_out.name)
            return tmp_out.name

    @staticmethod
    def _invert_mask(mask_l: "Image.Image") -> "Image.Image":
        """Return the logical inverse of a binary/grayscale mask (region to change)."""
        import numpy as np
        arr = np.asarray(mask_l.convert("L"))
        arr = 255 - arr
        return Image.fromarray(arr, mode="L")

    @staticmethod
    def _fetch_keep_mask(sam3_url, prompt, image, token) -> "Image.Image":
        """Ask the idv2v worker's SAM3 to segment the foreground object to keep.

        Returns the object's binary mask at the input resolution. The caller
        inverts it so only everything-else is regenerated.
        """
        import base64 as _b64
        import json as _json
        import urllib.request
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        payload = _json.dumps({
            "image": _b64.b64encode(buf.getvalue()).decode("ascii"),
            "mode": "auto",
            "prompt": prompt or "person",
        }).encode()
        req = urllib.request.Request(
            str(sam3_url).rstrip("/") + "/video-creator/v1/sam3",
            data=payload,
            headers={"Content-Type": "application/json", "X-Worker-Token": token},
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = _json.loads(resp.read().decode())
        mask = Image.open(io.BytesIO(_b64.b64decode(body["mask_b64"]))).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.NEAREST)
        return mask

    @torch.inference_mode()
    def edit_image(
        self,
        prompt: str,
        image_path: str,
        mask_path: str | None = None,
        keep_subject: bool = False,
        sam3_url: str | None = None,
        sam3_prompt: str = "person",
        keep_mask_b64: str | None = None,
        worker_token: str = "",
        strength: float = 0.6,
        num_steps: int = 9,
        seed: int = 42,
        guidance_scale: float | None = None,
    ) -> str:
        """Z-Image img2img / masked-inpaint edit of an image.

        - ``mask_path`` set                -> ZImageInpaintPipeline (only that region changes)
        - ``keep_subject`` True            -> fetch the object-to-keep from the idv2v worker's
                                              SAM3, invert it, and inpaint everything EXCEPT it
        - otherwise                        -> ZImageImg2ImgPipeline (whole-frame strength edit)

        Returns the path to the edited PNG (source copy on failure).
        """
        from PIL import Image

        tmp_out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_out.close()

        try:
            image = Image.open(image_path).convert("RGB")
            mask: "Image.Image | None" = None
            if keep_subject:
                if keep_mask_b64:
                    # Reuse a mask the caller (backend) already computed for this frame.
                    import base64 as _b64
                    keep_mask = Image.open(io.BytesIO(_b64.b64decode(keep_mask_b64))).convert("L")
                    if keep_mask.size != image.size:
                        keep_mask = keep_mask.resize(image.size, Image.NEAREST)
                else:
                    if not sam3_url:
                        raise RuntimeError("keep_subject requires sam3_url (idv2v worker)")
                    keep_mask = self._fetch_keep_mask(sam3_url, sam3_prompt, image, worker_token)
                mask = self._invert_mask(keep_mask)  # change everything else
            elif mask_path:
                mask = Image.open(mask_path).convert("L")
                if mask.size != image.size:
                    mask = mask.resize(image.size, Image.NEAREST)

            if self._zimage_pipe is None:
                with self._model_lock:
                    self._evict_video()
                    logger.info("Loading Z-Image-Turbo edit pipeline ...")
                    self._zimage_pipe = self._build_zimage_pipeline()
                    logger.info("Z-Image edit pipeline ready (%s)", os.environ.get("ZIMAGE_DTYPE", "bf16"))

            generator = torch.Generator(device=self._device).manual_seed(seed)
            kwargs: dict = {
                "prompt": prompt,
                "strength": strength,
                "num_inference_steps": num_steps,
                "generator": generator,
            }
            if mask is not None:
                kwargs["image"] = image
                kwargs["mask_image"] = mask
                pipe = self._zinpaint_pipe
            else:
                kwargs["image"] = image
                pipe = self._zimg2img_pipe
            if guidance_scale is not None:
                kwargs["guidance_scale"] = guidance_scale

            result = pipe(**kwargs)
            img = result.images[0]
            img.save(tmp_out.name)
            return tmp_out.name
        except Exception as exc:
            logger.warning("Image edit failed (%s), returning source-copy placeholder", exc)
            try:
                Image.open(image_path).convert("RGB").save(tmp_out.name)
            except Exception:
                pass
            return tmp_out.name

    # ------------------------------------------------------------------
    # Prompt enhancement (Gemma)
    # ------------------------------------------------------------------

    @staticmethod
    def _decode_enhance_image(image_base64: str) -> "torch.Tensor":
        """Decode a base64 image the same way the desktop's i2v enhancer does.

        ``decode_image`` returns a numpy HxWxC uint8 array; it is resized to a
        max side of 896 (aspect-ratio preserving) and returned as a uint8 tensor
        for the Gemma processor.
        """
        import base64 as _b64

        from ltx_pipelines.utils.media_io import decode_image, resize_aspect_ratio_preserving

        img_bytes = _b64.b64decode(image_base64)
        # Pick a real extension by sniffing the bytes so decode_image can find
        # its format regardless of what produced the base64 stream.
        try:
            from PIL import Image as _PILImage

            fmt = (_PILImage.open(io.BytesIO(img_bytes)).format or "PNG").lower()
            ext = "jpg" if fmt in ("jpeg", "jpg") else "png"
        except Exception:
            ext = "png"

        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
        try:
            tmp.write(img_bytes)
            tmp.close()
            image = decode_image(image_path=tmp.name)
            return resize_aspect_ratio_preserving(torch.tensor(image), 896).to(torch.uint8)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    @torch.inference_mode()
    def enhance_prompt(
        self,
        prompt: str,
        image_base64: str | None = None,
        seed: int | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Enhance a prompt with the Gemma text encoder.

        Loads Gemma on demand for the call and frees it afterward, so the runner
        only pays the VRAM cost while an enhance request is actually in flight.
        The resident video/image pipelines are evicted first to make room and
        are reloaded lazily on the next generation (via ``_ensure_pipeline`` /
        ``generate_image``).

        When ``image_base64`` is provided the enhancement is image-conditioned
        (i2v), mirroring the desktop's ``enhance_i2v``; otherwise it is a plain
        text rewrite (t2v).
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        # The enhancement model is the provisioned Gemma QAT q4_0 text encoder
        # (Lightricks/gemma-3-12b-it-qat-q4_0 -> TEXT_ENCODER_ROOT, i.e.
        # /models/gemma). Fail with an actionable message if it isn't present so
        # an unprovisioned box surfaces a clear error instead of a cryptic
        # find_matching_file glob miss.
        if not os.path.isdir(self._gemma_root):
            raise RuntimeError(
                "Gemma text encoder not found at TEXT_ENCODER_ROOT (%s). "
                "Provision it with provision_models.py "
                "(Lightricks/gemma-3-12b-it-qat-q4_0) or point TEXT_ENCODER_ROOT "
                "at the gemma folder." % self._gemma_root
            )
        try:
            from ltx_core.utils import find_matching_file as _find
            _find(self._gemma_root, "model*.safetensors")
        except Exception as exc:
            raise RuntimeError(
                "No Gemma weights (model*.safetensors) under TEXT_ENCODER_ROOT "
                "(%s): %s" % (self._gemma_root, exc)
            )

        # Enhancement on a different GPU than the video pipeline has no VRAM
        # contention, so the resident diffusion pipeline stays loaded. Only
        # when they share a GPU do we evict the pipelines to make room.
        same_gpu = str(self._enhance_device) == str(self._device)

        with self._model_lock:
            if same_gpu:
                # Free the big generation pipelines so Gemma fits.
                self._evict_video()
                self._evict_zimage()
                self._free_vram()

            from ltx_core.loader.registry import DummyRegistry
            from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
            from ltx_core.text_encoders.gemma import (
                GEMMA_LLM_KEY_OPS,
                GEMMA_MODEL_OPS,
                GemmaTextEncoderConfigurator,
                module_ops_from_gemma_root,
            )
            from ltx_core.utils import find_matching_file
            from ltx_pipelines.utils.gpu_model import gpu_model

            module_ops = module_ops_from_gemma_root(self._gemma_root)
            model_folder = find_matching_file(self._gemma_root, "model*.safetensors").parent
            weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]

            def _build() -> Any:
                builder = SingleGPUModelBuilder(
                    model_path=tuple(weight_paths),
                    model_class_configurator=GemmaTextEncoderConfigurator,
                    model_sd_ops=GEMMA_LLM_KEY_OPS,
                    module_ops=(GEMMA_MODEL_OPS, *module_ops),
                    registry=DummyRegistry(),
                )
                return builder.build(device=self._enhance_device, dtype=torch.bfloat16).eval()

            logger.info(
                "Enhancing prompt via local Gemma (device=%s, len=%d, image=%s)",
                self._enhance_device, len(prompt), image_base64 is not None,
            )
            with gpu_model(_build()) as text_encoder:
                if image_base64 is not None:
                    resolved = system_prompt or text_encoder.default_gemma_i2v_system_prompt
                    image = self._decode_enhance_image(image_base64)
                    messages: list[dict[str, object]] = [
                        {"role": "system", "content": resolved},
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                            ],
                        },
                    ]
                    return _gemma_generate(text_encoder, messages, image=image, seed=seed)

                resolved = system_prompt or text_encoder.default_gemma_t2v_system_prompt
                messages = [
                    {"role": "system", "content": resolved},
                    {"role": "user", "content": f"user prompt: {prompt}"},
                ]
                return _gemma_generate(text_encoder, messages, image=None, seed=seed)

    # ------------------------------------------------------------------
    # Video I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _concat_videos(inputs: list[str], output: str) -> None:
        """Concatenate videos using ffmpeg. All inputs must have the same resolution/fps."""
        import subprocess
        # Build concat list
        concat_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
        try:
            for inp in inputs:
                concat_file.write(f"file '{inp}'\n")
            concat_file.close()

            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", concat_file.name,
                    "-c", "copy",
                    output,
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        finally:
            concat_file.close()
            if os.path.exists(concat_file.name):
                os.unlink(concat_file.name)

    @staticmethod
    def _trim_video(src: str, start_sec: float, end_sec: float, output: str) -> None:
        """Trim a video to [start_sec, end_sec] using ffmpeg."""
        import subprocess
        duration = end_sec - start_sec
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src,
                "-ss", str(start_sec),
                "-t", str(duration),
                "-c", "copy",
                output,
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        """Run a small generation to warm up GPU memory."""
        self._ensure_pipeline()
        tiling_config = self.default_tiling_config()
        chunks = self.video_chunks_number(9, tiling_config)

        try:
            video, audio, _out_frames, _out_tiling = self._call_pipeline(
                self._pipeline,
                prompt="warmup test",
                seed=42,
                height=256,
                width=384,
                num_frames=9,
                frame_rate=8.0,
                images=[],
                tiling_config=tiling_config,
            )
            self.encode_video_output(
                video=video,
                fps=8,
                output_path=output_path,
                video_chunks_number_value=chunks,
            )
            _memlog("warmup encode done")
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
