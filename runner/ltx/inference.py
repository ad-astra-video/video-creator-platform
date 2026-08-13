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

    # ------------------------------------------------------------------
    # Pipeline lifecycle
    # ------------------------------------------------------------------

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

        self._pipeline = DistilledPipeline(
            distilled_checkpoint_path=self._checkpoint,
            gemma_root=self._gemma_root,
            spatial_upsampler_path=self._upsampler_path,
            loras=lora_entries,
            device=self._device,
            quantization=quantization,
            offload_mode=offload_mode,
        )
        logger.info("DistilledPipeline loaded (offload=%s, fp8=%s)",
                    offload_mode, quantization is not None)

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
        """Drop the video pipeline from GPU. Called before the image model loads."""
        if self._pipeline is not None:
            logger.info("Evicting video pipeline from GPU before loading image model")
            self._pipeline = None
            self._free_vram()

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
                or self._zimage_pipe is not None
                or self._zimg2img_pipe is not None
                or self._zinpaint_pipe is not None
            ):
                logger.info("Freeing full engine (video + image pipelines) from GPU")
                self._pipeline = None
                self._zimage_pipe = None
                self._zimg2img_pipe = None
                self._zinpaint_pipe = None
                self._free_vram()

    def set_loras(self, loras: list[tuple[str, float]] | None) -> None:
        """Set the desired LoRA set for the next generation.

        _ensure_pipeline triggers a pipeline reload when this differs from the
        set currently baked into the loaded DistilledPipeline."""
        with self._model_lock:
            self._loras = list(loras) if loras else []

    def _ensure_pipeline(self) -> None:
        """Load the video pipeline, first evicting the image pipeline so only one
        model occupies VRAM (bf16 Z-Image + video DiT cannot coexist on 32 GB).

        Reloads when the requested LoRA set changed: LoRAs are baked into the
        DistilledPipeline at construction, so a different set needs a reload (the
        server's generation lock serializes this)."""
        with self._model_lock:
            if self._pipeline is None or self._loras != self._loaded_loras:
                if self._pipeline is not None:
                    logger.info("Requested LoRA set changed — reloading pipeline")
                self._evict_zimage()
                self._load_pipeline()
                self._loaded_loras = list(self._loras)

    def _pad_latent_frames(self, latent: torch.Tensor, pad_frames: int, at: str) -> torch.Tensor:
        """Zero-pad a latent on its temporal axis (dim 2): front for ``start``, back for
        ``end``. Mirrors LTX-Desktop's LTXRetakePipeline._pad_latent_frames."""
        if pad_frames <= 0:
            return latent
        pad_shape = list(latent.shape)
        pad_shape[2] = pad_frames
        pad = torch.zeros(pad_shape, device=latent.device, dtype=latent.dtype)
        return torch.cat([pad, latent] if at == "start" else [latent, pad], dim=2)

    def _extend_audio_conditioner(self) -> Any:
        """Lazily build the audio latent encoder (AudioConditioner) needed to reproduce
        LTX-Desktop's extend (encode source audio -> pad -> regenerate). Cached; None on
        failure so extend degrades to video-only rather than crashing."""
        if self._audio_cond is not _AUDIO_COND_MISSING:
            return self._audio_cond
        try:
            from ltx_pipelines.utils.blocks import AudioConditioner
            self._audio_cond = AudioConditioner(
                self._checkpoint,
                dtype=torch.bfloat16,
                device=self._device,
            )
            logger.info("Built AudioConditioner for extend audio regeneration")
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
        from ltx_core.model.video_vae import TilingConfig
        return TilingConfig.default()

    @staticmethod
    def video_chunks_number(num_frames: int, tiling_config) -> int:
        from ltx_core.model.video_vae import get_video_chunks_number
        return int(get_video_chunks_number(num_frames, tiling_config))

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
    ) -> None:
        """Text-to-video generation (optionally applying the given catalog LoRAs)."""
        if loras is not None:
            self.set_loras(loras)
        self._ensure_pipeline()
        tiling_config = self.default_tiling_config()
        chunks = self.video_chunks_number(num_frames, tiling_config)

        video, audio = self._pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=float(fps),
            images=[],
            tiling_config=tiling_config,
        )

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
    ) -> None:
        """Image-to-video generation (optionally applying the given catalog LoRAs)."""
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

        if loras is not None:
            self.set_loras(loras)
        self._ensure_pipeline()

        # Decode base64 to temporary PNG
        img_bytes = base64.b64decode(image_base64)
        tmp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            tmp_img.write(img_bytes)
            tmp_img.close()

            tiling_config = self.default_tiling_config()
            chunks = self.video_chunks_number(num_frames, tiling_config)

            video, audio = self._pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=float(fps),
                images=[_LtxImageInput(tmp_img.name, frame_idx=0, strength=1.0)],
                tiling_config=tiling_config,
            )

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
    ) -> None:
        """Extend a video by appending/prepending frames - faithful reproduction of
        LTX-Desktop's LTXRetakePipeline.extend (services/retake_pipeline/ltx_retake_pipeline.py).

        Strategy (mirrors the desktop): encode the WHOLE source video (and audio) to a
        latent, zero-pad the latent on the time axis by ``extend_frames`` (front/back per
        ``mode``), then regenerate ONLY the new region under a TemporalRegionMask - the
        source stays frozen while the newly drawn frames are governed by the TEXT PROMPT
        (the diffusion's actual conditioning), with a 0.5s seam feather into the source.
        Audio is regenerated too. This is what makes the extension FOLLOW the prompt -
        unlike the old single-edge-frame image-to-video continuation.
        """
        self._ensure_pipeline()
        pipe = self._pipeline
        dtype = torch.bfloat16
        device = self._device

        from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.model.video_vae import (
            SpatialTilingConfig,
            TemporalTilingConfig,
            TilingConfig,
            get_video_chunks_number,
        )
        from ltx_core.types import AudioLatentShape, VideoLatentShape
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES as _distilled_sigmas
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import audio_latent_from_file, video_latent_from_file
        from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
        from ltx_pipelines.utils.types import ModalitySpec

        # Save source video from base64.
        src_bytes = base64.b64decode(video_base64)
        src_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        try:
            src_file.write(src_bytes)
            src_file.close()

            tiling = TilingConfig.default()
            encoding_tiling = TilingConfig(
                spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
                temporal_config=TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=16),
            )

            # --- Encode the WHOLE source video + audio to latents (tiled) ---
            output_shape = get_videostream_metadata(src_file.name)
            initial_video_latent = pipe.image_conditioner(
                lambda enc: video_latent_from_file(
                    video_encoder=enc,
                    file_path=src_file.name,
                    output_shape=output_shape,
                    dtype=dtype,
                    device=device,
                    tiling_config=encoding_tiling,
                )
            )
            audio_cond = self._extend_audio_conditioner()
            initial_audio_latent = (
                audio_cond(
                    lambda enc: audio_latent_from_file(
                        audio_encoder=enc,
                        file_path=src_file.name,
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
            # Seam feather: regenerate MASK_DELTA_SECONDS of real source adjacent to the new
            # region so the frozen -> generated boundary blends (matches the desktop's 0.5s).
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

            decoded_audio = pipe.audio_decoder(audio_state.latent)
            decoded_video = pipe.video_decoder(video_state.latent, tiling, generator)
            video_chunks = get_video_chunks_number(target_shape.frames, tiling)
            encode_video(
                video=decoded_video,
                fps=int(fps),
                audio=decoded_audio,
                output_path=output_path,
                video_chunks_number=video_chunks,
            )
        finally:
            if os.path.exists(src_file.name):
                os.unlink(src_file.name)

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

                video, audio = self._pipeline(
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
            video, audio = self._pipeline(
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
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
