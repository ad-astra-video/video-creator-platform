"""Gemma 3 LLM support for the ID-V2V worker.

Adds two capabilities to the restyle worker using the already-provisioned
Lightricks Gemma 3 12B QAT text encoder
(Lightricks/gemma-3-12b-it-qat-q4_0-unquantized, on the shared /models/gemma
mount that the LTX runner drops it onto):

  * prompt enhancement    (rewrite/expand the user's restyle prompt)
  * automatic captioning  (describe a sampled source-video clip, then feed that
                           description into the restyle task as its text prompt)

It loads the FULL ``Gemma3ForConditionalGeneration`` (including the SigLIP
vision tower) via stock ``transformers`` (already present in this image), so
both text-only enhancement and image-conditioned captioning work off one model.

VRAM / placement:
  Gemma is loaded on a SEPARATE GPU (``GEMMA_GPU_DEVICE``, default cuda:1) from
  the resident id-v2v DiT/VACE (``GPU_DEVICE``, default cuda:0). The .151 box
  has GPUs 1/2 free, so Gemma never contends with the diffusion model and no
  eviction of the resident worker model is required. It is loaded lazily and
  then held warm (a 12B model costs seconds-to-tens-of-seconds to read off
  disk; reloading per request would add that to every job).

Threading:
  A single module-level instance is shared by all request threads; a lock
  serializes concurrent enhance/caption calls. Generation runs with
  ``torch.no_grad``/``inference_mode`` so it is safe inside a worker thread.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import TYPE_CHECKING

import torch

from . import config

logger = logging.getLogger("video_creator.runner.idv2v.gemma")

# System prompts. Kept deliberately close to the stock LTX t2v/i2v enhancer
# behaviour so enhanced output reads like the rest of the app.
ENHANCE_T2V_SYSTEM_PROMPT = (
    "You are an expert at writing prompts for video generation models. "
    "Rewrite and expand the user's prompt into a vivid, detailed video "
    "description that captures the scene, subject, style and motion. "
    "Keep it to a single natural-language paragraph. Do not add "
    "instructions, headings, or preamble."
)

# Dedicated prompt for the RESTYLE task's enhance (run.py _gemma_stage). Kept
# separate from ENHANCE_T2V_SYSTEM_PROMPT because the generic video-create
# Enhance path must stay untouched — only the restyle task needs the color-
# fidelity constraint (the restyled first frame sets the color palette, and it
# must not drift/warm/cool as the video progresses).
RESTYLE_ENHANCE_SYSTEM_PROMPT = (
    "You are an expert at writing prompts for video generation models. "
    "Rewrite and expand the user's prompt into a vivid, detailed video "
    "description that captures the scene, subject, style and motion. "
    "This video is a RESTYLE of a first frame: the opening frame has "
    "already been restyled into a specific look. State explicitly, in the "
    "prompt itself, that the entire video must keep the exact same color "
    "palette and tone as that restyled first frame, and that colors must "
    "stay constant for the whole clip (no gradual color shift, drift, or "
    "warming/cooling as the video progresses). Keep it to a single "
    "natural-language paragraph. Do not add instructions, headings, or "
    "preamble."
)

IMAGE_ENHANCE_SYSTEM_PROMPT = (
    "You are an expert at writing image-editing prompts. Given a reference "
    "image and the user's edit direction, rewrite and expand the direction into "
    "a vivid, detailed prompt describing the desired NEW version of the image (the "
    "edit result). Preserve the original composition: the framing, camera angle, "
    "subject scale, position and layout must stay exactly as in the reference "
    "image, and say so explicitly in the prompt (e.g. \"keep the same framing and "
    "camera angle\"). Change only the style, materials, lighting mood and other "
    "visual attributes the user asks for. Keep it to a single natural-language "
    "paragraph. Do not add instructions, headings, or preamble."
)

CAPTION_SYSTEM_PROMPT = (
    "You are a video analyst. Look at the provided frames from a single "
    "video clip and write a concise natural-language description of the "
    "scene and what is happening: the setting, the subject(s), their "
    "appearance, actions and any motion you can infer between frames, and "
    "the overall mood. Write it as one flowing paragraph suitable to use "
    "as the text prompt for a video-restyle generation task. Focus on what "
    "is visible; do not invent objects that are not in the frames."
)

# How many evenly-spaced frames to feed the vision model for captioning, and
# the max side they're downscaled to (the SigLIP tower expects ~896).
CAPTION_MAX_FRAMES = 4
CAPTION_MAX_SIDE = 896
MAX_NEW_TOKENS = 384


class GemmaEnhancer:
    """Lazy, resident Gemma 3 LLM for prompt enhance + video captioning."""

    def __init__(
        self,
        root: str | None = None,
        device: str | None = None,
        attn_impl: str | None = None,
        evict_cb=None,
    ) -> None:
        self._root = root or config.GEMMA_ROOT
        self._device = device or config.gemma_device()
        self._attn_impl = attn_impl or config.GEMMA_ATTN_IMPL
        self._evict_cb = evict_cb
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    # -- lifecycle --------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        """Load the Gemma 3 model + processor (lazily, once)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            # Gemma shares GPU_DEVICE with the resident id-v2v model (they cannot
            # coexist on one card). If the caller wired an evict hook, run it now
            # to free the video model's VRAM before we allocate Gemma.
            if self._evict_cb is not None:
                self._evict_cb()

            t0 = time.time()
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration

            if not os.path.isdir(self._root):
                raise RuntimeError(
                    "Gemma checkpoint not found at GEMMA_ROOT (%s). Provision "
                    "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized there." % self._root
                )
            logger.info(
                "Loading Gemma 3 from %s on %s (attn=%s) ...",
                self._root, self._device, self._attn_impl,
            )
            model = Gemma3ForConditionalGeneration.from_pretrained(
                self._root,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                attn_implementation=self._attn_impl,
            )
            model = model.to(self._device).eval()
            processor = AutoProcessor.from_pretrained(self._root)
            self._processor = processor
            self._model = model
            logger.info(
                "Gemma 3 loaded in %.1fs (device=%s)", time.time() - t0, self._device,
            )

    def unload(self) -> None:
        """Drop the model and free its GPU memory."""
        with self._lock:
            self._model = None
            self._processor = None
            torch.cuda.empty_cache()

    # -- generation -------------------------------------------------------

    def _generate(
        self,
        messages: list[dict],
        images,
        seed: int | None,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> str:
        self.ensure_loaded()
        proc = self._processor
        model = self._model
        text = proc.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = proc(text=text, images=images, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            if seed is not None:
                torch.manual_seed(seed)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                pad_token_id=proc.tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs.input_ids.shape[1]:]
        return proc.tokenizer.decode(generated, skip_special_tokens=True).strip()

    # -- public ops -------------------------------------------------------

    def enhance_text(self, prompt: str, seed: int | None = None) -> str:
        """Rewrite/expand a text prompt via Gemma (text-only, t2v)."""
        self.ensure_loaded()
        messages = [
            {"role": "system", "content": ENHANCE_T2V_SYSTEM_PROMPT},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]
        return self._generate(messages, images=None, seed=seed)

    def enhance_restyle(self, prompt: str, seed: int | None = None) -> str:
        """Rewrite/expand a text prompt for the RESTYLE task (text-only).

        Same as :meth:`enhance_text` but uses the restyle-specific system
        prompt (RESTYLE_ENHANCE_SYSTEM_PROMPT), which injects the color-
        fidelity constraint so the generated video keeps the restyled first
        frame's color palette instead of drifting as it progresses.
        """
        self.ensure_loaded()
        messages = [
            {"role": "system", "content": RESTYLE_ENHANCE_SYSTEM_PROMPT},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]
        return self._generate(messages, images=None, seed=seed)

    def enhance_image(
        self, prompt: str, image, seed: int | None = None
    ) -> str:
        """Enhance an image-edit prompt using Gemma's vision path.

        ``image`` is a PIL RGB image (the frame being edited); it is fed to the
        SigLIP tower as visual context alongside the user's edit direction, so
        the enhanced prompt can describe the actual edit result. Mirrors the
        LTX runner's i2v enhance (896 max-side, aspect-preserved).
        """
        self.ensure_loaded()
        tensor = _frame_to_tensor(image, CAPTION_MAX_SIDE)
        content: list[dict] = [
            {"type": "image"},
            {"type": "text", "text": f"user edit prompt: {prompt}"},
        ]
        messages = [
            {"role": "system", "content": IMAGE_ENHANCE_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return self._generate(messages, images=[tensor], seed=seed)

    def caption_video(self, frames, seed: int | None = None) -> str:
        """Caption a list of PIL RGB frames ("describe this video" -> prompt).

        ``frames`` may be a single PIL image or a list. Frames are downscaled
        to CAPTION_MAX_SIDE and fed as Gemma vision images.
        """
        self.ensure_loaded()
        if isinstance(frames, (list, tuple)):
            imgs = list(frames)
        else:
            imgs = [frames]
        tensors = [_frame_to_tensor(f, CAPTION_MAX_SIDE) for f in imgs]
        n = len(tensors)
        content: list[dict] = [{"type": "image"} for _ in range(n)]
        content.append({"type": "text", "text": "Describe this video clip."})
        messages = [
            {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ]
        return self._generate(messages, images=tensors, seed=seed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _frame_to_tensor(pil_image, max_side: int) -> torch.Tensor:
    """Convert a PIL RGB image to a uint8 HWC tensor, scaled to max_side while
    preserving aspect ratio (matches the LTX runner's i2v prompt-enhance — a
    square center-crop would discard scene content the caption needs)."""
    import numpy as np
    from PIL import Image

    img = pil_image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize(
            (max(2, int(round(w * scale))), max(2, int(round(h * scale)))),
            Image.BICUBIC,
        )
    arr = np.asarray(img, dtype=np.uint8).copy()  # writable copy -> HxWxC
    return torch.from_numpy(arr)


# Module-level shared instance (one Gemma per worker process).
_shared: GemmaEnhancer | None = None
_shared_lock = threading.Lock()


def get_enhancer() -> GemmaEnhancer:
    """Return the process-wide GemmaEnhancer (created on first use)."""
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = GemmaEnhancer()
    return _shared


def configure_evict_cb(evict_cb) -> None:
    """Wire an eviction hook onto the process-wide enhancer.

    Called before Gemma allocates VRAM on the shared GPU. The worker passes a
    callback that evicts the resident id-v2v model so both models can share one
    card (they cannot coexist on 32 GB). Hogging the eviction here keeps every
    Gemma entry point (restyle stage + /prompt-enhance) consistent.
    """
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = GemmaEnhancer()
        _shared._evict_cb = evict_cb
