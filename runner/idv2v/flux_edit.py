"""FLUX.2 [klein] 4B image editor — first-frame styling for the id-v2v worker.

Wires Black Forest Labs' FLUX.2 [klein] 4B (the 4B distilled image-editing +
generation model) in as the image-edit model that styles the restyle first
frame. All heavy imports (torch, the BFL `flux2` package, transformers) happen
LAZILY inside `load()` so this module is importable in GPU-less test venvs —
the same staged pattern the worker already uses for diffsynth/Gemma.

Model facts (researched from the official repo black-forest-labs/flux2):
  * FLUX.2 [klein] 4B is guidance- AND step-distilled:
        num_steps = 4, guidance = 1.0   (fixed by BFL; no CFG)
    (The 50-step "klein-base-4B" is for fine-tuning/control, not prod editing.)
  * Editing is SINGLE-REFERENCE conditioning (no ControlNet/mask needed for a
    style transform): encode the frame through the FLUX.2 AE -> `ref_tokens`,
    run the distilled `denoise` with `img_cond_seq=ref_tokens`, decode. The
    prompt describes the DESIRED (styled) result; the reference anchors the
    content.
  * Three components must be resident to edit:
      - the 4B flow transformer  (KLEIN4B_MODEL,   ~8 GB bf16)
      - the FLUX.2 autoencoder   (KLEIN4B_AE)   — shared with FLUX.2 [dev]
      - a Qwen3 4B text embedder (KLEIN4B_TEXT_ENC, hidden states [9,18,27])
        which is a SEPARATE LLM from Gemma and must share the card with the
        video model via the evict-churn lifecycle below.

GPU / eviction: the editor cannot coexist on one card with the id-v2v
DiT/VACE (~19.5 GB) or a Gemma LLM, so callers run the enhance-before-evict
sequence (see server.py) and this editor evicts any resident video model via
its `evict_cb` before allocating, then is itself evicted before the video
model loads again.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger("video_creator.runner.idv2v.flux_edit")


def resolve_dims(width: int, height: int, max_side: int, base: int = 16) -> tuple[int, int]:
    """Scale ``(width, height)`` to fit ``max_side`` on the long edge, round to
    a multiple of ``base`` (the FLUX.2 latent is /16). Preserves aspect ratio."""
    if max_side and max(width, height) > max_side:
        scale = max_side / max(width, height)
        nw = int(round(width * scale) // base * base)
        nh = int(round(height * scale) // base * base)
    else:
        nw = width // base * base
        nh = height // base * base
    return max(base, nw), max(base, nh)


class FluxKleinEditor:
    """A single, swappable FLUX.2 [klein] 4B editor (one per worker process).

    Loads lazily on first use, then stays resident until `unload()`. Ownership
    mirrors the id-v2v `ModelManager` and Gemma `GemmaEnhancer`: a loose
    module-level instance the worker evicts before loading anything else on the
    shared GPU.
    """

    def __init__(self, device: str | None = None, evict_cb=None) -> None:
        self._device = _normalize_device(device or config.klein4b_device())
        self._evict_cb = evict_cb
        self._steps = config.klein4b_steps()
        self._guidance = config.klein4b_guidance()
        self._lock = threading.Lock()
        self._model = None
        self._ae = None
        self._text_encoder = None
        self._max_side = config.KLEIN4B_MAX_SIDE

    # -- lifecycle --------------------------------------------------------

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    def ensure_loaded(self) -> None:
        """Load the FLUX.2 klein 4B flow + AE + Qwen3 text encoder (once)."""
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if self._evict_cb is not None:
                self._evict_cb()
            if not config.klein4b_enabled():
                raise RuntimeError(
                    "FLUX.2 Klein 4B is not enabled (KLEIN4B_ENABLED=%r, model=%s). "
                    "Provision flux-2-klein-4b.safetensors + ae.safetensors on /models/flux2, "
                    "or set KLEIN4B_ENABLED=1." % (config.KLEIN4B_ENABLED, config.KLEIN4B_MODEL)
                )

            # Give the BFL loaders the weight paths via their env-var contract.
            os.environ.setdefault("KLEIN_4B_MODEL_PATH", config.KLEIN4B_MODEL)
            os.environ.setdefault("AE_MODEL_PATH", config.KLEIN4B_AE)

            import torch
            from flux2.sampling import denoise  # noqa: F401  (smoke: package wired)
            from flux2.text_encoder import load_qwen3_embedder
            from flux2.util import load_ae, load_flow_model

            logger.info(
                "Loading FLUX.2 klein 4B editor on %s (model=%s, ae=%s, te=%s, "
                "steps=%d, guidance=%.2f) ...",
                self._device, config.KLEIN4B_MODEL, config.KLEIN4B_AE,
                config.KLEIN4B_TEXT_ENC, self._steps, self._guidance,
            )
            model = load_flow_model("flux.2-klein-4b", device=self._device)
            ae = load_ae("flux.2-klein-4b", device=self._device)
            text_encoder = load_qwen3_embedder(variant="4B", device=self._device)
            model.eval()
            ae.eval()
            text_encoder.eval()
            self._model = model
            self._ae = ae
            self._text_encoder = text_encoder
            logger.info("FLUX.2 klein 4B editor loaded on %s", self._device)

    def unload(self) -> None:
        """Drop the editor + free its GPU memory (safe to call when unloaded)."""
        with self._lock:
            self._model = None
            self._ae = None
            self._text_encoder = None
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    # -- editing ----------------------------------------------------------

    def edit(
        self,
        image: "Image.Image",
        prompt: str,
        seed: int,
        width: int | None = None,
        height: int | None = None,
        max_side: int | None = None,
    ) -> "Image.Image":
        """Single-reference image edit: style ``image`` according to ``prompt``.

        Args:
            image: PIL RGB frame to edit (the restyle first frame).
            prompt: style prompt describing the DESIRED (styled) result.
            seed: integer seed for reproducibility.
            width/height: output dims (defaults to the source, capped at
                KLEIN4B_MAX_SIDE). Must be multiples of 16.
            max_side: optional long-edge cap (defaults to config).

        Returns:
            PIL RGB styled image.
        """
        self.ensure_loaded()
        import torch
        from einops import rearrange
        from flux2.sampling import (
            batched_prc_img,
            batched_prc_txt,
            denoise,
            encode_image_refs,
            get_schedule,
            scatter_ids,
        )

        src_w, src_h = image.size
        cap = max_side or self._max_side
        if width is None or height is None:
            width, height = resolve_dims(src_w, src_h, cap)
        else:
            width, height = resolve_dims(width, height, cap)

        # Context: the (distilled) klein model needs only the prompt — no
        # negative / empty pair (guidance_distilled -> `denoise`, not denoise_cfg).
        ctx = self._text_encoder([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)

        # Reference conditioning: encode the source frame through the AE.
        ref_tokens, ref_ids = encode_image_refs(self._ae, [image])
        ref_tokens = ref_tokens.to(self._device)
        ref_ids = ref_ids.to(self._device)

        # Output latent + schedule.
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device=self._device).manual_seed(int(seed))
        randn = torch.randn(shape, generator=generator,
                            dtype=torch.bfloat16, device=self._device)
        x, x_ids = batched_prc_img(randn)
        timesteps = get_schedule(self._steps, x.shape[1])

        with torch.no_grad():
            x = denoise(
                self._model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps,
                guidance=self._guidance,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
            x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
            x = self._ae.decode(x).float()

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        out = (127.5 * (x + 1.0)).cpu().byte().numpy()
        from PIL import Image
        return Image.fromarray(out, mode="RGB")


# ---------------------------------------------------------------------------
# Helpers + module-level singleton
# ---------------------------------------------------------------------------

def _normalize_device(device: str) -> str:
    """Normalize a CUDA device spec ('0' | 'cuda:0') to the form BFL/transformers
    expects ('cuda:0'). Non-cuda specs pass through."""
    s = str(device).strip()
    if not s:
        return "cuda:0"
    if s.isdigit():
        return f"cuda:{s}"
    return s


# Single editor shared by all request threads (like GemmaEnhancer / ModelManager).
_shared: FluxKleinEditor | None = None
_shared_lock = threading.Lock()


def get_editor() -> FluxKleinEditor:
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = FluxKleinEditor()
    return _shared


def configure_evict_cb(evict_cb) -> None:
    """Wire an eviction hook onto the process-wide editor.

    Runs before FLUX.2 Klein allocates VRAM on the shared video GPU. The worker
    passes a callback that evicts the resident id-v2v model (and, if needed,
    Gemma) so the editor can take the card.
    """
    global _shared
    with _shared_lock:
        if _shared is None:
            _shared = FluxKleinEditor()
        _shared._evict_cb = evict_cb


def evict_editor() -> None:
    """Unload the process-wide editor if it is resident (safe to call always)."""
    global _shared
    with _shared_lock:
        if _shared is not None and _shared.is_ready:
            _shared.unload()
