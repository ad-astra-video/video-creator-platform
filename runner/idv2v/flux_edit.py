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
      - a Qwen3 4B text embedder (KLEIN4B_TEXT_ENC, hidden states [9,18,27], bf16)
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




def _tiled_decode(
    ae,
    z,
    tile: int = 256,  # latent-space tile (z is (1, C, H, W), latent /16)
    overlap: int = 32,
    device: str = "cuda:0",
) -> "torch.Tensor":
    """Decode a (possibly large) AE latent in overlapping spatial tiles.

    The FLUX.2 AE decode on the full 1920x1080 latent in ONE call is the second
    big VRAM spike after the encode. Tiling keeps peak activation memory flat
    (~a tile, not the whole frame) so native output resolution is VRAM-bound
    only by the flow denoise, not the AE.

    Overlapping tiles are blended linearly by distance-from-overlap-edge so
    there are no visible seams.
    """
    import torch
    z = z.to(device)
    _, C, H, W = z.shape
    import einops
    from PIL import Image as _PIL

    SCALE = 16  # flow latent is (1,128,H//16,W//16); decode -> 16x pixels
    out_h, out_w = H * SCALE, W * SCALE
    # preallocate output + weight in pixel space (float, on CPU to not blow VRAM)
    acc = torch.zeros((3, out_h, out_w), dtype=torch.float32, device="cpu")
    wacc = torch.zeros((1, out_h, out_w), dtype=torch.float32, device="cpu")

    # tile indices in latent space; step = tile - overlap
    step = max(1, tile - overlap)
    ys = list(range(0, max(1, H - tile + 1), step)) or [0]
    if ys[-1] != H - tile:
        ys.append(H - tile)
    xs = list(range(0, max(1, W - tile + 1), step)) or [0]
    if xs[-1] != W - tile:
        xs.append(W - tile)

    import torch.nn.functional as F
    for y0 in set(ys):
        y0 = max(0, min(y0, H - tile))
        for x0 in set(xs):
            x0 = max(0, min(x0, W - tile))
            zt = z[:, :, y0:y0 + tile, x0:x0 + tile]
            with torch.no_grad():
                dec = ae.decode(zt).float()          # (1,3, hh, ww)
            dec = dec[0].cpu()                        # (3, hh, ww)
            # pixel-space tile origin
            py0, px0 = y0 * SCALE, x0 * SCALE
            hh, ww = dec.shape[1], dec.shape[2]
            # weights: 1 in middle, ramp 0->1 across overlap margins
            wy = torch.ones((hh, 1), dtype=torch.float32)
            wx = torch.ones((1, ww), dtype=torch.float32)
            ovy = min(overlap * SCALE, hh // 2)
            ovx = min(overlap * SCALE, ww // 2)
            if ovy > 0:
                ramp = torch.arange(ovy, dtype=torch.float32) / ovy
                wy[:ovy, 0] = ramp
                wy[-ovy:, 0] = ramp.flip(0)
            if ovx > 0:
                ramp = torch.arange(ovx, dtype=torch.float32) / ovx
                wx[0, :ovx] = ramp
                wx[0, -ovx:] = ramp.flip(0)
            wgt = (wy * wx)
            acc[:, py0:py0 + hh, px0:px0 + ww] += dec.clamp(-1, 1) * wgt
            wacc[0, py0:py0 + hh, px0:px0 + ww] += wgt
            del dec, zt, wgt

    out = acc / wacc.clamp(min=1e-6)
    return out.unsqueeze(0)  # (1,3,out_h,out_w) CPU float32

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
        self._strength = _clamp01(config.klein4b_strength())
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
        """Load the FLUX.2 klein 4B flow + AE + Qwen3 text encoder.

        The Qwen3 4B text encoder may be evicted separately right after it
        produces the prompt conditioning (see evict_text_encoder) to trim VRAM
        on the shared card during sampling. It is lazily re-loaded here on the
        next edit when the flow + AE are already resident.
        """
        if self._model is not None and self._text_encoder is not None:
            return
        with self._lock:
            if self._model is not None and self._text_encoder is not None:
                return
            if self._model is not None:
                # Flow + AE resident but text encoder was evicted -> load just it.
                self._load_text_encoder_locked()
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
            from flux2.util import load_ae, load_flow_model

            logger.info(
                "Loading FLUX.2 klein 4B editor on %s (model=%s, ae=%s, te=%s, "
                "steps=%d, guidance=%.2f) ...",
                self._device, config.KLEIN4B_MODEL, config.KLEIN4B_AE,
                config.KLEIN4B_TEXT_ENC, self._steps, self._guidance,
            )
            model = load_flow_model("flux.2-klein-4b", device=self._device)
            ae = load_ae("flux.2-klein-4b", device=self._device)
            self._model = model
            self._ae = ae
            self._load_text_encoder_locked()
            logger.info("FLUX.2 klein 4B editor loaded on %s", self._device)

    def _load_text_encoder_locked(self) -> None:
        """Load/reload the Qwen3 4B text encoder (assumes self._lock held)."""
        if self._text_encoder is not None:
            return
        # Load the Qwen3 text embedder directly with the configured repo id.
        # NB: we deliberately do NOT use flux2's load_qwen3_embedder() — it
        # hardcodes the Qwen/Qwen3-4B-FP8 suffix, which (with torch_dtype=None
        # and an fp8 checkpoint that carries a quantization_config) materializes
        # the encoder at full bf16 AND requires the `kernels` finegrained-fp8
        # runtime at forward time. The default KLEIN4B_TEXT_ENC is the plain
        # bf16 Qwen/Qwen3-4B (same weights BFL bundles in the klein repo), which
        # loads ~8 GB, needs no kernels pkg, and matches the ~13 GB reference
        # footprint.
        from flux2.text_encoder import Qwen3Embedder
        text_encoder = Qwen3Embedder(
            model_spec=config.KLEIN4B_TEXT_ENC, device=self._device
        )
        text_encoder.eval()
        self._text_encoder = text_encoder
        logger.info("FLUX.2 klein Qwen3 4B text encoder loaded on %s", self._device)

    def evict_text_encoder(self) -> None:
        """Drop only the Qwen3 4B text encoder's weights.

        Call after the prompt conditioning (ctx/ctx_ids) has been produced —
        denoise consumes those tensors, not the encoder, so its ~4 GB is freed
        for the flow + AE sampling loop on the shared card. Re-loaded lazily on
        the next edit via ensure_loaded.
        """
        with self._lock:
            if self._text_encoder is None:
                return
            self._text_encoder = None
            try:
                import gc
                gc.collect()
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            logger.info("Evicted FLUX.2 klein Qwen3 4B text encoder (freed GPU memory)")

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
        strength: float | None = None,
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
        from PIL import Image
        from einops import rearrange
        from flux2.sampling import (
            batched_prc_img,
            batched_prc_txt,
            denoise,
            encode_image_refs,
            get_schedule,
            scatter_ids,
        )

        # OUTPUT resolution: use the caller's explicit width/height, else the
        # source (capped at KLEIN4B_MAX_SIDE so a stray huge input can't blow
        # the card).  This is the resolution the flow denoises its latent at
        # and what comes back from decode -> a 1920x1080 output stays 1920x1080.
        cap = max_side or self._max_side
        if width is None or height is None:
            out_w, out_h = resolve_dims(image.size[0], image.size[1], cap)
        else:
            out_w, out_h = resolve_dims(width, height, cap)

        # REFERENCE-ENCODE resolution: independent of the output. The reference
        # only CONDITIONS the edit (ref_tokens), it does not set output size, so
        # we cap it to keep the AE encode fast and VRAM-light.  Native 1080p is
        # carried by the OUTPUT latent / tiled decode, not by encoding the ref
        # at 1080p (which is what previously OOM'd).
        ref_side = config.KLEIN4B_REF_SIDE
        ref_img = image
        if ref_side and max(image.size) > ref_side:
            ref_w, ref_h = resolve_dims(image.size[0], image.size[1], ref_side)
            ref_img = image.resize((ref_w, ref_h), Image.Resampling.LANCZOS)
        width, height = out_w, out_h

        # Context: the (distilled) klein model needs only the prompt — no
        # negative / empty pair (guidance_distilled -> `denoise`, not denoise_cfg).
        ctx = self._text_encoder([prompt]).to(torch.bfloat16)
        ctx, ctx_ids = batched_prc_txt(ctx)
        # The prompt conditioning is now fully captured in ctx/ctx_ids; the 4B
        # encoder's weights are no longer needed for denoise. Evict it to keep
        # the card footprint down to flow + AE while sampling (~4 GB freed).
        # Lazily re-loaded on the next edit via ensure_loaded.
        self.evict_text_encoder()

        # Reference conditioning: encode the (capped) source frame through the AE.
        ref_tokens, ref_ids = encode_image_refs(self._ae, [ref_img])
        ref_tokens = ref_tokens.to(self._device)
        ref_ids = ref_ids.to(self._device)

        # Output latent + schedule. FIDELITY KNOB: strength<1.0 anchors the
        # denoise start on the AE-encoded source (img2img-style) so the edit
        # stays truer to the original; strength=1.0 (default) is the stock
        # pure-noise re-imagine. If image-init isn't available (BFL API /
        # VRAM) we fall back to the stock path so it can never regress.
        shape = (1, 128, height // 16, width // 16)
        generator = torch.Generator(device=self._device).manual_seed(int(seed))
        eff = _clamp01(strength if strength is not None else self._strength)
        init = None
        if eff < 1.0:
            try:
                init = self._image_init(ref_img, height, width, eff, generator)
            except Exception as exc:
                logger.warning(
                    "klein image-init at strength %.2f failed (%s); "
                    "falling back to full re-imagine", eff, exc)
                init = None
        if init is None:
            randn = torch.randn(shape, generator=generator,
                                dtype=torch.bfloat16, device=self._device)
            x, x_ids = batched_prc_img(randn)
            timesteps = get_schedule(self._steps, x.shape[1])
        else:
            x, x_ids, timesteps = init

        with torch.no_grad():
            x = denoise(
                self._model, x, x_ids, ctx, ctx_ids,
                timesteps=timesteps,
                guidance=self._guidance,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
            x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
            # Tile the decode so a native-HD output latent doesn't spike VRAM.
            x = _tiled_decode(self._ae, x, device=self._device)

        x = x.clamp(-1, 1)
        x = rearrange(x[0], "c h w -> h w c")
        out = (127.5 * (x + 1.0)).byte().numpy()
        from PIL import Image
        return Image.fromarray(out, mode="RGB")

    def _image_init(self, ref_img, out_h, out_w, strength, generator):
        """Build a noised source-anchored init so klein stays true to the original.

        EXPERIMENTAL / PENDING ON-BOX VALIDATION (the step-distilled klein model
        and the exact BFL flux2 API surface can't be exercised off-GPU, and .151
        is unreachable). Encodes the source frame at the OUTPUT latent resolution
        and starts the (truncated) denoise from that latent instead of pure noise
        -- lower strength keeps the result closer to the source. Runs
        `round(steps*strength)` (>=1) steps. Raises on any failure so the caller
        falls back to the stock full-re-imagine path (never a silent wrong output).

        Note: the init encode runs at the output res (latent must match the
        flow's), unlike the conditioning ref which stays capped at
        KLEIN4B_REF_SIDE -- on a small-VRAM card this encode may OOM and the
        strength=1.0 fallback engages.
        """
        import torch
        from flux2.sampling import batched_prc_img, get_schedule
        from flux2.util import encode_image

        init_img = ref_img.resize((out_w, out_h))
        z = encode_image(self._ae, [init_img]).to(torch.bfloat16).to(self._device)
        z = z[:, :, : out_h // 16, : out_w // 16]
        n_steps = max(1, round(self._steps * strength))
        x, x_ids = batched_prc_img(z)
        timesteps = get_schedule(n_steps, x.shape[1])
        return x, x_ids, timesteps


# ---------------------------------------------------------------------------
# Helpers + module-level singleton
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    """Clamp ``v`` to [0, 1] (image-init strength range)."""
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 1.0


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
