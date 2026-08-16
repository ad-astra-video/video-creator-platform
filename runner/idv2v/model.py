"""ID-V2V model loader and inference pipeline (ported from the id-v2v runner).

Loads the ID-V2V model — a Wan 2.1 I2V-14B DiT + VACE-14B ControlNet video
model (from Eyeline-Labs/ID-V2V, wrapped by the `diffsynth` fork) — on GPU.

Two knobs make this fit a 32 GB RTX 5090 instead of a 96 GB card:

* INT8 QUANTIZATION of the video model: the 14B DiT and its VACE controlnet are
  quantized in-place with torchao `int8_weight_only()` (~28 GB bf16 -> ~14 GB
  int8). T5 text-encoder + VAE + tokenizer stay bf16 but live on the CPU offload
  pool.
* CPU OFFLOAD: every ModelConfig uses offload_device="cpu", pipeline uses
  enable_vram_management(vram_buffer=10). Layers move to GPU on demand.

The worker keeps the model warm between calls; the live-runner tells it to
/evict (dropping the pipe + empty_cache) before another worker needs the GPU.

Grounded in the reference pipeline:
    c:\\dev\\id-v2v\\runner\\src\\idv2v_runner\\model.py
    Eyeline-Labs/ID-V2V  src/idv2v/inference/pipeline.py
"""

import asyncio
import gc
import logging
import os
import time

import torch

from . import config

logger = logging.getLogger("video_creator.runner.idv2v.model")

# ---------------------------------------------------------------------------
# Video processing helpers — ported verbatim from the reference pipeline.py so
# clip scheduling, center-crop and frame slicing match exactly.
# ---------------------------------------------------------------------------


def center_crop_and_resize(img, width: int, height: int):
    """Center-crop + resize a PIL RGB image to (width, height) using BICUBIC."""
    from PIL import Image
    w, h = img.size
    target_aspect = height / width
    aspect = h / w
    if (h == height) and (w == width):
        return img
    if abs(aspect - target_aspect) < 1e-6:
        return img.resize((width, height), Image.BICUBIC)
    if aspect > target_aspect:  # too tall -> resize width, crop height
        new_w, new_h = width, int(aspect * width)
        resized = img.resize((new_w, new_h), Image.BICUBIC)
    else:  # too wide -> resize height, crop width
        new_h, new_w = height, int(height / aspect)
        resized = img.resize((new_w, new_h), Image.BICUBIC)
    rw, rh = resized.size
    left, top = (rw - width) // 2, (rh - height) // 2
    return resized.crop((left, top, left + width, top + height))


def compute_clip_schedule(total_frames: int, num_frames_per_clip: int):
    """Compute (start, end) frame indices for multi-clip generation.

    Regular clips advance by stride = num_frames_per_clip - 1 (1-frame overlap);
    the last clip is anchored at the end so it always has exactly
    num_frames_per_clip frames.
    """
    if total_frames <= num_frames_per_clip:
        return [(0, total_frames)]
    clips = []
    start = 0
    stride = num_frames_per_clip - 1
    while start + num_frames_per_clip < total_frames:
        clips.append((start, start + num_frames_per_clip))
        start += stride
    clips.append((total_frames - num_frames_per_clip, total_frames))
    return clips


def slice_frames(all_frames, start: int, end: int):
    """Slice frames[start:end]; pad by repeating the last frame if needed."""
    n = len(all_frames)
    if end <= n:
        return all_frames[start:end]
    result = list(all_frames[start:n])
    result += [all_frames[-1]] * (end - n)
    return result


class _ChunkedFFN(torch.nn.Module):
    """Wraps a block's FFN so its sequence dim is processed in chunks, capping
    the ffn1/ffn2 intermediate activation size at `chunk_tokens` rows. The FFN
    (Linear->GELU->Linear) is elementwise-per-token, so chunking the sequence
    dim is mathematically EXACT — it only trades a small loop iteration for a
    ~3x smaller peak activation, which is what lets 720p/81-frame fit in a
    31 GB GPU alongside the resident fp8 DiT."""
    def __init__(self, inner, chunk_tokens=32768):
        super().__init__()
        self.inner = inner
        self.chunk_tokens = chunk_tokens

    def forward(self, x):
        if x.dim() == 2 and x.shape[0] > self.chunk_tokens:
            return torch.cat([self.inner(c) for c in x.split(self.chunk_tokens, dim=0)], dim=0)
        return self.inner(x)


class ModelManager:
    """Loads ID-V2V (int8 DiT+VACE) with CPU offload for a 32 GB 5090.

    Instance-stated on purpose so the worker owns/evicts its model: construct
    one per worker (or one per repo of the process), call ``load()`` to build it,
    keep it warm, and ``evict()`` to free GPU/CPU memory when another worker
    needs the card.
    """

    def __init__(self, device: str = ""):
        self.device = self._normalize_device(device or config.GPU_DEVICE)
        self._pipe = None
        self._torch_dtype = torch.bfloat16
        self._enc_paths = {}
        self._tokenizer_path = None
        self._posi_context = None
        self._nega_context = None
        # Model variant ("fast" | "regular") this instance loads. Set via
        # set_variant() before load(); defaults from config.
        self.variant = config.DEFAULT_MODEL_VARIANT
        self.hf_subfolder = config.subfolder_for(self.variant)

    def set_variant(self, variant: str) -> None:
        """Select the model variant to load on the next load() (no-op now).
        `variant` is normalized (unknown -> config default); the effective
        subfolder honors the IDV2V_HF_SUBFOLDER env override."""
        self.variant = config._norm_variant(variant)
        self.hf_subfolder = config.subfolder_for(self.variant)
        logger.info("idv2v variant set: %s (hf subfolder=%r)", self.variant, self.hf_subfolder)

    @staticmethod
    def _normalize_device(device: str) -> str:
        """Normalize a bare CUDA index (e.g. env "0") to a torch-valid device
        string ("cuda:0"). diffsynth's enable_vram_management/get_vram calls
        torch.cuda.mem_get_info(self.device), which rejects a bare "0" with
        "Invalid device string", so a proper device is required there."""
        d = str(device).strip()
        if d.startswith(("cuda", "cpu", "meta", "mps", "xpu")):
            return d
        if d and (d.isdigit() or (d[0] == "-" and d[1:].isdigit())):
            return f"cuda:{d}"
        return d

    @property
    def device_name(self) -> str:
        return self.device

    # -- lifecycle ----------------------------------------------------------
    async def load(self):
        """Build the WanVideoPipeline, quantize the video model to int8, offload."""
        if self._pipe is not None:
            return
        logger.info(
            "Loading ID-V2V on %s (quant=%s offload=%s vram_buffer=%d) ...",
            self.device, config.IDV2V_QUANT, config.IDV2V_OFFLOAD,
            config.IDV2V_VRAM_BUFFER,
        )
        start = time.time()

        try:
            import torchao
            from torchao.quantization import int8_weight_only, quantize_
            self._torchao = torchao  # keep a handle; used below
        except Exception as exc:  # pragma: no cover - quantization optional at runtime
            logger.warning("torchao not importable (%s); int8 quant disabled", exc)
            self._torchao = None

        # Load all pieces in a thread executor so we don't block the event loop
        # during the multi-minute model build.
        self._pipe = await asyncio.to_thread(self._build_pipeline)

        # Pre-quantized FP8-from-HF carries the quantization already baked in the
        # weights, so we skip the runtime int8 quantization entirely for it.
        if (
            config.IDV2V_SOURCE != "hf-fp8"
            and config.IDV2V_QUANT == "int8"
            and self._torchao is not None
        ):
            self._quantize_int8()

        self._enable_offload()

        logger.info(
            "Model loaded in %.1fs (source=%s, quant=%s, %s)",
            time.time() - start,
            config.IDV2V_SOURCE,
            "fp8-from-hf" if config.IDV2V_SOURCE == "hf-fp8" else config.IDV2V_QUANT,
            "CPU offload" if config.IDV2V_OFFLOAD else "no offload",
        )

    def evict(self) -> None:
        """Drop the pipeline and free GPU/CPU memory. Safe to call when unloaded."""
        if self._pipe is not None:
            logger.info("Evicting ID-V2V pipeline (freeing GPU/CPU memory)")
        self._pipe = None
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def _build_pipeline(self):
        """Construct the diffsynth WanVideoPipeline (runs in a worker thread)."""
        from diffsynth.pipelines.wan_video_new_multiVace_svi import (
            ModelConfig,
            WanVideoPipeline,
        )

        # DiT + VACE come from the finetuned idv2v.pth, so we SKIP loading the
        # base I2V DiT (same fast path as the reference script). We only load the
        # single-file env models the WanVideoPipeline needs: T5 text-encoder, VAE
        # and CLIP image-encoder, plus the google/* tokenizer.
        #
        # diffsynth resolves each file as local_model_path/<model_id>/<pattern>.
        # We downloaded the whole Wan-AI/Wan2.1-I2V-14B-720P repo (which via
        # redirect_common_files carries T5/VAE/CLIP/tokenizer inside it), so all
        # model configs point at that one repo with a single-file pattern each —
        # never a glob, because a multi-file glob resolves to a LIST and crashes
        # model_manager.match() ("'list' has no attribute 'endswith'").
        I2V_REPO = "Wan-AI/Wan2.1-I2V-14B-720P"
        model_configs = [
            ModelConfig(model_id=I2V_REPO, origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth",  offload_device="cpu"),  # T5 text encoder
            ModelConfig(model_id=I2V_REPO, origin_file_pattern="Wan2.1_VAE.pth",                   offload_device="cpu"),  # VAE
            ModelConfig(model_id=I2V_REPO, origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),  # CLIP
        ]

        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=self._torch_dtype,
            device=self.device,
            use_usp=False,                 # single-GPU 5090 -> no USP sequence parallel
            model_configs=model_configs,
            tokenizer_config=ModelConfig(
                model_id=I2V_REPO,
                origin_file_pattern="google/*",
                offload_device="cpu",
            ),
            # local_model_path is the HF-cache ROOT (/models), NOT dirname of the
            # repo dir. diffsynth builds the file as
            # local_model_path/<model_id>/<pattern>, and model_id already carries
            # "Wan-AI/Wan2.1-I2V-14B-720P". WAN_MODEL_DIR IS .../Wan-AI/Wan2.1-I2V-14B-720P,
            # so dirname(WAN_MODEL_DIR) = /models/Wan-AI (wrong -> double "Wan-AI");
            # we need two levels up to reach /models.
            local_model_path=os.path.dirname(os.path.dirname(config.WAN_MODEL_DIR.rstrip("/"))),  # /models (HF cache root)
            checkpoint_path=None,
            skip_download=True,
            redirect_common_files=False,
        )

        # Remember the on-disk paths of the encoders so we can lazily RELOAD them
        # for the conditioning phases after evicting them from RAM at /load.
        # (The pipeline's ModelManager is a local that's discarded after
        # from_pretrained, so reload must re-instantiate from the checkpoint.)
        wan_dir = config.WAN_MODEL_DIR.rstrip("/")
        self._enc_paths = {
            "wan_video_text_encoder": os.path.join(wan_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
            "wan_video_image_encoder": os.path.join(
                wan_dir, "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        }
        self._tokenizer_path = os.path.join(wan_dir, "google", "umt5-xxl")
        logger.info("Encoder reload paths: t5=%s clip=%s tok=%s",
                    bool(self._enc_paths["wan_video_text_encoder"]),
                    bool(self._enc_paths["wan_video_image_encoder"]),
                    bool(self._tokenizer_path))

        # Fast-load the finetuned DiT + VACE. Two possible sources:
        #   "hf-fp8" -> stream the pre-quantized per-channel FP8 (native fp8, no
        #               runtime int8; weights stay fp8 and dequant per-layer).
        #   "local"  -> load idv2v.pth (memory-mapped) and quantize at runtime.
        #
        # On the STAGED lifecycle the DiT + VACE are NOT loaded here: doing so
        # would push peak RAM to T5(11) + CLIP(~2) + VAE(0.5) + dit/vace-fp8(19.5)
        # ~= 33 GB -> OOM. Instead _ensure_dit_vace() loads the native-fp8
        # DiT/VACE lazily, only AFTER the T5 has been freed (see the staged
        # driver). That keeps peak RAM ~= 22 GB and removes the need to
        # int8-quantize the T5/CLIP at all.
        if not config.IDV2V_STAGED:
            self._load_dit_vace(pipe)
            assert pipe.dit is not None and pipe.dit.has_image_input, "Expected I2V DiT"

        if config.IDV2V_STAGED:
            # True one-at-a-time staging: do NOT keep T5 (11 GB) or CLIP (2.5 GB)
            # resident from /load — they are lazily reloaded for their conditioning
            # phase and released again. This keeps the SAM3-preprocessing baseline
            # at just VAE (~0.5 GB) so a 720p/81-frame request fits in ~24 GB RAM
            # (holding the full encoder set resident OOMs at that size).
            self._release_encoder(pipe, "wan_video_text_encoder", "text_encoder")
            self._release_encoder(pipe, "wan_video_image_encoder", "image_encoder")

        # Single-GPU (no USP): enable per-layer vram offloading for the encode
        # sub-models (T5/CLIP/VAE). The DiT/VACE are FP8Linear/FP8ConvNd - not
        # torch.nn.Linear - so diffsynth's vram_management does not wrap them;
        # the staged driver places the fp8 DiT/VACE on GPU explicitly.
        pipe.enable_vram_management(vram_buffer=config.IDV2V_VRAM_BUFFER)

        return pipe

    def _load_dit_vace(self, pipe, torch_dtype=torch.bfloat16):
        """Load the finetuned DiT + VACE from the configured source."""
        if config.IDV2V_SOURCE == "hf-fp8":
            self._load_fp8_native(pipe)
        elif os.path.isfile(config.MODEL_CHECKPOINT):
            self._load_finetuned_dit_vace(pipe, config.MODEL_CHECKPOINT, torch_dtype=torch_dtype)
        else:
            raise FileNotFoundError(
                f"idv2v checkpoint not found: {config.MODEL_CHECKPOINT}. "
                "Run download_models.sh first."
            )

    def _enable_block_cpu_offload(self, pipe, num_blocks=8):
        """Move the last `num_blocks` DiT blocks to CPU and compute them there
        (their weights stay resident, input+/output hops GPU<->CPU once per
        forward). Frees ~num_blocks/40 of the fp8 model's GPU residency for the
        720p/81-frame denoise activations — the only knob that reduces the fixed
        per-step activation peak (a hidden/residual over the ~300k-token seq that
        tiling, FFN chunking and VAE eviction cannot). Full bf16/fp8 quality is
        preserved; it is simply slower.
        """
        num_blocks = int(num_blocks)
        if num_blocks <= 0:
            # full-GPU mode: 0 means "don't offload anything". Note
            # blocks[-0:] == blocks[0:] in Python (since -0 == 0), so we
            # MUST short-circuit here rather than compute slices[0:].
            return
        blocks = pipe.dit.blocks
        if len(blocks) == 0:
            return
        off = list(blocks[-num_blocks:])
        dev = self.device
        for blk in off:
            orig_fwd = blk.forward  # bound method
            blk.to("cpu")
            def wrapped(x, context, t_mod, freqs, cam_emb=None, context2=None,
                        _fwd=orig_fwd, _dev=dev):
                x = x.to("cpu")
                if context is not None:
                    context = context.to("cpu")
                if cam_emb is not None:
                    cam_emb = cam_emb.to("cpu")
                if context2 is not None:
                    context2 = context2.to("cpu")
                t_mod = t_mod.to("cpu")
                freqs = freqs.to("cpu")
                out = _fwd(x, context, t_mod, freqs, cam_emb=cam_emb, context2=context2)
                return out.to(_dev)
            blk.forward = wrapped
        logger.info("Staged: offloaded %d DiT blocks to compute on CPU (frees ~%.1f GB GPU)",
                    len(off), len(off) * 0.36)

    def _enable_vace_cpu(self, pipe):
        """Compute the whole VACE ControlNet on CPU and free ~2 GB GPU.

        diffsynth's Wan/VACE pipeline has NO tiled-attention implementation (the
        `tiled` kwarg only exists for SD/SDXL/SD3/SD-video), so the VACE
        self-attention's rope_apply still materializes a ~2.88 GB tensor from the
        ~104k-token 720p/81-frame sequence. That is the exact peak that OOMs the
        31 GB GPU once the fp8 DiT is resident. Running the ~2 GB VACE on CPU
        removes that peak (and frees its weights) while the wrapper keeps the
        returned per-block embeddings flowing back to GPU for the DiT.
        """
        vace = getattr(pipe, "vace", None)
        if vace is None:
            return
        if not getattr(pipe, "_vace_cpu_wrapped", False):
            orig_fwd = vace.forward
            dev = self.device
            def _wrap(*args, **kwargs):
                a2 = tuple(x.to("cpu") if torch.is_tensor(x) and x.is_cuda else x for x in args)
                k2 = {k: (v.to("cpu") if torch.is_tensor(v) and v.is_cuda else v) for k, v in kwargs.items()}
                out = orig_fwd(*a2, **k2)
                if torch.is_tensor(out):
                    return out.to(dev)
                if isinstance(out, (tuple, list)):
                    return [o.to(dev) if torch.is_tensor(o) else o for o in out]
                if isinstance(out, dict):
                    return {kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in out.items()}
                return out
            vace.forward = _wrap
            pipe._vace_cpu_wrapped = True
        vace.to("cpu")
        torch.cuda.empty_cache()
        logger.info("Staged: VACE computed on CPU (frees ~2 GB GPU)")

    def _enable_ffn_chunking(self, pipe, chunk_tokens=32768):
        """Wrap every DiT block's FFN with _ChunkedFFN to cap the per-layer FFN
        intermediate at `chunk_tokens` sequence rows. Removes the ~2.88 GB
        peak (104k x ffn_dim x 2) that otherwise OOMs 720p/81-frame denoise."""
        for blk in pipe.dit.blocks:
            if hasattr(blk, "ffn") and not isinstance(blk.ffn, _ChunkedFFN):
                blk.ffn = _ChunkedFFN(blk.ffn, chunk_tokens)
        logger.info("Staged: FFN sequence chunking enabled (%d tokens/chunk)", chunk_tokens)

    def _ensure_dit_vace(self):
        """Staged path: lazily load DiT+VACE after the T5 has been freed, then
        place the fp8 weights on GPU (they dequant per-layer inside forward)."""
        pipe = self._pipe
        if pipe.dit is not None:
            return
        logger.info("Staged: loading DiT + VACE (native fp8) after T5 freed ...")
        self._load_dit_vace(pipe)
        assert pipe.dit is not None and pipe.dit.has_image_input, "Expected I2V DiT"
        pipe.dit.to(self.device)
        pipe.vace.to(self.device)
        torch.cuda.empty_cache()
        logger.info("Staged: DiT + VACE placed on %s (fp8 resident)", self.device)
        self._enable_ffn_chunking(pipe)
        self._enable_block_cpu_offload(
            pipe, num_blocks=int(os.environ.get("IDV2V_DIT_CPU_BLOCKS", "16")))

    def _make_dit_pre(self, prev_holder):
        def pre(mod, args):
            mod.to(self.device)
            p = prev_holder[0]
            if p is not None and p is not mod:
                p.to("cpu")
            prev_holder[0] = mod
        return pre

    @staticmethod
    def _make_dit_post():
        def post(mod, inp, out):
            mod.to("cpu")
        return post

    def _setup_dit_stream(self, pipe):
        """Register forward hooks on the DiT blocks so they stream GPU<->CPU
        during the denoise (at most ~2 blocks resident on GPU at once).

        The fp8 DiT (~16 GB) fully GPU-resident + 720p/81-frame denoise
        activations exceed the 31 GB GPU. Streaming the blocks frees ~16 GB of
        VRAM for activations; fp8 weights are only device-cast (never
        dtype-cast) so fp8 efficiency is preserved. After the clip completes,
        _run_staged_clip restores every block to GPU (low-RAM baseline)."""
        if getattr(self, "_dit_stream_hooked", False):
            return
        blocks = getattr(pipe.dit, "blocks", [])
        prev_holder = [None]
        for blk in blocks:
            blk.register_forward_pre_hook(self._make_dit_pre(prev_holder))
            blk.register_forward_hook(self._make_dit_post())
        self._dit_stream_hooked = True
        logger.info("Staged: DiT block streaming enabled (%d blocks)", len(blocks))

    def _restore_dit_to_gpu(self, pipe):
        """Move every DiT block back onto the GPU (post-denoise), returning to
        the low-RAM baseline so preprocessing of the next request is not
        saddled with ~16 GB of fp8 blocks sitting in CPU RAM."""
        if not getattr(self, "_dit_stream_hooked", False):
            return
        pipe.dit.to(self.device)
        torch.cuda.empty_cache()

    def _load_finetuned_dit_vace(self, pipe, checkpoint_path, torch_dtype=torch.bfloat16):
        """Instantiate empty DiT + VACE and load finetuned weights (mmap)."""
        from diffsynth import load_state_dict
        from diffsynth.models.utils import init_weights_on_device
        from diffsynth.models.wan_video_dit import WanModel
        from diffsynth.models.wan_video_vace import VaceWanModel

        I2V_14B_DIT_CONFIG = {
            "has_image_input": True, "patch_size": [1, 2, 2], "in_dim": 36,
            "dim": 5120, "ffn_dim": 13824, "freq_dim": 256, "text_dim": 4096,
            "out_dim": 16, "num_heads": 40, "num_layers": 40, "eps": 1e-6,
        }
        VACE_14B_CONFIG = {
            "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35), "vace_in_dim": 96,
            "patch_size": (1, 2, 2), "has_image_input": False, "dim": 5120,
            "num_heads": 40, "ffn_dim": 13824, "eps": 1e-6,
        }

        logger.info("Instantiating empty DiT + VACE (int8 target) ...")
        with init_weights_on_device():
            pipe.dit = WanModel(**I2V_14B_DIT_CONFIG)
            pipe.vace = VaceWanModel(**VACE_14B_CONFIG)

        logger.info("Loading checkpoint %s ...", checkpoint_path)
        state_dict = load_state_dict(checkpoint_path)
        state_dict_vace = {k: v for k, v in state_dict.items() if "vace" in k}
        state_dict_dit = {k: v for k, v in state_dict.items() if "vace" not in k}
        pipe.dit.load_state_dict(state_dict_dit, assign=True)
        pipe.vace.load_state_dict(state_dict_vace, assign=True)
        pipe.dit = pipe.dit.to(dtype=torch_dtype)
        pipe.vace = pipe.vace.to(dtype=torch_dtype)
        logger.info(
            "Loaded DiT (%d params) + VACE (%d params)",
            len(state_dict_dit), len(state_dict_vace),
        )

    def _load_fp8_native(self, pipe):
        """Load the pre-quantized per-channel FP8 checkpoint WITHOUT diffsynth's loader.

        Replaces the WanModel/VaceWanModel Linear/Conv leaves with FP8Linear/FP8ConvNd
        and streams the fp8 weights in, so there is never a whole-model bf16 build:
        peak RAM = fp8-resident model (~19.5 GB) + one shard + one layer. Weights stay
        fp8 and dequantize per-layer inside each wrapper's forward.
        """
        from diffsynth.models.utils import init_weights_on_device
        from diffsynth.models.wan_video_dit import WanModel
        from diffsynth.models.wan_video_vace import VaceWanModel
        from .fp8_loader import (
            replace_quantisable_layers,
            materialize_meta,
            load_fp8_into_models,
            snapshot_fp8_checkpoint,
        )

        I2V_14B_DIT_CONFIG = {
            "has_image_input": True, "patch_size": [1, 2, 2], "in_dim": 36,
            "dim": 5120, "ffn_dim": 13824, "freq_dim": 256, "text_dim": 4096,
            "out_dim": 16, "num_heads": 40, "num_layers": 40, "eps": 1e-6,
        }
        VACE_14B_CONFIG = {
            "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35), "vace_in_dim": 96,
            "patch_size": (1, 2, 2), "has_image_input": False, "dim": 5120,
            "num_heads": 40, "ffn_dim": 13824, "eps": 1e-6,
        }

        logger.info("Instantiating empty DiT + VACE (native fp8 target) ...")
        with init_weights_on_device():
            pipe.dit = WanModel(**I2V_14B_DIT_CONFIG)
            pipe.vace = VaceWanModel(**VACE_14B_CONFIG)

        # Load the fp8 model DIRECTLY onto the compute device, never staging the
        # ~19.5 GB fp8 weights in system RAM. .8 has only ~24 GB free RAM, so a
        # CPU-staged build + the resident encoders + SAM3 subprocess OOMs (137).
        # GPU has 32 GB, so the fp8 model + denoise activations fit there.
        dev = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
        n_dit = replace_quantisable_layers(pipe.dit, device=dev)
        n_vace = replace_quantisable_layers(pipe.vace, device=dev)
        # init_weights_on_device() leaves the small non-quantised leaves on the
        # meta device; rewrite them to real device params so the loader can copy_.
        m_dit = materialize_meta(pipe.dit, device=dev, dtype=self._torch_dtype)
        m_vace = materialize_meta(pipe.vace, device=dev, dtype=self._torch_dtype)
        logger.info(
            "Replaced quantisable layers: dit=%d vace=%d; meta leaves materialised: dit=%d vace=%d",
            n_dit, n_vace, m_dit, m_vace,
        )

        shards = snapshot_fp8_checkpoint(
            config.HF_REPO, config.HF_TOKEN or None,
            subfolder=self.hf_subfolder, local_dir=config.IDV2V_MODEL_DIR,
        )
        counts = load_fp8_into_models(pipe.dit, shards, pipe.vace, device=dev)
        # Safety: any fp32 leave (not in ckpt, or a diff synth fp32 default) would
        # break the bf16 LayerNorms. Keep every non-fp8 param bf16.
        for mod_ in (pipe.dit, pipe.vace):
            for pp in mod_.parameters():
                if pp.dtype in (torch.float32, torch.float64):
                    pp.data = pp.data.to(self._torch_dtype)
        logger.info("Native FP8 loaded: %s", counts)

    def _load_fp8_from_hf(self, pipe):
        """Load pre-quantized per-channel FP8 DiT + VACE from HF_REPO.

        The pipeline (diffsynth WanVideoPipeline) has no native fp8 matmul, so we
        dequantize each weight fp8*scale -> bf16 at load and assign it into the
        empty DiT/VACE. This keeps the download/disk small (~19.5 GB vs ~78 GB)
        and preserves per-channel FP8 quality WITHOUT the runtime int8 conversion.
        """
        from diffsynth.models.utils import init_weights_on_device
        from diffsynth.models.wan_video_dit import WanModel
        from diffsynth.models.wan_video_vace import VaceWanModel

        I2V_14B_DIT_CONFIG = {
            "has_image_input": True, "patch_size": [1, 2, 2], "in_dim": 36,
            "dim": 5120, "ffn_dim": 13824, "freq_dim": 256, "text_dim": 4096,
            "out_dim": 16, "num_heads": 40, "num_layers": 40, "eps": 1e-6,
        }
        VACE_14B_CONFIG = {
            "vace_layers": (0, 5, 10, 15, 20, 25, 30, 35), "vace_in_dim": 96,
            "patch_size": (1, 2, 2), "has_image_input": False, "dim": 5120,
            "num_heads": 40, "ffn_dim": 13824, "eps": 1e-6,
        }

        logger.info("Instantiating empty DiT + VACE (hf-fp8 target) ...")
        with init_weights_on_device():
            pipe.dit = WanModel(**I2V_14B_DIT_CONFIG)
            pipe.vace = VaceWanModel(**VACE_14B_CONFIG)

        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        logger.info("Resolving FP8 checkpoint shards ... variant=%s", self.variant)
        from .fp8_loader import snapshot_fp8_checkpoint
        shards = snapshot_fp8_checkpoint(
            config.HF_REPO, config.HF_TOKEN or None,
            subfolder=self.hf_subfolder, local_dir=config.IDV2V_MODEL_DIR,
        )
        if not shards:
            raise FileNotFoundError(f"No FP8 .safetensors found for variant {self.variant!r}")

        dit_sd, vace_sd = {}, {}
        n_fp8 = 0
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as f:
                keys = f.keys()
                for k in keys:
                    if k.endswith(".weight_scale") or k.endswith(".comfy_quant"):
                        continue
                    if k.endswith(".weight") and (k[:-len(".weight")] + ".weight_scale") in keys:
                        qdata = f.get_tensor(k)                      # fp8_e4m3fn
                        scale = f.get_tensor(k[:-len(".weight")] + ".weight_scale")  # [Cout,1,...] bf16
                        val = (qdata.to(torch.float32) * scale.to(torch.float32)).to(self._torch_dtype)
                        n_fp8 += 1
                    else:
                        val = f.get_tensor(k).to(self._torch_dtype)  # plain bf16 param
                    if k.startswith("vace."):
                        vace_sd[k[len("vace."):]] = val
                    elif k.startswith("dit."):
                        dit_sd[k[len("dit."):]] = val
                    else:
                        dit_sd[k] = val

        logger.info("Dequantized %d fp8 weights from %d shards", n_fp8, len(shards))
        pipe.dit.load_state_dict(dit_sd, assign=True)
        pipe.vace.load_state_dict(vace_sd, assign=True)
        pipe.dit = pipe.dit.to(dtype=self._torch_dtype)
        pipe.vace = pipe.vace.to(dtype=self._torch_dtype)

    def _quantize_int8(self):
        """Quantize the video model (DiT + VACE) to int8 weights in-place."""
        if not hasattr(self, "_torchao") or self._torchao is None:
            return
        from torchao.quantization import int8_weight_only, quantize_

        q = int8_weight_only()
        logger.info("Quantizing video model (DiT + VACE) to int8 ...")
        t0 = time.time()
        quantize_(self._pipe.dit, q)
        if self._pipe.vace is not None:
            quantize_(self._pipe.vace, q)
        torch.cuda.empty_cache()
        logger.info("int8 quantization done in %.1fs", time.time() - t0)

    def _enable_offload(self):
        """Ensure pipeline is on the offloaded (CPU) execution path."""
        if config.IDV2V_OFFLOAD:
            hook = getattr(self._pipe, "enable_model_cpu_offload", None)
            if callable(hook):
                try:
                    hook()
                except Exception as exc:  # pragma: no cover - optional
                    logger.warning(
                        "model_cpu_offload hook failed (%s); using vram_management", exc
                    )

    # -- staged (RAM-bounded) lifecycle --------------------------------------
    def _release_encoder(self, pipe, model_name, attr):
        """Truly drop a model from RAM.

        The pipeline's ModelManager is local to from_pretrained and discarded, so
        the only hard ref to each encoder is pipe.<attr> (and pipe.prompter for
        T5). Clearing those + gc is enough to free the ~11 GB T5 / ~2.5 GB CLIP.
        Reload later is done by _ensure_text_encoder/_ensure_image_encoder from
        the captured on-disk paths. Returns True.
        """
        if getattr(pipe, attr, None) is not None:
            setattr(pipe, attr, None)
        if model_name == "wan_video_text_encoder":
            pipe.prompter = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Staged: released %s from RAM", model_name)
        return True

    def _ensure_text_encoder(self, pipe):
        """Lazily reload the T5 text-encoder + prompter (evicted at /load) for the
        prompt-embed phase. Called only on the first clip; released right after."""
        if pipe.text_encoder is not None:
            return
        from diffsynth.models.model_manager import ModelManager
        from diffsynth.prompters import WanPrompter
        path = self._enc_paths.get("wan_video_text_encoder")
        if not path or not self._tokenizer_path:
            raise RuntimeError("T5 text-encoder path/tokenizer not captured")
        mgr = ModelManager()
        mgr.load_model(path, device="cpu", torch_dtype=self._torch_dtype)
        pipe.text_encoder = mgr.fetch_model("wan_video_text_encoder")
        pipe.prompter = WanPrompter(tokenizer_path=self._tokenizer_path)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(self._tokenizer_path)
        pipe.text_encoder = pipe.text_encoder.to(device=self.device, dtype=self._torch_dtype)
        logger.info("Staged: reloaded T5 text-encoder + prompter on %s", self.device)

    def _ensure_image_encoder(self, pipe):
        """Lazily reload the CLIP image-encoder (evicted at /load) for the
        ImageEmbedder phase. Released again after the last clip."""
        if pipe.image_encoder is not None:
            return
        from diffsynth.models.model_manager import ModelManager
        path = self._enc_paths.get("wan_video_image_encoder")
        if not path:
            raise RuntimeError("CLIP image-encoder path not captured")
        mgr = ModelManager()
        mgr.load_model(path, device="cpu", torch_dtype=self._torch_dtype)
        pipe.image_encoder = mgr.fetch_model("wan_video_image_encoder")
        pipe.image_encoder = pipe.image_encoder.to(device=self.device, dtype=self._torch_dtype)
        logger.info("Staged: reloaded CLIP image-encoder on %s", self.device)

    @staticmethod
    def _safe_load_to_device(pipe, model_names):
        """Like BasePipeline.load_models_to_device but skips sub-models that are
        None (e.g. the freed T5). Preserves its managed-vs-plain offload/onload
        semantics so GPU residency matches the ordinary pipeline exactly."""
        model_names = set(model_names)
        for name, model in pipe.named_children():
            if model is None:
                continue
            if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                for module in model.modules():
                    if not hasattr(module, "offload"):
                        continue
                    if name in model_names:
                        module.onload()
                    else:
                        module.offload()
            else:
                if name in model_names:
                    model.to(pipe.device)
                else:
                    model.cpu()
        torch.cuda.empty_cache()

    def _run_staged_clip(
        self, pipe, *, prompt, negative_prompt, input_image, random_ref_frame,
        vace_video, vace_video_mask, seed, num_inference_steps, cfg_scale,
        vace_scale, width, height, num_frames, ref_pad_num, first_clip,
        last_clip, _prog=None,
    ):
        """Run ONE clip through the staged phases, replicating diffsynth's
        WanVideoPipeline.__call__ (L604-742) but holding only the sub-models each
        phase needs. The T5 is used only for the first clip's text encode and is
        freed before the native-fp8 DiT/VACE load; CLIP/VAE stay resident and the
        DiT/VACE stay resident from clip 0 onward (peak CPU RAM ~= 22 GB)."""
        from diffsynth.pipelines.wan_video_new_multiVace_svi import (
            WanVideoUnit_PromptEmbedder,
            WanVideoUnit_ImageEmbedder,
        )

        scheduler = pipe.scheduler
        scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=5.0)

        inputs_posi = {
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": None,
            "input_video": None, "denoising_strength": 1.0,
            "control_video": None, "reference_image": None,
            "camera_control_direction": None, "camera_control_speed": 1 / 54,
            "camera_control_origin": (
                0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0,
                0, 0, 0, 1, 0, 0, 0, 0, 1, 0,
            ),
            "vace_video": vace_video, "vace_video_mask": vace_video_mask,
            "vace_reference_image": None, "vace_scale": vace_scale,
            "seed": seed, "rand_device": "cpu",
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": False, "sigma_shift": 5.0,
            "motion_bucket_id": None,
            # Tiled attention caps denoise activation VRAM — required to fit the
            # fp8 DiT+VACE (~19.5 GB) + 720p/81-frame activations in 31 GB GPU.
            "tiled": True, "tile_size": (30, 52), "tile_stride": (15, 26),
            "sliding_window_size": None, "sliding_window_stride": None,
            "use_multi_control_vace": True,
            "ref_pad_num": ref_pad_num, "random_ref_frame": random_ref_frame,
        }

        # -- unit phase (encode conditioning into the inputs_* dicts) --------
        for unit in pipe.units:
            if isinstance(unit, WanVideoUnit_PromptEmbedder):
                if first_clip:
                    # Reload T5 (evicted at /load) just for this text-encode.
                    self._ensure_text_encoder(pipe)
                    inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                        unit, pipe, inputs_shared, inputs_posi, inputs_nega
                    )
                    # Cache the per-side T5 contexts (same prompt every clip).
                    self._posi_context = inputs_posi.get("context")
                    self._nega_context = inputs_nega.get("context")
                    # Truly release T5 + prompter from RAM (manager-aware) so it is
                    # NOT resident through the rest of encode/denoise. This is the
                    # ~11 GB win that keeps 720p/81-frame requests under the
                    # ~24 GB RAM ceiling of .8.
                    self._release_encoder(pipe, "wan_video_text_encoder", "text_encoder")
                else:
                    # Reuse clip-0 context; the T5 is no longer needed.
                    inputs_posi["context"] = self._posi_context
                    inputs_nega["context"] = self._nega_context
                continue
            if isinstance(unit, WanVideoUnit_ImageEmbedder):
                self._ensure_image_encoder(pipe)
            inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
                unit, pipe, inputs_shared, inputs_posi, inputs_nega
            )

        # Free the CLIP image-encoder on the LAST clip as soon as its embedding
        # is done — BEFORE the fp8 DiT/VACE loads — so the GPU is clean when the
        # ~16 GB fp8 model materializes on it. (Previously clip was released only
        # AFTER the dit load, stealing ~2.5 GB of VRAM at exactly the wrong time.)
        # Only on the last clip because earlier clips' ImageEmbedder re-runs.
        if last_clip:
            self._release_encoder(pipe, "wan_video_image_encoder", "image_encoder")

        # -- stage memory: bring up DiT+VACE (T5 + CLIP already released) -------
        self._ensure_dit_vace()             # loads native-fp8 DiT/VACE on GPU

        # -- denoise loop (replicates __call__ L709-726) ---------------------
        self._safe_load_to_device(pipe, pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        # Re-establish the DiT-block CPU offload AFTER _safe_load_to_device:
        # that call moved the whole pipe.dit (all 40 blocks) back onto the GPU,
        # which would otherwise strand the wrapped blocks' fp8 weights on cuda
        # while their inputs hop to CPU (device mismatch in self_attn.q). Running
        # every clip keeps the offloaded blocks resident on CPU for the whole
        # denoise; the forward wrap is idempotent and the blk.to("cpu") is cheap.
        self._enable_block_cpu_offload(
            pipe, num_blocks=int(os.environ.get("IDV2V_DIT_CPU_BLOCKS", "16")))
        # The VAE is not used during the denoise (only VAE-encoded latents are,
        # which are already on GPU) — evict it to CPU to give the 720p/81-frame
        # denoise activations ~0.5 GB more VRAM. VACE runs on GPU; the DiT-block
        # CPU offload (IDV2V_DIT_CPU_BLOCKS, default 16) funds VACE's ~2.88 GB
        # rope_apply self-attention peak so model+activations stay under 31 GB.
        # VAE is not used during the denoise; evict it to CPU for ~0.5 GB more VRAM.
        # VACE stays on GPU (it is in in_iteration_models) and runs fast; the
        # DiT-block CPU offload funds VACE's rope_apply activation peak.
        if pipe.vae is not None:
            pipe.vae.cpu()
        torch.cuda.empty_cache()
        # The bf16/fp8 model runs in bf16 and cannot take fp32 conditionings
        # (LayerNorms fail with "expected scalar type BFloat16 but found Float").
        # Cast once for the constant conditionings; the noisy latent is re-cast
        # each iteration because the scheduler may return a different dtype.
        # NOTE: use self._torch_dtype (bf16), NOT pipe.torch_dtype — with a fp8
        # DiT diffsynth's dtype detection can drift (e.g. to float32).
        cvt = lambda t: t.to(dtype=self._torch_dtype, device=pipe.device)
        for kk in ("context", "clip_feature", "y", "vace_context", "input_latents"):
            v = inputs_shared.get(kk)
            if isinstance(v, torch.Tensor):
                inputs_shared[kk] = cvt(v)
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
                inputs_shared[kk] = [cvt(t) for t in v]
        def bf16_model_fn(**kw):
            """model_fn_wan_video shim: cast every floating-point conditioning
            tensor to bf16 before it reaches the fp8/bf16 DiT. Catches any fp32
            that leaks in regardless of which dict it arrived through."""
            for k, v in kw.items():
                if isinstance(v, torch.Tensor) and v.is_floating_point():
                    if v.dtype not in (self._torch_dtype, torch.float8_e4m3fn):
                        kw[k] = v.to(self._torch_dtype)
                elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
                    kw[k] = [
                        t.to(self._torch_dtype) if t.is_floating_point() and t.dtype != self._torch_dtype else t
                        for t in v
                    ]
            return pipe.model_fn(**kw)

        for progress_id, timestep in enumerate(scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=self._torch_dtype, device=pipe.device)
            inputs_shared["latents"] = inputs_shared["latents"].to(
                dtype=self._torch_dtype, device=pipe.device
            )
            noise_pred_posi = bf16_model_fn(
                **models, **inputs_shared, **inputs_posi, timestep=timestep,
            )
            if cfg_scale != 1.0:
                noise_pred_nega = bf16_model_fn(
                    **models, **inputs_shared, **inputs_nega, timestep=timestep,
                )
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi
            inputs_shared["latents"] = scheduler.step(
                noise_pred, scheduler.timesteps[progress_id], inputs_shared["latents"]
            )
            if _prog is not None:
                _prog((progress_id + 1) / len(scheduler.timesteps))

        if _prog is not None:
            _prog(1.0, "decoding", "Decoding video...")
        # -- decode (VAE reloaded for decode; __call__ L734-738) --------------
        self._safe_load_to_device(pipe, ["vae"])
        video = pipe.vae.decode(
            inputs_shared["latents"], device=pipe.device, tiled=False,
            tile_size=(30, 52), tile_stride=(15, 26),
        )
        video = pipe.vae_output_to_video(video)
        self._safe_load_to_device(pipe, [])
        return video

    def _infer_staged(
        self, *, prompt, negative_prompt, input_image, condition_videos,
        keyframes=None, width=1280, height=720, num_frames=81,
        max_frames=None, num_inference_steps=30, cfg_scale=5.0,
        vace_scale=1.0, ref_pad_num=-1, seed=123,
        progress_cb=None,
    ):
        """Staged RAM-bounded clip-by-clip generation (same output as infer()).

        The T5 text-encoder is loaded, used once for the prompt context, then
        freed before the native-fp8 DiT/VACE load, so peak RAM stays ~22 GB.
        """
        pipe = self._pipe
        keyframes = keyframes or []
        from PIL import Image

        if max_frames is not None:
            condition_videos = [c[:max_frames] for c in condition_videos]
        total_frames = len(condition_videos[0])

        white = Image.new("RGB", (width, height), (255, 255, 255))
        black = Image.new("RGB", (width, height), (0, 0, 0))
        kf_set = {idx for idx, _ in keyframes}
        full_mask = [black if i in kf_set else white for i in range(total_frames)]

        for idx, kf_img in keyframes:
            resized = center_crop_and_resize(kf_img, width, height)
            for c in condition_videos:
                c[idx] = resized

        clip_schedule = compute_clip_schedule(total_frames, num_frames)
        logger.info("Clip schedule (%d clips): %s", len(clip_schedule), clip_schedule)

        self._posi_context = None
        self._nega_context = None

        all_clips = []
        current_input_image = input_image

        with torch.no_grad():
            for clip_idx, (frame_start, frame_end) in enumerate(clip_schedule):
                clip_seed = seed
                if clip_idx > 0:
                    splice_idx = clip_schedule[clip_idx][0] - clip_schedule[clip_idx - 1][0]
                    current_input_image = all_clips[-1][splice_idx]

                clip_end = frame_start + num_frames
                clip_conditions = [
                    slice_frames(c, frame_start, clip_end) for c in condition_videos
                ]
                clip_mask_single = slice_frames(full_mask, frame_start, clip_end)
                clip_mask = [clip_mask_single] * len(condition_videos)

                logger.info("Clip %d/%d frames=[%d,%d) seed=%d",
                            clip_idx + 1, len(clip_schedule), frame_start, frame_end,
                            clip_seed)
                def _prog(within, stage="generating", message=None):
                    if progress_cb is None:
                        return
                    overall = (clip_idx + within) / len(clip_schedule)
                    if message is None:
                        step_no = min(num_inference_steps, int(round(within * num_inference_steps)))
                        message = f"clip {clip_idx+1}/{len(clip_schedule)} step {step_no}/{num_inference_steps}"
                    try:
                        progress_cb(overall, stage, message)
                    except Exception:
                        pass
                generated = self._run_staged_clip(
                    pipe,
                    prompt=prompt, negative_prompt=negative_prompt,
                    input_image=current_input_image, random_ref_frame=input_image,
                    vace_video=clip_conditions, vace_video_mask=clip_mask,
                    seed=clip_seed, num_inference_steps=num_inference_steps,
                    cfg_scale=cfg_scale, vace_scale=vace_scale,
                    width=width, height=height, num_frames=num_frames,
                    ref_pad_num=ref_pad_num, first_clip=(clip_idx == 0),
                    last_clip=(clip_idx == len(clip_schedule) - 1),
                    _prog=_prog,
                )
                all_clips.append(generated)

        if len(clip_schedule) == 1:
            combined = all_clips[0]
            if total_frames < num_frames:
                combined = combined[:total_frames]
        else:
            combined = list(all_clips[0])
            for i in range(1, len(clip_schedule)):
                overlap = clip_schedule[i - 1][1] - clip_schedule[i][0]
                combined = combined[:-overlap] + list(all_clips[i])

        logger.info("Stitched %d clips into %d frames",
                    len(clip_schedule), len(combined))
        return combined

    # -- inference ----------------------------------------------------------
    def infer(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        input_image,                    # PIL stylized first frame (I2V anchor)
        condition_videos,               # list of frame-lists (VACE control videos)
        keyframes=None,                 # list of (frame_index:int, image:PIL)
        width: int = 1280,
        height: int = 720,
        num_frames: int = 81,           # frames per clip
        max_frames: int | None = None,  # cap total output frames
        num_inference_steps: int = 30,
        cfg_scale: float = 5.0,
        vace_scale: float = 1.0,
        ref_pad_num: int = -1,
        seed: int = 123,
        progress_cb=None,
    ):
        """Run ID-V2V clip-by-clip generation (ported from the reference).

        Returns: list of PIL RGB frames (the combined video).
        """
        pipe = self._pipe
        if config.IDV2V_STAGED:
            return self._infer_staged(
                prompt=prompt, negative_prompt=negative_prompt,
                input_image=input_image, condition_videos=condition_videos,
                keyframes=keyframes, width=width, height=height,
                num_frames=num_frames, max_frames=max_frames,
                num_inference_steps=num_inference_steps, cfg_scale=cfg_scale,
                vace_scale=vace_scale, ref_pad_num=ref_pad_num, seed=seed,
                progress_cb=progress_cb,
            )
        keyframes = keyframes or []   # [(index, PIL image), ...]
        from PIL import Image

        if max_frames is not None:
            condition_videos = [c[:max_frames] for c in condition_videos]
        total_frames = len(condition_videos[0])

        white = Image.new("RGB", (width, height), (255, 255, 255))
        black = Image.new("RGB", (width, height), (0, 0, 0))
        kf_set = {idx for idx, _ in keyframes}
        full_mask = [black if i in kf_set else white for i in range(total_frames)]

        for idx, kf_img in keyframes:
            resized = center_crop_and_resize(kf_img, width, height)
            for c in condition_videos:
                c[idx] = resized

        clip_schedule = compute_clip_schedule(total_frames, num_frames)
        logger.info("Clip schedule (%d clips): %s", len(clip_schedule), clip_schedule)

        all_clips = []
        current_input_image = input_image

        with torch.no_grad():
            for clip_idx, (frame_start, frame_end) in enumerate(clip_schedule):
                clip_seed = seed
                if clip_idx > 0:
                    splice_idx = clip_schedule[clip_idx][0] - clip_schedule[clip_idx - 1][0]
                    current_input_image = all_clips[-1][splice_idx]

                clip_end = frame_start + num_frames
                clip_conditions = [
                    slice_frames(c, frame_start, clip_end) for c in condition_videos
                ]
                clip_mask_single = slice_frames(full_mask, frame_start, clip_end)
                clip_mask = [clip_mask_single] * len(condition_videos)

                logger.info("Clip %d/%d frames=[%d,%d) seed=%d",
                            clip_idx + 1, len(clip_schedule), frame_start, frame_end,
                            clip_seed)
                generated = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    input_image=current_input_image,
                    random_ref_frame=input_image,
                    ref_pad_num=ref_pad_num,
                    vace_video=clip_conditions,
                    vace_video_mask=clip_mask,
                    seed=clip_seed,
                    num_inference_steps=num_inference_steps,
                    use_multi_control_vace=True,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    cfg_scale=cfg_scale,
                    tiled=False,
                    vace_scale=vace_scale,
                )
                all_clips.append(generated)

        if len(clip_schedule) == 1:
            combined = all_clips[0]
            if total_frames < num_frames:
                combined = combined[:total_frames]
        else:
            combined = list(all_clips[0])
            for i in range(1, len(clip_schedule)):
                overlap = clip_schedule[i - 1][1] - clip_schedule[i][0]
                combined = combined[:-overlap] + list(all_clips[i])

        logger.info("Stitched %d clips into %d frames",
                    len(clip_schedule), len(combined))
        return combined

    @property
    def is_ready(self) -> bool:
        return self._pipe is not None

    @property
    def precision(self) -> str:
        if config.IDV2V_SOURCE == "hf-fp8":
            return "fp8-from-hf"
        return config.IDV2V_QUANT


def health_check(model) -> dict:
    """Health payload — model status."""
    return {
        "status": "ok" if model.is_ready else "loading",
        "device": model.device,
        "model_loaded": model.is_ready,
        "precision": model.precision,
        "variant": model.variant,
        "hf_subfolder": model.hf_subfolder,
        "offload": config.IDV2V_OFFLOAD,
    }

