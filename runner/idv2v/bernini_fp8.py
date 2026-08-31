"""fp8-aware loader/fill for the full Bernini-14b ("Bernini-R-Diffusers-v2")
checkpoint, used by the wan-worker Bernini rail.

WHY (the upstream loader can't read this checkpoint):
  * ``BerniniModel.from_pretrained`` goes through transformers' standard
    meta-load, which assigns raw tensors into the model's float ``Linear``
    weights. The 14b-quantised checkpoint stores the Wan renderer weights
    (``diff_dec.*`` / ``diff_dec_low.*``, ndim>=2) as *weight-only fp8*:
    the raw e4m3 bits in a ``uint8`` safetensors tensor plus a per-output-
    channel float32 ``_scale`` sibling (``<name>.weight`` /
    ``<name>.weight_scale``). ``uint8`` cannot be assigned into a float
    Linear -> RuntimeError "Only Tensors of floating point ... can require
    gradients".
  * Everything else in the checkpoint (mllm, t5, vit_decoder, connector,
    norms/biases) is already bf16 and loads like the 1.3b.

WHY fp8-resident (not dequant-to-bf16-on-load):
  * The renderer is ~28.6B params; bf16 would be ~57GB - it cannot fit the
    32GB RTX 5090. Storing the fp8 weights resident (dequant per-layer in
    forward) keeps the renderer ~28.6GB / half of bf16, matching the quant
    config's "storage-only; no GEMM speedup" intent.

Recall the documented decode:
    w = (u8.view-as-e4m3.to(float32) * scale[:,None]).to(bf16)   # [out,in]
"""

from __future__ import annotations

import gc
import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger("video_creator.runner.idv2v.bernini_fp8")

# Diagnostic A/B (2026-08-30): force bf16 activations — skip the per-GEMM
# fp8-activation _scaled_mm path and instead dequant the fp8 weights to bf16
# and matmul in bf16 (weights stay fp8 for storage). Isolates whether the
# fp8-activation quantization path is the ~126x latent-inflation source.
FORCE_BF16_ACTIVATIONS = False


# ---------------------------------------------------------------------------
# fp8-resident layers: weight stays e4m3 on the compute device, dequant to the
# activation dtype per call (per-output-channel scale broadcasts over the full
# weight shape).
# ---------------------------------------------------------------------------


class FP8Linear(nn.Module):
    """fp8-resident Linear backed by native FP8 tensor-core GEMM via
    ``torch._scaled_mm`` (Blackwell executes fp8 without dequantising to bf16).

    Storage: ``weight`` is e4m3 ``(K,N)`` -- already transposed from the
    checkpoint's ``(N,K)`` at fill time -- so ``_scaled_mm(xq (M,K), weight
    (K,N))`` needs no runtime transpose or duplicate copy. ``scale`` is the
    per-output-channel scale ``(N,)``. ``forward`` quantises the activation to
    fp8 per-tensor, runs the fp8 GEMM with ``scale_b=1`` (this torch accepts
    only per-tensor scales), rescales the output by the per-channel weight
    scale, then adds bias. Dims not 16-aligned (a ``_scaled_mm`` requirement)
    fall back to the bf16 dequant path.
    """

    _F8_MAX = 448.0  # float8_e4m3fn finite max

    def __init__(self, in_features: int, out_features: int, has_bias: bool = False,
                 device: Optional[str] = None):
        super().__init__()
        dev = device or "cpu"
        self.in_features, self.out_features = in_features, out_features
        self.weight = nn.Parameter(
            torch.empty(in_features, out_features, dtype=torch.float8_e4m3fn, device=dev),
            requires_grad=False,
        )
        # Bernini fp8 scale convention: scale.shape == weight.shape[:-1] (drops the
        # last / in-dim). For the (K,N) weight that's (N,) -- per-output-channel.
        self.scale = nn.Parameter(
            torch.ones(out_features, dtype=torch.bfloat16, device=dev), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, dtype=torch.bfloat16, device=dev),
                         requires_grad=False)
            if has_bias
            else None
        )
        # Optional runtime low-rank LoRA term (rzgar LightX2V 4-step adapter):
        #   lora_a shape (rank, in), lora_b shape (out, rank) -- bf16
        #   forward adds alpha * lora_b(lora_a(x)).
        # We keep the fp8 base GEMM untouched and add the low-rank term on the
        # bf16 output, so the native-res fp8 VRAM budget is preserved (dequant-
        # to-bf16-and-apply would balloon ~14GB->~28GB and OOM at 848x480).
        self.lora_a = None
        self.lora_b = None
        self.lora_alpha = 1.0
        # Optional full-rank delta (e.g. the output-head adapter `head.head.diff`
        # (N,K)): added as x @ lora_full.t() -- separate from the low-rank term.
        self.lora_full = None
        # _scaled_mm requires K (trailing dim of mat1) and N % 16 == 0.
        self._scaled = bool(getattr(torch, "_scaled_mm", None)) and (
            in_features % 16 == 0 and out_features % 16 == 0
        )
        self._warned = False

    def _apply_lora(self, xin: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Add alpha * lora_b(lora_a(xin)) to an already-computed output.

        xin: (M, K) input (pre-projection). out: (M, N) fp8/base output.
        No-op when no LoRA is attached (common case -> zero overhead).
        """
        if self.lora_a is None:
            return out
        # Device-agnostic LoRA term. lora_a/lora_b/lora_full are plain tensors (not
        # nn.Parameters), so the pipeline's later .to(cuda) does NOT move them; and
        # apply() may have parked them on CPU (the renderer builds on CPU then moves
        # at render). Move each to the INPUT's device+dtype here at forward time so
        # no CPU tensor ever leaks into the matmul ("mat2 is on cpu").
        dev, d = xin.device, xin.dtype
        h = xin @ self.lora_a.to(device=dev, dtype=d).t()  # (M, rank)
        term = self.lora_alpha * (h @ self.lora_b.to(device=dev, dtype=d).t())  # (M, N)
        out = out + term
        if self.lora_full is not None:
            out = out + xin @ self.lora_full.to(device=dev, dtype=d).t()
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # VeOmni's fp8 Wan path can feed float8 activations; normalize to a real
        # dtype for both paths (the rest of the model runs bf16).
        if x.dtype == torch.float8_e4m3fn:
            x = x.to(torch.bfloat16)
        if not FORCE_BF16_ACTIVATIONS and self._scaled and x.is_cuda and self.weight.is_cuda:
            try:
                N, K = self.out_features, self.in_features
                orig = x.shape
                x2 = x.contiguous().reshape(-1, K)
                amax = x2.abs().amax().clamp_min(1e-12).float()
                q = (self._F8_MAX / amax).float()
                xq = (x2.float() * q).to(torch.float8_e4m3fn)
                o = self._scaled_mm_padded(xq, q, N, K)
                o = o * self.scale.to(torch.bfloat16).unsqueeze(0)  # per-channel rescale
                if self.bias is not None:
                    o = o + self.bias.unsqueeze(0).to(torch.bfloat16)
                o = self._apply_lora(x2, o)
                return o.reshape(*orig[:-1], N)
            except Exception as exc:  # noqa: BLE001 - fall back to bf16 dequant
                if not self._warned:
                    self._warned = True
                    logger.warning("bernini_fp8: scaled_mm py (%s) on %s", exc,
                                   f"{self.in_features}x{self.out_features}")
        w = self.weight.t().to(x.dtype) * self.scale.to(x.dtype).unsqueeze(-1)  # (N,K) dequant
        b = self.bias.to(x.dtype) if self.bias is not None else None
        out = F.linear(x, w, b)
        if self.lora_a is not None:
            xin = x.reshape(-1, self.in_features)
            out = self._apply_lora(xin, out.reshape(-1, self.out_features)).reshape(
                x.shape[:-1] + (self.out_features,)
            )
        return out



    def _scaled_mm_padded(self, xq, q, N, K):
        """torch._scaled_mm with a shape-alignment fallback: pad M/K to 16 and
        retry before the bf16 dequant path (the fp8 tensor-core path rejects
        misaligned dims, e.g. the 256x5120 text cross-attn projection)."""
        ones = torch.ones((), device=xq.device, dtype=torch.float32)
        scale_a = (1.0 / q).reshape(())
        try:
            return torch._scaled_mm(xq, self.weight, scale_a=scale_a,
                                    scale_b=ones, out_dtype=torch.bfloat16)
        except Exception:
            Mc = xq.shape[0]
            Mp, Kp = ((Mc + 15) // 16) * 16, ((K + 15) // 16) * 16
            if Mp == Mc and Kp == K:
                raise
            xq_p = torch.zeros((Mp, Kp), device=xq.device, dtype=xq.dtype)
            xq_p[:Mc, :K] = xq
            w_p = torch.zeros((Kp, N), device=self.weight.device, dtype=self.weight.dtype)
            w_p[:K, :N] = self.weight
            o = torch._scaled_mm(xq_p, w_p, scale_a=scale_a, scale_b=ones,
                                 out_dtype=torch.bfloat16)
            return o[:Mc, :N]


class FP8ConvNd(nn.Module):
    """Conv wrapper for patch-embed / proj convs (1D/2D). fp8 [Cout,Cin,*ks], scale [Cout,1,..]."""

    def __init__(self, weight_shape, has_bias: bool, stride, padding, device: Optional[str] = None):
        super().__init__()
        dev = device or "cpu"
        self.weight = nn.Parameter(
            torch.empty(weight_shape, dtype=torch.float8_e4m3fn, device=dev), requires_grad=False
        )
        # Bernini fp8 scale convention: scale.shape == weight.shape[:-1]
        self.scale = nn.Parameter(
            torch.ones(weight_shape[:-1], dtype=torch.bfloat16, device=dev), requires_grad=False
        )
        self.bias = (
            nn.Parameter(torch.empty(weight_shape[0], dtype=torch.bfloat16, device=dev),
                         requires_grad=False)
            if has_bias
            else None
        )
        self.stride, self.padding = stride, padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.float8_e4m3fn:
            x = x.to(torch.bfloat16)
        w = self.weight.to(x.dtype) * self.scale.to(x.dtype).unsqueeze(-1)
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


def replace_quantisable_layers(model: nn.Module, device: Optional[str] = None) -> int:
    """Swap quantisable leaves (Linear/Conv) for fp8 wrappers in place; returns count."""
    n = 0
    for name, child in list(model.named_children()):
        cls = _quantisable_layer(child)
        if cls is not None:
            has_bias = getattr(child, "bias", None) is not None
            if cls == "FP8Linear":
                new = FP8Linear(child.in_features, child.out_features, has_bias, device)
            else:
                new = FP8ConvNd(tuple(child.weight.shape), has_bias,
                                child.stride, child.padding, device)
            setattr(model, name, new)
            n += 1
        else:
            n += replace_quantisable_layers(child, device)
    return n


# ---------------------------------------------------------------------------
# Streaming fill (one shard at a time; fp8 weight + scale resident).
# ---------------------------------------------------------------------------


def decode_fp8_weight(u8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """u8 (e4m3 bits) * per-row scale -> bf16, any ndim.

    Checkpoint convention: scale.shape == weight.shape[:-1] (per output channel
    for 2-D, e.g. the 3-D adaLN scale_shift_table has weight (1,6,5120) and
    scale (1,6)). Broadcast over the trailing dim for ANY ndim by reshaping
    scale to weight.shape[:-1]+(1,). ``scale[:, None]`` is WRONG for ndim>2: it
    inserts a new axis mid-shape ((1,6) -> (1,1,6)) instead of at the end, which
    mis-broadcasts -- the original green-video root cause.
    """
    return (u8.view(torch.float8_e4m3fn).to(torch.float32)
            * scale.to(torch.float32).reshape(u8.shape[:-1] + (1,))).to(torch.bfloat16)


def collect_scale_keys(shards: List[str]) -> set:
    scales: set = set()
    for sh in shards:
        try:
            from safetensors import safe_open
            with safe_open(sh, framework="pt", device="cpu") as f:
                scales.update(k for k in f.keys() if k.endswith("_scale"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("collect_scale_keys shard %s: %s", os.path.basename(sh), exc)
    return scales


def _read_shard(sh: str):
    """Load every tensor of one safetensors shard into a dict (CPU).

    Runs in a worker thread for the parallel path. Returns (shard_path,
    {name: tensor}) dict. With mmap-backed safe_open this is mostly zero-copy
    views; the real disk->host reads happen when the main thread copies each
    tensor into the model (device transfer), so the window of host RAM held is
    bounded by the number of shards in flight (= jobs).
    """
    from safetensors import safe_open

    with safe_open(sh, framework="pt", device="cpu") as f:
        return sh, {k: f.get_tensor(k) for k in f.keys()}


def _fill_shard(params: Dict, tens: Dict, counts: Dict, device: Optional[str],
                layout: str, in_scope, scale_keys: set) -> None:
    """Copy one already-loaded shard's tensors into the model params.

    Shared by the serial and parallel fill paths so both produce identical
    results; counts is mutated in place.
    """
    for k in sorted(tens.keys()):
        if not in_scope(k):
            continue
        if k.endswith("_scale"):
            continue  # handled with its weight
        p = params.get(k)
        if p is None:
            counts["skipped"] += 1
            continue
        # fp8 marker is the `_scale` sibling, not a `.weight` suffix:
        # non-Linear fp8 weights (e.g. `scale_shift_table`, an nn.Parameter
        # with an adaLN `_scale`) have keys that do NOT end in `.weight`.
        # The old `k.endswith(".weight")` gate plain-copied their raw uint8
        # e4m3 BYTES into the bf16 param, corrupting the modulation table
        # -> every block's adaLN scale/shift (incl. timestep conditioning)
        # was garbage -> ~186x velocity / flat-green latent.
        is_fp8 = (k + "_scale") in scale_keys
        if is_fp8:
            w = tens[k]
            s = tens[k + "_scale"]
            if p.dtype == torch.float8_e4m3fn:
                wq = w.view(torch.float8_e4m3fn)
                # FP8Linear stores (K,N) = transpose of the legacy
                # checkpoint's (N,K) so _scaled_mm needs no runtime
                # transpose/dup; only 2-D linear weights are transposed
                # (convs stay as-is). Pre-baked ("k_n") shards are already
                # (K,N) -> straight copy, no transpose (a shape heuristic
                # can't distinguish square K==N orientations; the manifest
                # `layout` field disambiguates).
                if layout != "k_n" and p.dim() == 2 and p.shape[0] == w.shape[1] and p.shape[1] == w.shape[0]:
                    wq = wq.t().contiguous()
                # fp8-resident wrapper: weight + separate per-channel scale
                p.data.copy_(wq.to(device or p.device))
                sp = params.get(k[: -len(".weight")] + ".scale")
                if sp is not None:
                    sp.data.copy_(s.to(sp.dtype).to(device or sp.device))
                    counts["fp8scale"] += 1
                counts["fp8"] += 1
            else:
                # plain param that was fp8-quantized: dequant -> bf16
                p.data.copy_(decode_fp8_weight(w, s).to(p.dtype).to(device or p.device))
                counts["dequant"] += 1
            del w, s
        else:
            p.data.copy_(tens[k].to(p.dtype).to(device or p.device))
            counts["plain"] += 1


def stream_fill(
    model: nn.Module,
    shards: List[str],
    scale_keys: Optional[set] = None,
    device: Optional[str] = None,
    prefixes: Optional[List[str]] = None,
    layout: str = "n_k",
    jobs: Optional[int] = None,
) -> Dict[str, int]:
    """Fill the (renderer) module weights directly from the fp8 safetensors shards.

    - ``<name>.weight`` + a ``_scale`` sibling -> fp8-resident (weight e4m3 +
      separate scale), or dequant to bf16 when the param isn't an fp8 wrapper.
    - everything else -> plain copy into the matching bf16 param.
    Returns counts (skipped = ckpt keys with no matching model param).

    ``prefixes`` (e.g. ["diff_dec","diff_dec_low"]) restrict which keys are
    considered; by default all keys that map to a model param are filled.

    ``jobs``: >1 enables a bounded-parallel load. Disk read + safe_open +
    deserialize for up to jobs shards run in worker threads while the main
    thread copies completed shards into the model (device transfer), which is
    the throughput limiter on a big checkpoint. Defaults to the runner's core
    count minus 2. jobs=1 is the exact legacy serial path (identical results,
    just no parallelism).
    """
    if scale_keys is None:
        scale_keys = collect_scale_keys(shards)
    params = {name: p for name, p in model.named_parameters()}
    counts = {"fp8": 0, "fp8scale": 0, "dequant": 0, "plain": 0, "skipped": 0, "missing": 0}

    def _in_scope(k: str) -> bool:
        return prefixes is None or k.split(".")[0] in prefixes

    if jobs is None:
        # Parallel depth derived from the runner's core count minus 2 (the two
        # reserved for the model/GPU pipelines). Sensible always-on default, not
        # a config toggle.
        jobs = max(1, (os.cpu_count() or 2) - 2)

    need_fill = len(shards) > 1 and jobs and jobs > 1
    if need_fill:
        # Parallel: read shards on up to jobs worker threads; consume in
        # submission order (deterministic, matches serial bit-for-bit).
        with ThreadPoolExecutor(max_workers=min(jobs, len(shards))) as ex:
            for _sh, tens in ex.map(_read_shard, shards):
                _fill_shard(params, tens, counts, device, layout, _in_scope, scale_keys)
                del tens
        return counts
    # Serial (legacy) path: one shard at a time in the main thread.
    for sh in shards:
        _sh, tens = _read_shard(sh)
        _fill_shard(params, tens, counts, device, layout, _in_scope, scale_keys)
        del tens
        gc.collect()
    return counts


# ---------------------------------------------------------------------------
# SageAttention on the WIT-CFG renderer (14b fp8).
#
# The renderer dispatches attention via transformers' `ALL_ATTENTION_FUNCTIONS`
# (a shared dict on modeling_utils that the VeOmni Wan diffusers model indexes
# by `processor.attn_implementation`). That dict has NO "sageattention" key, so
# the native knob can't reach it (FA2/FA3 are first-class). We register a
# `sageattention` entry matching the ALL_ATTENTION_FUNCTIONS convention:
#   input : (B, heads, seq, head_dim)          <- call site passes query.transpose(1,2)
#   output: (B, seq, heads, head_dim), None    <- sageattn gives (B,H,S,D); transpose back
# Verified on this GPU (RTX 5090 sm_120): sageattn is bit-exact vs SDPA and fast.
# ---------------------------------------------------------------------------

def _wan_sage_attention_forward(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask=None,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    **kwargs,
):
    """ALL_ATTENTION_FUNCTIONS-compatible attention_interface backed by SageAttention."""
    from sageattention import sageattn

    try:
        attn_output = sageattn(query, key, value)
    except Exception as exc:  # noqa: BLE001 - unsupported head_dim/etc -> SDPA fallback
        logger.warning("bernini_fp8: sageattn failed (%s) - falling back to SDPA", exc)
        attn_output = F.scaled_dot_product_attention(
            query, key, value, attn_mask=None, dropout_p=dropout, is_causal=False
        )
    return attn_output.transpose(1, 2), None


# Attention implementation to force on the renderer. For A/B testing:
#   "sageattention"      -> registered above (needs the key injection)
#   "flash_attention_2"  -> first-class ALL_ATTENTION_FUNCTIONS key (real FA2)
#   "sdpa" / "eager"     -> baseline
_ATTN_IMPL = "sageattention"


def _enable_attention(model: nn.Module, impl: str) -> int:
    """Force every Wan attention processor + transformer config to use `impl`.
    Registers the sageattention key only when sageattention is requested.
    Returns number of switches. Idempotent (key registered once)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    if impl == "sageattention" and "sageattention" not in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS["sageattention"] = _wan_sage_attention_forward
    n = 0
    for mod in model.modules():
        proc = getattr(mod, "processor", None)
        if proc is not None and getattr(proc, "attn_implementation", None) is not None:
            proc.attn_implementation = impl
            n += 1
        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = impl
            n += 1
    return n


# ---------------------------------------------------------------------------
# Build the full Bernini model with an fp8 renderer.
# ---------------------------------------------------------------------------


def _load_fp8_manifest(config_dir: str, shards: List[str]):
    """Read the baked fp8 manifest (fp8 key list + ``k_n`` layout) if present.

    Returns ``(scale_keys, layout)``. When a manifest exists the caller skips
    ``collect_scale_keys`` (the per-shard header scan) AND knows the 2-D fp8
    linears are already ``(K,N)`` (no ambiguous, square-unsafe transpose).
    Falls back to a header scan + legacy ``"n_k"`` layout for stock/pre-bake
    repos so old checkpoints still load unchanged.
    """
    import json as _json
    mpath = os.path.join(config_dir, "fp8_manifest.json")
    if os.path.exists(mpath):
        try:
            with open(mpath) as fh:
                m = _json.load(fh)
            fw = m.get("fp8_weights")
            if fw:
                layout = m.get("layout", "n_k")
                keys = {k + "_scale" for k in fw}
                logger.info(
                    "bernini_fp8: fp8_manifest layout=%s fp8_weights=%d (no shard scan)",
                    layout, len(keys))
                return keys, layout
        except Exception as exc:  # noqa: BLE001 - never fail load on a bad manifest
            logger.warning("bernini_fp8: manifest read failed (%s); falling back to scan", exc)
    return collect_scale_keys(shards), "n_k"


def bernini_shards(model_dir: str) -> List[str]:
    sub = "bernini"
    return sorted(glob.glob(os.path.join(model_dir, sub, "model-*.safetensors")))


def build_fp8_model(
    config_dir: str,
    config,
    device: Optional[str] = None,
    prefixes: Optional[List[str]] = None,
):
    """Construct BerniniModel(config) and swap the renderer to fp8. Returns model."""
    from bernini.models.bernini import BerniniModel

    # Constructing the graph (MLLM + Wan transformers + connector) runs default
    # random init over ~14B params (~128s of kaiming_uniform_/uniform_/normal_),
    # and every value is overwritten from the checkpoint by stream_fill below
    # (missing:0) - so that init is pure wasted work. No-op the random
    # initializers for the construction window; keep cheap deterministic inits
    # (zeros_/ones_/constant_) so buffers etc. still init. torch.nn.init.* IS the
    # feed-through for Linear/embedding reset and _initialize_weights, so this
    # covers the RNG cost. (The meta-device construct trick is NOT used here:
    # BerniniModel internally from_pretrains the T5 text encoder, which refuses
    # to run under a meta default device.)
    import torch.nn.init as _torch_init
    _skip_init = ("uniform_", "normal_", "kaiming_uniform_", "kaiming_normal_",
                  "xavier_uniform_", "xavier_normal_", "trunc_normal_")
    _saved_init = {n: getattr(_torch_init, n) for n in _skip_init}
    def _noop_init(*a, **k):
        return None
    for _n in _skip_init:
        setattr(_torch_init, _n, _noop_init)
    # transformers' _initialize_weights/_init_weights call the DIRECT tensor
    # methods (weight.data.normal_(...)), which BYPASS torch.nn.init - so the
    # torch.nn.init patch alone leaves that full normal_ cost running (~34s
    # on this model). Neutralise the direct Tensor.normal_/uniform_ methods
    # for the construction window too. All such calls hit weight params that
    # stream_fill overwrites (missing:0); bias/padding_idx zero_ still runs.
    _T = torch.Tensor
    _saved_tmethod = {"normal_": _T.normal_, "uniform_": _T.uniform_}
    def _noop_self(self, *a, **k):
        return self
    for _m in _saved_tmethod:
        setattr(_T, _m, _noop_self)
    try:
        model = BerniniModel(config)
    finally:
        for _n, _f in _saved_init.items():
            setattr(_torch_init, _n, _f)
        for _m, _f in _saved_tmethod.items():
            setattr(_T, _m, _f)
    # Swap quantisable layers inside the two renderer /GEN_Wanx22/ containers only
    # (mllm/t5/connector/vit_decoder stay bf16 plain weights).
    for attr in ("diff_dec", "diff_dec_low"):
        m = getattr(model, attr, None)
        if m is not None:
            n = replace_quantisable_layers(m, device)
            logger.info("bernini_fp8: replaced %d layers in %s", n, attr)
    shards = bernini_shards(config_dir)
    scale_keys, layout = _load_fp8_manifest(config_dir, shards)
    counts = stream_fill(model, shards, scale_keys, device=device, prefixes=prefixes,
                         layout=layout)
    logger.info("bernini_fp8: fill %s", counts)
    for attr in ("diff_dec", "diff_dec_low"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "transformer_2") and getattr(m, "transformer_2", None) is None:
            pass
    # pipeline wiring: transformer_2 lives in diff_dec_low, attach to diff_dec
    if model.diff_dec is not None and model.diff_dec_low is not None:
        setattr(model.diff_dec, "transformer_2", model.diff_dec_low.transformer_2)
    n_attn = _enable_attention(model, _ATTN_IMPL)
    logger.info("bernini_fp8: attention=%s forced on %d attn processors", _ATTN_IMPL, n_attn)
    n_mm = sum(1 for m in model.modules() if isinstance(m, FP8Linear) and m._scaled)
    n_fb = sum(1 for m in model.modules() if isinstance(m, FP8Linear) and not m._scaled)
    logger.info("bernini_fp8: fp8 _scaled_mm linears=%d (dequant fallback=%d)", n_mm, n_fb)
    model.eval()
    return model


def build_fp8_pipeline(config_dir: str, device, **config_overrides):
    """Replicate ``BerniniPipeline.from_pretrained`` but with an fp8 renderer.

    The stock 14b path calls ``BerniniModel.from_pretrained``, which reads each
    shard's raw tensors straight through transformers' meta-load and dies on the
    ``uint8`` e4m3 weights ("Only Tensors of floating point ... can require
    gradients"). We build the model with :func:`build_fp8_model` (fp8-resident
    renderer) and otherwise mirror the upstream pipeline construction exactly:
    mllm-as-text-encoder, T5 tokenizer, VIT processor, Wan VAE.
    """
    from bernini.models.bernini import BerniniConfig
    from bernini.pipeline import _localize_bernini_config, BerniniPipeline
    from transformers import AutoTokenizer, AutoProcessor
    from diffusers.models import AutoencoderKLWan

    config = BerniniConfig.from_pretrained(config_dir, **config_overrides)
    _localize_bernini_config(config, config_dir)

    model = build_fp8_model(config_dir, config)  # build + fp8 fill + wiring + eval

    t5_tokenizer = AutoTokenizer.from_pretrained(
        config.t5_tokenizer_path,
        subfolder=config.t5_tokenizer_subfolder,
        trust_remote_code=True,
    )
    vit_processor = AutoProcessor.from_pretrained(
        config.processor_config_path,
        subfolder=config.processor_subfolder,
        padding_side="right",
        trust_remote_code=True,
    )
    vae = AutoencoderKLWan.from_pretrained(
        config.vae_model_path,
        subfolder=config.vae_subfolder,
        torch_dtype=torch.float32,
    )
    vae.eval()
    vae.requires_grad_(False)
    return BerniniPipeline(config, model, vae, t5_tokenizer, vit_processor, device)


def is_fp8_model(config_dir: str) -> bool:
    """Whether a model dir is the fp8-quantised Bernini (has quantization_config.json)."""
    return os.path.exists(os.path.join(config_dir, "quantization_config.json"))
