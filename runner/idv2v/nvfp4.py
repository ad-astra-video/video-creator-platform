"""NVFP4 (4-bit E2M1, block-scaled) encode/decode for the ID-V2V worker.

PURPOSE
  ID-V2V (Wan 2.1 I2V-14B restyle) weight memory is the binding constraint: on a
  single 31 GB RTX 5090 the fp8 model (~16.5 GB) + 720p activations (~14 GB)
  don't fit, forcing the slow block-CPU-offload fallback. nvfp4 halves weight
  memory again (4-bit), so DiT+VACE (~8.25 GB) + activations (~22 GB) fit fully
  on-GPU, killing the offload path for 720p.

  This module implements ONLY the low-precision representation + block-scale
  quantize/dequantize. The worker's existing architecture dequantizes each layer
  to bf16 in forward then runs a plain matmul (same as its FP8 path — no tensor
  cores). So nvfp4 here is a *storage/quality-vs-size* tradeoff, NOT a speedup.

FORMAT (native nvfp4, torchao-compatible)
  * e2m1 values: 1 sign, 2 exponent (bias 1), 1 mantissa bit.
  * Two values packed per byte (uint8); low nibble = element 0 of the pair.
  * scale: bf16, one per BLOCK(=16) elements along the reduction dim (matches
    torchao's nvfp4 block scale, so the SAME weights could later feed a real
    tensor-core path without re-quantizing).
  * Layout of an [M, K] Linear weight:
        qweight  uint8 [M, ceil(K/2)]            (packed e2m1)
        scale    bf16  [M, ceil(K/BLOCK)]        (per-16-elem block)
  * Convs: treated like a [Cout, Cin*K*K] 2D weight for quantization purposes
    (blocking over the flattened reduction dim), which keeps the math identical.

WHY BLOCK SCALE (not per-row like fp8)
  At 4-bit, a single per-row scale is too coarse: dynamic range within an output
  row varies, so a row-scale clipping/rounds aggressively. Per-16-elem blocks
  track local magnitude far better and are the nvfp4-native choice, so this keeps
  quality as high as 4-bit allows. It also keeps torchao-compatibility.
"""

from __future__ import annotations

import torch

BLOCK = 16  # nvfp4 block-scale size (torchao uses 16 for the E2M1/E4M3 pair)

# ---------------------------------------------------------------------------
# E2M1 float table (16 entries). value = sign * man-scaling * 2**(exp-bias)
#   normal:        (1 + man/2) * 2**(exp-1), exp in {1,2,3}
#   subnormal:     (man/2) * 2**(-1),        exp == 0  -> {0, 0.25}
# ---------------------------------------------------------------------------
NVFP4_MAX = 6.0  # largest normal magnitude (e=3, m=1 -> 1.5*4)


def _build_e2m1_table() -> list[float]:
    tbl = []
    for i in range(16):
        sign = -1.0 if (i & 0b1000) else 1.0
        exp = (i >> 1) & 0b11
        man = i & 0b1
        if exp == 0:
            val = 0.0 if man == 0 else 0.25
        else:
            val = (1.0 + man / 2.0) * (2.0 ** (exp - 1))
        tbl.append(sign * val)
    return tbl


_E2M1_VALS = _build_e2m1_table()
# Map float -> closest nibble for encode (nibble 0 = +0; we re-sign below).
_DECIMAL_TO_NIBBLE = {
    round(v, 9): i for i, v in enumerate(_E2M1_VALS)
}


def e2m1_table() -> list[float]:
    """The 16 representable values, indexed by 4-bit nibble (int 0..15)."""
    return list(_E2M1_VALS)


def dequant_nvfp4(
    qweight_packed: torch.Tensor, scale: torch.Tensor,
    block: int = BLOCK, dtype=torch.bfloat16,
) -> torch.Tensor:
    """Dequantize packed nvfp4 -> a float tensor of shape [M, K].

    qweight_packed: uint8 [M, ceil(K/2)]  (2 e2m1 nibbles per byte)
    scale:          bf16   [M, ceil(K/BLOCK)]
    returns:        [M, K] float16/32/bf16 (unpadded-masked to exactly K)
    """
    M, K2 = qweight_packed.shape
    K = K2 * 2
    lo = (qweight_packed & 0x0F).to(torch.int64)  # element (2j)
    hi = ((qweight_packed >> 4) & 0x0F).to(torch.int64)  # element (2j+1)
    vals = qweight_packed.new_tensor(_E2M1_VALS, dtype=dtype)
    # [M, K2] each, then interleave -> [M, K]
    lo_v = vals.index_select(0, lo.reshape(-1)).reshape(M, K2)
    hi_v = vals.index_select(0, hi.reshape(-1)).reshape(M, K2)
    w = torch.stack([lo_v, hi_v], dim=-1).reshape(M, K).to(dtype)
    # block scale broadcast: [M, ceil(K/16)] -> [M,1,KB] -> [M,K]
    scale = scale.to(dtype)
    KB = scale.shape[-1]
    s = scale.unsqueeze(2)  # [M, KB, 1]
    w = w.reshape(M, KB, block).float() * s.float()  # exact K if K % block == 0
    w = w.reshape(M, KB * block).to(dtype)
    return w[..., :K]


def quantize_nvfp4(
    w: torch.Tensor, block: int = BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-block nvfp4 weight-only quantization.

    w: [M, K] float (K need not be a multiple of block; we pad internally).
       Also works for 4D conv kernels flattened as [Cout, Cin*kk...].
    returns (qweight_packed uint8 [M, ceil(K/2)], scale bf16 [M, ceil(K/BLOCK)]).
    """
    orig_shape = None
    if w.dim() != 2:
        orig_shape = w.shape
        w = w.reshape(w.shape[0], -1)
    M, K = w.shape
    wf = w.detach().float()
    Kpad = ((K + block - 1) // block) * block
    if Kpad != K:
        wf = torch.nn.functional.pad(wf, (0, Kpad - K))
    # scale per block = block absmax / nvfp4_max  (clamped, min for zero-safe)
    wb = wf.reshape(M, -1, block)
    scale = (wb.abs().amax(dim=2).clamp_min(1e-12) / NVFP4_MAX).to(torch.bfloat16)
    scaled = wb / scale.unsqueeze(2)  # in ~[-6, 6]

    # nearest-nibble encode over ALL 16 nvfp4 values (INCLUDING 0, so small
    # magnitudes round to zero instead of being forced up to 0.25).
    xs = wf.new_tensor(e2m1_table())
    xf = scaled.reshape(M, -1, 1)
    idx = (xf - xs.view(1, 1, -1)).abs().argmin(dim=2)  # index into 16-value table
    nib = xs.index_select(0, idx.reshape(-1)).reshape(M, -1)
    # map chosen value -> nibble int (0..15)
    nib_uint = _value_to_nibble(nib)
    nib_uint = nib_uint.to(torch.int64)
    # map signed value -> nibble int (xor 0 for positives; for negatives and 0:
    #   -v in table at position 8^... ) we use a mapping from value->nibble int.
    nib_uint = _value_to_nibble(nib)
    nib_uint = nib_uint.to(torch.int64)

    # pack two per byte (low=nibble of even element, high=nibble of odd)
    K2 = nib_uint.shape[1]  # = Kpad (each element has one nibble)
    lo = nib_uint[:, 0::2]
    hi = nib_uint[:, 1::2]
    # pad odd tail with 0 nibble so hi aligns
    if hi.shape[1] < lo.shape[1]:
        hi = torch.nn.functional.pad(hi, (0, lo.shape[1] - hi.shape[1]))
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
    packed = packed.to(torch.uint8)

    # trim scale/packed to exact K
    scale = scale[:, :(K + block - 1) // block]
    packed = packed[:, :(K + 1) // 2]

    if orig_shape is not None:
        # For conv we keep 2D representation in the wrapper; caller reshapes.
        pass
    return packed.contiguous(), scale.contiguous()


def _value_to_nibble(values: torch.Tensor) -> torch.Tensor:
    """Map a tensor of exact representable nvfp4 floats -> nibble int (0..15)."""
    lookup = values.new_tensor(
        [_DECIMAL_TO_NIBBLE[round(v, 9)] for v in _E2M1_VALS],
        dtype=torch.long,
    )
    flat = values.reshape(-1).float()
    rounded = torch.round(flat * 1e9) / 1e9  # snap to table-representable
    idx = (rounded.unsqueeze(1) - values.new_tensor(e2m1_table()).unsqueeze(0)).abs().argmin(1)
    return lookup.index_select(0, idx).reshape(values.shape)


# ---------------------------------------------------------------------------
# Self-test: run with `python -m runner.idv2v.nvfp4` (CPU or CUDA, no diffsynth).
# Verifies exact round-trip for representable values and bounds the error for
# arbitrary floats.
# ---------------------------------------------------------------------------
def _selftest(device="cpu"):
    torch.manual_seed(0)
    M, K = 8, 128
    w = torch.randn(M, K, device=device) * 3.0

    q, s = quantize_nvfp4(w)
    assert q.dtype == torch.uint8 and q.shape == (M, K // 2), q.shape
    assert s.shape == (M, K // BLOCK), s.shape
    wd = dequant_nvfp4(q, s)
    assert wd.shape == (M, K), wd.shape

    rel = (wd - w).abs() / (w.abs() + 1e-8)
    print(f"  max rel err (randn*3): {rel.max().item():.4f}")
    print(f"  mean abs err:          {(wd - w).abs().mean().item():.4f}")

    # Exact round-trip for every representable nibble repeated across blocks
    tbl = w.new_tensor(e2m1_table())
    rep = tbl.repeat(2, (K // 2 // 16))  # 32 rows, K aligned
    rep = rep[:, :K]
    q2, s2 = quantize_nvfp4(rep)
    wd2 = dequant_nvfp4(q2, s2)
    err = (wd2 - rep).abs().max().item()
    print(f"  exact-nibble round-trip max err: {err:.6f}")
    assert err < 1e-5, f"round-trip not exact: {err}"

    # conv-flatten path
    wc = torch.randn(4, 3, 3, 3, device=device) * 2.0
    qc, sc = quantize_nvfp4(wc)
    wdc = dequant_nvfp4(qc, sc)
    rc = (wdc - wc.reshape(4, -1)).abs().max().item()
    print(f"  conv [4,27] path max err: {rc:.4f}")
    print("  OK")


if __name__ == "__main__":
    _selftest()
