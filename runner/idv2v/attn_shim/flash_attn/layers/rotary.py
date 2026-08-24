"""Torch reimplementation of `flash_attn.layers.rotary.apply_rotary_emb`.

Supplied for the import guard in Bernini's modeling (when transformers'
`is_flash_attn_2_available()` returns True it imports this). The SDPA path never
calls it; this is a correct standalone implementation for the dormant case.
"""
import torch


def rotate_half(x):
    x = x.reshape(*x.shape[:-1], 2, -1)
    x1, x2 = x[..., 0, :], x[..., 1, :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x, cos, sin, position_ids=None, interleaved=False,
                     seqlen_offsets=0, _flash_supports_window_size=None):
    """x: (batch, seq, nheads, headdim); cos/sin: (seq, headdim) or broadcastable."""
    # If position_ids provided, gather the relevant cos/sin rows.
    ndim = x.dim()
    # cos/sin usually (seq, headdim) -> align to (1, seq, 1, headdim)
    cos = cos.unsqueeze(0).unsqueeze(2) if cos.dim() == 2 else cos
    sin = sin.unsqueeze(0).unsqueeze(2) if sin.dim() == 2 else sin
    if position_ids is not None:
        pos = position_ids[..., : cos.shape[-2]].permute(0, -1, *([1] * (cos.dim() - 2)))
        cos = cos.expand_as(pos) if cos.shape != pos.shape else cos
    return (x * cos) + (rotate_half(x) * sin)
