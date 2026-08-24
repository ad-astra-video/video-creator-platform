"""Pure-torch `flash_attn` compatibility shim for Bernini's Qwen2.5-VL MLLM.

Bernini's MLLM runs on torch SDPA (config `mllm_attn_implementation="sdpa"`); the
`flash_attn` package is only required to satisfy `modeling_qwen2_5_vl.py`'s
import-time guard (raises if neither `flash_attn` nor `flash_attn_interface` is
importable). The flash functions below are thus never called in the sdpa path;
they are provided as correct SDPA-backed (with optional SageAttention) fallbacks
so the module imports and any dormant flash branch also works.

No flash-attn / flash-attn2 C extension is built or installed.
"""
import torch
import torch.nn.functional as F


def _attempt_sage_dense(q, k, v, softmax_scale, causal):
    """Route dense attention through SageAttention when it's installed.

    q,k,v already transposed to (batch, nheads, seq, headdim) [HND layout].
    Returns the HND output tensor, or None if SageAttention isn't usable.
    """
    try:
        from sageattention import sageattn
    except Exception:  # not installed / import error -> fall back to SDPA
        return None
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return None
    try:
        return sageattn(q, k, v, qk_scale=softmax_scale, tensor_layout="HND",
                        is_causal=bool(causal))
    except Exception:
        return None


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), alibi_slopes=None, deterministic=False,
                    return_attn_probs=False, softcap=None, **kwargs):
    """Dense attention. q,k,v: (batch, seq, nheads, headdim)."""
    del window_size, alibi_slopes, deterministic, softcap  # tolerated, unused
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    qh = q.transpose(1, 2).contiguous()  # (b, h, s, d)
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    out = _attempt_sage_dense(qh, kh, vh, scale, causal)
    if out is None:
        out = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=None, dropout_p=dropout_p,
            is_causal=causal, scale=scale)
    out = out.transpose(1, 2).contiguous()  # back to (b, s, h, d)
    if return_attn_probs:
        return out, None
    return out


def _varlen_to_dense(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                     max_seqlen_k, softmax_scale, causal, dropout_p):
    """Reconstruct a (B, max_seq, H, D) dense batch from varlen inputs and run
    per-sequence attention (SageAttention when available, else SDPA)."""
    import torch as t
    out_all = []
    n = len(cu_seqlens_q) - 1
    for i in range(n):
        s0, s1 = int(cu_seqlens_q[i]), int(cu_seqlens_q[i + 1])
        k0, k1 = int(cu_seqlens_k[i]), int(cu_seqlens_k[i + 1])
        qi = q[s0:s1]
        ki = k[k0:k1]
        vi = v[k0:k1]
        # pad to max_seqlen
        qp = t.zeros(1, max_seqlen_q, qi.shape[1], qi.shape[2], dtype=qi.dtype,
                     device=qi.device)
        kp = t.zeros(1, max_seqlen_k, ki.shape[1], ki.shape[2], dtype=ki.dtype,
                     device=ki.device)
        vp = t.zeros(1, max_seqlen_k, vi.shape[1], vi.shape[2], dtype=vi.dtype,
                     device=vi.device)
        sl_q = qi.shape[0]
        sl_k = ki.shape[0]
        qp[:, :sl_q] = qi
        kp[:, :sl_k] = ki
        vp[:, :sl_k] = vi
        qh = qp.transpose(1, 2).contiguous()
        kh = kp.transpose(1, 2).contiguous()
        vh = vp.transpose(1, 2).contiguous()
        scale = softmax_scale if softmax_scale is not None else qi.shape[-1] ** -0.5
        # causal over the padded seq: only keep lower-triangular within sl_q
        out = _attempt_sage_dense(qh, kh, vh, scale, causal and sl_q == sl_k)
        if out is None:
            out = F.scaled_dot_product_attention(
                qh, kh, vh, attn_mask=None, dropout_p=0.0,
                is_causal=causal and sl_q == sl_k, scale=scale)
        out = out.transpose(1, 2)  # (1, seq, h, d)
        out_all.append(out[0, :sl_q])
    return t.cat(out_all, dim=0)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                           max_seqlen_k, dropout_p=0.0, softmax_scale=None,
                           causal=False, window_size=(-1, -1), softcap=None,
                           alibi_slopes=None, deterministic=False,
                           return_attn_probs=False, **kwargs):
    """Variable-length attention (NHD layout, concatenated across batch)."""
    del window_size, softcap, alibi_slopes, deterministic
    out = _varlen_to_dense(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                           max_seqlen_k, softmax_scale, causal, dropout_p)
    if return_attn_probs:
        return out, None
    return out
