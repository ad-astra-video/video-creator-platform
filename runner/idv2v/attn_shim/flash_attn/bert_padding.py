"""Torch reimplementation of `flash_attn.bert_padding.unpad_input`.

Used by `transformers.modeling_flash_attention_utils._upad_input` which
Bernini's modeling imports. Only reached on the (non-default) flash path.
"""
import torch
import torch.nn.functional as F


def unpad_input(hidden_states, attention_mask, length=None):
    """Unpad hidden states along the sequence dim per the attention mask.

    Args:
        hidden_states: (batch, seq, ...)
        attention_mask: (batch, seq) with 1/True = attend
        length: optional explicit sequence length to slice to first
    Returns:
        (unpad_states, indices, cu_seq_lens, max_seqlen_in_batch)
    """
    batch, seq = hidden_states.shape[0], hidden_states.shape[1]
    if length is not None and length < seq:
        hidden_states = hidden_states[:, :length]
        attention_mask = attention_mask[:, :length]
    mask = attention_mask
    # work with bool or float masks
    seqlens = mask.long().sum(dim=-1)  # (batch,)
    max_seqlen_in_batch = int(seqlens.max().item())
    indices = torch.nonzero(mask.reshape(-1) != 0, as_tuple=False).flatten()
    cu_seq_lens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    unpad = hidden_states.reshape(-1, *hidden_states.shape[2:])[indices]
    return unpad, indices, cu_seq_lens, max_seqlen_in_batch
