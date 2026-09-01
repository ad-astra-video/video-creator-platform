#!/usr/bin/env python3
"""Build-time patch: route Bernini's Wan-transformer attention through
SageAttention (dense, single-segment path) with FA2/SDPA fallback.

Bernini's `attention.py` ships an FA3 -> FA2 -> SDPA varlen backend. The
wan-worker image bakes Bernini from the upstream repo at build time, so this
script edits the baked `/opt/bernini/src/bernini/attention.py` in place (the
same convention as `bernini_progress_patch.py`). SageAttention is already in
the Bernini venv (inherited via --system-site-packages); we only add a
dispatch for it:

  * `_select_backend()` prefers SageAttention (dense kernel) when importable.
  * `varlen_attention()` uses SageAttention for the single-contiguous-segment
    case (the Wan batch=1 path) and keeps FA2/SDPA for genuine multi-sample
    varlen / any import failure.

Idempotent: guarded by a marker comment; re-runs are no-ops.

Usage: python bernini_sage_patch.py [target_path]
       (default target is the baked in-image path)
"""
import py_compile
import sys

MARK = "# == Hermes SageAttention bridge (NHD dense) =="
DEFAULT = "/opt/bernini/src/bernini/attention.py"

NEW_SELECT_BACKEND = """\
def _select_backend():
    global _BACKEND, _flash_varlen, _sage
    if _BACKEND is not None:
        return
    try:
        from sageattention import sageattn

        _sage = sageattn
        _BACKEND = "sage"
        return
    except Exception:
        pass
    try:
        from flash_attn_interface import flash_attn_varlen_func  # FA3

        _flash_varlen, _BACKEND = flash_attn_varlen_func, "fa3"
        return
    except Exception:
        pass
    try:
        from flash_attn import flash_attn_varlen_func  # FA2

        _flash_varlen, _BACKEND = flash_attn_varlen_func, "fa2"
        return
    except Exception:
        pass
    _BACKEND = "sdpa"
"""


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    with open(target, "r", encoding="utf-8", newline="") as f:
        src = f.read()
    nl = "\r\n" if "\r\n" in src else "\n"

    if MARK in src:
        print(f"[bernini_sage_patch] already applied to {target} (idempotent no-op)")
        return 0

    def require(anchor: str, what: str) -> None:
        n = src.count(anchor)
        if n != 1:
            raise SystemExit(
                f"[bernini_sage_patch] {what}: anchor not unique (count={n}): {anchor!r}"
            )

    # A. module-level _sage handle, before _select_backend
    require("def _select_backend():", "module _sage")
    src = src.replace(
        "def _select_backend():",
        f"_sage = None{nl}{nl * 2}def _select_backend():",
        1,
    )

    # B. replace the whole _select_backend body (robust span edit) with the
    #    SageAttention-first version. Spans between the two unique defs.
    require("def _select_backend():", "select backend span")
    require("def get_attention_backend() -> str:", "get_attention_backend def")
    start = src.index("def _select_backend():")
    end = src.index("def get_attention_backend() -> str:")
    new_fn = NEW_SELECT_BACKEND.replace("\n", nl).rstrip(nl) + nl + nl + nl
    src = src[:start] + new_fn + src[end:]

    # C. add the dense single-segment helper before varlen_attention
    require("def varlen_attention(", "varlen_attention def")
    helper = (
        "def _sage_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal):\n"
        f"    {MARK}\n"
        '    """Dense single-segment attention via SageAttention (layout NHD:\n'
        "    [B, seq, H, D]). Only correct when each sample is one contiguous\n"
        "    segment (the Wan batch=1 path). Approximate kernel (~1e-3 error).\n"
        "    Multi-sample varlen never reaches here (dispatch guards it).\"\"\"\n"
        "    if _sage is None:\n"
        "        _select_backend()\n"
        "    sm_scale = float(q.shape[2]) ** -0.5\n"
        "    out = _sage(\n"
        "        q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),\n"
        '        tensor_layout="NHD", is_causal=causal, sm_scale=sm_scale,\n'
        "    )\n"
        "    return out.squeeze(0)\n"
        "\n"
        "\n"
    ).replace("\n", nl)
    src = src.replace("def varlen_attention(", helper + "def varlen_attention(", 1)

    # D. dispatch to sage for the single-segment case
    require('    if _BACKEND == "fa3":', "fa3 dispatch anchor")
    dispatch = (
        '    if _BACKEND == "sage" and len(cu_seqlens_q) == 2 and q.dim() == 3:\n'
        "        return _sage_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, causal)\n"
        "\n"
    ).replace("\n", nl)
    src = src.replace('    if _BACKEND == "fa3":', dispatch + '    if _BACKEND == "fa3":', 1)

    with open(target, "w", encoding="utf-8", newline="") as f:
        f.write(src)

    py_compile.compile(target, doraise=True)
    print(f"[bernini_sage_patch] applied to {target}; py_compile OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
