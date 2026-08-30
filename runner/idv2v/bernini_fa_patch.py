#!/usr/bin/env python3
"""Apply the Bernini SDPA-fallback patch to modeling_qwen2_5_vl.py.

Bernini's MLLM attention defaults to SDPA / flex_attention, but its module
hard-requires an importable flash_attn (fa2) or flash_attn_interface (fa3) at
import time and raises ValueError otherwise. There is no usable fa2/fa3 build
for consumer Blackwell (sm120 / RTX 50-series / cc 12.0: FA3 does not support
sm120; an fa2 sm120 build is a heavy from-source job), so we soften the gate:
with neither available, bind the flash helpers to None and let the code fall
through to its torch SDPA / flex_attention path.

Pure line-based text edit (no flash/CJK dependence). Asserts the expected
structure so a Bernini clone change fails loudly instead of silently.
"""
import sys

P = "/opt/bernini/src/bernini/models/modeling_qwen2_5_vl.py"


def main():
    lines = open(P, encoding="utf-8").read().split("\n")

    out, i, patched = [], 0, False
    while i < len(lines):
        line = lines[i]
        # Soften the hard `raise ValueError` gate when neither fa2 nor fa3 present.
        if (not patched and line == "else:"
                and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("raise ValueError")):
            out.append("else:")
            out.append("    # Hermes patch: no sm120-compatible flash-attn build"
                       " (FA3 unsupported on consumer Blackwell;")
            out.append("    # FA2 sm120 is a heavy source build). Bernini's MLLM"
                       " defaults to SDPA/flex_attention, so degrate gracefully")
            out.append("    # instead of hard-crashing at import time.")
            out.append("    flash_attn_func = None")
            out.append("    flash_attn_varlen_func = None")
            i += 2
            patched = True
            continue
        out.append(line)
        i += 1

    src = "\n".join(out)
    assert patched, "Bernini modeling file: fa2/fa3 raise gate not found; structure changed?"
    assert src.count("flash_attn_func = None") == 1, "patch not applied cleanly"

    # Also define _flash_supports_window_size on the no-flash path (only set inside
    # the `is_flash_attn_2_available()` branch, so it would otherwise be undefined).
    block = "else:\n    apply_rotary_emb = None"
    assert src.count(block) == 1, "apply_rotary_emb/else block not found"
    src = src.replace(block, block + "\n    _flash_supports_window_size = False")

    open(P, "w", encoding="utf-8").write(src)
    print("bernini_fa_patch: OK (%s)" % P)


if __name__ == "__main__":
    sys.exit(main())
