"""Runtime low-rank LoRA adapter for the fp8 Bernini renderer (rzgar 4-step).

Applies ``rzgar/Bernini-R-LightX2V-4step-loras`` -- a LightX2V-native *delta*
adapter (NOT diffusers format) -- on top of the already-built fp8 renderer,
at worker startup, reversibly, without a fused on-disk turbo checkpoint.
``TurboLora.apply()`` / ``.restore()`` flip the per-job turbo toggle on the
resident model.

Why low-rank-on-the-fly instead of dequant-and-apply:
  * The renderer stores weights as fp8 (K,N) + per-channel scale; at native
    848x480 we run at ~31.7/32 GB. Dequantising the targeted layers to bf16
    and adding the delta would balloon the renderer toward bf16 (~2x) and OOM.
  * So each targeted ``FP8Linear`` keeps its fp8 base GEMM and *adds* the
    low-rank term ``alpha * lora_b(lora_a(x))`` in bf16 -- ~1-2% extra flops,
    ~zero resident-VRAM growth. Tiny deltas (biases, norm weights) are patched
    directly in bf16 and restored on ``restore()``.

Key map (LightX2V ``diffusion_model.blocks.N`` -> our BerniniModel renderer);
both expert files share the same key structure (values differ):
  self_attn.<q,k,v,o>    -> <exp>.blocks.N.attn1.<to_q,to_k,to_v,to_out.0>
  cross_attn.<q,k,v,o>   -> <exp>.blocks.N.attn2.<to_q,to_k,to_v,to_out.0>
  (self|cross)_attn.<norm_q,norm_k>.diff -> attnN.<norm_q,norm_k> (scale delta)
  ffn.0 / ffn.2          -> <exp>.blocks.N.ffn.net.0.proj / ffn.net.2
  norm3                  -> <exp>.blocks.N.norm2
  head / text_embedding / time_embedding / time_projection -> suffix-resolved
  High-noise file -> diff_dec.transformer ; low-noise file -> diff_dec_low.transformer_2
"""

from __future__ import annotations

import logging

import torch

from bernini_fp8 import FP8Linear

logger = logging.getLogger("video_creator.runner.idv2v.bernini_lora")

_ATTN_KIND = {"self_attn": "attn1", "cross_attn": "attn2"}
_QKV = {"q": "to_q", "k": "to_k", "v": "to_v", "o": "to_out.0"}
_OTHER_TOP = ("head", "text_embedding", "time_embedding", "time_projection")
# LoRA top-level body -> (our renderer-relative path, mode) ; mode "full" = the
# adapter stores a whole-weight delta (head), "lora" = rank-reduced lora pair.
_OTHER_MAP = {
    "head.head": ("proj_out", "full"),
    "text_embedding.0": ("condition_embedder.text_embedder.linear_1", "lora"),
    "text_embedding.2": ("condition_embedder.text_embedder.linear_2", "lora"),
    "time_embedding.0": ("condition_embedder.time_embedder.linear_1", "lora"),
    "time_embedding.2": ("condition_embedder.time_embedder.linear_2", "lora"),
    "time_projection.1": ("condition_embedder.time_proj", "lora"),
}


def _disc(k: str) -> str:
    """Discriminating token: name before a trailing '.weight' (lora_down/lora_up
    carry .weight; diff/diff_b/alpha are stored bare)."""
    if k.endswith(".weight"):
        return k[: -len(".weight")].rpartition(".")[2]
    return k.rpartition(".")[2]


def _resolve_rel(src_key: str):
    """Map a LightX2V key to a renderer-relative name + apply kind.
    Returns (rel, kind) with kind in {lora_linear,norm_scale,norm_bias,other},
    or (None, None) when unmappable."""
    toks = src_key.split(".")
    if not toks or toks[0] != "diffusion_model":
        return None, None
    if toks[1] == "blocks":
        n = toks[2]
        rest = toks[3:]
        if not rest:
            return None, None
        k = rest[0]
        if k in _ATTN_KIND:
            attn = _ATTN_KIND[k]
            sub = rest[1] if len(rest) > 1 else ""
            if sub in _QKV:
                return f"blocks.{n}.{attn}.{_QKV[sub]}", "lora_linear"
            if sub in ("norm_q", "norm_k"):
                return f"blocks.{n}.{attn}.{sub}", "norm_scale"
            return None, None
        if k == "ffn":
            sub = rest[1] if len(rest) > 1 else ""
            if sub == "0":
                return f"blocks.{n}.ffn.net.0.proj", "lora_linear"
            if sub == "2":
                return f"blocks.{n}.ffn.net.2", "lora_linear"
            return None, None
        if k == "norm3":
            return f"blocks.{n}.norm2", "norm_scale"
        return None, None
    if toks[1] in _OTHER_TOP:
        return ".".join(["@" + toks[1]] + toks[2:]), "other"
    return None, None


def _find_module_by_suffix(root, suffix: str):
    best = None
    for name, mod in root.named_modules():
        if not name:
            continue
        if name == suffix or name.endswith("." + suffix):
            if best is None or len(name) > len(best[0]):
                best = (name, mod)
    return best[1] if best else None


def _find_param_by_suffix(root, suffix: str):
    hit = None
    for pname, p in root.named_parameters():
        if pname == suffix or pname.endswith("." + suffix):
            if hit is None or len(pname) > len(hit[0]):
                hit = (pname, p)
    return hit


class TurboLora:
    """Load the rzgar 4-step LoRA once and apply/restore it on the fp8 renderer.

    ``linear`` : rel -> {"lora_a","lora_b","alpha","diff_b"} (low-rank, on FP8Linear)
    ``patch``  : list of (param, delta) for direct bf16 patches (biases, norms)
                 with the original value stored so restore() is exact.
    also keeps ``picked``/``missed`` coverage counts.
    """

    def __init__(self, model, high_ckpt: str, low_ckpt: str, device=None):
        self.model = model
        self.linear = {}   # rel -> dict
        self.patch = []    # list of (param, orig, delta)
        self.missed = []   # sample of unmapped keys (first 10)
        self.missed_n = 0
        self.applied_n = 0
        self.active = False
        self.device = device or "cpu"
        self._load(high_ckpt, low_ckpt)
        logger.info("bernini_lora: loaded lora (linears=%d patches=%d missed=%d)",
                    len(self.linear), len(self.patch), self.missed_n)

    # ---- load -------------------------------------------------------------
    def _load(self, high_ckpt, low_ckpt):
        from safetensors import safe_open
        experts = [
            (high_ckpt, self.model.diff_dec.transformer, "diff_dec.transformer"),
            (low_ckpt, self.model.diff_dec_low.transformer_2, "diff_dec_low.transformer_2"),
        ]
        for ckpt, root, exp in experts:
            mods = {name: m for name, m in root.named_modules() if name}
            with safe_open(ckpt, framework="pt", device="cpu") as f:
                for k in sorted(f.keys()):
                    disc = _disc(k)
                    rel, kind = _resolve_rel(k)
                    if rel is None:
                        self._miss(k)
                        continue
                    t = f.get_tensor(k).to(torch.bfloat16)  # kept CPU; moved at apply/restore
                    if kind == "lora_linear":
                        m = mods.get(rel)
                        if m is None or not isinstance(m, FP8Linear):
                            self._miss(k)
                            continue
                        slot = self.linear.setdefault(exp + "." + rel, {"alpha": 1.0})
                        if disc == "lora_down":
                            slot["lora_a"] = t
                        elif disc == "lora_up":
                            slot["lora_b"] = t
                        elif disc == "alpha":
                            slot["alpha"] = float(t.reshape(()).item())
                        elif disc == "diff_b":
                            if m.bias is not None:
                                slot["diff_b"] = t
                            else:
                                self._miss(k)
                        else:
                            self._miss(k)
                    elif kind == "norm_scale":
                        m = mods.get(rel)
                        if m is None:
                            self._miss(k)
                            continue
                        if disc == "diff":
                            w = getattr(m, "weight", None)
                            if w is not None and tuple(w.shape) == tuple(t.shape):
                                self._patch_w(w, t)
                                continue
                        elif disc == "diff_b":
                            b = getattr(m, "bias", None)
                            if b is not None:
                                self._patch_w(b, t)
                                continue
                        self._miss(k)
                    else:  # "other"
                        self._load_other(root, exp, rel, disc, t)

    def _load_other(self, root, exp, rel, disc, t):
        body = rel.lstrip("@")
        # rel carries the trailing discriminator (diff / diff_b / lora_down /
        # lora_up / alpha); strip it so `body` is the component path.
        if body.endswith(".weight"):
            body = body[: -len(".weight")]
        if disc and body.endswith("." + disc):
            body = body[: -(len(disc) + 1)]
        mapped = _OTHER_MAP.get(body)
        if mapped is None:
            self._miss_key(body)
            return
        actual, mode = mapped
        mods = {name: m for name, m in root.named_modules() if name}
        m = mods.get(actual)
        if m is None or not isinstance(m, FP8Linear):
            self._miss_key(body)
            return
        slot = self.linear.setdefault(exp + "." + actual, {"alpha": 1.0})
        if mode == "full":  # head: stash the whole (N,K) delta
            if disc == "diff":
                slot["lora_full"] = t
                return
            if disc == "diff_b" and m.bias is not None:
                slot["diff_b"] = t
                return
            self._miss_key(body + "." + disc)
            return
        if disc == "lora_down":
            slot["lora_a"] = t
        elif disc == "lora_up":
            slot["lora_b"] = t
        elif disc == "alpha":
            slot["alpha"] = float(t.reshape(()).item())
        elif disc == "diff_b" and m.bias is not None:
            slot["diff_b"] = t
        else:
            self._miss_key(body + "." + disc)
            return

    def _miss(self, k):
        self.missed_n += 1
        if len(self.missed) < 10:
            self.missed.append(k)

    def _miss_key(self, name):
        self.missed_n += 1
        if len(self.missed) < 10:
            self.missed.append(name)

    def _patch_w(self, p, delta):
        orig = p.detach().clone().cpu()  # CPU-copy so apply/restore is device-agnostic
        self.patch.append((p, orig, delta))

    # ---- apply / restore --------------------------------------------------
    def apply(self):
        if self.active:
            return
        for rel, slot in self.linear.items():
            m = self._module(rel)
            if m is None:
                continue
            dev = m.weight.device if m.weight is not None else self.device
            if "lora_a" in slot:
                m.lora_a = slot["lora_a"].to(dev)
            if "lora_b" in slot:
                m.lora_b = slot["lora_b"].to(dev)
            if "lora_full" in slot:
                m.lora_full = slot["lora_full"].to(dev)
            # LightX2V alpha is the standard alpha/rank LoRA scale (rank==alpha
            # here -> strength 1.0, matching rzgar's documented strength 1.0).
            ra = slot.get("lora_a")
            m.lora_alpha = slot["alpha"] / int(ra.shape[0]) if ra is not None else 1.0
            if "diff_b" in slot and m.bias is not None:
                m.bias.data.add_(slot["diff_b"].to(m.bias.dtype).to(m.bias.device))
        for p, orig, delta in self.patch:
            p.data.copy_((orig + delta).to(p.dtype).to(p.device))
        self.active = True

    def restore(self):
        if not self.active:
            return
        for rel, slot in self.linear.items():
            m = self._module(rel)
            if m is None:
                continue
            m.lora_a = None
            m.lora_b = None
            m.lora_alpha = 1.0
            m.lora_full = None
            if "diff_b" in slot and m.bias is not None:
                m.bias.data.sub_(slot["diff_b"].to(m.bias.dtype).to(m.bias.device))
        for p, orig, delta in self.patch:
            p.data.copy_(orig.to(p.dtype).to(p.device))
        self.active = False

    def _module(self, rel):
        """Resolve a rel path against both experts (rel already expert-scoped
        in linear/patch; for the low-rank linears rel is the full path)."""
        if rel.startswith("diff_dec.") or rel.startswith("diff_dec_low."):
            return _submodule_from_full(self.model, rel)
        # rel is relative to an expert (unused: linear keys are stored absolute)
        return None


def _submodule_from_full(model, path):
    cur = model
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def apply_rzgar_lora(model, high_ckpt, low_ckpt, device=None):
    """One-shot convenience: load and apply immediately (returns TurboLora)."""
    tl = TurboLora(model, high_ckpt, low_ckpt, device=device)
    tl.apply()
    return tl
