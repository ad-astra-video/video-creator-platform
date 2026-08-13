"""Isolated functional test for fp8_loader: routing, classification, fp8+plain fills.

Run: backend/.venv/Scripts/python.exe runner/idv2v/_test_fp8_loader.py
"""
import os, sys, tempfile, torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_loader import (
    FP8Linear, FP8ConvNd,
    replace_quantisable_layers, load_fp8_into_models,
)

# --- tiny stand-ins for diffsynth's WanModel / VaceWanModel -----------------
class TinyBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.to_q = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)      # has .weight/.bias (plain bf16 leaves)
        self.out = nn.Linear(d, d)

class TinyDiT(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.patch = nn.Conv1d(d, d, 1)  # conv -> FP8ConvNd
        self.blocks = nn.ModuleList([TinyBlock(d) for _ in range(2)])
        self.has_image_input = True

class TinyVace(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.proj = nn.Linear(d, d)

H = 42  # deterministic weights

def fresh_source():
    torch.manual_seed(H)
    return TinyDiT(), TinyVace()

# --- build real model + surgery (from a fresh pair) -------------------------
src_dit, src_vace = fresh_source()   # clean source weights for the checkpoint
dit = TinyDiT(); vace = TinyVace()   # targets for surgery + load
# copy source weights into targets so 'orig' tracking matches exactly
src_dit.load_state_dict(dit.state_dict())  # same arch
n_dit = replace_quantisable_layers(dit)
n_vace = replace_quantisable_layers(vace)
assert n_dit == 5, n_dit   # 1 conv + 2 blocks*(2 linear)
assert n_vace == 1, n_vace
print("surgery ok: dit=%d vace=%d" % (n_dit, n_vace))

# --- synthesize an FP8 checkpoint (per-channel E4M3) from the source model --
sd = {}
def quantize_linear(w):
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8)
    q = (w.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    return q, scale
def add_linear(prefix, lin: nn.Linear):
    q, s = quantize_linear(lin.weight.detach())
    sd[prefix + ".weight"] = q; sd[prefix + ".weight_scale"] = s
    sd[prefix + ".comfy_quant"] = torch.tensor([0], dtype=torch.uint8)
    sd[prefix + ".bias"] = lin.bias.detach()
def add_norm(prefix, ln: nn.LayerNorm):
    sd[prefix + ".weight"] = ln.weight.detach()
    sd[prefix + ".bias"] = ln.bias.detach()
def add_conv(prefix, c: nn.Conv1d):
    # store conv weight in NATIVE shape [Cout, Cin, K]; per-channel scale [Cout,1,..]
    w = c.weight.detach()                       # [8,8,1]
    scale = w.abs().amax(dim=tuple(range(1, w.dim())), keepdim=True).clamp_min(1e-8)
    q = (w.to(torch.float32) / scale).to(torch.float8_e4m3fn)
    sd[prefix + ".weight"] = q; sd[prefix + ".weight_scale"] = scale
    sd[prefix + ".bias"] = c.bias.detach()

for i, b in enumerate(src_dit.blocks):
    add_linear(f"dit.blocks.{i}.to_q", b.to_q)
    add_norm(f"dit.blocks.{i}.norm", b.norm)
    add_linear(f"dit.blocks.{i}.out", b.out)
add_conv("dit.patch", src_dit.patch)
add_linear("vace.proj", src_vace.proj)

# original-target values to compare against (pre-surgery source)
orig = {}
for m, pre in ((src_dit, "dit."), (src_vace, "vace.")):
    for name, p in m.named_parameters():
        orig[pre + name] = p.detach().clone()

with tempfile.TemporaryDirectory() as td:
    shard = os.path.join(td, "model-00001-of-00001.safetensors")
    from safetensors.torch import save_file
    save_file(sd, shard)
    counts = load_fp8_into_models(dit, [shard], vace)

print("counts:", counts)
assert counts["fp8"] >= 6, counts
assert counts["plain"] >= 6, counts   # 2 norms*(w+b) + patch bias + linear biases

# --- verify fp8 dequant round-trips to the original bf16 --------------------
def walk(m, dotted):
    cur = m
    for part in dotted.split("."):
        cur = getattr(cur, part)
    return cur

for i in range(2):
    for att in ("to_q", "out"):
        lp = f"blocks.{i}.{att}"
        lp_mod = walk(dit, lp)
        w_loaded = lp_mod.weight.to(torch.float32) * lp_mod.scale.to(torch.float32)
        w_orig = orig[f"dit.{lp}.weight"].to(torch.float32)
        err = (w_loaded - w_orig).abs().max().item()
        assert err < 0.1, (lp, err)
        print(f"{lp}: max err {err:.2e} (fp8 round-trip)")

# plain norm weight must be preserved bit-exact
err = (dict(dit.named_parameters())["blocks.0.norm.weight"] - orig["dit.blocks.0.norm.weight"]).abs().max().item()
assert err == 0, err
print("norm weight bit-exact: ok")

# vace fp8
vm = walk(vace, "proj")
w_loaded = vm.weight.to(torch.float32) * vm.scale.to(torch.float32)
err = (w_loaded - orig["vace.proj.weight"].to(torch.float32)).abs().max().item()
assert err < 0.1, err
print("vace fp8 round-trip: OK, err %.2e" % err)

# patch conv (4D-stored-flat fp8) round-trip
pm = walk(dit, "patch")
w_loaded = pm.weight.to(torch.float32) * pm.scale.to(torch.float32)
err = (w_loaded - orig["dit.patch.weight"].to(torch.float32)).abs().max().item()
assert err < 0.1, err
print("patch conv fp8 round-trip: OK, err %.2e" % err)

# --- FP8Linear computes correctly (dequant in forward) ----------------------
dit.eval()
x = torch.randn(3, 8)
y = dit.blocks[0].to_q(x)
m = walk(dit, "blocks.0.to_q")
w = m.weight.to(x.dtype) * m.scale.to(x.dtype)
yref = torch.nn.functional.linear(x, w, m.bias.to(x.dtype))
assert torch.allclose(y, yref, atol=1e-5), (y - yref).abs().max()
print("FP8Linear forward == F.linear(dequant(w), x): OK")

print("\nALL TESTS PASSED")
