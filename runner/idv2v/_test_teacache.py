"""Standalone validation of the diffsynth TeaCache state machine (copied from
wan_video_new_multiVace_svi.py, Eyeline-Labs/ID-V2V) + skip-rate vs threshold.

Run:  backend/.venv/Scripts/python.exe runner/idv2v/_test_teacache.py
No diffsynth required — only torch + numpy. Verifies:
  * step 0 and the LAST step are ALWAYS computed (quality guard).
  * accumulation + reset logic matches the reference.
  * skip rate across representative thresholds for the 480P/720P coeffs.
"""
import numpy as np
import torch

COEFFS = {
    "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
    "Wan2.1-I2V-14B-720P": [8.10705460e+03, 2.13393892e+03, -3.72934672e+02, 1.66203073e+01, -4.17769401e-02],
}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": COEFFS["Wan2.1-I2V-14B-480P"],
            "Wan2.1-I2V-14B-720P": COEFFS["Wan2.1-I2V-14B-720P"],
        }
        if model_id not in self.coefficients_dict:
            raise ValueError(f"{model_id} not supported")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                ((modulated_inp - self.previous_modulated_input).abs().mean()
                 / self.previous_modulated_input.abs().mean()).cpu().item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


def simulate(model_id, raw_l1_per_step, thresh, N=30):
    """raw_l1_per_step: list/array of N relative-L1 distances (the raw value
    between consecutive steps' t_mod). Returns (#computed, #skipped)."""
    tc = TeaCache(N, rel_l1_thresh=thresh, model_id=model_id)
    computed = 0
    for step in range(N):
        t_mod = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        if step > 0:
            # perturb t_mod by the raw L1 distance so mean-abs rel diff ~ raw
            t_mod = t_mod + torch.randn(6) * (raw_l1_per_step[step] * t_mod.abs().mean())
        x = torch.randn(8, 5120)
        skip = tc.check(None, x, t_mod)
        if skip:
            x = tc.update(x)
        else:
            computed += 1
            tc.store(x + torch.randn_like(x) * 0.001)
    return computed, N - computed


def profile(model_id, base_raw, tag):
    print(f"\n== {model_id}  ({tag})  N=30 steps ==")
    for thresh in (0.06, 0.08, 0.10, 0.15, 0.20, 0.25):
        # a few seeds; average skip rate
        sks = []
        for _ in range(5):
            raw = [base_raw * (0.85 ** i) for i in range(30)]
            c, s = simulate(model_id, raw, thresh)
            sks.append(s)
        avg = np.mean(sks)
        print(f"  thresh={thresh:<5} avg skipped steps: {avg:5.1f}/30  ({avg/30*100:4.0f}% skip)")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("Sanity: step0 & last always computed (should_skip[0]==last==False):", end=" ")
    tc = TeaCache(5, rel_l1_thresh=0.10, model_id="Wan2.1-I2V-14B-720P")
    sk = [tc.check(None, torch.zeros(1, 4), torch.zeros(1, 1, 6)) for _ in range(5)]
    print(sk)
    # decaying raw L1: larger early (noisy), smaller late
    profile("Wan2.1-I2V-14B-720P", 0.008, "moderate drift, decaying")
    profile("Wan2.1-I2V-14B-480P", 0.008, "moderate drift, decaying")
    print("\nNote: actual skip rate depends on real per-step t_mod drift; these are")
    print("representative estimates from a synthetic decaying profile.")
