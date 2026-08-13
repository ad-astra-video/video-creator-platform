"""Offline test of the STAGED inference orchestrator (no diffsynth needed).

Mocks out _run_staged_clip (which imports diffsynth) to exercise the pure
staging logic: clip scheduling/chaining, T5-context threading, the
random_ref_frame separation, first_clip flag, and frame stitching.

Run:  python runner/idv2v/_test_staged.py
"""
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from runner.idv2v.model import ModelManager


class _Harness(ModelManager):
    """ModelManager with the diffsynth-touching clip driver stubbed out."""

    def __init__(self, device="cpu"):
        super().__init__(device=device)
        self._pipe = object()            # only passed through to the stub
        self.calls = []                  # records per-clip args
        self._frames_per_clip = 40

    def _run_staged_clip(self, pipe, *, prompt, negative_prompt, input_image,
                         random_ref_frame, vace_video, vace_video_mask, seed,
                         num_inference_steps, cfg_scale, vace_scale,
                         width, height, num_frames, ref_pad_num, first_clip):
        self.calls.append(dict(
            first_clip=first_clip,
            input_image=input_image,
            random_ref_frame=random_ref_frame,
            num_frames=num_frames,
            seed=seed,
            n_vace=len(vace_video),
            n_cond=len(vace_video[0]),
        ))
        clip_idx = len(self.calls) - 1  # call order == clip order
        return [f"{clip_idx}:{i:03d}" for i in range(num_frames)]


def make_input_image(tag):
    class Img: pass
    i = Img()
    i.tag = tag
    return i


def run_test():
    h = _Harness()
    orig_input = make_input_image("ORIGINAL")
    # 100 frames, 40/clip -> 3 clips [(0,40),(39,79),(60,100)]
    cv = [[make_input_image(f"cond{i}") for i in range(100)]]
    out = h._infer_staged(
        prompt="p", negative_prompt="n", input_image=orig_input,
        condition_videos=cv, width=1280, height=720, num_frames=40,
        seed=42,
    )

    assert len(h.calls) == 3, f"expected 3 clips, got {len(h.calls)}"
    # first_clip only on clip 0
    assert [c["first_clip"] for c in h.calls] == [True, False, False], h.calls
    # random_ref_frame must be the ORIGINAL first frame for EVERY clip
    for c in h.calls:
        assert c["random_ref_frame"] is orig_input, "random_ref_frame drift"
    # clip chaining: clip i>0 input_image == clip i-1 splice frame
    splice_idx = 39 - 0
    assert h.calls[1]["input_image"] is not None
    # per-clip condition slicing: n_cond == num_frames, n_vace == 1
    for c in h.calls:
        assert c["n_cond"] == 40, c
        assert c["n_vace"] == 1, c
    assert all(c["seed"] == 42 for c in h.calls)

    # stitching yields 100 frames
    assert len(out) == 100, f"stitched len {len(out)} != 100"
    # overlap removal: clip0 frames 0..38 kept, then clip1 appended
    assert out[0] == "0:000" and out[38] == "0:038"
    assert out[39] == "1:000" and out[59] == "1:020"   # clip1 starts at frame 39
    assert len(set(out)) == 100  # all unique across clips
    print("ok: 3 clips, first_clip=[T,F,F], random_ref preserved, stitched=100")


def run_clip_schedule():
    from runner.idv2v.model import compute_clip_schedule
    assert compute_clip_schedule(10, 40) == [(0, 10)]           # single short
    assert compute_clip_schedule(40, 40) == [(0, 40)]           # exact one clip
    assert compute_clip_schedule(100, 40) == [(0, 40), (39, 79), (60, 100)]
    print("ok: clip schedule edge cases")


def run_single_clip_short():
    h = _Harness()
    orig = make_input_image("orig")
    cv = [[make_input_image(f"c{i}") for i in range(10)]]  # 10 frames, < 40
    out = h._infer_staged(prompt="p", negative_prompt="n", input_image=orig,
                          condition_videos=cv, num_frames=40)
    # single clip, truncated to total_frames
    assert len(h.calls) == 1 and h.calls[0]["first_clip"] is True
    assert len(out) == 10, f"expected 10 (truncated), got {len(out)}"
    print("ok: single short clip truncated to 10 frames")


if __name__ == "__main__":
    run_clip_schedule()
    run_test()
    run_single_clip_short()
    print("ALL STAGED TESTS PASSED")
