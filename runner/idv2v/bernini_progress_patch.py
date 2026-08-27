#!/usr/bin/env python3
"""Thread a per-step ``progress_cb`` through BOTH Bernini samplers.

Mirrors how the ID-V2V worker reports denoise progress (see ``model.py``'s
``infer(progress_cb=...)`` -> ``_prog((progress_id + 1) / len(timesteps))``).
This patch adds ``progress_cb=None`` to the two sampling entry points and calls
it with ``(fraction, step, total)`` at the top of each denoise step, and adds
``progress_cb`` (forwarded into the sampler) to both pipeline ``__call__``s:

* ``wan_diffusion.py::sample``                — the 1.3B RendererPipeline path
* ``wan_diffusion.py::sample_bernini_wvitcfg``— the 14B BerniniPipeline path
  (the model deployed for the deck: ``model_type == "bernini"``)
* ``pipeline.py::BerniniRendererPipeline.__call__`` -> ``sample()``
* ``pipeline.py::BerniniPipeline.__call__``   -> ``sample_bernini_wvitcfg()``

It is a build-time (root) patch applied to the venv copy of the ByteDance
source, like ``bernini_fa_patch.py``. It is TOLERANT: if an upstream anchor
isn't found (source drift / different commit) it logs and skips — per-step
progress degrades gracefully to the existing coarse stage progress, and
generation still works.
"""
import sys

WAN_DIFFUSION = "/opt/bernini/src/bernini/models/wan_diffusion.py"
PIPELINE = "/opt/bernini/src/bernini/pipeline.py"

_HOOK = (
    "            if progress_cb is not None:\n"
    "                progress_cb((t_idx + 1) / len(timesteps), "
    "t_idx + 1, len(timesteps))"
)


def _log(msg: str) -> None:
    print(f"bernini_progress_patch: {msg}", flush=True)


def _insert_after(src: str, anchor: str, text: str, what: str) -> tuple:
    """Insert ``text`` on its own line immediately after ``anchor`` (must be
    present exactly once). Returns (new_src, ok). Idempotent per anchor+text."""
    if anchor + "\n" + text in src:
        return src, True  # already applied at this anchor
    n = src.count(anchor)
    if n != 1:
        _log(f"WARN anchor for {what} not found/ambiguous (count={n}); skipping")
        return src, False
    return src.replace(anchor, anchor + "\n" + text, 1), True


def _insert_after_each(src: str, anchor: str, text: str, what: str) -> tuple:
    """Insert ``text`` after EVERY occurrence of ``anchor`` (e.g. the same
    denoise-loop opener in both samplers). Returns (new_src, ok)."""
    if anchor + "\n" + text in src:
        return src, True  # already applied at every anchor
    n = src.count(anchor)
    if n == 0:
        _log(f"WARN anchor for {what} not found; skipping")
        return src, False
    return src.replace(anchor, anchor + "\n" + text), True


def _patch_wan_diffusion() -> bool:
    try:
        src = open(WAN_DIFFUSION, encoding="utf-8").read()
    except OSError as exc:
        _log(f"SKIP (cannot read {WAN_DIFFUSION}: {exc})")
        return False

    # 1.3B sampler signature (sample()).
    src, ok = _insert_after(
        src, "    def sample(\n        self,\n        prompt_embeds=None,",
        "        progress_cb=None,",
        "sample() signature")
    if not ok:
        return False

    # 14B sampler signature (sample_bernini_wvitcfg()).
    src, ok = _insert_after(
        src, "        **kwargs,\n    ):",
        "        progress_cb=None,",
        "sample_bernini_wvitcfg() signature")
    if not ok:
        return False

    # Per-step hook at the top of every denoise loop (both samplers share the
    # same loop opener, so this covers sample() AND sample_bernini_wvitcfg()).
    src, ok = _insert_after_each(
        src, "        for t_idx, t in enumerate(timesteps):", _HOOK,
        "denoise loops")
    if not ok:
        return False

    open(WAN_DIFFUSION, "w", encoding="utf-8").write(src)
    _log(f"OK ({WAN_DIFFUSION})")
    return True


def _patch_pipeline() -> bool:
    try:
        src = open(PIPELINE, encoding="utf-8").read()
    except OSError as exc:
        _log(f"SKIP (cannot read {PIPELINE}: {exc})")
        return False

    # 1.3B BerniniRendererPipeline.__call__ -> sample().
    sig1 = "    def __call__(\n        self,\n        prompt: str,\n" \
           "        *,\n        neg_prompt: str = \"\","
    src, ok = _insert_after(
        src, sig1, "        progress_cb=None,",
        "BerniniRendererPipeline.__call__ signature")
    if not ok:
        return False

    # Forward progress_cb into the sampler calls (both already-forwarded when
    # the marker is present -> idempotent across rebuilds).
    if "progress_cb=progress_cb," not in src:
        fwd1 = "            momentum=momentum,\n        )"
        if src.count(fwd1) != 1:
            _log("WARN 1.3B sample() call anchor not found/ambiguous; skipping forward")
            return False
        src = src.replace(fwd1,
                          "            momentum=momentum,\n            progress_cb=progress_cb,\n        )",
                          1)

        # 14B BerniniPipeline.__call__ -> sample_bernini_wvitcfg().
        sig2 = "        max_sequence_length: int = 512,\n    ):"
        src, ok = _insert_after(
            src, sig2, "        progress_cb=None,",
            "BerniniPipeline.__call__ signature")
        if not ok:
            return False
        fwd2 = "            device=device,\n        )"
        if src.count(fwd2) != 1:
            _log("WARN 14B sample_bernini_wvitcfg() call anchor not found/ambiguous; skipping forward")
            return False
        src = src.replace(fwd2,
                          "            device=device,\n            progress_cb=progress_cb,\n        )",
                          1)

    open(PIPELINE, "w", encoding="utf-8").write(src)
    _log(f"OK ({PIPELINE})")
    return True


def main() -> int:
    _patch_wan_diffusion()
    _patch_pipeline()
    return 0


if __name__ == "__main__":
    sys.exit(main())
