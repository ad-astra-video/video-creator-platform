"""Build-time source patch: "plan once across the full timeline" for the
deployed 14B Bernini pipeline (upstream `BerniniPipeline.__call__`, no
render_prepared seam in the image).

Adds:
  * BerniniPipeline.plan_full(...) -- run the MAR planner ONCE over the whole
    source timeline, returning shared text/T5 conditioning + the frame-ordered
    ViT plan tensor (`wotxt_wvit`) + per-frame token offsets.
  * `__call__(plan=None, ...)` -- when a precomputed per-chunk plan is given,
    skip the embeds-build / sample_vit_embed / T5-encode and reassemble the
    four cond_embeds from the plan slice, so every chunk renders from ONE
    coherent global plan instead of re-planning per chunk.

cond_embeds from feat_from_planner_to_renderer are token subsets under masks:
  C_wtxt_wvit   text+vit (interleaved)      -- renderer conditioning sequence
  C_wtxt_wvovit text only                   -- SHARED across chunks
  C_wotxt_wvit  vit only (frame-ordered)    -- SLICE per chunk
  C_wotxt_wvovit base                       -- SHARED
We assume the ViT tokens are the trailing contiguous block (text-then-vit), so
chunk_wtxt_wvit = cat([C_wtxt_wvovit, chunk_vit]). Validated on-device.
"""

PIPELINE = "/opt/bernini/src/bernini/pipeline.py"

_PLAN_FULL = r'''
    @torch.no_grad()
    def plan_full(
        self,
        sample,
        prompt: str,
        neg_prompt: str = "",
        num_frames: int = 89,
        planning_step: int = 25,
        vit_txt_cfg: float = 1.4,
        vit_img_cfg: float = 1.2,
        vit_denoising_step: int = 3,
        t5_tokenizer=None,
        weight_dtype=None,
    ):
        """Plan the FULL source timeline ONCE.

        Reuses the normal __call__ plumbing (transform -> embeds -> planner)
        for the whole clip, then returns shared text/T5 conditioning plus the
        frame-ordered ViT plan (``wotxt_wvit``) and per-frame token offsets so
        a later ``__call__(plan=...)`` can render each chunk from a slice of
        one coherent global plan. ``sample`` is the full-source preprocessed
        dict (from preprocess_inputs).
        """
        import torch
        device = self.device
        wd = weight_dtype or getattr(self, "weight_dtype", torch.bfloat16)
        # move planner components to GPU (mirrors __call__ preamble)
        self.model.mllm.to(device).to(wd)
        if self.connector is not None:
            self.connector.to(device=device, dtype=wd)
        if getattr(self.model, "vit_decoder", None) is not None:
            self.model.vit_decoder.to(device=device, dtype=wd)
        torch.cuda.empty_cache()
        input_dict = self.transform_inputs(
            sample, num_frames, task_name="v2v", neg_prompt=neg_prompt,
        )

        def _mv(o):
            if isinstance(o, torch.Tensor):
                return o.to(device)
            if isinstance(o, dict):
                return {k: _mv(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_mv(v) for v in o]
            if isinstance(o, tuple):
                return tuple(_mv(v) for v in o)
            return o
        input_dict = _mv(input_dict)
        inputs = input_dict["inputs"]
        un = input_dict["uncond_inputs"]
        ic = input_dict["imgcond_inputs"]

        def _emb(inp):
            return self.model.format_mllm_inputs_embeds(
                input_ids=inp["input_ids"], visual_embeds=inp["visual_embeds"],
                visual_input_mask=inp["visual_input_token_mask"],
                visual_output_mask=inp["visual_output_token_mask"],
            ).to(wd)
        in_e, un_e, ic_e = _emb(inputs), _emb(un), _emb(ic)

        def _post(e, mask):
            return self.model.post_process_input_embeds(
                e.unsqueeze(0), mask, tgt_vit_mask=None, inference=True)["input_embeds"]
        in_p, un_p, ic_p = (_post(x, m) for x, m in
                            ((in_e, inputs["visual_output_token_mask"]),
                             (un_e, un["visual_output_token_mask"]),
                             (ic_e, ic["visual_output_token_mask"])))

        ret = self.sample_vit_embed(
            input_embeds=in_p, attention_mask_4d=inputs["attention_mask_4d"].unsqueeze(0),
            position_ids=inputs["position_ids"].unsqueeze(0),
            visual_output_token_mask=inputs["visual_output_token_mask"],
            uncond_input_embeds=un_p, uncond_position_ids=un["position_ids"].unsqueeze(0),
            uncond_attention_mask_4d=un["attention_mask_4d"].unsqueeze(0),
            uncond_visual_output_token_mask=un["visual_output_token_mask"],
            imgcond_input_embeds=ic_p, imgcond_position_ids=ic["position_ids"].unsqueeze(0),
            imgcond_attention_mask_4d=ic["attention_mask_4d"].unsqueeze(0),
            imgcond_visual_output_token_mask=ic["visual_output_token_mask"],
            planning_step=planning_step, vit_txt_cfg=vit_txt_cfg,
            vit_img_cfg=vit_img_cfg, vit_denoising_step=vit_denoising_step,
        )
        if self.connector is not None:
            self.connector.to("cpu")
        if getattr(self.model, "vit_decoder", None) is not None:
            self.model.vit_decoder.to("cpu")
        import gc
        gc.collect(); torch.cuda.empty_cache()

        if getattr(self.model, "t5_text_encoder", None) is not None:
            self.model.t5_text_encoder.to(device)
        from .pipeline import _get_t5_text_ids, _prompt_clean
        t5_ids, t5_mask = _get_t5_text_ids(prompt, t5_tokenizer or self.t5_tokenizer)
        t5_embeds = self.model.get_t5_text_embeddings_sample(
            t5_ids.to(device), t5_mask.to(device))
        neg_ids, neg_mask = _get_t5_text_ids(
            _prompt_clean(neg_prompt), t5_tokenizer or self.t5_tokenizer)
        neg_t5_embeds = self.model.get_t5_text_embeddings_sample(
            neg_ids.to(device), neg_mask.to(device))
        if getattr(self.model, "t5_text_encoder", None) is not None:
            self.model.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()

        # Frame-ordered per-video-clip token offsets (inline; self-contained).
        # The true token divisor is NOT the image merge^2 (4): video rows have
        # an extra temporal factor, so a row [t,h,w] maps to (t*h*w)//D tokens.
        # Derive D from the real visual-token count so the offsets always sum
        # exactly to len(wotxt_wvit).
        grid = sample["video_grid_thw"]
        _raw_rows = [int(t) * int(h) * int(w) for (t, h, w) in grid]
        _raw_sum = sum(_raw_rows)
        _T = ret["cond_embeds_wotxt_wvit"].shape[1] if ret["cond_embeds_wotxt_wvit"] is not None else 0
        _D = max(1, round(_raw_sum / _T)) if _T else 8
        _c, offsets = 0, []
        for _row in _raw_rows:
            _nt = _row // _D
            offsets.append((_c, _c + _nt))
            _c += _nt

        return dict(
            wtxt_wvovit=ret["cond_embeds_wtxt_wovit"].cpu() if ret["cond_embeds_wtxt_wovit"] is not None else None,
            wotxt_wvit=ret["cond_embeds_wotxt_wvit"].cpu() if ret["cond_embeds_wotxt_wvit"] is not None else None,
            wotxt_wvovit=ret["cond_embeds_wotxt_wovit"].cpu() if ret["cond_embeds_wotxt_wovit"] is not None else None,
            vit_frame_offsets=offsets,
            n_frames=int(num_frames),
            t5_embeds=t5_embeds.cpu(),
            neg_t5_embeds=neg_t5_embeds.cpu(),
        )
'''

# __call__ signature: add plan=None after max_sequence_length
_SIG_OLD = "        max_sequence_length: int = 512,\n"
_SIG_NEW = "        max_sequence_length: int = 512,\n        plan=None,\n"

# Branch inserted at the __call__ embeds-build anchor: when plan given, skip
# planner+T5 and rebuild cond_embeds from the slice.
_CALL_ANCHOR = "        input_embeds = self.model.format_mllm_inputs_embeds(\n"
_IF_BODY = (
    "        if plan is not None:\n"
    "            _wvit = plan[\"wotxt_wvit\"].to(device)\n"
    "            _wtxt = plan[\"wtxt_wvovit\"].to(device)\n"
    "            _wovit = plan[\"wotxt_wvovit\"].to(device)\n"
    "            cond_embeds_wtxt_wvit = torch.cat([_wtxt, _wvit], dim=1)\n"
    "            cond_embeds_wtxt_wovit = _wtxt\n"
    "            cond_embeds_wotxt_wvit = _wvit\n"
    "            cond_embeds_wotxt_wovit = _wovit\n"
    "            _t5 = plan[\"t5_embeds\"].to(device)\n"
    "            _nt5 = plan[\"neg_t5_embeds\"].to(device)\n"
    "            cond_embeds_wtxt_wvit = torch.cat([_t5, cond_embeds_wtxt_wvit], dim=1)\n"
    "            if cond_embeds_wtxt_wovit is not None:\n"
    "                cond_embeds_wtxt_wovit = torch.cat([_t5, cond_embeds_wtxt_wovit], dim=1)\n"
    "            if cond_embeds_wotxt_wvit is not None:\n"
    "                cond_embeds_wotxt_wvit = torch.cat([_nt5, cond_embeds_wotxt_wvit], dim=1)\n"
    "            cond_embeds_wotxt_wovit = torch.cat([_nt5, cond_embeds_wotxt_wovit], dim=1)\n"
    "            # release the MLLM/planner so the renderer fits (normal path does\n"
    "            # this after the planner; the plan branch must do it too)\n"
    "            self.model.mllm.to(\"cpu\")\n"
    "            if self.connector is not None:\n"
    "                self.connector.to(\"cpu\")\n"
    "            if getattr(self.model, \"vit_decoder\", None) is not None:\n"
    "                self.model.vit_decoder.to(\"cpu\")\n"
    "            torch.cuda.empty_cache()\n"
)
_ELSE_A = "        else:\n"
_ELSE_T = "            input_embeds = self.model.format_mllm_inputs_embeds(\n"


_BLOCK_END = (
    "        cond_embeds_wotxt_wovit = torch.cat([neg_t5_embeds, cond_embeds_wotxt_wovit], dim=1)\n"
)


def _reindent(block, add=4):
    out = []
    for ln in block.splitlines(keepends=True):
        out.append(" " * add + ln if ln.strip() else ln)
    return "".join(out)


def _apply(path=PIPELINE):
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()

    # 1. insert OR replace plan_full (before the last def __call__)
    idx_call = src.rfind("    def __call__(")
    assert idx_call != -1, "__call__ not found"
    if "def plan_full(" in src:
        p0 = src.find("    @torch.no_grad()\n    def plan_full(")
        nxt = src.find("    @torch.no_grad()\n    def ", p0 + 1)
        if nxt == -1 or nxt > idx_call:
            nxt = idx_call
        src = src[:p0] + _PLAN_FULL + src[nxt:]
        idx_call = src.rfind("    def __call__(")  # recompute (insert shifted)
    else:
        src = src[:idx_call] + _PLAN_FULL + src[idx_call:]

    # 2. add plan=None to __call__ signature (after max_sequence_length param)
    if "plan=None," not in src:
        after = src[idx_call:]
        sig_rel = after.find(_SIG_OLD)
        assert sig_rel != -1, "__call__ signature anchor missing"
        pos = idx_call + sig_rel
        src = src[:pos] + _SIG_NEW + src[pos + len(_SIG_OLD):]

    # 3. wrap the planner+T5 block in "if plan is not None: ... else:".
    # The block runs from __call__'s embeds-build anchor to the last T5-concat
    # line; we re-indent it +4 to nest under `else:`. If the branch already
    # exists, replace only its if-body (keeps the else+reindented block).
    after = src[idx_call:]
    if_start = after.find("        if plan is not None:\n")
    if if_start != -1:
        else_start = after.find(_ELSE_A, if_start)
        assert else_start != -1, "plan-branch else anchor missing"
        pos_if = idx_call + if_start
        pos_else = idx_call + else_start
        src = src[:pos_if] + _IF_BODY + src[pos_else:]
    else:
        s_rel = after.find(_CALL_ANCHOR)
        e_rel = after.find(_BLOCK_END)
        assert s_rel != -1, "__call__ embeds anchor missing"
        assert e_rel != -1, "__call__ t5-concat end anchor missing"
        s = idx_call + s_rel
        e = idx_call + e_rel + len(_BLOCK_END)
        block = src[s:e]
        src = src[:s] + _IF_BODY + _ELSE_A + _reindent(block) + src[e:]

    with open(path, "w", encoding="utf-8") as f:
        f.write(src)

    import py_compile
    py_compile.compile(path, doraise=True)
    print("[bernini_planonce_patch] applied to", path, "; py_compile OK", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", default=PIPELINE)
    a = ap.parse_args()
    _apply(a.pipeline)
