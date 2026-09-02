"""Model-selection tests for the model-aware /load path.

The wan-worker (the id-v2v worker code) can serve two model families — the
diffsynth id-v2v pipe (restyle) and the Bernini subprocess (edit/t2v). The
live-runner derives which model to make resident from the route
(``model_for_endpoint``) and the worker resolves the ``model`` /load arg via
``config.resolve_model``. These are pure constants/helpers (no torch/aiohttp
server), so they run anywhere.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from runner.live_runner.routing import (  # noqa: E402
    BERNINI_ENDPOINTS,
    ROUTES,
    model_for_endpoint,
)
from runner.idv2v import config  # noqa: E402


def test_bernini_routes_default_to_bernini_1_3b():
    for ep in ("bernini-t2v", "bernini-v2v", "bernini-r2v"):
        assert model_for_endpoint(ep) == "bernini-1.3b"
        assert ep in BERNINI_ENDPOINTS


def test_bernini_route_prefers_body_engine():
    assert model_for_endpoint("bernini-v2v", {"engine": "bernini-14b"}) == "bernini-14b"
    assert model_for_endpoint("bernini-r2v", {"engine": "bernini-1.3b"}) == "bernini-1.3b"
    # Non-string / empty engine falls back to the default.
    assert model_for_endpoint("bernini-v2v", {"engine": 0}) == "bernini-1.3b"


def test_bernini_route_prefers_body_model_short_id():
    # The frontend sends the SHORT engine id in `model` (bernini-delivery.ts:
    # `model: target.engine` -> "1.3b"/"14b"), NOT `engine`. This is the
    # regression: model_for_endpoint must honour `model`, or a 14b request wars
    # the 1.3b default and (with a device-less generation) runs 1.3b.
    assert model_for_endpoint("bernini-v2v", {"model": "14b"}) == "bernini-14b"
    assert model_for_endpoint("bernini-v2v", {"model": "1.3b"}) == "bernini-1.3b"
    assert model_for_endpoint("bernini-t2v", {"model": "bernini-14b"}) == "bernini-14b"
    # An explicit `model` (short id) wins over a stale `engine`.
    assert model_for_endpoint("bernini-r2v", {"model": "14b", "engine": "bernini-1.3b"}) == "bernini-14b"
    # Unknown model string falls back to the default.
    assert model_for_endpoint("bernini-v2v", {"model": "idv2v"}) == "bernini-1.3b"


def test_restyle_route_loads_idv2v():
    assert model_for_endpoint("restyle") == "idv2v"


def test_every_route_resolves_a_model():
    # Model selection is threaded across ALL workers: no route falls back to
    # "worker picks on its own" — model_for_endpoint returns a value for every
    # endpoint in ROUTES (single-model workers get their family label; the
    # wan-worker gets bernini vs idv2v, which is where the value bites).
    for ep in ROUTES:
        assert model_for_endpoint(ep) is not None, f"{ep} has no model mapping"


def test_single_model_worker_families():
    # LTX / gemma / vp workers are single-model: their routes carry family labels.
    for ep in ("t2v", "i2v", "a2v", "extend", "retake",
               "extract-conditioning", "ic-lora-generate"):
        assert model_for_endpoint(ep) == "ltx"
    for ep in ("prompt-enhance", "suggest-gap-prompt", "chat", "suggest-layers"):
        assert model_for_endpoint(ep) == "gemma"
    for ep in ("process", "fps-boost", "upscale", "ffmpeg"):
        assert model_for_endpoint(ep) == "vp"


def test_image_route_model_is_advisory():
    # image-worker selects the real model per-request from the body engine, so
    # /load model is a warm hint: body engine wins, else the family label.
    for ep in ("image", "edit", "layer", "style-frame"):
        assert model_for_endpoint(ep) == "image"
    assert model_for_endpoint("edit", {"engine": "hydream"}) == "hydream"
    assert model_for_endpoint("image", {"engine": "qwen-edit"}) == "qwen-edit"


def test_resolve_model_maps_load_arg_to_kind():
    # Bernini ids -> bernini kinds (first-class load participants), idv2v
    # variants/blank -> idv2v.
    assert config.resolve_model("bernini-1.3b") in config.BERNINI_MODELS
    assert config.resolve_model("bernini-14b") in config.BERNINI_MODELS
    assert config.resolve_model("bernini") == "bernini-14b"
    for m in ("", "fast", "regular", "idv2v"):
        assert config.resolve_model(m) == "idv2v"


def test_bernini_models_contains_both_kinds():
    assert "bernini-1.3b" in config.BERNINI_MODELS
    assert "bernini-14b" in config.BERNINI_MODELS


def test_normalize_target_device_shapes():
    # Device comes from the live-runner: int index, bare "N", or "cuda:N" all
    # normalize to a full cuda device; empty stays empty (worker rejects it) —
    # there is NO invented default.
    from runner.idv2v.bernini import _normalize_target_device
    assert _normalize_target_device(3) == "cuda:3"
    assert _normalize_target_device("2") == "cuda:2"
    assert _normalize_target_device("cuda:1") == "cuda:1"
    # With no configured device and none supplied, result is empty (not cuda:0).
    import runner.idv2v.bernini as bernini_mod
    old_gpu = config.GPU_DEVICE
    old_bernini = config.BERNINI_GPU_DEVICE
    try:
        config.GPU_DEVICE = ""
        config.BERNINI_GPU_DEVICE = ""
        # _normalize_target_device reads module-level constants through config,
        # which in bernini.py are imported as `from . import config`; it reads
        # config.GPU_DEVICE / config.BERNINI_GPU_DEVICE live.
        assert _normalize_target_device(None) == ""
    finally:
        config.GPU_DEVICE = old_gpu
        config.BERNINI_GPU_DEVICE = old_bernini
