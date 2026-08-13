"""Validation that worker auth rejects token-less / mismatched requests.

The live-runner -> worker link is authenticated with a shared WORKER_TOKEN sent
in the X-Worker-Token header. This proves that (a) a request without the header,
and (b) a request with the wrong token, are both rejected, while (c) the correct
token is accepted. Model import is stubbed so the test runs without diffsynth/torch.
"""

import importlib
import sys
import types

import pytest

# --- stub the heavy model module before importing server -------------------
from aiohttp import web
from types import SimpleNamespace

stub_model = types.ModuleType("runner.idv2v.model")
stub_run = types.ModuleType("runner.idv2v.run")


def fake_health_check(*_a, **_k):
    return {"model_loaded": False}


stub_model.ModelManager = object  # type: ignore[attr-defined]
stub_model.health_check = fake_health_check  # type: ignore[attr-defined]


@pytest.fixture()
def server_pkg(monkeypatch):
    import importlib

    for name in list(sys.modules):
        if name.startswith("runner.idv2v"):
            del sys.modules[name]
    monkeypatch.setitem(sys.modules, "runner.idv2v.model", stub_model)
    monkeypatch.setitem(sys.modules, "runner.idv2v.run", stub_run)

    from runner.idv2v import config
    monkeypatch.setattr(config, "WORKER_TOKEN", "test-shared-token-123")
    # Reload config-dependent module state
    monkeypatch.setitem(sys.modules, "runner.idv2v.config", config)

    server = importlib.import_module("runner.idv2v.server")
    return server


def _make_request(token: str | None) -> SimpleNamespace:
    """Minimal request-like object exposing only `.headers` (what _require_token reads)."""
    return SimpleNamespace(headers={"X-Worker-Token": token} if token else {})


def test_blank_token_rejected(server_pkg) -> None:
    with pytest.raises(web.HTTPForbidden):
        server_pkg._require_token(_make_request(None))


def test_wrong_token_rejected(server_pkg) -> None:
    with pytest.raises(web.HTTPForbidden):
        server_pkg._require_token(_make_request("wrong-token"))


def test_correct_token_accepted(server_pkg) -> None:
    # Should not raise.
    server_pkg._require_token(_make_request("test-shared-token-123"))
