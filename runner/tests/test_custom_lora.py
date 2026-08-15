"""GPU-independent tests for the runner's CUSTOM (client-supplied URL) LoRA path.

Runs without torch / ltx / a GPU. Covers the custom-LoRA safeguards:
  - https-only + host allowlist (hf.co)
  - .safetensors-only
  - streaming size cap
  - optional sha256 verification
  - optional per-request HF token (authorization header)
  - one-shot cleanup + exclusion from the catalog LRU budget

Run:
    python -m pytest tests/test_custom_lora.py -q
"""

import hashlib
import http.server
import os
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from runner.ltx import loracache  # noqa: E402


def _mk_cache(tmp_path):
    c = loracache.LoraCache(
        cache_dir=str(tmp_path / "loras"),
        size_gb=1.0,
        catalog={},
        hf_token=None,
    )
    return c


# ---------------------------------------------------------------------------
# URL validation (allowlist / https / extension)
# ---------------------------------------------------------------------------

class TestValidation:
    def test_https_required(self, tmp_path):
        c = _mk_cache(tmp_path)
        with pytest.raises(ValueError):
            c._validate_custom_url("http://huggingface.co/u/r/resolve/main/x.safetensors")

    def test_host_allowlist(self, tmp_path):
        c = _mk_cache(tmp_path)
        with pytest.raises(ValueError):
            c._validate_custom_url("https://evil.example.com/u/r/resolve/main/x.safetensors")
        with pytest.raises(ValueError):
            c._validate_custom_url("https://huggingface.co.evil.com/u/r/x.safetensors")

    def test_hf_ok(self, tmp_path):
        c = _mk_cache(tmp_path)
        url = "https://huggingface.co/some/user/repo/resolve/main/x.safetensors"
        assert c._validate_custom_url(url) == url

    def test_hf_subdomain_ok(self, tmp_path):
        c = _mk_cache(tmp_path)
        url = "https://cdn-lfs.huggingface.co/some/blob.safetensors"
        assert c._validate_custom_url(url) == url

    def test_extension_required(self, tmp_path):
        c = _mk_cache(tmp_path)
        with pytest.raises(ValueError):
            c._validate_custom_url("https://huggingface.co/u/r/resolve/main/x.pt")
        with pytest.raises(ValueError):
            c._validate_custom_url("https://huggingface.co/u/r/")


# ---------------------------------------------------------------------------
# Download flow with safeguards (validation bypassed via a local http server)
# ---------------------------------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    seen_headers = {}
    body = b""

    def do_GET(self):  # noqa: N802
        _Handler.seen_headers = dict(self.headers)
        self.send_response(200)
        self.send_header("Content-Length", str(len(_Handler.body)))
        self.end_headers()
        self.wfile.write(_Handler.body)

    def log_message(self, *a):  # silence
        pass


@pytest.fixture
def served(tmp_path, monkeypatch):
    """Start a local file server; monkeypatch validation so download_custom
    streams from it (bypassing the hf.co allowlist used for prod)."""
    blob = b"\x00" * 4096
    _Handler.body = blob
    _Handler.seen_headers = {}
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/x.safetensors", blob
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def bypass(monkeypatch):
    monkeypatch.setattr(loracache.LoraCache, "_validate_custom_url",
                        lambda self, u: u)


def test_download_ok_and_cleanup(tmp_path, served, bypass):
    c = _mk_cache(tmp_path)
    url, blob = served
    p = c.download_custom(url)
    assert os.path.isfile(p)
    assert os.path.basename(p).endswith(".safetensors")
    with open(p, "rb") as fh:
        assert fh.read() == blob
    # one-shot cleanup removes the file
    c.remove_custom(p)
    assert not os.path.exists(p)

    # custom files are excluded from the catalog disk budget
    p2 = c.download_custom(url)
    assert c._total_bytes() == 0
    c.remove_custom(p2)


def test_download_size_cap(tmp_path, served, bypass):
    c = _mk_cache(tmp_path)
    c._max_custom_bytes = 1024  # served blob is 4096 bytes
    url, _ = served
    with pytest.raises(ValueError):
        c.download_custom(url)
    assert not os.listdir(c._custom_dir)  # partial cleaned up


def test_download_sha_mismatch(tmp_path, served, bypass):
    c = _mk_cache(tmp_path)
    url, blob = served
    wrong = hashlib.sha256(b"other").hexdigest()
    with pytest.raises(ValueError):
        c.download_custom(url, sha256=wrong)
    assert not os.listdir(c._custom_dir)


def test_download_sha_ok(tmp_path, served, bypass):
    c = _mk_cache(tmp_path)
    url, blob = served
    good = hashlib.sha256(blob).hexdigest()
    p = c.download_custom(url, sha256=good)
    assert os.path.isfile(p)
    c.remove_custom(p)


def test_download_sends_token(tmp_path, served, bypass):
    c = _mk_cache(tmp_path)
    url, _ = served
    p = c.download_custom(url, token="hf_secret")
    assert _Handler.seen_headers.get("Authorization") == "Bearer hf_secret"
    c.remove_custom(p)


def test_sweep_removes_orphans(tmp_path, served, bypass):
    c = _mk_cache(tmp_path)
    url, _ = served
    p = c.download_custom(url)
    assert os.path.isfile(p)
    # Age the file beyond the TTL and sweep
    old = time.time_ns() - (c._custom_ttl + 60) * 1_000_000_000
    os.utime(p, ns=(old, old))
    c._sweep_custom()
    assert not os.path.exists(p)


import time  # noqa: E402  (used by test_sweep_removes_orphans)
