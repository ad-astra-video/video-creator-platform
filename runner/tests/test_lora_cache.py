"""GPU-independent tests for the runner's LoRA catalog cache (budget + LRU).

Runs WITHOUT torch / ltx / a GPU. Constructs LoraCache with explicit args so it
never touches runner.config or the network.

Run:
    python -m pytest tests/test_lora_cache.py -q
or standalone:
    python tests/test_lora_cache.py
"""

import os
import sys

# Make the `runner` package importable without installing it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest  # noqa: E402

from runner.ltx import loracache  # noqa: E402

_MIB = 1024 * 1024


def _fake_catalog():
    """Narrowed {id: entry} catalog — the shape LoraCache consumes directly."""
    def entry(lora_id, size):
        return {
            "repo_id": f"R/{lora_id}",
            "variants": {
                f"{lora_id}.safetensors": {
                    "variant_id": "default", "label": "Default", "size_bytes": size,
                }
            },
            "default_filename": f"{lora_id}.safetensors",
            "recommended_strength": 1.0,
        }

    return {
        "a": entry("a", 4 * _MIB),
        "b": entry("b", 4 * _MIB),
        "c": entry("c", 4 * _MIB),
    }


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A LoraCache with ~10 MiB budget and 4 MiB LoRAs; downloads write real files."""
    c = loracache.LoraCache(
        cache_dir=str(tmp_path / "loras"),
        size_gb=0.01,          # ~10.49 MiB budget
        catalog=_fake_catalog(),
        hf_token=None,
    )
    downloads = {"count": 0}

    def fake_download(repo_id, filename, dest):
        downloads["count"] += 1
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        size = c._catalog[os.path.basename(os.path.dirname(dest))][  # lora_id dir
            "variants"][filename]["size_bytes"]
        with open(dest, "wb") as fh:
            fh.write(b"\0" * size)

    monkeypatch.setattr(c, "_download", fake_download)
    return c, downloads


def test_catalog_only_rejects_unknown(cache):
    c, _ = cache
    with pytest.raises(KeyError):
        c.ensure("nope")


def test_catalog_only_rejects_unknown_file(cache):
    c, _ = cache
    with pytest.raises(KeyError):
        c.ensure("a", "missing.safetensors")


def test_downloads_and_caches(cache):
    c, dl = cache
    p1 = c.ensure("a")
    assert os.path.isfile(p1)
    assert dl["count"] == 1
    # Reuse — no second download.
    p2 = c.ensure("a")
    assert p2 == p1
    assert dl["count"] == 1


def test_lru_eviction(cache):
    c, _ = cache
    c.ensure("a")               # 4 MiB
    c.ensure("b")               # 4 MiB (total 8 MiB < ~10.5 MiB)
    c.ensure("a")               # touch a -> b is now LRU
    c.ensure("c")               # +4 MiB = 12 MiB > budget -> evict b (oldest)
    present = {os.path.basename(f) for f in c._cached_files()}
    assert present == {"a.safetensors", "c.safetensors"}


def test_single_large_lora_allowed(cache):
    """A LoRA bigger than the whole budget is still servable (evicts all else)."""
    small, _ = cache
    # Fresh cache with a 1 MiB budget holding a single 4 MiB LoRA.
    c = loracache.LoraCache(
        cache_dir=str(os.path.join(str(small.cache_dir), "big")),
        size_gb=0.001,           # ~1.05 MiB budget
        catalog=_fake_catalog(),
        hf_token=None,
    )
    # No monkeypatch here — bypass _download by writing the file ourselves.
    os.makedirs(os.path.join(c.cache_dir, "a"), exist_ok=True)
    dest = os.path.join(c.cache_dir, "a", "a.safetensors")
    with open(dest, "wb") as fh:
        fh.write(b"\0" * (4 * _MIB))
    # ensure() sees the file already present and returns it.
    assert c.ensure("a") == dest


def test_parse_catalog_narrows():
    raw = {
        "schema_version": 1,
        "loras": [
            {
                "id": "fpv-motion",
                "download": {
                    "repo_id": "chsengni/ltx2.3-fpv-motion",
                    "variants": [{
                        "id": "default", "label": "Default",
                        "filename": "ltx2.3_fpv_motion.safetensors",
                        "size_bytes": 308666208,
                    }],
                },
                "recommended_strength": 1.0,
            },
            {"id": "no-file", "download": {"repo_id": "org/x"}},  # skipped
        ],
    }
    reg = loracache._parse_catalog(raw)
    assert "fpv-motion" in reg
    assert reg["fpv-motion"]["repo_id"] == "chsengni/ltx2.3-fpv-motion"
    assert reg["fpv-motion"]["default_filename"] == "ltx2.3_fpv_motion.safetensors"
    assert "no-file" not in reg


if __name__ == "__main__":
    # Minimal standalone runner (no pytest).
    c = loracache.LoraCache(
        cache_dir="/tmp/lora_test_cache",
        size_gb=0.01,
        catalog=_fake_catalog(),
        hf_token=None,
    )
    print("cache dir:", c.cache_dir, "budget(MiB):", round(c.budget_bytes / _MIB, 2))
    print("done")
