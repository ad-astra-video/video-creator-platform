"""Disk-bounded LRU cache for catalog LoRAs on the LTX remote runner.

The operator controls how much disk the LoRA cache may use via
``LORA_CACHE_SIZE_GB`` (see ``runner.config``). On first use of a catalog LoRA
the cache downloads its weights into ``LORA_CACHE_DIR``; when a new download
would exceed the budget it evicts the least-recently-used files to make room.

The catalog is the same ``lora_catalog.json`` the LTX-Desktop app uses: by
default the runner downloads it from the main LTX-Desktop repo
(``LORA_CATALOG_SOURCE``); an operator can point that at a different URL or a
local file. Only LoRAs that appear in the catalog can be fetched (no arbitrary
repo from a remote client).

This module is deliberately GPU-independent (file-size + mtime bookkeeping), so
it is unit-testable without torch / ltx / a GPU.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time

logger = logging.getLogger(__name__)

_ONE_GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def _read_source(source: str) -> str:
    """Read a catalog source — an http(s) URL or a local file path."""
    if not source:
        raise ValueError("LoRA catalog source is empty (check LORA_CATALOG_SOURCE)")
    if source.startswith(("http://", "https://")):
        import requests
        resp = requests.get(source, timeout=15)
        resp.raise_for_status()
        return resp.text
    with open(source, encoding="utf-8") as fh:
        return fh.read()


def _parse_catalog(data: dict) -> dict:
    """Narrow a parsed LoraCatalogFile dict to ``{id: entry}``.

    Only the fields the runner needs are kept:
      entry = {
        "repo_id": str,
        "variants": {filename: {"variant_id", "label", "size_bytes"}},
        "default_filename": str,
        "recommended_strength": float,
      }
    """
    registry: dict = {}
    for item in data.get("loras", []) or []:
        lora_id = item.get("id")
        download = item.get("download") or {}
        if not lora_id or not download.get("repo_id"):
            logger.warning("Skipping catalog entry missing id/repo_id: %r", item)
            continue
        variants = {}
        for v in download.get("variants", []) or []:
            fname = v.get("filename")
            if not fname:
                continue
            variants[fname] = {
                "variant_id": v.get("id"),
                "label": v.get("label", ""),
                "size_bytes": v.get("size_bytes", 0),
            }
        # Support a bare `filename` fallback when no variants are declared.
        if not variants:
            bare = download.get("filename")
            if bare:
                variants[bare] = {"variant_id": "default", "label": "Default", "size_bytes": 0}
        if not variants:
            logger.warning("Catalog entry %r has no downloadable file; skipping", lora_id)
            continue
        registry[lora_id] = {
            "repo_id": download["repo_id"],
            "variants": variants,
            "default_filename": download.get("default_filename")
            or (download["variants"][0]["filename"] if download.get("variants") else download.get("filename"))
            or next(iter(variants)),
            "recommended_strength": float(item.get("recommended_strength", 1.0)),
        }
    return registry


def load_catalog(source: str) -> dict:
    """Fetch + parse a LoraCatalogFile from *source* -> narrowed registry."""
    raw = _read_source(source)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LoRA catalog at {source} is not valid JSON: {exc}") from exc
    return _parse_catalog(data)


# ---------------------------------------------------------------------------
# LoraCache
# ---------------------------------------------------------------------------

class LoraCache:
    """Download + LRU-evict catalog LoRAs within an operator-set disk budget."""

    def __init__(
        self,
        cache_dir: str | None = None,
        size_gb: float | None = None,
        catalog: dict | None = None,
        source: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        # Import config lazily and only when the caller didn't supply an explicit
        # value, so tests (which pass every arg) stay hermetic.
        if cache_dir is not None and size_gb is not None and hf_token is not None:
            LORA_CACHE_DIR, LORA_CACHE_SIZE_GB, LORA_HF_TOKEN = cache_dir, size_gb, hf_token
        else:
            from runner.ltx.config import (
                LORA_CACHE_DIR,
                LORA_CACHE_SIZE_GB,
                LORA_HF_TOKEN,
            )
        if catalog is None:
            if not source:
                from runner.ltx.config import LORA_CATALOG_SOURCE
                source = LORA_CATALOG_SOURCE
            self._catalog = load_catalog(source)
        else:
            self._catalog = catalog

        self.cache_dir = cache_dir or LORA_CACHE_DIR
        self.budget_bytes = int((size_gb if size_gb is not None else LORA_CACHE_SIZE_GB) * _ONE_GB)
        self._hf_token = hf_token if hf_token is not None else LORA_HF_TOKEN
        self._lock = threading.RLock()
        os.makedirs(self.cache_dir, exist_ok=True)

    # -- public ------------------------------------------------------------

    @property
    def catalog(self) -> dict:
        return self._catalog

    def ensure(self, lora_id: str, filename: str | None = None) -> str:
        """Return the local path to *lora_id*, downloading + LRU-evicting as needed.

        Raises ``KeyError`` if the id/filename isn't in the catalog (catalog-only).
        """
        with self._lock:
            entry = self._catalog.get(lora_id)
            if entry is None:
                raise KeyError(
                    f"catalog has no LoRA {lora_id!r} — refusing to download an "
                    "arbitrary model (only catalog LoRAs are allowed)"
                )
            fname = filename or entry["default_filename"]
            if fname not in entry["variants"]:
                raise KeyError(
                    f"LoRA {lora_id!r} has no file {fname!r} "
                    f"(known: {sorted(entry['variants'])})"
                )

            dest = os.path.join(self.cache_dir, lora_id, fname)
            if os.path.isfile(dest):
                self._touch(dest)
                return dest

            os.makedirs(os.path.dirname(dest), exist_ok=True)
            self._download(entry["repo_id"], fname, dest)
            # The new file is already on disk, so _total_bytes() now includes it.
            self._evict_to_fit(keep=[dest])
            self._touch(dest)
            size = os.path.getsize(dest)
            logger.info("LoRA %s ready at %s (%.1f MiB)", lora_id, dest, size / (1024 ** 2))
            return dest

    # -- internals ---------------------------------------------------------

    def _download(self, repo_id: str, filename: str, dest: str) -> None:
        import huggingface_hub
        from huggingface_hub import hf_hub_download

        rel_dir = os.path.dirname(dest)
        # Write into a per-repo temp dir, then surface the file to `dest` so an
        # interrupted download never leaves a partial file at the final path.
        try:
            landing = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=rel_dir,
                local_dir_use_symlinks=False,
                token=self._hf_token,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to download LoRA {repo_id}/{filename}: {exc}") from exc

        # hf places the file under rel_dir (mirroring repo subpaths). Locate and
        # move it to `dest` if it didn't land exactly there.
        if os.path.abspath(landing) != os.path.abspath(dest) and os.path.isfile(landing):
            os.replace(landing, dest)
        if not os.path.isfile(dest):
            found = None
            for root, _dirs, files in os.walk(rel_dir):
                for f in files:
                    if f == filename and os.path.join(root, f) != dest:
                        found = os.path.join(root, f)
                        break
                if found:
                    break
            if found is None:
                raise RuntimeError(f"LoRA download did not land at {filename} under {rel_dir}")
            os.replace(found, dest)
        # Drop HF's cache metadata so disk accounting stays clean.
        hf_cache = os.path.join(rel_dir, ".cache")
        if os.path.isdir(hf_cache):
            shutil.rmtree(hf_cache, ignore_errors=True)

    def _touch(self, path: str) -> None:
        # mtime (ns) is the LRU clock. Bump it strictly above every other cached
        # file so consecutive uses within the same wall-clock second still order
        # correctly (deterministic LRU even under coarse timestamp granularity).
        now = time.time_ns()
        max_ns = 0
        for f in self._cached_files():
            if f == path:
                continue
            try:
                st = os.stat(f)
            except OSError:
                continue
            if st.st_mtime_ns > max_ns:
                max_ns = st.st_mtime_ns
        target = max(now, max_ns + 1)
        try:
            os.utime(path, ns=(target, target))
        except OSError:
            pass

    def _cached_files(self):
        for root, _dirs, files in os.walk(self.cache_dir):
            for f in files:
                yield os.path.join(root, f)

    def _total_bytes(self) -> int:
        return sum(os.path.getsize(f) for f in self._cached_files())

    def _evict_to_fit(self, keep: list[str] | None = None) -> None:
        keep = keep or []
        # Evict LRU until the on-disk total is under budget. If we can't reach
        # under budget (e.g. a single LoRA larger than the whole budget), evict
        # everything except `keep` and let that one exceed the budget.
        while self._total_bytes() > self.budget_bytes:
            candidates = sorted(self._cached_files(),
                      key=lambda f: (os.stat(f).st_mtime_ns, f))  # LRU = oldest
            victim = next((c for c in candidates if c not in keep), None)
            if victim is None:
                break
            try:
                os.remove(victim)
                logger.info("Evicted LRU LoRA file %s to stay under %d GiB budget",
                            victim, self.budget_bytes / _ONE_GB)
            except OSError:
                break
