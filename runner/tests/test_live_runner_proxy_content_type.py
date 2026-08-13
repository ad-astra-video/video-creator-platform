"""Regression test: live-runner proxy relays a worker response with a charset.

The worker sends Content-Type like "application/json; charset=utf-8". The old
proxy passed that raw header into web.Response(content_type=...), which aiohttp
rejects with "charset must not be in content_type argument" — surfacing the
successful restyle result as a 500 in the live-runner. The fix parses the media
type and charset separately (aiohttp's resp.content_type / resp.charset) and
relays raw body bytes. GPU/worker-free.
"""

import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402


class FakeWorkerManager:
    """Minimal stand-in: the real ensure() just records the resident name."""

    def __init__(self):
        self.resident = None

    async def ensure(self, name: str):
        self.resident = name


async def _run_proxy(content_type: str, body: bytes, status: int = 200) -> web.Response:
    """Boot a fake worker returning the given content-type/body, then call proxy()."""
    from runner.live_runner.routing import proxy

    worker_resp = None

    async def worker_handler(request):
        nonlocal worker_resp
        # Set Content-Type via the header (not content_type=) so aiohttp accepts
        # a charset-bearing value exactly as a real worker would send it.
        return web.Response(status=status, body=body,
                            headers={"Content-Type": content_type})

    worker_app = web.Application()
    worker_app.router.add_post("/video-creator/v1/image", worker_handler)
    worker_client = TestClient(TestServer(worker_app))
    await worker_client.start_server()
    try:
        wm = FakeWorkerManager()
        session = worker_client.session
        # point proxy at the fake worker instead of a Docker DNS name
        import runner.live_runner.config as cfg
        orig = cfg.WORKERS["ltx-worker"]
        cfg.WORKERS["ltx-worker"] = f"http://{worker_client.host}:{worker_client.port}"
        try:
            worker_resp = await proxy(wm, session, "tok", "ltx-worker", "image", {})
        finally:
            cfg.WORKERS["ltx-worker"] = orig
        assert wm.resident == "ltx-worker"
        return worker_resp
    finally:
        await worker_client.close()


@pytest.mark.parametrize("ctype", [
    "application/json; charset=utf-8",
    "application/json",
    "image/png",
])
def test_proxy_relays_charset_content_type(ctype):
    body = json.dumps({"video_base64": "x"}).encode()
    resp = asyncio.run(_run_proxy(ctype, body))
    assert resp.status == 200
    # Body relayed byte-for-byte.
    assert resp.body == body


def test_proxy_relays_error_body():
    err_body = b'{"error":"boom"}'
    resp = asyncio.run(_run_proxy("application/json; charset=utf-8", err_body, status=502))
    assert resp.status == 502
    assert err_body in resp.body
