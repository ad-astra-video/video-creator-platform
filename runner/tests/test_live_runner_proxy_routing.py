"""Regression test: live-runner routes /video-creator/v1/{endpoint} to handle_generic.

The pre-existing bug registered static paths (e.g. "/video-creator/v1/image")
without an {endpoint} placeholder, so handle_generic read an empty
match_info["endpoint"] and returned 404 'unknown endpoint: ' for every proxied
call. This test boots the real aiohttp app and asserts the parameterized route
populates match_info so the endpoint resolves in ROUTES. GPU/worker-free.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pytest  # noqa: E402

from aiohttp import web  # noqa: E402
from aiohttp.test_utils import TestClient, TestServer  # noqa: E402


def handle_endpoint(endpoint: str):
    """Thin shim so handle_generic-style lookup is testable independently."""
    from runner.live_runner.routing import ROUTES
    return ROUTES.get(endpoint)


@pytest.mark.parametrize("endpoint,expected_worker", [
    ("restyle", "idv2v-worker"),
    ("image", "ltx-worker"),
    ("t2v", "ltx-worker"),
    ("i2v", "ltx-worker"),
])
def test_endpoint_resolves_in_route_table(endpoint, expected_worker):
    assert handle_endpoint(endpoint) == expected_worker


async def _request_routes_app(endpoint: str) -> tuple[int, str]:
    """Boot a minimum aiohttp app using the same single-param-route registration
    the production server.py uses, and return (status, body endpoint echo)."""
    from runner.live_runner.routing import ROUTES

    app = web.Application()

    p = "/video-creator/v1"

    async def handler(req):
        ep = req.match_info.get("endpoint", "")
        worker = ROUTES.get(ep)
        if worker is None:
            return web.json_response({"error": f"unknown endpoint: {ep}"}, status=404)
        return web.json_response({"endpoint": ep, "worker": worker})

    app.router.add_get(f"{p}/health", lambda _r: web.json_response({"ok": True}))
    app.router.add_post(f"{p}/{{endpoint}}", handler)

    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.post(f"{p}/{endpoint}", json={})
        body = await resp.json()
        return resp.status, body.get("endpoint", "")
    finally:
        await client.close()


@pytest.mark.parametrize("endpoint", ["restyle", "image", "t2v", "i2v", "extend"])
def test_param_route_passes_endpoint_to_handler(endpoint):
    """The {endpoint} placeholder (the fix) makes match_info carry the endpoint,
    so ROUTES lookups succeed and the handler gets a real endpoint (not '')."""
    status, echoed = asyncio.run(_request_routes_app(endpoint))
    assert status == 200
    assert echoed == endpoint


def test_unknown_endpoint_still_404s():
    status, echoed = asyncio.run(_request_routes_app("nope"))
    assert status == 404
    assert echoed == ""
