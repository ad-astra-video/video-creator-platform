"""Shared pytest configuration for runner tests.

Loads pytest-asyncio so async ``def test_...`` functions and async fixtures are
handled. Set ``asyncio_mode = auto`` (either via -o asyncio_mode=auto on the
command line or a pytest.ini in the repo root) so async tests run without
@asyncio.mark. Existing sync tests that call ``asyncio.run(...)`` themselves are
unaffected.
"""

pytest_plugins = ["pytest_asyncio"]
