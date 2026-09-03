"""Self-test for the generic engine-subprocess harness (runner.common.engineproc).

Uses a REAL python subprocess as the fake model child (no torch/GPU) to prove:
  * start() spawns the child and waits for its ready handshake;
  * run() round-trips one JSON op and forwards progress lines;
  * run() raises on a child-declared error and evicts the child;
  * run() raises + evicts on timeout;
  * stop() terminates the process (the mechanism that destroys a real CUDA
    context — the child's exit is what frees the GPU).

Each test drives ONE event loop end-to-end (subprocess transports are tied to
their loop; reusing an EngineProc across several asyncio.run() loops breaks on
Windows' proactor transport, so each test keeps a single loop lifetime).
"""

import asyncio
import os
import sys

import pytest

from runner.common import engineproc

CHILD = """
import json, sys, time
print(json.dumps({"type": "ready", "ok": True}), flush=True)
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    op = req.get("op"); args = req.get("args") or {}
    if op == "echo":
        print(json.dumps({"type": "progress", "step": 1, "total_steps": 3}), flush=True)
        print(json.dumps({"ok": True, "result": args}), flush=True)
    elif op == "boom":
        print(json.dumps({"ok": False, "error": "kaboom"}), flush=True)
    elif op == "hang":
        time.sleep(30)
        print(json.dumps({"ok": True, "result": "never"}), flush=True)
    else:
        print(json.dumps({"ok": False, "error": "unknown op"}), flush=True)
"""


def _argv():
    return [sys.executable, "-c", CHILD]


def _alive(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) is unsupported on Windows (WinError 87) — probe the
        # process handle instead.
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@pytest.fixture()
def loop():
    l = asyncio.new_event_loop()
    asyncio.set_event_loop(l)
    yield l
    try:
        l.run_until_complete(asyncio.sleep(0))
    except Exception:
        pass
    l.close()
    asyncio.set_event_loop(None)


def test_start_waits_for_ready_and_run_roundtrips(loop):
    async def scenario():
        ep = engineproc.EngineProc("test", _argv(), startup_timeout=20, job_timeout=20)
        await ep.start()
        assert ep._ready
        progress = []
        res = await ep.run("echo", {"a": 1},
                           progress_cb=lambda p: progress.append(p))
        assert res == {"a": 1}
        assert any(p.get("step") == 1 for p in progress), "progress must forward"
        await ep.stop()

    loop.run_until_complete(scenario())


def test_run_declared_error_raises_and_evicts(loop):
    async def scenario():
        ep = engineproc.EngineProc("test", _argv(), startup_timeout=20, job_timeout=20)
        await ep.start()
        with pytest.raises(engineproc.EngineProcError, match="kaboom"):
            await ep.run("boom")
        assert ep._proc is None or ep._proc.returncode is not None
        await ep.stop()

    loop.run_until_complete(scenario())


def test_run_timeout_raises_and_evicts(loop):
    async def scenario():
        ep = engineproc.EngineProc("test", _argv(), startup_timeout=20, job_timeout=20)
        await ep.start()
        with pytest.raises(engineproc.EngineProcError, match="timed out"):
            await ep.run("hang", timeout=1)
        assert ep._proc is None or ep._proc.returncode is not None
        await ep.stop()

    loop.run_until_complete(scenario())


def test_stop_terminates_child(loop):
    async def scenario():
        ep = engineproc.EngineProc("test", _argv(), startup_timeout=20, job_timeout=20)
        await ep.start()
        pid = ep._proc.pid
        await ep.stop()
        assert ep._proc is None
        assert not _alive(pid), "child must be gone after stop (kills CUDA ctx)"

    loop.run_until_complete(scenario())


def test_double_stop_is_safe(loop):
    async def scenario():
        ep = engineproc.EngineProc("test", _argv(), startup_timeout=20, job_timeout=20)
        await ep.start()
        await ep.stop()
        await ep.stop()

    loop.run_until_complete(scenario())
