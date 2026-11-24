"""Exercise admission, leases and recovery on real worker pipes."""

import asyncio
from contextlib import asynccontextmanager
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import pytest

from tomojax.cli.mcp import supervisor as workers
from tomojax.cli.mcp.ledger import Ledger
from tomojax.cli.mcp.scratch import scratch
from tomojax.cli.mcp.tools.references import resolve

pytestmark = pytest.mark.surface


@asynccontextmanager
async def session(directory):
    parent, child = socket.socketpair()
    parent.setblocking(False)
    ledger = Ledger(directory / "state")
    server = workers.Supervisor(
        ledger,
        directory,
        directory / "cache",
        argv=(
            sys.executable,
            str(Path(__file__).with_name("worker_fixture.py")),
            str(child.fileno()),
        ),
        pass_fds=(child.fileno(),),
    )
    try:
        async with asyncio.timeout(20):
            yield server, await server.worker(), parent
    finally:
        await server.close()
        ledger.close()
        parent.close()
        child.close()


def test_slots_progress_and_off_queue_controls(tmp_path):
    async def run():
        async with session(tmp_path) as (_, g, port):
            wait = {"type": "inspect", "path": "wait"}
            rid, first = g.submit(wait)
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"G"
            jobs = [g.submit({"type": "inspect", "path": "stderr"})[1] for _ in range(61)]
            with pytest.raises(ValueError, match="62 jobs"):
                g.submit(wait)
            assert (await g.request({"type": "status"}))["requests"][rid] == "running"
            await g.request({"type": "control", "target": rid, "action": "cancel"})
            port.sendall(b"C")
            with pytest.raises(workers.WorkerError, match="cancelled"):
                await first.result
            await asyncio.gather(*(job.result for job in jobs))
            _, ticks = g.submit({"type": "inspect", "path": "ticks"})
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"D"
            await ticks.result
            assert ticks.activity.progress["message"] == "complete"

    asyncio.run(run())


def test_cancel_deadline_without_ack_and_shutdown_in_flight(tmp_path, monkeypatch):
    monkeypatch.setattr(workers, "CANCEL_SECONDS", 0.12)

    async def run():
        async with session(tmp_path) as (server, g, port):
            caller = asyncio.create_task(g.request({"type": "inspect", "path": "wait"}))
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"G"
            os.kill(g.process.pid, signal.SIGSTOP)
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
            await g.retired.wait()
            assert "deadline" in server.report["reason"]
            replacement = await server.worker()
            _, wait = replacement.submit({"type": "inspect", "path": "wait"})
            await server.close()
            with pytest.raises(workers.WorkerLost, match="shutdown"):
                await wait.result

    asyncio.run(run())


def test_oom_and_cancel_leave_other_work_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(workers, "CANCEL_SECONDS", 0.3)

    async def run():
        async with session(tmp_path) as (_, g, port):
            opened = await g.request({"type": "open", "kind": "scan", "path": "x"})
            old = opened["value"]["handle"]
            with pytest.raises(workers.WorkerError) as failure:
                await g.request({"type": "inspect", "path": "jax-oom"})
            assert failure.value.kind == "out_of_memory"
            with pytest.raises(workers.WorkerError) as invalid:
                await g.request({"type": "inspect", "path": "invalid"})
            assert invalid.value.path == "config.filter"
            assert old in (await g.request({"type": "status"}))["handles"]
            rid, first = g.submit({"type": "inspect", "path": "wait"})
            _, next_job = g.submit({"type": "inspect", "path": "wait"})
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"G"
            await g.request({"type": "control", "target": rid, "action": "cancel"})
            port.sendall(b"C")
            with pytest.raises(workers.WorkerError):
                await first.result
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"G"
            await asyncio.sleep(0.35)
            port.sendall(b"C")
            assert (await next_job.result)["value"]["report"]["path"] == "wait"

    asyncio.run(run())


def test_executor_exit_recovers_and_never_replays(tmp_path):
    async def run():
        async with session(tmp_path) as (server, g, _):
            with pytest.raises(ValueError, match="1 MiB"):
                g.submit({"type": "inspect", "path": "x" * (1024**2)})
            _, first = g.submit({"type": "inspect", "path": "executor-exit"})
            _, queued = g.submit({"type": "open", "kind": "scan", "path": "x"})
            for job in (first, queued):
                with pytest.raises(workers.WorkerLost):
                    await job.result
            fresh = await server.worker()
            assert fresh is not g
            assert not (await fresh.request({"type": "status"}))["handles"]

    asyncio.run(run())


def test_scratch_lock_outlives_either_owner(tmp_path):
    with scratch(tmp_path) as (path, fd):
        child = subprocess.Popen([sys.executable, "-c", "pass"], pass_fds=(fd,))
        assert child.wait(timeout=5) == 0
        with scratch(tmp_path):
            assert path.is_dir()
    # Model abrupt parent death by acquiring the worker lease in a process that
    # passes the same locked description to its surviving worker, then _exit.
    program = """
import os,sys,subprocess
from pathlib import Path
from tomojax.cli.mcp.scratch import scratch
with scratch(Path(sys.argv[1])) as (path,fd):
    subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read()'],pass_fds=(fd,),
                     stdout=subprocess.DEVNULL)
    print(path,flush=True)
    os._exit(0)
"""
    with subprocess.Popen(
        [sys.executable, "-c", program, str(tmp_path)], stdin=-1, stdout=-1, text=True
    ) as parent:
        orphan = Path(parent.stdout.readline().strip())
        assert parent.wait(timeout=5) == 0
        with scratch(tmp_path):
            assert orphan.is_dir()
        parent.stdin.close()
        with (orphan / "lock").open("rb") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
        with scratch(tmp_path):
            assert not orphan.exists()


def test_ledger_and_reference_grammar(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    ledger = Ledger(tmp_path)
    record = {"kind": "scan", "worker": "h0"}
    first = ledger.issue("scan", record)

    def ref(value, **kwargs):
        return resolve(value, base=tmp_path, ledger=ledger, **kwargs)

    assert ref(first)["member"] == "projections"
    assert ref(first, field="source")["member"] == "self"
    assert ref(first + ".angles") == {"type": "member", "handle": "h0", "member": "angles"}
    assert ref("x.h5::/a::/b")["dataset"] == "/a::/b"
    assert ref("./@x")["path"] == str(tmp_path / "@x")
    assert ref("~/y")["path"] == str(tmp_path / "y")
    artifact = ledger.issue("array", {"path": "pixels.npy", "artifact": "a7"})
    assert ref(artifact) == {"type": "npy", "path": "pixels.npy", "artifact": "a7"}
    other = Ledger(tmp_path)
    with pytest.raises(ValueError, match="another server"):
        other.lookup(first)
    other.close()
    ledger.closed(artifact)
    ledger.close()
    ledger = Ledger(tmp_path)
    with pytest.raises(ValueError, match="closed"):
        ledger.lookup(artifact)
    with pytest.raises(ValueError, match="lost"):
        ledger.lookup(first)
    live = ledger.issue("scan", record)
    with pytest.raises(ValueError, match=live):
        ledger.lookup("@scan-999-" + ledger.tag)
    ledger.close()
    (tmp_path / "ledger").unlink()
    ledger = Ledger(tmp_path)
    with pytest.raises(ValueError, match="earlier ledger"):
        ledger.lookup(first)
    ledger.close()


def test_nested_mask_and_pose_table_resolution(tmp_path):
    from tomojax.cli.mcp.tools.references import wire_arguments
    from tomojax.cli.operations import Poses, Reconstruct

    ledger = Ledger(tmp_path)
    try:
        handle = ledger.issue("scan", {"kind": "scan", "worker": "root"})
        config = {"type": "FISTAConfig", "support": handle + ".projections"}
        request = wire_arguments(
            {"type": "reconstruct", "source": handle, "config": config},
            Reconstruct,
            base=tmp_path,
            ledger=ledger,
        )
        assert request["source"]["member"] == "self"
        assert request["config"]["support"] == {
            "type": "member",
            "handle": "root",
            "member": "projections",
        }
        edited = wire_arguments(
            {"type": "poses", "source": "./@input", "poses": "table.h5::/x::/y"},
            Poses,
            base=tmp_path,
            ledger=ledger,
        )
        assert edited["source"]["path"] == str(tmp_path / "@input")
        assert edited["poses"] == {
            "type": "hdf5",
            "path": str(tmp_path / "table.h5"),
            "dataset": "/x::/y",
        }
    finally:
        ledger.close()


def test_mcp_close_preserves_admitted_array_consumer(tmp_path):
    from tomojax.cli.mcp.core import Core

    async def run():
        async with session(tmp_path) as (server, g, port):
            core = Core(server)
            _, producing = g.submit({"type": "inspect", "path": "preview"})
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"G"
            await g.request({"type": "status"})
            storage = producing.activity.progress["arrays"]["frame"]["storage"]
            path = Path(storage["path"])
            ref = server.ledger.issue("array", storage)
            pinned = resolve(ref, base=tmp_path, ledger=server.ledger)
            _, job = g.submit(
                {
                    "type": "run",
                    "operation": {
                        "type": "backproject",
                        "source": {"type": "path", "path": "scan"},
                        "projections": pinned,
                    },
                }
            )
            reply = await core.server.call_tool("control", {"action": "close", "target": ref})
            assert not reply.is_error
            assert path.exists()
            port.sendall(b"C")
            await producing.result
            assert (await job.result)["value"]["report"]["sum"] == 99
            assert not path.exists()

    asyncio.run(run())


def test_rollover_waits_for_terminal(tmp_path, monkeypatch):
    monkeypatch.setattr(workers, "ID_LIMIT", 131)

    async def run():
        async with session(tmp_path) as (server, g, port):
            rid, first = g.submit({"type": "inspect", "path": "wait"})
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"G"
            await g.request({"type": "status"})
            await g.request({"type": "status"})
            with pytest.raises(ValueError, match="budget"):
                g.submit({"type": "open", "path": "x", "kind": "scan"})
            await g.request({"type": "control", "action": "cancel", "target": rid})
            port.sendall(b"C")
            with pytest.raises(workers.WorkerError):
                await first.result
            await g.retired.wait()
            assert await server.worker() is not g

    asyncio.run(run())


def test_early_cancellation_is_delivered_at_submission(tmp_path):
    async def run():
        async with session(tmp_path) as (_, g, port):
            _, blocker = g.submit({"type": "inspect", "path": "wait"})
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"G"
            activity = workers.Activity()
            activity.request_cancel()
            token = workers.activity.set(activity)
            try:
                with pytest.raises(workers.WorkerError) as failure:
                    await g.request({"type": "inspect", "path": "notify"})
                assert failure.value.kind == "cancelled"
            finally:
                workers.activity.reset(token)
            port.sendall(b"C")
            await blocker.result

    asyncio.run(run())


def test_abrupt_server_loss_retires_ledger_ids(tmp_path):
    program = """
import os,sys
from pathlib import Path
from tomojax.cli.mcp.ledger import Ledger
ledger = Ledger(Path(sys.argv[1]))
print(ledger.issue('scan', {'worker':'old'}),flush=True)
os._exit(0)
"""
    ref = subprocess.check_output([sys.executable, "-c", program, str(tmp_path)], text=True).strip()
    ledger = Ledger(tmp_path)
    try:
        with pytest.raises(ValueError, match="lost"):
            ledger.lookup(ref)
    finally:
        ledger.close()
