"""Exercise admission, cancellation and worker loss on real worker pipes."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import socket
import sys

import pytest

from tomojax.cli.mcp import supervisor as workers
from tomojax.cli.mcp.ledger import Ledger
from tomojax.cli.mcp.tools.references import resolve

pytestmark = pytest.mark.surface


@asynccontextmanager
async def session(directory):
    parent, child = socket.socketpair()
    parent.setblocking(False)
    server = workers.Supervisor(
        Ledger(),
        directory,
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
        parent.close()
        child.close()


def test_slots_progress_and_off_queue_controls(tmp_path):
    async def run():
        async with session(tmp_path) as (_, g, port):
            wait = {"type": "inspect", "path": "wait"}
            rid, first = g.submit(wait)
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"G"
            jobs = [g.submit({"type": "inspect", "path": "stderr"})[1] for _ in range(3)]
            assert (await g.request({"type": "status"}))["requests"][rid] == "running"
            await g.request({"type": "control", "target": rid, "action": "cancel"})
            port.sendall(b"C")
            with pytest.raises(workers.WorkerError, match="canceled"):
                await first.result
            await asyncio.gather(*(job.result for job in jobs))
            _, ticks = g.submit({"type": "inspect", "path": "ticks"})
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"D"
            await ticks.result
            assert ticks.activity.progress["message"] == "complete"

    asyncio.run(run())


def test_shutdown_fails_work_in_flight(tmp_path):
    async def run():
        async with session(tmp_path) as (server, g, port):
            _, wait = g.submit({"type": "inspect", "path": "wait"})
            assert await asyncio.get_running_loop().sock_recv(port, 1) == b"G"
            await server.close()
            with pytest.raises(workers.WorkerLost, match="shutdown"):
                await wait.result

    asyncio.run(run())


def test_oom_and_cancel_leave_other_work_intact(tmp_path):
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


def test_reference_grammar(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    ledger = Ledger()
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
    ledger.closed(artifact)
    with pytest.raises(ValueError, match=f"Unknown reference {artifact}. Live handles: {first}"):
        ref(artifact)


def test_nested_mask_and_pose_table_resolution(tmp_path):
    from tomojax.cli.mcp.tools.references import wire_arguments
    from tomojax.cli.operations import Poses, Reconstruct

    ledger = Ledger()
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
                assert failure.value.kind == "canceled"
            finally:
                workers.activity.reset(token)
            port.sendall(b"C")
            await blocker.result

    asyncio.run(run())


@pytest.mark.parametrize("cancel_first", [True, False])
def test_canceled_view_releases_pixels_and_coordinates(tmp_path, monkeypatch, cancel_first):
    from tomojax.cli.mcp.tools.seeing import temporary

    async def run():
        supervisor = workers.Supervisor(Ledger(), tmp_path)
        try:
            g = await supervisor.worker()
            sim = await g.request({"type": "run", "operation": {"type": "simulate", "size": 8}})
            source = dict(type="member", handle=sim["value"]["handle"], member="projections")

            async def render():
                async with temporary(g, {"type": "read", "source": source}):
                    pytest.fail("canceled read reached renderer")

            receive = g.receive
            files = []

            def cancel_at_terminal(row):
                array = row.get("result", {}).get("value", {}).get("array")
                if not array:
                    receive(row)
                    return
                files.extend(
                    [
                        Path(array["storage"]["path"]),
                        Path(array["axes"][0]["coordinates"]["values"]["path"]),
                    ]
                )
                assert all(path.is_file() for path in files)
                if cancel_first:
                    task.cancel()
                    asyncio.get_running_loop().call_soon(receive, row)
                else:
                    receive(row)
                    task.cancel()

            monkeypatch.setattr(g, "receive", cancel_at_terminal)
            task = asyncio.create_task(render())
            with pytest.raises(asyncio.CancelledError):
                await task
            async with asyncio.timeout(5):
                while any(path.exists() for path in files):
                    await g.request({"type": "status"})
            assert len(files) == 2
        finally:
            await supervisor.close()

    asyncio.run(run())
