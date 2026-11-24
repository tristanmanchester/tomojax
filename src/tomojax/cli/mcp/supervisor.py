"""Own worker generations, request admission and terminal delivery on one event loop."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
import os
import signal
import sys
from typing import TYPE_CHECKING, Any

import msgspec

from tomojax.cli._commit import commit
from tomojax.cli.mcp.scratch import scratch
from tomojax.cli.products import Staged

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tomojax.cli.mcp.ledger import Ledger

type Object = dict[str, Any]
FAST = {"status", "control", "close", "release", "describe", "devices"}
CANCEL_SECONDS = 30
ID_LIMIT = 100_000


class WorkerLost(RuntimeError):
    """The worker was lost before it returned a terminal. Its work must never be replayed."""


class WorkerProtocolError(RuntimeError):
    """The worker rejected a command before admission."""


class WorkerError(ValueError):
    """A terminal failure, including its worker kind and parameter path."""

    def __init__(self, error: Object) -> None:
        self.kind = error.get("kind", "invalid_request")
        self.path = error.get("path")
        super().__init__(error["message"])


@dataclass
class Activity:
    """A task's most recent worker event and cancellation route."""

    updated: datetime | None = None
    progress: Object | None = None
    cancel: Callable[[], None] | None = None
    requested: bool = False
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    stages: dict[int, str] = field(default_factory=dict)
    devices: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def record(self, event: Object) -> None:
        """Keep the active labels even when a newer worker event has no label."""
        self.updated = datetime.now(UTC)
        self.progress = event
        kind = event["event"]
        if kind == "started":
            self.stages[event["task"]] = " ".join(filter(None, (event["name"], event["label"])))
        elif kind == "finished":
            self.stages.pop(event["task"], None)
        elif kind == "note":
            self.notes.append(event["message"])
        elif kind == "placed":
            self.devices.update(event["devices"])
        self.changed.set()

    def status(self) -> tuple[str, float | None]:
        """Describe the innermost active stage and its latest estimate."""
        event = self.progress or {}
        label = next(reversed(self.stages.values()), "worker")
        if event.get("event") != "advanced":
            return f"waiting: {label}", None
        eta = event.get("eta")
        metrics = [f"{name} {value}" for name, value in event["metrics"].items()]
        position = (
            f"{event['done']}/{event['total']}"
            if event.get("total") is not None
            else str(event["done"])
        )
        if eta is not None:
            metrics.append(f"eta {eta:g} s")
        return " · ".join([f"{label} {position}", *metrics]), eta

    def request_cancel(self) -> None:
        """Carry an early cancellation across submission."""
        self.requested = True
        if self.cancel:
            self.cancel()


activity: ContextVar[Activity | None] = ContextVar("tomojax_job", default=None)


@dataclass
class Pending:
    """The state owned until one terminal or generation loss."""

    result: asyncio.Future[Object]
    execution: bool
    activity: Activity
    accept: Callable[[Object], Object] | None
    deadline: asyncio.TimerHandle | None = None
    drain_resume: bool = False


class Generation:
    """A single worker's correlation namespace and admission state."""

    def __init__(self, prefix: int) -> None:
        self.commits: set[asyncio.Task[None]] = set()
        self.prefix = prefix
        self.count = 0
        self.pending: dict[str, Pending] = {}
        self.ready: asyncio.Future[Object] = asyncio.get_running_loop().create_future()
        self.exit = asyncio.Event()
        self.retired = asyncio.Event()
        self.reason = "Worker exited"
        self.process: asyncio.subprocess.Process | None = None

    def stop(self, reason: str) -> None:
        """Wake the generation owner and seal admission immediately."""
        if not self.exit.is_set():
            self.reason = reason
            self.exit.set()

    def submit(
        self, command: Object, *, accept: Callable[[Object], Object] | None = None
    ) -> tuple[str, Pending]:
        """Write the entire request atomically with reservation of its pending slot."""
        if self.exit.is_set() or self.process is None or self.process.stdin is None:
            raise WorkerLost(self.reason)
        execution = command["type"] not in FAST
        if execution and sum(p.execution for p in self.pending.values()) >= 62:
            raise ValueError("62 jobs pending. Check status before submitting more work.")
        target = self.pending.get(command.get("target", ""))
        action = command.get("action")
        if self.count >= ID_LIMIT - 128:
            self.admit_during_drain(action, target)
        rid = f"{self.prefix}-{self.count + 1}"
        data = msgspec.json.encode({"id": rid, "command": command})
        if len(data) > 1024 * 1024:
            raise ValueError("Worker request exceeds 1 MiB")
        job = Pending(
            asyncio.get_running_loop().create_future(),
            execution,
            activity.get() or Activity(),
            accept,
        )
        self.count += 1
        self.pending[rid] = job
        self.process.stdin.write(data + b"\n")
        if action == "cancel" and target and target.execution and target.deadline is None:
            target.deadline = asyncio.get_running_loop().call_later(
                CANCEL_SECONDS, self.stop, f"Cancel deadline expired for {command['target']}"
            )
        return rid, job

    def admit_during_drain(self, action: str | None, target: Pending | None) -> None:
        """Reserve a resume and a cancel for each of the 62 remaining jobs."""
        if target and target.execution:
            if action == "cancel" and target.deadline is None:
                return
            if action == "resume" and not target.drain_resume:
                target.drain_resume = True
                return
        if not self.pending:
            self.stop("Worker request ID budget exhausted")
        raise ValueError("Worker request budget exhausted. Retry after pending jobs drain.")

    def cancel(self, rid: str) -> None:
        """Send one off-queue cancel without waiting for the worker to acknowledge."""
        if rid in self.pending and not self.exit.is_set() and self.pending[rid].deadline is None:
            _, ack = self.submit({"type": "control", "action": "cancel", "target": rid})
            ack.result.add_done_callback(lambda future: future.exception())

    async def request(
        self, command: Object, *, accept: Callable[[Object], Object] | None = None
    ) -> Object:
        """Deliver a terminal, forwarding client cancellation even during submission."""
        rid, job = self.submit(command, accept=accept)
        if job.execution:
            job.activity.cancel = lambda: self.cancel(rid)
            if job.activity.requested:
                self.cancel(rid)
        try:
            return await asyncio.shield(job.result)
        except asyncio.CancelledError:
            if job.execution:
                self.cancel(rid)
            job.result.add_done_callback(lambda future: future.exception())
            raise

    def receive(self, row: Object) -> None:
        """Accept progress or settle one terminal before dispatching another record."""
        if row["type"] == "ready":
            self.ready.set_result(row)
            return
        rid = row["request"]
        job = self.pending[rid]
        if row["type"] == "progress":
            job.activity.record(row["record"])
            return
        del self.pending[rid]
        if job.deadline:
            job.deadline.cancel()
        if job.execution:
            job.activity.cancel = None
        try:
            if row["type"] == "protocol_error":
                raise WorkerProtocolError(row["message"])
            if row["type"] != "result":
                raise WorkerError(row.get("error", row))
            value = row["result"]
            if value.get("value", {}).get("plan"):
                task = asyncio.create_task(self.install(job, value))
                self.commits.add(task)
                task.add_done_callback(self.commits.discard)
            else:
                job.result.set_result(job.accept(value) if job.accept else value)
        except Exception as err:
            job.result.set_exception(err)
        if not self.pending and self.count >= ID_LIMIT - 128:
            self.stop("Worker request ID budget exhausted")

    async def install(self, job: Pending, value: Object) -> None:
        """Settle an accepted plan independently of worker death and client cancellation."""
        try:
            files = value["value"]
            group = msgspec.convert(files.pop("plan"), type=list[Staged])
            files["recovery"] = await asyncio.to_thread(commit, group)
            job.result.set_result(job.accept(value) if job.accept else value)
        except OSError as err:
            job.result.set_exception(
                WorkerError({"kind": "input", "path": "path", "message": str(err)})
            )
        except Exception as err:
            job.result.set_exception(err)

    async def read(self, stream: asyncio.StreamReader) -> None:
        """Decode complete JSON lines without the asyncio readline size limit."""
        buffer = bytearray()
        try:
            while data := await stream.read(65536):
                chunks = data.split(b"\n")
                buffer.extend(chunks[0])
                for chunk in chunks[1:]:
                    self.receive(msgspec.json.decode(buffer))
                    buffer = bytearray(chunk)
        except (KeyError, ValueError, TypeError) as err:
            self.stop(f"Invalid worker response: {err}")
        finally:
            self.stop("Worker protocol closed")

    def fail(self, reason: str) -> None:
        """Settle outstanding work after EOF, shutdown or forced termination."""
        if not self.ready.done():
            self.ready.set_exception(WorkerLost(reason))
            self.ready.exception()
        for job in self.pending.values():
            if job.deadline:
                job.deadline.cancel()
            job.activity.cancel = None
            job.result.set_exception(WorkerLost(reason))
        self.pending.clear()


class Supervisor:
    """Keep exactly one worker and its scratch lease per server."""

    def __init__(
        self,
        ledger: Ledger,
        base: Path,
        cache: Path,
        *,
        argv: tuple[str, ...] = (
            sys.executable,
            "-m",
            "tomojax.cli.main",
            "serve",
            "--commit",
            "supervisor",
        ),
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        self.ledger = ledger
        self.base = base
        self.cache = cache
        self.argv = argv
        self.pass_fds = pass_fds
        self.current = Generation(1)
        self.report: Object = {}
        self.closing = False
        self.launch = asyncio.Event()
        self.launch.set()
        self.runner = asyncio.create_task(self.run())

    async def worker(self) -> Generation:
        """Wait for startup without coupling worker lifetime to a single MCP request."""
        g = self.current
        if g.exit.is_set():
            await g.retired.wait()
            g = self.current
        self.launch.set()
        await asyncio.shield(g.ready)
        return g

    async def restart(self) -> Object:
        """Restart the worker and return the list of invalidated references."""
        g = self.current
        g.stop("Worker restarted")
        self.launch.set()
        await g.retired.wait()
        await self.worker()
        return self.report

    async def close(self) -> None:
        """Kill and reap the group and remove the scratch directory."""
        self.closing = True
        self.current.stop("Server shutdown")
        self.launch.set()
        await self.runner

    async def run(self) -> None:
        """Recover accepted state once per generation, without replay."""
        while not self.closing:
            await self.launch.wait()
            g = self.current
            recovered: dict[str, list[str]] = {}
            try:
                recovered = await asyncio.to_thread(commit, [])
                lease = scratch(self.cache, report=recovered)
                path, fd = await asyncio.to_thread(lease.__enter__)
                self.report.update(recovered)
                try:
                    await self.serve_generation(g, path, fd)
                finally:
                    await asyncio.to_thread(lease.__exit__, None, None, None)
            except Exception as err:
                g.stop(str(err))
            lost = self.ledger.lost()
            self.report = {**recovered, "reason": g.reason, "lost": lost}
            started = g.ready.done()
            g.fail(
                f"{g.reason}. Lost references: {', '.join(lost) or 'none'}. Open inputs again. "
                "See status for recovered or discarded files."
            )
            if not started:
                # A broken installation should fail its first request, not spin on startup.
                self.launch.clear()
            if not self.closing:
                self.current = Generation(g.prefix + 1)
            g.retired.set()

    async def serve_generation(self, g: Generation, path: Path, fd: int) -> None:
        """Monitor both process exit and protocol EOF, draining both pipes independently."""
        proc = await asyncio.create_subprocess_exec(
            *self.argv,
            stdin=-1,
            stdout=-1,
            stderr=-1,
            start_new_session=True,
            cwd=self.base,
            env={**os.environ, "TMPDIR": str(path)},
            pass_fds=(*self.pass_fds, fd),
        )
        g.process = proc
        assert proc.stdout
        assert proc.stderr
        stderr = bytearray()

        async def drain(stream: asyncio.StreamReader) -> None:
            while data := await stream.read(65536):
                stderr.extend(data)
                del stderr[:-4096]

        reader = asyncio.create_task(g.read(proc.stdout))
        errors = asyncio.create_task(drain(proc.stderr))
        loop = asyncio.get_running_loop()
        pidfd = None
        try:
            # A fork can retain the pipes after the worker exits. Watch the leader
            # independently of pipe EOF, without waiting for inherited fds to close.
            try:
                pidfd = os.pidfd_open(proc.pid)
            except ProcessLookupError:
                g.stop("Worker exited")
            else:
                loop.add_reader(pidfd, g.stop, "Worker exited")
                await g.exit.wait()
        finally:
            if pidfd is not None:
                loop.remove_reader(pidfd)
                os.close(pidfd)
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await asyncio.gather(reader, errors, return_exceptions=True)
            # A malformed record can end the decoder with unread bytes in its pipe.
            while await proc.stdout.read(65536):
                pass
            await proc.wait()
            await asyncio.gather(*g.commits)
            if stderr and not g.ready.done():
                g.reason += ": " + stderr.decode(errors="replace")
