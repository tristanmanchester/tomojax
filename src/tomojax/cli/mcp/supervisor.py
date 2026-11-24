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
import tempfile
from typing import TYPE_CHECKING, Any

import msgspec

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tomojax.cli.mcp.ledger import Ledger

type Object = dict[str, Any]
FAST = {"status", "control", "close", "release", "describe", "devices"}


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


class Generation:
    """A single worker's correlation namespace and admission state."""

    def __init__(self, prefix: int) -> None:
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
        rid = f"{self.prefix}-{self.count + 1}"
        data = msgspec.json.encode({"id": rid, "command": command})
        if len(data) > 1024 * 1024:
            raise ValueError("Worker request exceeds 1 MiB")
        job = Pending(
            asyncio.get_running_loop().create_future(),
            command["type"] not in FAST,
            activity.get() or Activity(),
            accept,
        )
        self.count += 1
        self.pending[rid] = job
        self.process.stdin.write(data + b"\n")
        return rid, job

    def cancel(self, rid: str) -> None:
        """Send one off-queue cancel without waiting for the worker to acknowledge."""
        if rid in self.pending and not self.exit.is_set():
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
        if job.execution:
            job.activity.cancel = None
        try:
            if row["type"] == "protocol_error":
                raise WorkerProtocolError(row["message"])
            if row["type"] != "result":
                raise WorkerError(row.get("error", row))
            value = row["result"]
            job.result.set_result(job.accept(value) if job.accept else value)
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
            job.activity.cancel = None
            job.result.set_exception(WorkerLost(reason))
        self.pending.clear()


class Supervisor:
    """Keep exactly one worker and its scratch directory per server."""

    def __init__(
        self,
        ledger: Ledger,
        base: Path,
        *,
        argv: tuple[str, ...] = (sys.executable, "-m", "tomojax.cli.main", "serve"),
        pass_fds: tuple[int, ...] = (),
    ) -> None:
        self.ledger = ledger
        self.base = base
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
        """Kill and reap the worker and remove its scratch directory."""
        self.closing = True
        self.current.stop("Server shutdown")
        self.launch.set()
        await self.runner

    async def run(self) -> None:
        """Run one worker at a time. A lost worker's work is reported, never replayed."""
        while not self.closing:
            await self.launch.wait()
            g = self.current
            try:
                await self.serve_generation(g)
            except Exception as err:
                g.stop(str(err))
            lost = self.ledger.lost()
            self.report = {"reason": g.reason, "lost": lost}
            started = g.ready.done()
            g.fail(f"{g.reason}. Lost references: {', '.join(lost) or 'none'}. Open inputs again.")
            if not started:
                # A broken installation should fail its first request, not spin on startup.
                self.launch.clear()
            if not self.closing:
                self.current = Generation(g.prefix + 1)
            g.retired.set()

    async def serve_generation(self, g: Generation) -> None:
        """Run the worker in its own scratch directory until it exits or is stopped."""
        with tempfile.TemporaryDirectory(prefix="tomojax-mcp-") as scratch:
            proc = await asyncio.create_subprocess_exec(
                *self.argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                cwd=self.base,
                env={**os.environ, "TMPDIR": scratch},
                pass_fds=self.pass_fds,
            )
            g.process = proc
            assert proc.stdout
            assert proc.stderr
            stderr = bytearray()

            async def drain(stream: asyncio.StreamReader) -> None:
                while data := await stream.read(65536):
                    stderr.extend(data)
                    del stderr[:-4096]

            reader = asyncio.create_task(g.read(proc.stdout))  # stops g when the worker exits
            errors = asyncio.create_task(drain(proc.stderr))
            try:
                await g.exit.wait()
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                await proc.wait()
                await asyncio.gather(reader, errors, return_exceptions=True)
                if stderr and not g.ready.done():
                    g.reason += ": " + stderr.decode(errors="replace")
