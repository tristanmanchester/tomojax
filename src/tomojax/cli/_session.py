"""Serialised worker admission and publication. Only the adapter runs outside the monitor."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from functools import partial
import logging
from pathlib import Path
from threading import Condition

import msgspec

from tomojax import progress
from tomojax.cli import envelope as e, execution as ex, operations as op, products as p
from tomojax.cli._staging import Stages
from tomojax.errors import classify
from tomojax.wire import progress as w
from tomojax.wire.arrays import Coordinates, Member, Npy


@dataclass
class Resource:
    """The registry's initial reference is surrendered by close or release."""

    dispose: Callable[[], None]
    references: int = 1

    def pin(self, lifetime: ExitStack) -> None:
        self.references += 1
        lifetime.callback(self.drop)

    def drop(self) -> None:
        self.references -= 1
        if not self.references:
            self.dispose()


@dataclass
class Handle:
    bound: ex.BoundHandle
    manifest: p.Provenance
    size: int
    resource: Resource
    artifacts: set[str]


@dataclass(eq=False)
class Job:
    id: str
    command: op.ExecutionInput
    channel: e.ChannelRevision | None
    control: progress.Control
    stages: Stages
    leases: dict[Member, ex.Lease]
    lifetime: ExitStack
    published: set[str] = field(default_factory=set)


class Session:
    def __init__(
        self,
        root: Path,
        execute: ex.Adapter[op.ExecutionInput],
        emit: Callable[[e.Response], None],
        ready: e.Ready,
        description: e.Description,
    ) -> None:
        self.root = root
        self.execute = execute
        self.emit = emit
        self.description = description
        self.ready = ready
        self.seen: set[str] = set()
        self.revisions: dict[str, int] = {}
        self.queue: deque[Job] = deque()
        self.jobs: dict[str, Job] = {}
        self.handles: dict[str, Handle] = {}
        self.artifacts: dict[str, tuple[Path, Resource]] = {}
        self.snapshots: dict[tuple[str, str], p.Snapshot] = {}
        self.changed = Condition()
        self.closing = False

    def admit(self, request: e.Request) -> None:
        with self.changed:
            rid, cmd = request.id, request.command
            if rid in self.seen:
                self.emit(e.ProtocolError("invalid_request", "duplicate id", rid))
                return
            channel = None
            try:
                if max(len(rid.encode()), len((request.channel or "").encode())) > 256:
                    raise ValueError("request and channel IDs are limited to 256 UTF-8 bytes")
                if cmd.__struct_config__.tag not in self.ready.capabilities:
                    raise ValueError("command unavailable in this worker")
                if len(self.seen) >= 100000 and not isinstance(cmd, e.Shutdown):
                    raise ValueError("session id limit reached")
                if request.channel:
                    channel = e.ChannelRevision(
                        request.channel, self.revisions.get(request.channel, 0) + 1
                    )
                if isinstance(
                    cmd,
                    e.Describe
                    | e.Devices
                    | e.Status
                    | e.Shutdown
                    | e.Close
                    | e.Release
                    | e.ControlRequest,
                ):
                    self._validate_fast(cmd)
                    self._accept(rid, channel)
                    result = self._fast(cmd)
                    self.emit(e.ResultReply(rid, channel, result))
                else:
                    if len(self.jobs) >= 64 and not any(
                        j.channel and channel and j.channel.name == channel.name for j in self.queue
                    ):
                        raise ValueError("64 execution requests pending")
                    job = self._pin(rid, cmd, channel)
                    self._accept(rid, channel)
                    self.jobs[rid] = job
                    self.queue.append(job)
                    self.changed.notify_all()
            except Exception as err:
                if rid in self.seen:
                    self.emit(e.ErrorReply(rid, channel, classify(err)))
                else:
                    self.emit(e.ProtocolError("invalid_request", str(err), rid))

    def _pin(self, rid: str, cmd: op.ExecutionInput, channel: e.ChannelRevision | None) -> Job:
        with ExitStack() as pins:
            leases = {}
            for ref in dict.fromkeys(op.input_references(cmd)):
                if isinstance(ref, Member):
                    handle = self.handles[ref.handle]
                    leases[ref] = ex.Lease(handle.bound.members[ref.member], handle.manifest)
                    handle.resource.pin(pins)
                    for token in handle.artifacts.intersection(self.artifacts):
                        self.artifacts[token][1].pin(pins)
                elif ref.artifact:
                    path, resource = self.artifacts[ref.artifact]
                    if str(path) != ref.path:
                        raise ValueError("artifact path differs from issued path")
                    resource.pin(pins)
            return Job(
                rid,
                cmd,
                channel,
                progress.Control(),
                Stages(self.root),
                leases,
                pins.pop_all(),
            )

    def _accept(self, rid: str, channel: e.ChannelRevision | None) -> None:
        self.seen.add(rid)
        if channel:
            self.revisions[channel.name] = channel.revision
            for job in list(self.jobs.values()):
                if job.channel and job.channel.name == channel.name:
                    self._stop_job(job, superseded=True)

    def _validate_fast(self, cmd: e.Command) -> None:
        if isinstance(cmd, e.ControlRequest) and cmd.target not in self.jobs:
            raise ValueError("unknown active request")
        if isinstance(cmd, e.Close) and cmd.handle not in self.handles:
            raise ValueError("unknown handle")
        if isinstance(cmd, e.Release) and cmd.artifact not in self.artifacts:
            raise ValueError("unknown artifact")

    def _fast(self, cmd: e.Command) -> e.Description | e.DeviceList | e.StatusReport | e.Ack:
        if isinstance(cmd, e.Describe):
            return self.description
        if isinstance(cmd, e.Devices):
            return e.DeviceList(self.ready.devices)
        if isinstance(cmd, e.Status):
            return e.StatusReport(
                {k: "queued" if j in self.queue else j.control.state for k, j in self.jobs.items()},
                {k: h.size for k, h in self.handles.items()},
                tuple(self.snapshots.values()),
            )
        if isinstance(cmd, e.Close):
            self._close_handle(cmd.handle)
        elif isinstance(cmd, e.Release):
            self.artifacts.pop(cmd.artifact)[1].drop()
        elif isinstance(cmd, e.Shutdown):
            self.stop()
        elif isinstance(cmd, e.ControlRequest) and (job := self.jobs.get(cmd.target)):
            if cmd.action == "cancel":
                self._stop_job(job)
            elif cmd.action == "pause":
                job.control.pause()
            else:
                job.control.resume()
        return e.Ack()

    def _stop_job(self, job: Job, *, superseded: bool = False) -> None:
        job.control.cancel()
        if job in self.queue:
            self.queue.remove(job)
            kind = "superseded" if superseded else "cancelled"
            self._finish(job, e.ErrorReply(job.id, job.channel, w.Error(2, kind, kind)))

    def stop(self) -> None:
        with self.changed:
            self.closing = True
            for job in list(self.jobs.values()):
                self._stop_job(job)
            self.changed.notify_all()

    def close(self) -> None:
        with ExitStack() as cleanup:
            for _, resource in self.artifacts.values():
                cleanup.callback(resource.drop)
            self.artifacts.clear()
            for name in list(self.handles):
                cleanup.callback(self._close_handle, name)

    def _close_handle(self, name: str) -> None:
        handle = self.handles.pop(name)
        with ExitStack() as cleanup:
            cleanup.callback(handle.resource.drop)
            for token in handle.artifacts:
                if issued := self.artifacts.pop(token, None):
                    cleanup.callback(issued[1].drop)

    def _stale(self, job: Job) -> bool:
        return bool(job.channel and self.revisions[job.channel.name] != job.channel.revision)

    def work(self) -> None:
        while True:
            with self.changed:
                self.changed.wait_for(lambda: self.closing or self.queue)
                if self.closing:
                    return
                job = self.queue.popleft()
            self._execute(job)
            del job  # Release the last execution context while waiting for more work.

    def _execute(self, job: Job) -> None:
        context = ex.ExecutionContext(
            job.leases,
            job.control,
            partial(self._event, job),
            partial(self._sample, job),
            job.stages,
        )
        success = None
        failure = None
        try:
            job.control.checkpoint()
            success = self.execute(job.command, context)
        except Exception as err:
            failure = err
        with self.changed:
            if self._stale(job):
                error = w.Error(2, "superseded", "superseded")
            elif job.control.cancelled:
                error = w.Error(2, "cancelled", "cancelled")
            elif failure is not None:
                error = classify(failure)
            else:
                assert success is not None
                try:
                    result = self._commit(job, success)
                except Exception as err:
                    error = classify(err)
                else:
                    self._finish(job, e.ResultReply(job.id, job.channel, result))
                    return
            if success and isinstance(success.value, ex.NewHandle):
                try:
                    success.value.dispose()
                except Exception:
                    logging.exception("Failed to dispose unpublished handle for %s", job.id)
            self._finish(job, e.ErrorReply(job.id, job.channel, error))

    def _commit(self, job: Job, success: ex.Success) -> p.Output:
        value = success.value
        handle = None
        if isinstance(value, ex.NewHandle):
            bound = value.bind(job.id)
            handle = Handle(
                bound, success.manifest, value.logical_bytes, Resource(value.dispose), job.published
            )
            product = bound.product
        else:
            product = value
        result = p.Output(product, success.info, success.manifest)
        # Encode before changing any caller-owned destination.
        _ = msgspec.json.encode(result)
        with ExitStack() as undo:
            for token in tuple(job.stages.arrays):
                self._publish(job, token)
                undo.callback(self.artifacts.pop, token)
                undo.callback(self.artifacts[token][1].drop)
            job.stages.commit()
            undo.pop_all()
        if handle:
            self.handles[job.id] = handle
        return result

    def _publish(self, job: Job, token: str) -> None:
        public = job.stages.publish_array(token)
        self.artifacts[token] = public, Resource(partial(public.unlink, missing_ok=True))
        job.published.add(token)

    def _sample(self, job: Job, snapshot: p.Snapshot) -> None:
        with self.changed:
            if job.id in self.jobs and not self._stale(job):
                self.snapshots[snapshot.device, snapshot.scope] = snapshot

    def _event(self, job: Job, event: w.Event) -> None:
        with self.changed:
            if job.id not in self.jobs or self._stale(job):
                return
            if isinstance(event, w.Preview):
                if job.control.cancelled:
                    return
                with ExitStack() as undo:
                    for array in event.arrays.values():
                        refs = [
                            array.storage,
                            *(
                                ax.coordinates.values
                                for ax in array.axes
                                if isinstance(ax.coordinates, Coordinates)
                            ),
                        ]
                        for ref in refs:
                            if isinstance(ref, Npy) and ref.artifact in job.stages.arrays:
                                assert ref.artifact is not None
                                self._publish(job, ref.artifact)
                                undo.callback(self.artifacts.pop, ref.artifact)
                                undo.callback(self.artifacts[ref.artifact][1].drop)
                    self.emit(e.EventReply(job.id, job.channel, event))
                    undo.pop_all()
                return
            self.emit(e.EventReply(job.id, job.channel, event))

    def _finish(self, job: Job, terminal: e.ErrorReply | e.ResultReply) -> None:
        del self.jobs[job.id]
        try:
            with job.lifetime:
                job.stages.temporary.close()
        except Exception:
            # The commit decision is final. Report cleanup failures to the operator.
            logging.exception("Failed to dispose resources for %s", job.id)
        self.emit(terminal)
