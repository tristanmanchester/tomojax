"""Serialized worker admission and publication. Only the adapter runs outside the monitor."""

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
from tomojax.core.input_files import InputFiles, reading
from tomojax.errors import Canceled, classify
from tomojax.wire import progress as w
from tomojax.wire.arrays import Array, Coordinates, Member, Npy, Record


@dataclass
class Resource:
    """The registry's initial reference is surrendered by close or release."""

    dispose: Callable[[], None]
    protection: ExitStack = field(default_factory=ExitStack)
    references: int = 1

    def pin(self, lifetime: ExitStack) -> None:
        self.references += 1
        lifetime.callback(self.drop)

    def drop(self) -> None:
        self.references -= 1
        if not self.references:
            with self.protection:
                self.dispose()


@dataclass
class Artifact:
    path: Path
    resource: Resource
    manifest: p.Provenance
    inputs: InputFiles
    metadata: Array | None = None
    name: str | None = None


@dataclass
class Handle:
    bound: ex.BoundHandle
    manifest: p.Provenance
    size: int
    resource: Resource
    artifacts: set[str]
    inputs: InputFiles


@dataclass(eq=False)
class Job:
    id: str
    command: op.ExecutionInput
    channel: e.ChannelRevision | None
    control: progress.Control
    stages: Stages
    leases: dict[Member | Npy, ex.Lease]
    lifetime: ExitStack
    inputs: InputFiles
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
        self.namespace: dict[str, object] = {}
        self.namespace_bytes = 0
        self.seen: set[str] = set()
        self.revisions: dict[str, int] = {}
        self.queue: deque[Job] = deque()
        self.jobs: dict[str, Job] = {}
        self.handles: dict[str, Handle] = {}
        self.artifacts: dict[str, Artifact] = {}
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

    def python_inputs(self, cmd: op.Python) -> op.Python:
        """Bind the live identifiers under the admission lock, before close can retire them."""
        aliases = {ref: key for key, ref in cmd.aliases.items()}
        live: dict[str, Member | Npy] = {}
        for key in self.handles:
            member = Member(key, "self")
            live[aliases.get(member, key)] = member
        for token, artifact in self.artifacts.items():
            storage = Npy(str(artifact.path), token)
            if name := aliases.get(storage, artifact.name):
                live[name] = storage
        return msgspec.structs.replace(cmd, aliases=live)

    def _pin(self, rid: str, cmd: op.ExecutionInput, channel: e.ChannelRevision | None) -> Job:
        with ExitStack() as pins:
            leases: dict[Member | Npy, ex.Lease] = {}
            parents: set[InputFiles] = set()
            pending = list(op.input_references(cmd))
            if isinstance(cmd, op.Python):
                cmd = self.python_inputs(cmd)
                pending.extend(cmd.aliases.values())
            while pending:
                ref = pending.pop()
                if ref in leases:
                    continue
                metadata = None
                if isinstance(ref, Member):
                    handle = self.handles[ref.handle]
                    leases[ref] = ex.Lease(
                        handle.bound.members[ref.member],
                        handle.manifest,
                        owner=handle.bound.members.get("self"),
                    )
                    member = leases[ref].value
                    if isinstance(member, ex.RegisteredArray):
                        metadata = member.metadata
                    parents.add(handle.inputs)
                    handle.resource.pin(pins)
                    for token in handle.artifacts.intersection(self.artifacts):
                        self.artifacts[token].resource.pin(pins)
                elif ref.artifact:
                    artifact = self.artifacts[ref.artifact]
                    if str(artifact.path) != ref.path:
                        raise ValueError("artifact path differs from issued path")
                    parents.add(artifact.inputs)
                    artifact.resource.pin(pins)
                    metadata = artifact.metadata
                    leases[ref] = ex.Lease(ref, artifact.manifest, metadata)
                # Consumers can compare a registered input's physical coordinates.
                if metadata:
                    pending.extend(
                        ax.coordinates.values
                        for ax in metadata.axes
                        if isinstance(ax.coordinates, Coordinates)
                        and isinstance(ax.coordinates.values, Member | Npy)
                    )
            inputs = InputFiles(parents)
            pins.enter_context(inputs.pinned())
            return Job(
                rid,
                cmd,
                channel,
                progress.Control(),
                Stages(self.root),
                leases,
                pins.pop_all(),
                inputs,
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
                self.namespace_bytes,
            )
        if isinstance(cmd, e.Close):
            self._close_handle(cmd.handle)
        elif isinstance(cmd, e.Release):
            self.artifacts.pop(cmd.artifact).resource.drop()
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
            kind = "superseded" if superseded else "canceled"
            self._finish(job, e.ErrorReply(job.id, job.channel, w.Error(2, kind, kind)))

    def stop(self) -> None:
        with self.changed:
            self.closing = True
            for job in list(self.jobs.values()):
                self._stop_job(job)
            self.changed.notify_all()

    def close(self) -> None:
        with ExitStack() as cleanup:
            for artifact in self.artifacts.values():
                cleanup.callback(artifact.resource.drop)
            self.artifacts.clear()
            for name in list(self.handles):
                cleanup.callback(self._close_handle, name)

    def _close_handle(self, name: str) -> None:
        handle = self.handles.pop(name)
        with ExitStack() as cleanup:
            cleanup.callback(handle.resource.drop)
            for token in handle.artifacts:
                if issued := self.artifacts.pop(token, None):
                    cleanup.callback(issued.resource.drop)

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

    def measure_namespace(self, size: int) -> None:
        with self.changed:
            self.namespace_bytes = size

    def _execute(self, job: Job) -> None:
        context = ex.ExecutionContext(
            job.leases,
            job.control,
            partial(self._event, job),
            partial(self._sample, job),
            job.stages,
            self.namespace,
            self.measure_namespace,
        )
        success = None
        failure = None
        stopped = None
        try:
            job.control.boundary()
            with reading(
                *map(Path, op.input_files(job.command)), group=job.inputs, owner="request"
            ):
                success = self.execute(job.command, context)
        except Exception as err:
            failure = classify(err)
            if isinstance(err, Canceled):
                stopped = failure
        with self.changed:
            if self._stale(job):
                error = w.Error(2, "superseded", "superseded")
            elif job.control.canceled:
                # A checkpointed run says where its saved state is.
                error = stopped or w.Error(2, "canceled", "canceled")
            elif failure is not None:
                error = failure
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

    def bind_roots(self, job: Job, success: ex.Success) -> tuple[p.Product, dict[str, Handle]]:
        """Build every handle descriptor before installing the first root or artifact."""
        value = success.value
        roots: dict[str, Handle] = {}
        if isinstance(value, ex.NewHandle):
            bound = value.bind(job.id)
            roots[job.id] = Handle(
                bound,
                success.manifest,
                value.logical_bytes,
                Resource(value.dispose),
                job.published,
                job.inputs,
            )
            product = bound.product
        else:
            product = value
        for name, root in success.roots:
            bound = root.bind(name)
            roots[name] = Handle(
                bound,
                success.manifest,
                root.logical_bytes,
                Resource(root.dispose),
                set(),
                job.inputs,
            )
            assert isinstance(product, p.PythonResult)
            product.handles[name] = bound.product
        return product, roots

    def _commit(self, job: Job, success: ex.Success) -> p.Output:
        with ExitStack() as undo:
            product, roots = self.bind_roots(job, success)
            result = p.Output(product, success.info, success.manifest)
            # Encode before changing any caller-owned destination.
            encoded = msgspec.json.encode(result)
            if isinstance(product, p.PythonResult) and len(encoded) > e.INLINE_LIMIT:
                raise ValueError(
                    "Python result too large for the worker pipe. "
                    "Return a smaller value or register arrays."
                )
            for token in tuple(job.stages.arrays):
                self._publish(job, token, success.manifest, undo)
            job.stages.commit()
            self._register_arrays(result)
            self.name_arrays(job.id, product)
            for root in roots.values():
                root.resource.protection.enter_context(job.inputs.pinned())
                _ = undo.callback(root.resource.protection.close)
            self.handles.update(roots)
            undo.pop_all()
        return result

    def name_arrays(self, request: str, product: p.Product) -> None:
        """Remember which published arrays are top-level Python inputs."""
        names = product.handles if isinstance(product, p.PythonResult) else {request: product}
        for name, member in names.items():
            if isinstance(member, p.ArrayResult | p.Projections | p.Volume):
                array = getattr(member, member.__struct_fields__[0])
                storage = array.storage
                if isinstance(storage, Npy) and storage.artifact in self.artifacts:
                    self.artifacts[storage.artifact].name = name

    def _register_arrays(self, value: object) -> None:
        if isinstance(value, Array):
            if isinstance(value.storage, Npy) and value.storage.artifact in self.artifacts:
                self.artifacts[value.storage.artifact].metadata = value
        elif isinstance(value, Record):
            for name in value.__struct_fields__:
                self._register_arrays(getattr(value, name))
        elif isinstance(value, tuple):
            for item in value:
                self._register_arrays(item)
        elif isinstance(value, dict):
            for item in value.values():
                self._register_arrays(item)

    def _publish(self, job: Job, token: str, manifest: p.Provenance, undo: ExitStack) -> None:
        public = job.stages.publish_array(token)
        self.artifacts[token] = Artifact(
            public, Resource(partial(public.unlink, missing_ok=True)), manifest, job.inputs
        )
        self.artifacts[token].resource.protection.enter_context(job.inputs.pinned())
        undo.callback(self.artifacts.pop, token)
        undo.callback(self.artifacts[token].resource.drop)
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
                if job.control.canceled:
                    return
                # Preview files are reusable before the adapter returns its final manifest.
                command = job.command.operation if isinstance(job.command, op.Run) else job.command
                parameters = msgspec.to_builtins(command)
                parents = tuple(job.leases[r].provenance for r in op.input_references(job.command))
                manifest = p.Provenance(parameters.pop("type"), parameters, parents)
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
                                self._publish(job, ref.artifact, manifest, undo)
                    self._register_arrays(event.arrays)
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
