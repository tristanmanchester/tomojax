"""Subprocess executor fixture. Its separate socket gates adapter work boundaries and returns."""

import json
import os
from pathlib import Path
import resource
import socket
import sys

import jax
import msgspec
import numpy as np

from tomojax import progress
from tomojax.cli import envelope as e, execution as ex, operations as op, products as p
from tomojax.cli.transport import serve
from tomojax.errors import InvalidParameter
from tomojax.wire import arrays as a, progress as w


def handle():
    data = np.arange(24, dtype=np.int16).reshape(2, 3, 4)

    def bind(name):
        meta = a.Array(
            a.Member(name, "volume"),
            data.shape,
            data.dtype.str,
            tuple(a.Axis(dim, "volume", "mm", a.Uniform(0, 1)) for dim in "zyx"),
            "1/mm",
            "attenuation",
        )
        volume = ex.RegisteredArray(data, meta)
        return ex.BoundHandle(
            p.Reconstruction(name, meta, None, "fbp"), {"self": volume, "volume": volume}
        )

    return ex.NewHandle(data.nbytes, bind, lambda: None)


def fail(case):
    if case == "invalid":
        raise InvalidParameter("config.filter", "unknown filter")
    if case.startswith("jax"):

        class Status(jax.errors.JaxRuntimeError):
            @property
            def error_code(self):
                return 8 if case == "jax-oom" else 13

        raise RuntimeError("operation failed") from Status("unrelated message")
    # Fail one real allocation in this isolated process, then restore its limit.
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    mapped = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
    try:
        resource.setrlimit(resource.RLIMIT_AS, (1, hard))
        np.empty(mapped + 1, dtype=np.uint8)
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


class Stub:
    def __init__(self, port):
        self.port = port

    def gate(self):
        self.port.sendall(b"G")
        assert self.port.recv(1) == b"C"

    def groups(self, operation, context):
        path = Path(operation.path)
        dests = [path, path.with_suffix(".json"), path.with_suffix(".manifest")]
        for dest in dests:
            stage = context.staging.output(dest, force=operation.force)
            if dest.suffix != ".manifest" or path.stem != "partial":
                stage.write_text("replacement " + dest.suffix)
        result = ex.Success(
            p.Files(tuple(map(str, dests))), w.Info({}, {}), p.Provenance("export", {}, ())
        )
        self.gate()
        return result

    def wait(self, context, name, label, total, unit, *, message=None):
        def relay(event):
            context.event(msgspec.convert(progress.to_json(event), type=w.Event))

        with progress.watch(relay, control=context.control):
            if message:
                progress.note("info", message)
            with progress.task(name, label, total, unit):
                self.gate()
                context.control.boundary()
                self.port.sendall(b"R")

    def execute(self, command, context):
        if isinstance(command, op.Plane):
            return ex.browse_plane(command, context)
        if isinstance(command, op.Run) and isinstance(command.operation, op.Export):
            return self.groups(command.operation, context)
        if isinstance(command, op.Run) and isinstance(command.operation, op.Reconstruct):
            source = command.operation.source
            self.wait(
                context,
                "reconstruct",
                "scan",
                10,
                "iterations",
                message=f"Opening {Path(source.path).name}",
            )
        if isinstance(command, op.Open):
            value = handle()
        elif isinstance(command, op.Inspect):
            value = self.inspect(command, context)
        elif isinstance(command, op.Run) and isinstance(command.operation, op.Backproject):
            storage = command.operation.projections
            value = p.Inspection({"sum": int(np.load(storage.path).sum())})
        else:
            value = p.Validated()
        context.snapshot(p.Snapshot(4.25, "cpu:0", "host", 1234, None, None))
        return ex.Success(
            value,
            w.Info({}, {}),
            p.Provenance(type(command).__name__, {}, context.lineage(command)),
        )

    def inspect(self, command, context):
        if command.path == "wait":
            self.wait(context, "inspect", "input", None, "")
        elif command.path == "executor-exit":
            raise SystemExit(3)
        elif command.path == "stderr":
            os.write(2, b"x" * (2 * 1024 * 1024))
        elif command.path == "ticks":
            for done in range(1, 100):
                context.event(w.Advanced(2, 1, done, done, 99, {}, None))
            context.event(w.Note(2, 1, 100, "info", "complete"))
            self.port.sendall(b"D")
        elif command.path == "notify":
            self.port.sendall(b"N")
        elif command.path == "preview":
            stage = context.staging.array()
            np.save(stage.path, np.asarray(99))
            arr = a.Array(stage.reference, (), "<i8", (), "1", "counts")
            context.event(w.Preview(2, 1, 0, {"frame": arr}))
            self.gate()
        else:
            fail(command.path)
        return p.Inspection({"path": command.path})


def worker(port):
    files = Path(__file__).resolve().parents[1] / "docs/protocol/transcripts/session.jsonl"
    ready = msgspec.convert(json.loads(files.read_text().splitlines()[0]), type=e.Ready)
    return serve(
        Stub(port).execute,
        stdin=0,
        stdout=1,
        ready=ready,
        description=e.Description({"large": "x" * 32768}),
    )


if __name__ == "__main__":
    with socket.socket(fileno=int(sys.argv[1])) as port:
        raise SystemExit(worker(port))
