"""Reconstruct a simulated scan across a disconnect and verify a slice on both machines."""

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import hashlib
from io import BytesIO
import os
from pathlib import Path
import sys
from uuid import uuid4

import imagecodecs
import msgspec
import numpy as np

from tomojax.cli import envelope as e
from tomojax.wire.arrays import Hdf5


class _Profile(msgspec.Struct):
    results: str = "~"


class _Array(msgspec.Struct):
    storage: Hdf5
    shape: tuple[int, ...]


class _Stored(msgspec.Struct):
    arrays: dict[str, _Array]


class _Location(msgspec.Struct):
    path: str


class _Placement(msgspec.Struct):
    event: str
    devices: tuple[str, ...] = ()


class _Record(msgspec.Struct):
    record: _Placement | None = None


class _Job(msgspec.Struct):
    directory: str
    state: str
    detail: str
    receipt: _Location | None = None
    offset: int = 0
    records: tuple[_Record, ...] = ()


class _Jobs(msgspec.Struct):
    jobs: tuple[_Job, ...]


class _Download(msgspec.Struct):
    sha256: str
    bytes: int


class _Reply(msgspec.Struct):
    type: str
    request: str | None = None
    bytes: int = 0
    result: msgspec.Raw = msgspec.Raw(b"null")


class Client:
    """One outstanding request on a worker connection, including its binary payload."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        assert process.stdin is not None
        self.reader = process.stdout
        self.writer = process.stdin
        self.number = 0

    async def call[T](self, command: object, schema: type[T]) -> tuple[T, bytes]:
        """Read the request through its terminal, returning credit as blocks arrive."""
        self.number += 1
        rid = str(self.number)
        self.writer.write(msgspec.json.encode({"id": rid, "command": command}) + b"\n")
        data = bytearray()
        while line := await self.reader.readline():
            row = msgspec.json.decode(line, type=_Reply)
            if row.type in ("progress", "executor"):
                continue
            if row.request != rid:
                raise ValueError(f"Unexpected worker reply: {line!r}")
            if row.type == "block":
                data.extend(await self.reader.readexactly(row.bytes))
                self.writer.write(msgspec.json.encode(e.Credit(row.bytes)) + b"\n")
            elif row.type == "result":
                return msgspec.json.decode(row.result, type=schema), bytes(data)
            else:
                raise RuntimeError(line.decode())
        raise EOFError("Connection ended before the worker replied")

    async def plane(self, file: str, key: str, encoding: e.PixelEncoding) -> tuple[e.Pixels, bytes]:
        """Fetch the middle plane at its stored dimensions."""
        stored, _ = await self.call({"type": "stored", "path": file}, _Stored)
        array = stored.arrays[key]
        depth, rows, cols = array.shape
        index = depth // 2
        frame, payload = await self.call(
            e.Fetch(
                storage=array.storage,
                shape=array.shape,
                roi=((index, index + 1), (0, rows), (0, cols)),
                encoding=encoding,
            ),
            e.Pixels,
        )
        if (
            frame.shape != (1, rows, cols)
            or frame.bytes != len(payload)
            or frame.encoding != encoding
        ):
            raise ValueError(f"Incomplete {encoding} plane: {frame}")
        return frame, payload


@asynccontextmanager
async def connect(name: str, *, drop: bool = False) -> AsyncIterator[Client]:
    """Start a fresh worker and wait for its numerical executor."""
    sibling = Path(sys.executable).with_name("tomojax-worker")
    process = await asyncio.create_subprocess_exec(
        sibling if sibling.is_file() else "tomojax-worker",
        "connect",
        name,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        limit=16 << 20,
    )
    try:
        client = Client(process)
        ready = msgspec.json.decode(await client.reader.readline(), type=e.WorkerReady)
        if ready.jobs == "inline":
            raise ValueError("Choose a connection with process or Slurm jobs")
        print(f"Connected to {ready.host} ({ready.jobs})", flush=True)
        while True:
            event = msgspec.json.decode(await client.reader.readline(), type=e.ExecutorEvent)
            if event.state == "ready":
                break
            if event.state == "gone":
                raise RuntimeError(f"Executor stopped: {event.detail}")
        yield client
    finally:
        if drop and process.returncode is None:
            process.kill()
        if process.stdin:
            process.stdin.close()
        try:
            _ = await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.kill()
            _ = await process.wait()


async def exercise(name: str, parent: str, output: Path, *, require_gpu: bool) -> None:
    """Disconnect during reconstruction and reopen the job from its directory."""
    run = f"remote-check-{uuid4().hex}"
    async with connect(name, drop=True) as client:
        result, _ = await client.call(
            {
                "type": "project_results",
                "action": "claim",
                "project": run,
                "name": run,
                "parent": parent,
            },
            _Location,
        )
        root = result.path
        scan = f"{root}/inputs/scan.nxs"
        _ = await client.call(
            {"type": "python", "code": f"tj.save({scan!r}, tj.simulate(size=64, n_views=90).scan)"},
            msgspec.Raw,
        )
        frame, pixels = await client.plane(scan, "projections", "jpeg")
        projection = imagecodecs.jpeg8_decode(pixels)
        if projection.shape != frame.shape[1:] or np.ptp(projection) == 0:
            raise ValueError("Expected a nonconstant projection of the phantom")
        directory = f"{root}/jobs/{run}"
        op: dict[str, object] = {
            "type": "reconstruct",
            "source": {"type": "path", "path": scan},
            "method": "cgls",
            "iterations": 12,
        }
        script: list[dict[str, object]] = [
            {"id": f"{run}.1", "command": {"type": "run", "operation": op}}
        ]
        job, _ = await client.call(
            {"type": "submit", "directory": directory, "script": script}, _Job
        )
        while job.state in {"queued", "unknown"}:
            await asyncio.sleep(0.5)
            job, _ = await client.call({"type": "job", "directory": directory}, _Job)
        if job.state != "running":
            raise RuntimeError(f"Expected a running job before disconnect: {job}")
        print(f"Disconnecting during {directory}", flush=True)

    async with connect(name) as client:
        found, _ = await client.call({"type": "jobs", "root": root}, _Jobs)
        if [j.directory for j in found.jobs] != [directory]:
            raise RuntimeError(f"Reconnected worker did not find the submitted job: {found}")
        offset = 0
        devices: set[str] = set()
        while True:
            job, _ = await client.call(
                {"type": "job", "directory": directory, "offset": offset}, _Job
            )
            for record in job.records:
                if record.record and record.record.event == "placed":
                    devices.update(record.record.devices)
            # Drain all log pages even when the receipt appeared before reconnect.
            if job.receipt and offset == job.offset:
                break
            if job.state in {"failed", "canceled", "not_launched"}:
                raise RuntimeError(f"Job {job.state}: {job.detail}")
            offset = job.offset
            if not job.receipt:
                await asyncio.sleep(1)
        if require_gpu and not any(
            "gpu" in device.lower() or "cuda" in device.lower() for device in devices
        ):
            raise ValueError(f"No GPU placement in job records: {devices}")
        print(f"Job devices: {sorted(devices)}", flush=True)
        await verify_slice(client, root, job.receipt.path, output)


async def verify_slice(client: Client, root: str, path: str, output: Path) -> None:
    """Compare the lossless wire frame with an independent Python read of the saved volume."""
    frame, pixels = await client.plane(path, "volume", "float32")
    samples = imagecodecs.byteshuffle_decode(np.frombuffer(imagecodecs.zstd_decode(pixels), "<f4"))
    remote_slice = f"{root}/centre.npy"
    code = (
        f"v = tj.load_reconstruction({path!r}).volume\n"
        f"np.save({remote_slice!r}, np.asarray(v[len(v)//2], dtype='<f4'))"
    )
    _ = await client.call({"type": "python", "code": code}, msgspec.Raw)
    transfer, data = await client.call({"type": "download", "path": remote_slice}, _Download)
    if transfer.bytes != len(data) or hashlib.sha256(data).hexdigest() != transfer.sha256:
        raise ValueError("Downloaded slice failed its size or SHA-256 check")
    expected = np.lib.format.read_array(BytesIO(data), allow_pickle=False)
    if expected.shape != frame.shape[1:] or samples.tobytes() != expected.tobytes():
        raise ValueError("Lossless pixels differ from the axial slice read on the node")
    with output.open("xb") as local:
        _ = local.write(data)
    print(f"Matched {expected.size} samples; downloaded {output}", flush=True)


class _Options(argparse.Namespace):
    name: str
    output: Path | None
    require_gpu: bool


def main() -> None:
    """Create a new results directory on the connection and download one verified plane."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("name", help="Saved connection with a writable results folder")
    _ = parser.add_argument(
        "--output", type=Path, help="New local NPY file (default: unique filename)"
    )
    _ = parser.add_argument(
        "--require-gpu", action="store_true", help="Require a GPU placement record"
    )
    args = parser.parse_args(namespace=_Options())
    config = (
        Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        / "tomojax/connections.toml"
    )
    profile = msgspec.toml.decode(config.read_bytes(), type=dict[str, _Profile])[args.name]
    output = args.output or Path(f"remote-slice-{uuid4().hex[:8]}.npy")

    async def run() -> None:
        async with asyncio.timeout(1800):
            await exercise(args.name, profile.results, output, require_gpu=args.require_gpu)

    asyncio.run(run())


if __name__ == "__main__":
    main()
