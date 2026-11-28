"""Measure three Studio-style panes through a worker's stdio, including decoding."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from statistics import median
import sys
import tempfile
from time import monotonic

import imagecodecs
import msgspec
import numpy as np

from tomojax.cli import envelope as e


class Options(argparse.Namespace):
    """Dimensions of the run and the stdio process to start."""

    command: list[str]
    seconds: float
    repeats: int
    json_out: Path | None
    scratch: Path | None


@dataclass
class Pane:
    """A slot keeps its latest selected index until its moving frame arrives."""

    edge: int
    wanted: int = 0
    active: str | None = None


@dataclass
class Request:
    """Payload received so far for one selection."""

    pane: int
    index: int
    encoding: str
    phase: str
    sent: float
    interrupted_at: int | None = None
    data: bytearray = field(default_factory=bytearray)


@dataclass
class Frame:
    """Elapsed times from submission through receipt and decode."""

    pane: int
    index: int
    encoding: str
    phase: str
    bytes: int
    arrival_ms: float
    decoded_ms: float


type ViewerReply = e.ExecutorEvent | e.Block | e.ResultReply | e.ErrorReply


class Viewer:
    """Drive the three pane channels against one worker."""

    def __init__(
        self, process: asyncio.subprocess.Process, path: Path, shape: tuple[int, ...]
    ) -> None:
        assert process.stdin is not None
        assert process.stdout is not None
        self.decode: msgspec.json.Decoder[ViewerReply] = msgspec.json.Decoder(ViewerReply)
        self.pipe = process.stdin
        self.stream = process.stdout
        self.path = path
        self.shape = shape
        self.panes = [Pane(1024), Pane(512), Pane(512)]
        self.inflight: dict[str, Request] = {}
        self.frames: list[Frame] = []
        self.serial = 0
        self.phase = "scroll"
        self.interrupt_at = 0
        self.discarded: list[int] = []
        self.resume_at: list[int] = []

    def send(self, message: dict[str, object]) -> None:
        """Write one request or credit acknowledgement."""
        self.pipe.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")

    def request(self, slot: int, encoding: str) -> None:
        """Start a fetch at the resolution the pane uses for this encoding."""
        pane = self.panes[slot]
        step = 1024 // (min(pane.edge, 512) if encoding == "jpeg" else pane.edge)
        self.serial += 1
        rid = str(self.serial)
        pane.active = rid
        self.inflight[rid] = Request(slot, pane.wanted, encoding, self.phase, monotonic())
        self.send(
            {
                "id": rid,
                "channel": f"view.{slot}",
                "command": {
                    "type": "fetch",
                    "storage": {"type": "npy", "path": str(self.path)},
                    "shape": self.shape,
                    "stride": [1, step, step],
                    "roi": [[pane.wanted, pane.wanted + 1], [0, 1024], [0, 1024]],
                    "encoding": encoding,
                    "window": [-2, 2] if encoding == "jpeg" else None,
                },
            }
        )

    def move(self, slot: int, index: int) -> None:
        """Coalesce changes during a JPEG and replace a lossless fetch immediately."""
        pane = self.panes[slot]
        pane.wanted = index
        if pane.active is None or self.inflight[pane.active].encoding == "float32":
            self.request(slot, "jpeg")

    async def reply(self) -> None:
        """Consume one block or finish decoding a frame."""
        line = await self.stream.readline()
        row = self.decode.decode(line)
        if isinstance(row, e.ExecutorEvent):
            return
        rid = row.request
        req = self.inflight[rid]
        pane = self.panes[req.pane]
        if isinstance(row, e.Block):
            size = row.bytes
            req.data.extend(await self.stream.readexactly(size))
            self.send({"credit": size})
            if (
                self.interrupt_at
                and req.pane == 0
                and req.encoding == "float32"
                and pane.active == rid
                and len(req.data) >= self.interrupt_at
            ):
                req.interrupted_at = len(req.data)
                self.resume_at.append(len(req.data))
                self.interrupt_at = 0
                self.phase = "resume"
                self.move(0, (pane.wanted + 1) % self.shape[0])
            return
        del self.inflight[rid]
        if isinstance(row, e.ErrorReply) and row.error.kind == "superseded" and pane.active != rid:
            if req.interrupted_at is not None:
                self.discarded.append(len(req.data) - req.interrupted_at)
            return
        if not isinstance(row, e.ResultReply) or not isinstance(row.result, e.Pixels):
            raise RuntimeError(row)
        result = row.result
        if result.bytes != len(req.data) or result.encoding != req.encoding:
            raise ValueError("incomplete frame")
        arrival = monotonic()
        if req.encoding == "jpeg":
            samples = imagecodecs.jpeg8_decode(req.data)
        else:
            shuffled = np.frombuffer(imagecodecs.zstd_decode(req.data), dtype="<f4")
            samples = imagecodecs.byteshuffle_decode(shuffled)
        if samples.size != np.prod(result.shape):
            raise ValueError("decoded size differs from descriptor")
        frame = Frame(
            req.pane,
            req.index,
            req.encoding,
            req.phase,
            len(req.data),
            1000 * (arrival - req.sent),
            1000 * (monotonic() - req.sent),
        )
        self.frames.append(frame)
        if pane.active == rid:
            pane.active = None
            if req.encoding == "jpeg":
                self.request(req.pane, "float32" if pane.wanted == req.index else "jpeg")

    async def settle(self) -> None:
        """Receive outstanding requests through the final lossless frames."""
        while self.inflight:
            await self.reply()

    async def exercise(self, seconds: float, repeats: int) -> dict[str, object]:
        """Scroll the main pane, let it settle, then interrupt a known lossless frame halfway."""
        if (
            "fetch"
            not in msgspec.json.decode(
                await self.stream.readline(), type=e.WorkerReady
            ).capabilities
        ):
            raise ValueError("worker lacks fetch")
        started = monotonic()
        for slot in range(3):
            self.move(slot, slot)

        async def scroll() -> tuple[int, float]:
            while monotonic() < started + seconds:
                await asyncio.sleep(1 / 60)
                self.move(0, (self.panes[0].wanted + 1) % self.shape[0])

            return sum(
                f.pane == 0 and f.encoding == "jpeg" for f in self.frames
            ), monotonic() - started

        motion = asyncio.create_task(scroll())
        while not motion.done():
            await self.reply()
        count, duration = await motion
        self.phase = "settle"
        await self.settle()
        known = self.panes[0].wanted
        full = next(
            f.bytes for f in reversed(self.frames) if f.pane == 0 and f.encoding == "float32"
        )
        for _ in range(repeats):
            self.phase = "settle"
            self.move(0, (known + 2) % self.shape[0])
            await self.settle()
            self.interrupt_at = (full + 1) // 2
            self.move(0, known)
            await self.settle()
        moving = [
            f for f in self.frames if f.phase == "scroll" and f.pane == 0 and f.encoding == "jpeg"
        ]
        settled = [
            f
            for f in self.frames
            if f.phase == "settle" and f.pane == 0 and f.encoding == "float32"
        ]
        resumed = [f for f in self.frames if f.phase == "resume" and f.encoding == "jpeg"]
        return {
            "held_seconds": duration,
            "scroll_fps": count / duration,
            "moving_bytes": median(f.bytes for f in moving),
            "moving_ms": median(f.decoded_ms for f in moving),
            "moving_arrival_ms": median(f.arrival_ms for f in moving),
            "settled_bytes": median(f.bytes for f in settled),
            "settled_ms": median(f.decoded_ms for f in settled),
            "resume_ms": [f.decoded_ms for f in resumed],
            "interrupted_at_bytes": self.resume_at,
            "obsolete_bytes_after_move": self.discarded,
            "decoded_per_pane": dict(Counter(f.pane for f in self.frames)),
            "frames": [asdict(f) for f in self.frames],
        }


async def measure(args: Options, path: Path, shape: tuple[int, ...]) -> dict[str, object]:
    """Drive the production protocol and reap the command on exit."""
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    command = command or ["tomojax-worker", "serve", "--python", sys.executable]
    proc = await asyncio.create_subprocess_exec(*command, stdin=-1, stdout=-1)
    try:
        viewer = Viewer(proc, path, shape)
        record = await viewer.exercise(args.seconds, args.repeats)
        viewer.send({"id": "quit", "command": {"type": "shutdown"}})
        reply = msgspec.json.decode(await viewer.stream.readline(), type=e.ResultReply)
        if reply.request != "quit" or await proc.wait():
            raise RuntimeError("worker shutdown failed")
        return record
    finally:
        if proc.returncode is None:
            proc.kill()
            _ = await proc.wait()


def main() -> int:
    """Create synthetic noise on disk and print the frame timings."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--seconds", type=float, default=10, help="held scroll duration")
    _ = parser.add_argument("--repeats", type=int, default=5, help="half-transfer interruptions")
    _ = parser.add_argument("--json-out", type=Path, help="write every measured frame")
    _ = parser.add_argument("--scratch", type=Path, help="directory for the synthetic NPY stack")
    _ = parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="stdio worker or slowlink invocation"
    )
    args = parser.parse_args(namespace=Options())
    if args.seconds <= 0 or args.repeats < 1:
        parser.error("seconds and repeats must be positive")
    with tempfile.TemporaryDirectory(dir=args.scratch) as directory:
        path = Path(directory).resolve() / "planes.npy"
        rng = np.random.default_rng(20261128)
        shape = (64, 1024, 1024)
        data = np.lib.format.open_memmap(path, mode="w+", dtype="<f4", shape=shape)
        for index in range(shape[0]):
            data[index] = rng.standard_normal((1024, 1024), dtype=np.float32)
        del data
        record = asyncio.run(measure(args, path, shape))
    if args.json_out:
        _ = args.json_out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "frames"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
