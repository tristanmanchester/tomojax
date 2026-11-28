"""Run a stdio command through a full-duplex, bandwidth- and delay-limited link."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import os
import select
import subprocess
import sys
from time import monotonic


class Options(argparse.Namespace):
    """Wire characteristics in each direction."""

    rate: float
    delay_ms: float
    buffer: int
    kill_after: float
    command: list[str]


@dataclass
class Direction:
    """One independently paced byte stream."""

    source: int
    sink: int
    queue: deque[tuple[float, memoryview]] = field(default_factory=deque)
    buffered: int = 0
    available: float = 0
    eof: bool = False

    def receive(self, limit: int, rate: float, delay: float) -> None:
        """Assign the last byte a delivery time after serialization and propagation."""
        data = os.read(self.source, min(16384, limit - self.buffered))
        if data:
            self.available = max(self.available, monotonic()) + len(data) / rate
            self.queue.append((self.available + delay, memoryview(data)))
            self.buffered += len(data)
        else:
            self.eof = True

    def send(self) -> None:
        """Forward the due bytes that fit in the destination pipe."""
        at, data = self.queue.popleft()
        size = os.write(self.sink, data)
        self.buffered -= size
        if size < len(data):
            self.queue.appendleft((at, data[size:]))


def relay(argv: list[str], rate: float, delay: float, capacity: int, kill_after: float) -> int:
    """Delay queued bytes in both directions, preserving half-close and partial writes."""
    with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as child:
        assert child.stdin is not None
        assert child.stdout is not None
        up = Direction(sys.stdin.fileno(), child.stdin.fileno())
        down = Direction(child.stdout.fileno(), sys.stdout.fileno())
        deadline = monotonic() + kill_after if kill_after else float("inf")
        previous = os.get_blocking(down.sink)
        try:
            for direction in (up, down):
                os.set_blocking(direction.sink, False)
            while not (down.eof and not down.queue):
                if up.eof and not up.queue:
                    child.stdin.close()
                now = monotonic()
                if now >= deadline:
                    return 1
                active = (up, down)
                reads = [d.source for d in active if not d.eof and d.buffered < capacity]
                writes = [d.sink for d in active if d.queue and d.queue[0][0] <= now]
                until = min(
                    [deadline, *(d.queue[0][0] for d in active if d.queue and d.queue[0][0] > now)]
                )
                readable, writable, _ = select.select(
                    reads, writes, [], None if until == float("inf") else until - now
                )
                for direction in active:
                    if direction.sink in writable:
                        direction.send()
                    if direction.source in readable:
                        direction.receive(capacity, rate, delay)
            child.stdin.close()
            return child.wait()
        finally:
            os.set_blocking(down.sink, previous)
            if child.poll() is None:
                child.kill()


def main() -> int:
    """Run the command with a simulated link on its standard streams."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        "--rate", type=float, default=1200000, help="bytes per second per direction"
    )
    _ = parser.add_argument("--delay-ms", type=float, default=27, help="one-way propagation delay")
    _ = parser.add_argument(
        "--buffer", type=int, default=2 << 20, help="queued bytes per direction"
    )
    _ = parser.add_argument(
        "--kill-after", type=float, default=0, help="drop the link after this many seconds"
    )
    _ = parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(namespace=Options())
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.rate <= 0 or args.delay_ms < 0 or args.buffer < 1 or args.kill_after < 0:
        parser.error("supply a command, positive rate and buffer, and nonnegative times")
    try:
        return relay(command, args.rate, args.delay_ms / 1000, args.buffer, args.kill_after)
    except BrokenPipeError:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
