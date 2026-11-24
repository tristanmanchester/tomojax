"""Per-call native and Python stream capture with bounded UTF-8 text buffers."""

import array
import codecs
from contextlib import ExitStack, contextmanager
import ctypes
import fcntl
import os
import select
import sys
import termios
from threading import Lock, Thread


class Pipe:
    def __init__(self, source: int) -> None:
        self.source = source
        self.lock = Lock()
        self.first = self.last = ""
        self.count = 0
        self.decoder = codecs.getincrementaldecoder("utf8")("replace")
        self.done = False
        Thread(target=self.drain, daemon=True).start()

    def append(self, text: str):
        self.count += len(text)
        head = 10000 - len(self.first)
        self.first += text[:head]
        self.last = (self.last + text[head:])[-10000:]

    def drain(self):
        poll = select.poll()
        poll.register(self.source, select.POLLIN)
        try:
            while True:
                poll.poll()
                with self.lock:
                    try:
                        data = os.read(self.source, 65536)
                    except BlockingIOError:
                        continue
                    if not self.done:
                        self.append(self.decoder.decode(data, final=not data))
                    if not data:
                        return
        finally:
            with self.lock:
                os.close(self.source)
                self.source = -1

    def finish(self):
        with self.lock:
            if self.source >= 0:
                size = array.array("i", [0])
                fcntl.ioctl(self.source, termios.FIONREAD, size)
                remaining = size[0]
                while remaining:
                    chunk = os.read(self.source, min(65536, remaining))
                    self.append(self.decoder.decode(chunk))
                    remaining -= len(chunk)
                self.append(self.decoder.decode(b"", final=True))
            self.done = True
        missing = self.count - len(self.first) - len(self.last)
        return self.first + (f"\n[{missing} characters omitted]\n" if missing else "") + self.last


def flush():
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and not stream.closed:
            stream.flush()
    ctypes.CDLL(None).fflush(None)


@contextmanager
def capture():
    streams = []
    output = []
    flush()
    with ExitStack() as restore:
        for name, fd in (("stdout", 1), ("stderr", 2)):
            original = getattr(sys, name)
            restore.callback(setattr, sys, name, original)
            saved = os.dup(fd)
            restore.callback(os.close, saved)
            read, write = os.pipe()
            os.set_blocking(read, False)
            streams.append(Pipe(read))
            os.dup2(write, fd)
            os.close(write)
            restore.callback(os.dup2, saved, fd)
            # Give the call its own wrapper; closing or rebinding sys.stdout in
            # user code must not damage the worker's stream in the next call.
            wrapper = restore.enter_context(
                open(fd, "w", encoding=original.encoding, errors=original.errors, closefd=False)
            )
            setattr(sys, name, wrapper)
        try:
            yield output
        finally:
            try:
                flush()
            finally:
                restore.close()
                output.extend(stream.finish() for stream in streams)
