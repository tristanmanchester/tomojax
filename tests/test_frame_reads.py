"""Worker reads and bounded copies accept stacks that delegate to a lazy source."""

from pathlib import Path

import numpy as np

# check-public-imports: allow-private
from tomojax._data import Hdf5Frames

# check-public-imports: allow-private
from tomojax.cli._browse import pixels

# check-public-imports: allow-private
from tomojax.corrections import _batches

from ._helpers import write_raw_nxtomo


class Delegating:
    """A wrapper that forwards attribute access, as a proxy or lazy loader does."""

    def __init__(self, inner: object) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)

    def __getitem__(self, key: object) -> object:
        return self._inner[key]


def test_a_strided_worker_read_reaches_a_delegated_lazy_stack(tmp_path: Path) -> None:
    path = tmp_path / "raw.nxs"
    write_raw_nxtomo(path)
    stack = Hdf5Frames.open(str(path), "/entry/instrument/detector/data")
    ranges = (range(0, 4, 2), range(0, 2, 1), range(0, 2, 1))
    expected = np.asarray(stack)[0:4:2]
    np.testing.assert_array_equal(pixels(Delegating(stack), ranges), expected)


def test_a_delegated_shape_keeps_the_bounded_copy(monkeypatch) -> None:
    values = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    filled = []

    def fill(out, source, *, depth):
        filled.append(depth)
        out[:] = np.asarray(source[:])
        return out

    monkeypatch.setattr("tomojax.core.host_volume.fill", fill)
    copied = _batches.copy_projections(Delegating(values), [])
    np.testing.assert_array_equal(copied, values)
    assert filled  # sliced into a host-owned output, not read whole by np.array
