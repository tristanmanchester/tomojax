"""Streamed detector rows come from raw blocks sized for the file system and the host."""

import numpy as np
import pytest

from tomojax.corrections import _rows  # check-public-imports: allow-private
from tomojax.corrections.api import corrected_row_source

MiB, GiB = 1 << 20, 1 << 30


@pytest.mark.parametrize(
    ("views", "room", "rows"),
    [
        # Each view's read is a megabyte: 204 rows of 5,120 bytes.
        (8000, None, 204),
        (8000, 1000 * GiB, 204),
        # A host with 24 GiB to spare holds blocks of 1 GiB, the least there is.
        (8000, 24 * GiB, GiB // (8000 * 5120)),
        (8000, 8 * GiB, GiB // (8000 * 5120)),
        # Between the two: a twenty-fourth of the room.
        (8000, 96 * GiB, 4 * GiB // (8000 * 5120)),
        # Few views read a megabyte each within 1 GiB.
        (900, None, GiB // (900 * 5120)),
        # A block never has more rows than the detector.
        (50, None, 2160),
    ],
)
def test_blocks_read_a_megabyte_per_view_within_an_eighth_of_the_room(
    monkeypatch, views, room, rows
):
    monkeypatch.setattr(_rows, "host_available_memory_bytes", lambda: room)
    assert _rows._block_rows((views, 2160, 2560), 2) == rows


class Lazy:
    """Detector windows of a stack, read only as whole windows, counting the reads."""

    def __init__(self, data):
        self.data, self.shape, self.dtype = data, data.shape, data.dtype
        self.reads = []

    def window(self, rows, cols):
        parent = self

        class Window:
            def __getitem__(self, key):
                assert key == slice(None)
                parent.reads.append((rows.start, rows.stop))
                return parent.data[:, rows, cols].copy()

        return Window()


def test_rows_across_raw_blocks_match_the_stack_in_memory(monkeypatch):
    # Two-row blocks: a three-row request spans two of them.
    monkeypatch.setattr(_rows, "_BLOCK_BYTES", 2 * 6 * 5 * 2)
    monkeypatch.setattr(_rows, "_RUN_BYTES", 1)
    rng = np.random.default_rng(4)
    raw = rng.integers(300, 900, (6, 9, 5)).astype(">u2")
    settings = dict(
        flats=None, flat_positions=None, darks=None, white_level=1000.0, steps=(), epsilon=1e-6
    )
    lazy = Lazy(raw)
    requests = [(0, 2), (1, 4), (3, 4), (4, 7), (2, 3), (7, 9)]
    with (
        corrected_row_source(lazy, **settings) as streamed,
        corrected_row_source(raw, **settings) as resident,
    ):
        for first, stop in requests:
            got, want = streamed(first, stop), resident(first, stop)
            np.testing.assert_array_equal(got.projections, want.projections)
            np.testing.assert_allclose(
                got.projections, -np.log(raw[:, first:stop] / 1000.0), rtol=1e-6
            )
    assert set(lazy.reads) <= {(0, 2), (2, 4), (4, 6), (6, 8), (8, 9)}
