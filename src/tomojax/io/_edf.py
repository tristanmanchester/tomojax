"""Uncompressed EDF detector frames, with file-owned byte order and image dimensions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

_TYPES = {
    "UnsignedByte": "u1",
    "SignedByte": "i1",
    "UnsignedShort": "u2",
    "SignedShort": "i2",
    "UnsignedInteger": "u4",
    "SignedInteger": "i4",
    "FloatValue": "f4",
    "DoubleValue": "f8",
    "UnsignedLong": "u4",
    "SignedLong": "i4",
    "Unsigned64": "u8",
    "Signed64": "i8",
}


def edf_header(path: Path) -> tuple[dict[str, str], int]:
    """Read an EDF ASCII header; no binary image data is decoded."""
    with path.open("rb") as stream:
        header = stream.read(512)
        if not header.lstrip().startswith(b"{"):
            raise ValueError(f"{path} is not an EDF image")
        while b"}" not in header:
            block = stream.read(512)
            if not block:
                raise ValueError(f"{path} has an incomplete EDF header")
            header += block
    end = header.index(b"}") + 1
    fields = {
        key.strip(): value.strip()
        for item in header[:end].decode("latin1").strip().strip("{}").split(";")
        if "=" in item
        for key, value in (item.split("=", 1),)
    }
    return fields, int(fields.get("EDF_HeaderSize", ((end + 511) // 512) * 512))


@dataclass(frozen=True)
class EdfFrames:
    """A lazy stack of single-image EDF files; selections hold no open handles."""

    files: tuple[Path, ...]
    offsets: tuple[int, ...]
    page_shape: tuple[int, int]
    dtype: np.dtype
    rows: int
    cols: int
    first_row: int = 0
    first_col: int = 0
    _efficient_row_windows = True

    @classmethod
    def open(cls, files: list[Path]) -> EdfFrames:
        paths = tuple(file.absolute() for file in files)
        layouts, offsets = [], []
        for file in paths:
            fields, offset = edf_header(file)
            if fields.get("Compression", "None").lower() not in {"none", "no", ""}:
                raise ValueError(f"{file}: compressed EDF images are not supported")
            dtype = np.dtype(
                {"LowByteFirst": "<", "HighByteFirst": ">"}[fields["ByteOrder"]]
                + _TYPES[fields["DataType"]]
            )
            if int(fields.get("Dim_3", 1)) != 1:
                raise ValueError(f"{file}: expected a two-dimensional EDF detector image")
            shape = (int(fields["Dim_2"]), int(fields["Dim_1"]))
            if file.stat().st_size < offset + int(np.prod(shape)) * dtype.itemsize:
                raise ValueError(f"{file}: EDF pixel data is incomplete")
            layouts.append((shape, dtype))
            offsets.append(offset)
        if not layouts or len(set(layouts)) != 1:
            raise ValueError("EDF frames must have one image size and data type")
        shape, dtype = layouts[0]
        return cls(paths, tuple(offsets), shape, dtype, *shape)

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.files), self.rows, self.cols

    @property
    def ndim(self) -> int:
        return 3

    def __len__(self) -> int:
        return len(self.files)

    def frames_at(self, which: np.ndarray | slice) -> EdfFrames:
        selected = np.arange(len(self))[which]
        return replace(
            self,
            files=tuple(self.files[i] for i in selected),
            offsets=tuple(self.offsets[i] for i in selected),
        )

    def window(self, rows: slice, cols: slice) -> EdfFrames:
        r0, r1, rs = rows.indices(self.rows)
        c0, c1, cs = cols.indices(self.cols)
        if rs != 1 or cs != 1 or r1 <= r0 or c1 <= c0:
            raise ValueError("a window of frames is a non-empty block of rows and columns")
        return replace(
            self,
            rows=r1 - r0,
            cols=c1 - c0,
            first_row=self.first_row + r0,
            first_col=self.first_col + c0,
        )

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        if copy is False:
            raise ValueError("EDF frames cannot be converted to an array without copying")
        return np.asarray(self[:], dtype=dtype)

    def __getitem__(self, which: slice | int | np.ndarray) -> np.ndarray:
        if isinstance(which, int | np.integer):
            return self[np.asarray([which], np.int64)][0]
        selected = np.arange(len(self))[which]
        result = np.empty((len(selected), self.rows, self.cols), self.dtype)
        for destination, index in enumerate(selected):
            pixels = np.memmap(
                self.files[index],
                mode="r",
                dtype=self.dtype,
                offset=self.offsets[index],
                shape=self.page_shape,
            )
            result[destination] = pixels[
                self.first_row : self.first_row + self.rows,
                self.first_col : self.first_col + self.cols,
            ]
        return result
