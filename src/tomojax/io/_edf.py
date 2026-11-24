"""Uncompressed EDF detector frames, with file-owned byte order and image dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tomojax._data import FrameStack

_TYPES = {
    "UnsignedByte": "u1",
    "SignedByte": "i1",
    "UnsignedShort": "u2",
    "SignedShort": "i2",
    "UnsignedInteger": "u4",
    "SignedInteger": "i4",
    "FloatValue": "f4",
    "DoubleValue": "f8",
    # PyHST exports use the shorter floating-point type names.
    "Float": "f4",
    "Double": "f8",
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
class EdfFrames(FrameStack):
    """A lazy stack of single-image EDF files; selections hold no open handles."""

    files: tuple[Path, ...]
    offsets: tuple[int, ...]
    page_shape: tuple[int, int]
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
        return cls(
            paths,
            tuple(offsets),
            shape,
            frames=np.arange(len(paths)),
            dtype=dtype,
            rows=shape[0],
            cols=shape[1],
        )

    def _read(self, indices: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
        result = np.empty((len(indices), self.rows, self.cols), self.dtype)
        for destination, index in enumerate(indices):
            pixels = np.memmap(
                self.files[index],
                mode="r",
                dtype=self.dtype,
                offset=self.offsets[index],
                shape=self.page_shape,
            )
            result[destination] = pixels[rows, cols]
        return result
