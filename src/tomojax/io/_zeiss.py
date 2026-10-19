"""Zeiss Xradia OLE images, read lazily without keeping containers open."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import olefile

from tomojax._data import FrameStack
from tomojax.geometry import ConeBeam, Detector, Grid

from ._datasets import ProjectionDataset

if TYPE_CHECKING:
    from ._acquisition import RawAcquisition


@dataclass(frozen=True)
class _ZeissFrames(FrameStack):
    """A logical window of OLE image streams, in their recorded row order.

    Each indexing request opens its own container; neither discovery nor
    cropping reads image pixels. Reads touch only the requested image rows.
    """

    path: str
    streams: tuple[str, ...]
    image_shape: tuple[int, int]
    starts: tuple[int, ...]
    sector_size: int
    fat: np.ndarray | None = field(repr=False, compare=False)
    exposures: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def exposure_times(self) -> np.ndarray | None:
        """Recorded durations for this selection, in seconds."""
        return None if self.exposures is None else self.exposures[self.frames]

    def _read(self, indices: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
        result = np.empty((len(indices), self.rows, self.cols), self.dtype)
        source: Any
        with (
            olefile.OleFileIO(self.path) if self.fat is None else Path(self.path).open("rb")
        ) as source:
            row_bytes = self.image_shape[1] * self.dtype.itemsize
            for destination, index in enumerate(indices):
                offset, size = rows.start * row_bytes, self.rows * row_bytes
                if self.fat is None:
                    stream = source.openstream(self.streams[index])
                    stream.seek(offset)
                    raw = stream.read(size)
                else:
                    raw = _stream_range(
                        source, self.fat, self.sector_size, self.starts[index], offset, size
                    )
                frame = np.frombuffer(raw, dtype=self.dtype).reshape(self.rows, self.image_shape[1])
                result[destination] = frame[:, cols]
                del frame, raw
        return result


def _stream_entry(container: Any, name: str) -> Any:
    entry = container.root
    for part in name.split("/"):
        entry = next(child for child in entry.kids if child.name.casefold() == part.casefold())
    return entry


def _stream_range(
    file: Any, fat: np.ndarray, sector_size: int, sector: int, offset: int, size: int
) -> bytearray:
    """Read a logical byte range, coalescing adjacent physical FAT sectors.

    OleFileIO.openstream materializes an entire stream. Its directory and FAT
    locate the image bytes without decoding the rows outside a requested slab.
    The retained FAT is metadata only; each batch opens just the backing file.
    """
    first, head = divmod(offset, sector_size)
    for _ in range(first):
        sector = int(fat[sector])
    sectors = np.empty((head + size + sector_size - 1) // sector_size, np.int64)
    for index in range(len(sectors)):
        sectors[index] = sector
        sector = int(fat[sector])
    boundaries = np.r_[0, np.flatnonzero(np.diff(sectors) != 1) + 1, len(sectors)]
    result = bytearray(size)
    target = memoryview(result)
    for start, stop in pairwise(boundaries):
        begin = max(0, int(start) * sector_size - head)
        end = min(size, int(stop) * sector_size - head)
        physical = (int(sectors[start]) + 1) * sector_size + (head if start == 0 else 0)
        file.seek(physical)
        target[begin:end] = file.read(end - begin)
    return result


def _stack(
    container: Any,
    path: Path,
    streams: tuple[str, ...],
    shape: tuple[int, int],
    data_type: int,
    *,
    exposures: np.ndarray | None = None,
) -> _ZeissFrames:
    # Xradia's codes are not NumPy's dtype numbers. Both representations are
    # little-endian, including the independently typed reference image.
    types = {5: np.dtype("<u2"), 10: np.dtype("<f4")}
    if data_type not in types:
        raise ValueError(f"unsupported Zeiss image data type {data_type}")
    entries = [_stream_entry(container, name) for name in streams]
    fat = None if entries[0].size < container.minisectorcutoff else np.asarray(container.fat)
    return _ZeissFrames(
        str(path.absolute()),
        streams,
        shape,
        tuple(int(entry.isectStart) for entry in entries),
        int(container.sectorsize),
        fat,
        exposures,
        frames=np.arange(len(streams)),
        dtype=types[data_type],
        rows=shape[0],
        cols=shape[1],
    )


def _values(container: Any, name: str, dtype: str) -> np.ndarray:
    return np.frombuffer(container.openstream(name).read(), dtype=np.dtype(dtype))


def _scalar(container: Any, name: str, dtype: str = "<u4") -> float:
    return float(_values(container, name, dtype).item())


def _is_ole(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return file.read(8) == olefile.MAGIC


def _images(container: Any, path: Path) -> _ZeissFrames:
    count = int(_scalar(container, "ImageInfo/NoOfImages"))
    shape = (
        int(_scalar(container, "ImageInfo/ImageHeight")),
        int(_scalar(container, "ImageInfo/ImageWidth")),
    )
    dtype = int(_scalar(container, "ImageInfo/DataType"))
    streams = tuple(f"ImageData{i // 100 + 1}/Image{i + 1}" for i in range(count))
    return _stack(
        container,
        path,
        streams,
        shape,
        dtype,
        exposures=_exposures(container, "ImageInfo", count),
    )


def _exposures(container: Any, prefix: str, count: int) -> np.ndarray | None:
    for name in (f"{prefix}/ExpTimes", f"{prefix}/ExpTime", f"{prefix.rsplit('/', 1)[0]}/ExpTime"):
        if container.exists(name):
            return np.broadcast_to(_values(container, name, "<f4"), (count,)).copy()
    return None


def _dates(container: Any, name: str) -> np.ndarray:
    """Xradia date records are forty-byte local timestamps, including padding."""
    result = []
    for raw in _values(container, name, "S40"):
        value = bytes(raw).split(b"\0", 1)[0].decode("ascii")
        fmt = "%m/%d/%Y %H:%M:%S.%f" if "." in value else "%m/%d/%Y %H:%M:%S"
        result.append((datetime.strptime(value, fmt) - datetime(1970, 1, 1)).total_seconds())
    return np.asarray(result, np.float64)


def _references(
    container: Any, path: Path, counts: _ZeissFrames
) -> tuple[_ZeissFrames | None, np.ndarray | None, np.ndarray | None]:
    """Read flat counts with their durations and acquisition timestamps."""
    prefix = "MultiReferenceData"
    required = (
        f"{prefix}/Image1",
        f"{prefix}/DataType",
        f"{prefix}/TotalRefImages",
        f"{prefix}/ImageInfo/Date",
        "ImageInfo/Date",
    )
    if all(container.exists(name) for name in required):
        number = int(_scalar(container, f"{prefix}/TotalRefImages"))
        exposure = _exposures(container, f"{prefix}/ImageInfo", number)
        if exposure is not None and counts.exposure_times is not None:
            flat_time = _dates(container, f"{prefix}/ImageInfo/Date")
            view_time = _dates(container, "ImageInfo/Date")
            if flat_time.shape != (number,) or view_time.shape != (len(counts),):
                raise ValueError(
                    "Zeiss reference timestamps need one entry per image; pass flats=..."
                )
            flats = _stack(
                container,
                path,
                tuple(f"{prefix}/Image{i + 1}" for i in range(number)),
                counts.image_shape,
                int(_scalar(container, f"{prefix}/DataType")),
                exposures=exposure,
            )
            return flats, flat_time - view_time[0], view_time - view_time[0]
    if container.exists("ReferenceData/Image") and container.exists("ReferenceData/DataType"):
        flats = _stack(
            container,
            path,
            ("ReferenceData/Image",),
            counts.image_shape,
            int(_scalar(container, "ReferenceData/DataType")),
            exposures=_exposures(container, "ReferenceData/ImageInfo", 1),
        )
        return flats, None, None
    return None, None, None


def read_zeiss_images(path: Path) -> _ZeissFrames | None:
    """Read an explicitly selected calibration stack without projection geometry.

    Single XRM reference images use image/scout acquisition modes. Their image
    dimensions and type are sufficient for calibration, unlike sample scans.
    """
    if not _is_ole(path):
        return None
    with olefile.OleFileIO(str(path)) as container:
        if not container.exists("ImageInfo/NoOfImages"):
            return None
        return _images(container, path)


def _offsets(container: Any, count: int) -> np.ndarray | None:
    """Calibrated image offsets, in recorded detector pixels (not stage motion).

    ``AutoRecon/CenterShift`` subtracts from the horizontal image shifts. Its
    sign and recorded-pixel convention were checked against the supplied Zeiss
    reconstruction; acquisition binning is already in the pixel calibration.
    """
    names = ("alignment/x-shifts", "alignment/y-shifts", "AutoRecon/CenterShift")
    if not all(container.exists(name) for name in names):
        return None
    result = np.column_stack([_values(container, name, "<f4") for name in names[:2]])
    result = np.broadcast_to(result, (count, 2)).astype(np.float64).copy()
    result[:, 0] -= _scalar(container, names[2], "<f4")
    return result


def _cone_metadata(
    source: np.ndarray, detector_distance: np.ndarray, shifts: np.ndarray, detector: Detector
) -> dict[str, Any]:
    """Preserve per-view beam/detector calibration without resampling the counts.

    A detector-image translation is not a rigid object translation in cone
    geometry. Use actual detector centres, grouping only identical consecutive
    arrangements; don't approximate a pixel shift by object-plane motion.
    """
    arrangements = np.column_stack((source, source + detector_distance, shifts))
    boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(arrangements, axis=0), axis=1)) + 1]
    stops = np.r_[boundaries[1:], len(arrangements)]
    entries = []
    for first, stop in zip(boundaries, stops, strict=True):
        sod, sdd, u, v = arrangements[first]
        entry = {
            "views": int(stop - first),
            "cone_beam": ConeBeam(float(sod), float(sdd)).to_dict(),
            "detector": replace(
                detector, center=(float(u * detector.du), float(v * detector.dv))
            ).to_dict(),
        }
        entries.append(entry)
    return {"cone_segments": entries} if len(entries) > 1 else entries[0]


def _record(container: Any, path: Path, shape: tuple[int, int, int]) -> ProjectionDataset:
    """Keep incomplete calibration separate from the image storage contract."""
    count, rows, cols = shape
    pixel = (
        _scalar(container, "ImageInfo/pixelsize", "<f4") / 1000
        if container.exists("ImageInfo/pixelsize")
        else None
    )
    angles = (
        _values(container, "ImageInfo/Angles", "<f4").copy()
        if container.exists("ImageInfo/Angles")
        else np.empty(0, np.float32)
    )
    shifts = _offsets(container, count)
    cone = (
        _scalar(container, "ImageInfo/XrayGeometry") == 1
        if container.exists("ImageInfo/XrayGeometry")
        else None
    )
    metadata: dict[str, Any] = {}
    poses = None
    pitch = pixel
    source = distance = None
    if cone:
        if all(
            container.exists(name)
            for name in ("ImageInfo/StoRADistance", "ImageInfo/DtoRADistance")
        ):
            source = np.abs(_values(container, "ImageInfo/StoRADistance", "<f4")).astype(np.float64)
            distance = np.abs(_values(container, "ImageInfo/DtoRADistance", "<f4")).astype(
                np.float64
            )
            pitch = None if pixel is None else pixel * (source[0] + distance[0]) / source[0]
        else:
            pitch = None
    detector = None
    if pitch is not None and shifts is not None and cone is not None:
        detector = Detector(cols, rows, pitch, pitch)
        if cone and source is not None and distance is not None:
            metadata.update(_cone_metadata(source, distance, shifts, detector))
            if "detector" in metadata:
                detector = Detector.from_dict(metadata.pop("detector"))
                metadata.pop("views")
        elif np.any(shifts != shifts[0]):
            poses = np.zeros((count, 6), np.float32)
            poses[:, 3:5] = -pitch * shifts
        else:
            detector = replace(detector, center=tuple(float(x) for x in pitch * shifts[0]))
    metadata["detector_center_status"] = "recorded" if detector is not None else "not_recorded"
    if detector is None:
        # An individual detector override cannot resolve unknown per-view shifts.
        metadata["axis_status"] = "not_recorded"
    metadata["zeiss"] = {
        "path": str(path),
        "detector_offset_pixels": None if shifts is None else shifts.tolist(),
    }
    return ProjectionDataset(
        projections=np.broadcast_to(np.float32(0), (count, rows, cols)),
        angles=angles,
        detector=detector,
        grid=None if pixel is None else Grid(cols, cols, rows, pixel, pixel, pixel),
        geometry_type="cone" if cone else "parallel",
        geometry_metadata=metadata,
        align_params=poses,
        align_gauge=None if poses is None else {"pose_translation_frame": "detector"},
        source_path=str(path),
        source_format="zeiss_ole",
        sample_name=path.stem,
    )


def read_zeiss(path: Path, *, data_path: str | None = None) -> RawAcquisition | None:
    """Read an Xradia acquisition by its OLE content, regardless of its filename.

    Xradia stores angles in degrees, distances in mm, and ``pixelsize`` as
    object-plane micrometres. Counts and the independently typed embedded
    reference remain lazy. Reconstructed TXM volumes are not projection scans.
    """
    from ._acquisition import RawAcquisition

    if not _is_ole(path):
        return None
    with olefile.OleFileIO(str(path)) as container:
        if not container.exists("ImageInfo/NoOfImages"):
            return None
        if data_path is not None:
            raise ValueError("a Zeiss container holds one acquisition, not HDF5 data paths")
        mode = int(_scalar(container, "ImageInfo/AcquisitionMode"))
        if mode != 0:
            raise ValueError(f"{path} is a Zeiss image/scout (mode {mode}), not detector frames")
        counts = _images(container, path)
        record = _record(container, path, counts.shape)
        flats, flat_positions, view_positions = _references(container, path, counts)
    return RawAcquisition(
        record,
        counts,
        flats=flats,
        flat_positions=flat_positions,
        view_positions=view_positions,
        exposure_times=counts.exposure_times,
        flat_exposure_times=None if flats is None else flats.exposure_times,
    )
