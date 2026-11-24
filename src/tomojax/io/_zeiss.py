"""Zeiss Xradia OLE images, read lazily without keeping containers open."""

from __future__ import annotations

import array
from dataclasses import dataclass, field
from datetime import datetime
from itertools import pairwise
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import olefile

from tomojax._data import FrameStack
from tomojax.core.files import read_path
from tomojax.geometry import Detector

from ._datasets import ProjectionDataset
from ._view_offsets import DISTANCES_KEY, OFFSETS_KEY, with_view_offsets

if TYPE_CHECKING:
    from ._acquisition import RawAcquisition


class _OleFile(olefile.OleFileIO):
    """Build large allocation tables without copying their accumulated prefix."""

    fat: array.array[int]

    def loadfat_sect(self, sect: Any) -> int | None:
        # Keep olefile's sector decoding, reads and enclosing DIFAT validation.
        # Only table accumulation differs: extend is linear in the new block.
        sectors = sect if isinstance(sect, array.array) else self.sect2array(sect)
        last = None
        for value in sectors:
            last = int(value) & 0xFFFFFFFF
            if last in (olefile.ENDOFCHAIN, olefile.FREESECT):
                break
            self.fat.extend(self.sect2array(self.getsect(last)))
        return last


@dataclass(frozen=True)
class _ZeissFrames(FrameStack):
    """A logical window of OLE image streams, in their recorded row order.

    Xradia stores each image's columns in the opposite sense to detector u, so a
    read reverses them: with the recorded column order a reconstruction is the
    mirror image of the sample (see :func:`_record`). Each indexing request opens
    its own file; neither discovery nor cropping reads image pixels. Reads touch
    only the requested image rows.
    """

    path: str
    streams: tuple[str, ...]
    image_shape: tuple[int, int]
    starts: tuple[int, ...]
    sector_size: int
    fat: np.ndarray | None = field(repr=False, compare=False)
    exposures: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def read_width(self) -> int:
        """Number of source columns decoded for each requested detector row."""
        return self.image_shape[1]

    @property
    def exposure_times(self) -> np.ndarray | None:
        """Recorded durations for this selection, in seconds."""
        return None if self.exposures is None else self.exposures[self.frames]

    def _read(self, indices: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
        result = np.empty((len(indices), self.rows, self.cols), self.dtype)
        source: Any
        path = read_path(self.path)
        with _OleFile(path) if self.fat is None else Path(path).open("rb") as source:
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
                result[destination] = frame[:, ::-1][:, cols]
                del frame, raw
        return result


def _stream_entry(container: Any, name: str, *, path: Path) -> Any:
    entry = container.root
    for part in name.split("/"):
        try:
            entry = next(child for child in entry.kids if child.name.casefold() == part.casefold())
        except StopIteration as exc:
            raise ValueError(
                f"{path}: missing declared Zeiss image stream {name!r}; "
                "supply the complete acquisition file"
            ) from exc
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
    entries = [_stream_entry(container, name, path=path) for name in streams]
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


_DARK = "AcquisitionSettings/CameraStaticBackground"
_LINEAR = "ConfigureBackup/ConfigCamera/Camera 1/MaxLinearRange"


def _scalar(container: Any, name: str, dtype: str = "<u4") -> float:
    return float(_values(container, name, dtype).item())


def _text(container: Any, name: str) -> str:
    return container.openstream(name).read().split(b"\0", 1)[0].decode("latin-1").strip()


def _beam_hardening(container: Any, path: Path) -> dict[str, object] | None:
    """The reconstructor's beam hardening constant and the tube settings of a scan.

    XMReconstructor stores its settings in the reconstruction, so the
    ``<name>_recon.txm`` beside an acquisition is read when the acquisition has none.
    """
    record: dict[str, object] = {"vendor": "zeiss"}
    if container.exists("ImageInfo/XrayVoltage"):  # float32 for each view in every version
        voltage = _values(container, "ImageInfo/XrayVoltage", "<f4")
        record["kv"] = round(float(np.median(voltage)), 2)
    if container.exists("ImageInfo/SourceFilterName"):
        record["filter"] = _text(container, "ImageInfo/SourceFilterName")
    recon = path.with_name(f"{path.stem}_recon.txm")
    sources = [container]
    if not container.exists("ReconSettings/BeamHardening") and _is_ole(recon):
        sources.append(_OleFile(str(recon)))
    try:
        for source in sources:
            if source.exists("ReconSettings/BeamHardening"):
                constant = _scalar(source, "ReconSettings/BeamHardening", "<f4")
                record["constant"] = round(constant, 6)
                if source.exists("ReconSettings/BeamHardeningFileName"):
                    record["configuration"] = _text(source, "ReconSettings/BeamHardeningFileName")
                break
    finally:
        for source in sources[1:]:
            source.close()
    return record if len(record) > 1 else None


def _is_ole(path: Path) -> bool:
    if not path.is_file():
        return False
    with read_path(path).open("rb") as file:
        return file.read(8) == olefile.MAGIC


def _images(container: Any, path: Path) -> _ZeissFrames:
    # NoOfImages can be the planned count of an interrupted acquisition.
    # The numbered streams establish which images were actually recorded;
    # per-view metadata must still agree with their count, without truncation.
    indices = [
        int(match[1])
        for parts in container.listdir()
        if (match := re.fullmatch(r"ImageData\d+/Image(\d+)", "/".join(parts), re.IGNORECASE))
    ]
    if not indices:
        raise ValueError(f"{path}: no Zeiss image streams; supply the projection acquisition file")
    count = max(indices)
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
            values = _values(container, name, "<f4")
            try:
                return np.broadcast_to(values, (count,)).copy()
            except ValueError as exc:
                raise ValueError(
                    f"Zeiss {name} has {len(values)} exposure values for {count} stored images; "
                    "supply an acquisition with matching image and exposure records"
                ) from exc
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
    with _OleFile(str(path)) as container:
        if not container.exists("ImageInfo/NoOfImages"):
            return None
        return _images(container, path)


def _offsets(container: Any, count: int) -> tuple[np.ndarray | None, float | None]:
    """Recorded per-view image shifts in detector pixels, and the centre shift if calibrated.

    ``AutoRecon/CenterShift`` adds to the horizontal image shifts. Its sign was
    checked against the vendor's own reconstruction of a scan that records it:
    the other sign doubles every small feature into a ring. The horizontal shifts
    are returned along detector u, which runs against the stored columns.
    Acquisition binning is already in the pixel calibration.
    Only a reconstruction setup writes it, so a raw scan may lack it. An interrupted
    acquisition records its shift tables at the planned length (``NoOfImages``), the
    stored views first, while its angles and exposures cover the stored views only.
    """
    names = ("alignment/x-shifts", "alignment/y-shifts")
    if not all(container.exists(name) for name in names):
        return None, None
    planned = int(_scalar(container, "ImageInfo/NoOfImages"))
    columns = []
    for name in names:
        values = _values(container, name, "<f4")
        if len(values) == planned > count:  # the plan of an interrupted acquisition
            values = values[:count]
        if len(values) not in (1, count):
            raise ValueError(
                f"Zeiss {name} has {len(values)} {name.split('/')[1]} for {count} stored images "
                f"({planned} declared); supply an acquisition with matching shift records"
            )
        columns.append(np.broadcast_to(values, (count,)))
    shifts = np.column_stack(columns).astype(np.float64) * (-1.0, 1.0)
    center = "AutoRecon/CenterShift"
    return shifts, _scalar(container, center, "<f4") if container.exists(center) else None


def _record(container: Any, path: Path, shape: tuple[int, int, int]) -> ProjectionDataset:
    """Keep incomplete calibration separate from the image storage contract.

    The orientation is the vendor's: with the columns reversed and a view's angle
    taken as 180 degrees less the recorded one, a reconstruction matches Xradia's
    own volume of the same scan voxel for voxel, and a snail shell comes out with
    the handedness of the shell itself. The recorded columns and angles as they
    stand give its mirror image.
    """
    count, rows, cols = shape
    pixel = (
        _scalar(container, "ImageInfo/pixelsize", "<f4") / 1000
        if container.exists("ImageInfo/pixelsize")
        else None
    )
    angles = (
        (180.0 - _values(container, "ImageInfo/Angles", "<f4")).astype(np.float32)
        if container.exists("ImageInfo/Angles")
        else np.empty(0, np.float32)
    )
    shifts, center_shift = _offsets(container, count)
    cone = (
        _scalar(container, "ImageInfo/XrayGeometry") == 1
        if container.exists("ImageInfo/XrayGeometry")
        else None
    )
    metadata: dict[str, Any] = {}
    pitch = pixel
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
            arrangement = np.column_stack((source, source + distance))
            metadata[DISTANCES_KEY] = np.broadcast_to(arrangement, (count, 2)).tolist()
        else:
            pitch = None
    if shifts is not None:
        metadata[OFFSETS_KEY] = shifts.tolist()
    fields = [
        name
        for name, value in (
            ("alignment/x-shifts", shifts),
            ("alignment/y-shifts", shifts),
            ("AutoRecon/CenterShift", center_shift),
        )
        if value is not None
    ]
    if (
        fields
    ):  # the vendor's own corrections, applied with the poses (tj.load apply_poses=False omits)
        metadata["vendor_corrections"] = {"source": "zeiss", "fields": fields, "applied": True}
    if pixel is not None:  # stated by the file; the grid's shape comes from the detector
        metadata["grid_status"] = "derived"
        metadata["voxel_size"] = [pixel, pixel, pixel]
    metadata["zeiss"] = {
        "path": str(path),
        "declared_view_count": int(_scalar(container, "ImageInfo/NoOfImages")),
        "stored_view_count": count,
        "center_shift_pixels": center_shift,
    }
    hardening = _beam_hardening(container, path)
    if hardening is not None:
        metadata["beam_hardening"] = hardening
    record = ProjectionDataset(
        projections=np.broadcast_to(np.float32(0), (count, rows, cols)),
        angles=angles,
        grid=None,
        geometry_type="cone" if cone else "parallel",
        geometry_metadata=metadata,
        source_path=str(path),
        source_format="zeiss_ole",
        sample_name=path.stem,
    )
    if pitch is not None and shifts is not None and cone is not None:
        # The shifts are measured from the recorded centre; a raw scan with no
        # reconstruction setup records none, and its shifts are measured from the middle.
        center = (-pitch * (center_shift or 0.0), 0.0)
        record = with_view_offsets(record, Detector(cols, rows, pitch, pitch, center=center))
    known = record.detector is not None and center_shift is not None
    record.geometry_metadata["detector_center_status"] = "recorded" if known else "not_recorded"
    if (
        record.detector is None and pitch is not None
    ):  # the pitch without a whole detector, as partial records keep it
        record.geometry_metadata["detector_pixel_size"] = [pitch, pitch]
    if shifts is None or cone is None:
        # Without the shifts or the beam type a detector alone cannot fix the views;
        # with them, detector= supplies the centre the shifts are measured from.
        record.geometry_metadata["axis_status"] = "not_recorded"
    return record


def read_zeiss(path: Path, *, data_path: str | None = None) -> RawAcquisition | None:
    """Read an Xradia acquisition by its OLE content, regardless of its filename.

    Xradia stores angles in degrees, distances in mm, and ``pixelsize`` as
    object-plane micrometres. Counts and the independently typed embedded
    reference remain lazy. Reconstructed TXM volumes are not projection scans.
    """
    from ._acquisition import RawAcquisition

    if not _is_ole(path):
        return None
    with _OleFile(str(path)) as container:
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
        # The camera subtracted its static background (the dark) before writing the counts.
        removed = container.exists(_DARK) and _scalar(container, _DARK, "<i4") == 1
        dark_source = f"the instrument ({_DARK} = 1)" if removed else None
        linear = container.exists(_LINEAR)
        linear_limit = (float(_scalar(container, _LINEAR, "<i4")), _LINEAR) if linear else None
    return RawAcquisition(
        record,
        counts,
        flats=flats,
        flat_positions=flat_positions,
        view_positions=view_positions,
        exposure_times=counts.exposure_times,
        flat_exposure_times=None if flats is None else flats.exposure_times,
        darks=None
        if dark_source is None
        else np.broadcast_to(np.float32(0), (1, *counts.image_shape)),
        dark_source=dark_source,
        linear_limit=linear_limit,
    )
