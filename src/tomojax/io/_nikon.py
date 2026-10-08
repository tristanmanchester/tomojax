"""Read Nikon (X-Tek) lab CT scans: an ``.xtekct`` parameter file and TIFF projections."""
# pyright: reportUnknownMemberType=false, reportAny=false

from __future__ import annotations

import configparser
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import imageio.v3 as iio
import numpy as np

from tomojax.core.geometry.base import Detector, Grid
from tomojax.core.geometry.cone import ConeBeam

from ._datasets import ProjectionDataset
from ._tiff import tiff_files

if TYPE_CHECKING:
    from os import PathLike

LOG = logging.getLogger(__name__)


def _parameters(path: Path) -> dict[str, str]:
    parser = configparser.ConfigParser(strict=False, interpolation=None)
    parser.optionxform = str  # pyright: ignore[reportAttributeAccessIssue]
    text = path.read_text(encoding="utf-8", errors="replace")
    parser.read_string(text if text.lstrip().startswith("[") else "[XTekCT]\n" + text)
    section = next((s for s in parser.sections() if s.lower() == "xtekct"), None)
    if section is None:
        raise ValueError(f"{path} has no [XTekCT] section")
    return dict(parser[section])


def _number(params: dict[str, str], key: str, default: float | None = None) -> float:
    value = params.get(key)
    if value is None or not value.strip():
        if default is None:
            raise ValueError(f"the .xtekct file has no {key}")
        return float(default)
    return float(value)


def _read(file: Path) -> np.ndarray:
    return np.asarray(cast("object", iio.imread(file)), np.float32)


def _angles(directory: Path, params: dict[str, str], views: int) -> np.ndarray:
    """Angles from ``_ctdata.txt`` (measured stage angles), else the nominal steps."""
    for ctdata in sorted(directory.glob("*_ctdata.txt")):
        rows: list[tuple[int, float]] = []
        for line in ctdata.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.replace(",", " ").split()
            try:
                rows.append((int(parts[0]), float(parts[1])))
            except (IndexError, ValueError):
                continue
        if len(rows) == views:
            return np.asarray([angle for _, angle in sorted(rows)], np.float64)
        LOG.warning("%s lists %d angles for %d projections; ignoring it", ctdata, len(rows), views)
    start = _number(params, "InitialAngle", 0.0)
    step = _number(params, "AngularStep", 360.0 / views)
    return start + step * np.arange(views)


def _projection_files(directory: Path, params: dict[str, str]) -> list[Path]:
    name = params.get("Name", "").strip()
    prefix = name + params.get("InputSeparator", "_") if name else ""
    files = [file for file in tiff_files(directory) if file.name.startswith(prefix)]
    if not files:
        raise ValueError(f"no projection TIFFs named {prefix}NNNN.tif in {directory}")
    return files


def load_nikon_xtekct(
    path: str | PathLike[str],
    *,
    absorption: bool = True,
    reverse_angles: bool = False,
) -> ProjectionDataset:
    """Load a Nikon scan as a cone-beam dataset.

    Reads source and detector distances, detector pixels and offsets, the
    reconstruction volume and the white level from ``path`` (an ``.xtekct``
    file), the projections ``<Name>_NNNN.tif`` beside it, and the angles from
    ``_ctdata.txt`` when present (else ``InitialAngle`` and ``AngularStep``).
    Projections become absorption ``-log(I / WhiteLevel)`` unless
    ``absorption`` is False, with image rows flipped so detector v points up.
    Lengths stay in the file's units (mm).

    The axis offset and detector roll are left at zero: calibrate them with
    :func:`tomojax.recon.calibrate_cone_axis` or ``tomojax align --mode cor``.
    ``reverse_angles`` negates the angles, for a stage turning the other way;
    check the handedness of the reconstruction on a known sample.
    """
    file = Path(path)
    params = _parameters(file)
    files = _projection_files(file.parent, params)
    first = _read(files[0])
    nv, nu = int(first.shape[0]), int(first.shape[1])
    projections = np.empty((len(files), nv, nu), np.float32)
    white = _number(params, "WhiteLevel", float(np.iinfo(np.uint16).max))
    for i, tiff in enumerate(files):
        image = _read(tiff) if i else first
        if image.shape != (nv, nu):
            raise ValueError(f"{tiff} is {image.shape}, expected {(nv, nu)}")
        # TIFF rows run top to bottom; detector v runs up.
        image = image[::-1]
        projections[i] = -np.log(np.maximum(image, 1.0) / white) if absorption else image
    expected = (
        int(_number(params, "DetectorPixelsY", nv)),
        int(_number(params, "DetectorPixelsX", nu)),
    )
    if expected != (nv, nu):
        LOG.warning("projections are %s pixels; the .xtekct file lists %s", (nv, nu), expected)
    angles = _angles(file.parent, params, len(files))
    if reverse_angles:
        angles = -angles
    detector = Detector(
        nu=nu,
        nv=nv,
        du=_number(params, "DetectorPixelSizeX"),
        dv=_number(params, "DetectorPixelSizeY"),
        center=(
            _number(params, "DetectorOffsetX", 0.0),
            _number(params, "DetectorOffsetY", 0.0),
        ),
    )
    beam = ConeBeam(_number(params, "SrcToObject"), _number(params, "SrcToDetector"))
    grid = None
    if all(key in params for key in ("VoxelsX", "VoxelsY", "VoxelsZ")):
        voxel = detector.du / beam.magnification
        grid = Grid(
            nx=int(_number(params, "VoxelsX")),
            ny=int(_number(params, "VoxelsY")),
            nz=int(_number(params, "VoxelsZ")),
            vx=_number(params, "VoxelSizeX", voxel),
            vy=_number(params, "VoxelSizeY", voxel),
            vz=_number(params, "VoxelSizeZ", voxel),
        )
    recorded = {
        key: params[key]
        for key in (
            "ObjectOffsetX",
            "CentreOfRotationTop",
            "CentreOfRotationBottom",
            "ObjectRoll",
            "ObjectTilt",
        )
        if key in params
    }
    metadata: dict[str, object] = {
        "cone_beam": beam.to_dict(),
        "ingest_source": "nikon_xtekct",
        "nikon_xtekct": {"path": str(file), "white_level": white, **recorded},
    }
    return ProjectionDataset(
        projections=projections,
        angles=angles.astype(np.float32),
        detector=detector,
        grid=grid,
        geometry_type="cone",
        geometry_metadata=cast("dict[str, object]", metadata),
        sample_name=params.get("Name") or None,
        source_path=str(file),
        source_format="nikon_xtekct",
    )


__all__ = ["load_nikon_xtekct"]
