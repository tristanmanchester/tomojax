"""What a file holds and a volume for other software: :func:`inspect` and :func:`export`."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

if TYPE_CHECKING:
    from os import PathLike

    from tomojax._scan import Reconstruction
    from tomojax.io.api import Inspection


def inspect(path: str | PathLike[str]) -> Inspection:
    """Describe a dataset or acquisition and check that it can be reconstructed.

    Reads metadata and a bounded sample of the frames, never the whole stack: the
    projections' shape, type and statistics, the angles, geometry, detector,
    flats and darks, the corrections recorded, saved poses, memory estimates and a
    saved volume. ``valid`` and ``issues`` say what the check found missing.

    Parameters
    ----------
    path : str or path-like
        HDF5/NXtomo dataset, saved reconstruction, or recognised raw acquisition.

    Returns
    -------
    Inspection
        The report, a mapping; ``tomojax inspect --json`` prints the same.

    Raises
    ------
    ValueError
        If the projections or acquisition metadata have invalid dimensions.
    OSError
        If the input cannot be opened.

    Examples
    --------
    >>> import tomojax as tj
    >>> report = tj.inspect("scan.nxs")  # doctest: +SKIP
    >>> report["valid"], report["angles"]["coverage"]  # doctest: +SKIP
    """
    from tomojax.io import validate_dataset
    from tomojax.io.api import inspect_dataset

    issues = validate_dataset(Path(path))["issues"]
    report = inspect_dataset(Path(path))
    return cast("Inspection", {**report, "valid": not issues, "issues": issues})


def export(
    path: str | PathLike[str],
    reconstruction: Reconstruction,
    *,
    dtype: Literal["float32", "uint16"] = "float32",
    range: tuple[float, float] | None = None,  # noqa: A002 the command's --range
    prefix: str = "slice",
) -> None:
    """Write a reconstruction for other software: TIFF z-slices, or one raw file.

    ``path`` ending in ``.raw`` writes one little-endian, z-major file; otherwise
    ``path`` is a directory of z-slice TIFFs (y rows, x columns), numbered from
    the bottom of the volume. A JSON sidecar gives the shape, voxel size and value
    scaling. Each file is written beside its name and moved onto it when whole,
    so a failure leaves no partial file.

    Parameters
    ----------
    path : str or path-like
        Directory for TIFF slices, or a ``.raw`` file.
    reconstruction : Reconstruction
        The volume and its grid.
    dtype : {"float32", "uint16"}, optional
        Sample type; ``uint16`` maps ``range`` to 0..65535. Default is float32.
    range : tuple of float or None, optional
        The values ``uint16`` maps to 0 and 65535; None takes the 0.1 and 99.9
        percentiles of up to 16 slices.
    prefix : str, optional
        TIFF file name prefix. Default is "slice".

    Examples
    --------
    >>> import tomojax as tj
    >>> tj.export("slices", tj.load_reconstruction("recon.nxs"))  # doctest: +SKIP
    """
    volume = reconstruction.volume
    grid = reconstruction.grid
    write_volume(
        Path(path),
        lambda z: np.asarray(volume[z], np.float32),
        (int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])),
        (grid.vx, grid.vy, grid.vz),
        dtype=dtype,
        value_range=range,
        prefix=prefix,
    )


def write_volume(
    out: Path,
    plane: Callable[[int], np.ndarray],
    shape: tuple[int, int, int],
    voxel: tuple[float, float, float],
    *,
    dtype: str,
    value_range: Sequence[float] | None,
    prefix: str,
    source: str | None = None,
) -> Path:
    """Write ``shape`` (z, y, x) z-planes, each ``plane(z)``, as TIFFs or one raw file.

    Returns the sidecar's path.
    """
    import imageio.v3 as iio

    from tomojax.core.files import replaced_when_complete

    nz = shape[0]
    raw = out.suffix.lower() == ".raw"
    low, high = (0.0, 1.0)
    if dtype == "uint16":
        low, high = _value_range(plane, nz) if value_range is None else map(float, value_range)

    def convert(values: np.ndarray) -> np.ndarray:
        if dtype == "float32":
            return values
        scaled = (values - low) / (high - low) * 65535.0
        return np.clip(np.round(scaled), 0, 65535).astype(np.uint16)

    parts = f"volume {np.prod(shape) * np.dtype(dtype).itemsize / 2**30:.1f} GiB"
    if raw:
        little = np.dtype(dtype).newbyteorder("<")
        with replaced_when_complete(out, parts) as temporary, temporary.open("wb") as stream:
            for z in range(nz):
                _ = stream.write(convert(plane(z)).astype(little).tobytes())
        sidecar = out.with_suffix(".json")
    else:
        out.mkdir(parents=True, exist_ok=True)
        for z in range(nz):
            target = out / f"{prefix}_{z:05d}.tif"
            with replaced_when_complete(target, f"slice {z}") as temporary:
                iio.imwrite(temporary, convert(plane(z)), extension=".tif")
        sidecar = out / f"{prefix}.json"
    info: dict[str, object] = {
        "source": source,
        "format": "raw" if raw else "tiff",
        "dtype": dtype,
        "shape_zyx": list(shape),
        "voxel_size_xyz": list(voxel),
        "byte_order": "little" if raw else None,
        "value_range": [low, high] if dtype == "uint16" else None,
        "slice_axis": "z, from the bottom of the volume",
    }
    with replaced_when_complete(sidecar, "sidecar") as temporary:
        _ = temporary.write_text(json.dumps(info, indent=2) + "\n")
    return sidecar


def _value_range(plane: Callable[[int], np.ndarray], nz: int) -> tuple[float, float]:
    """The 0.1 and 99.9 percentiles of up to 16 evenly spaced slices."""
    count = min(nz, 16)
    picks = sorted({round(i * (nz - 1) / max(1, count - 1)) for i in range(count)})
    values = np.concatenate([plane(z).ravel() for z in picks])
    low, high = (float(v) for v in np.percentile(values[np.isfinite(values)], [0.1, 99.9]))
    return low, high if high > low else low + 1.0


__all__ = ["export", "inspect", "write_volume"]
