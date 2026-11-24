"""``inspect --preview-dir``: a dataset's central projection, slices and centre sweep as PNGs.

A saved reconstruction's planes are read from the file, plane by plane. A scan's
planes are reconstructed: each plane on its own where axial FBP allows, otherwise
one full reconstruction cut into every requested plane. A centre sweep runs one
candidate at a time and keeps only small thumbnails for its contact sheet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from typing import TYPE_CHECKING, Literal, cast

import h5py
import numpy as np

from tomojax import progress
from tomojax.cli import _images
from tomojax.io.api import (
    locate_frames,
    read_acquisition,
    saved_volume_axes,
    saved_without_frames,
)
from tomojax.wire.arrays import Axis as Registered, Uniform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import Image

    import tomojax as tj

type Axis = Literal["z", "y", "x"]

_VOLUME_PATH = "/entry/processing/tomojax/volume"
_NUMBER = r"[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?"
_SPEC = re.compile(rf"^{_NUMBER}(:{_NUMBER}:{_NUMBER}|(,{_NUMBER})*)$")
_PIXELS = (Registered("row", "image", "px", Uniform(0, 1)),) * 2  # samples kept as they are
LISTING = "inspect-previews.json"


def parse_slice(text: str) -> tuple[Axis, int]:
    """``AXIS=INDEX`` as an axis and a plane index."""
    axis, sep, index = text.partition("=")
    if not sep or axis not in ("z", "y", "x") or not index.isdigit():
        raise ValueError(f"--slice takes AXIS=INDEX with AXIS z, y or x, not {text!r}")
    return cast("Axis", axis), int(index)


def parse_centres(text: str) -> tuple[float, ...]:
    """``START:STOP:STEP`` (STOP included when on the step grid) or a comma list."""
    if not is_centres_value(text):
        raise ValueError(
            f"--centres takes START:STOP:STEP or a comma list of numbers, not {text!r}"
        )
    if ":" not in text:
        return tuple(float(part) for part in text.split(","))
    start, stop, step = (float(part) for part in text.split(":"))
    if not step > 0 or stop < start:
        raise ValueError(f"--centres {text}: STEP must be positive and STOP at least START")
    count = math.floor((stop - start) / step + 1e-9) + 1
    return tuple(round(start + i * step, 12) for i in range(count))


def is_centres_value(text: str) -> bool:
    """Whether ``text`` is a ``--centres`` value, so a leading minus is not an option."""
    return _SPEC.match(text) is not None


@dataclass
class Previews:
    """What a preview run wrote, in the order it was asked for."""

    files: dict[str, str] = field(default_factory=dict)
    slice_requests: list[dict[str, object]] = field(default_factory=list)
    sweep: list[dict[str, object]] = field(default_factory=list)
    apply: dict[str, object] | None = None
    notes: list[str] = field(default_factory=list)


def has_saved_volume(path: Path) -> bool:
    """Whether ``path`` is a TomoJAX file holding a reconstructed volume."""
    if not h5py.is_hdf5(path):
        return False
    with h5py.File(path, "r") as file:
        return isinstance(file.get(_VOLUME_PATH), h5py.Dataset)


def projection_image(path: Path) -> np.ndarray | None:
    """The central sample projection, or None for a reconstruction saved without frames."""
    if has_saved_volume(path) and saved_without_frames(path):
        return None
    raw = read_acquisition(path)
    if raw is not None:
        return np.asarray(raw.counts[raw.counts.shape[0] // 2], np.float32)
    located = locate_frames(str(path))
    samples = (
        located.stack.frames
        if located.image_key is None
        else np.flatnonzero(located.image_key == 0)
    )
    if not len(samples):
        raise ValueError(f"{path} holds no sample projections")
    with h5py.File(path, "r") as file:
        stack = cast("h5py.Dataset", file[located.stack.dataset])
        return np.asarray(stack[int(samples[len(samples) // 2])], np.float32)


def _saved_shape(path: Path) -> tuple[str, tuple[int, ...]]:
    with h5py.File(path, "r") as file:
        volume = cast("h5py.Dataset", file[_VOLUME_PATH])
        attribute = file["/entry/processing/tomojax"].attrs.get("volume_axes_order")
        order = saved_volume_axes(None if attribute is None else str(attribute), path).lower()
        return order, tuple(int(n) for n in volume.shape)


def saved_plane(path: Path, axis: Axis, index: int) -> np.ndarray:
    """Plane ``index`` across ``axis`` of a saved volume, shown as (z, y, x) minus ``axis``."""
    order, _ = _saved_shape(path)
    with h5py.File(path, "r") as file:
        volume = cast("h5py.Dataset", file[_VOLUME_PATH])
        key: list[int | slice] = [slice(None)] * 3
        key[order.index(axis)] = index
        plane = np.asarray(volume[tuple(key)], np.float32)
    remaining = [a for a in order if a != axis]
    return np.transpose(plane, [remaining.index(a) for a in "zyx" if a != axis])


def _picture(image: np.ndarray, label: str) -> Image.Image:
    """``image`` as it was sampled (capped or enlarged), scaled and labelled."""
    shape = (int(image.shape[0]), int(image.shape[1]))
    shown = _images.resample(image, _images.display_shape(shape, _PIXELS, _images.PLANE_SIDE))
    return _images.labelled(_images.to_uint8(shown, _images.limits(shown)), label)


def destinations(directory: Path, names: Sequence[str]) -> list[Path]:
    """The files a preview run will write in ``directory``."""
    return [directory / name for name in names]


def slice_name(axis: str, index: int) -> str:
    """The file of plane ``index`` across ``axis``."""
    return f"slice_{axis}_{index}.png"


def centre_name(ordinal: int, count: int, centre: float) -> str:
    """The file of sweep candidate ``ordinal``: unique even for repeated centres."""
    width = max(2, len(str(count - 1)))
    return f"centre_{ordinal:0{width}d}_{centre:+.2f}px.png"


def write_saved_slices(
    path: Path, directory: Path, requests: Sequence[tuple[Axis, int]] | None, out: Previews
) -> None:
    """Read and write planes of a saved reconstruction (its central planes by default)."""
    order, shape = _saved_shape(path)
    sizes = dict(zip(order, shape, strict=True))
    central: list[tuple[Axis, int]] = [
        ("z", sizes["z"] // 2),
        ("y", sizes["y"] // 2),
        ("x", sizes["x"] // 2),
    ]
    wanted = list(requests or central)
    for axis, index in wanted:
        if not 0 <= index < sizes[axis]:
            raise ValueError(f"--slice {axis}={index} is past the volume's {sizes[axis]} planes")
    for axis, index in dict.fromkeys(wanted):
        name = slice_name(axis, index)
        plane = saved_plane(path, axis, index)
        _images.save_png(_picture(plane, f"{axis} = {index} · saved volume"), directory / name)
        out.files[name] = str(directory / name)
    for axis, index in wanted:
        out.slice_requests.append(
            {"axis": axis, "index": index, "path": str(directory / slice_name(axis, index))}
        )


def write_scan_slices(
    scan: tj.Scan, directory: Path, requests: Sequence[tuple[Axis, int]] | None, out: Previews
) -> None:
    """Reconstruct the requested planes of a scan (its central z plane by default)."""
    from tomojax import interactive
    from tomojax.cli._results import unit_of
    from tomojax.core.registration import registered_plane, volume_axes

    central: list[tuple[Axis, int]] = [("z", scan.grid.nz // 2)]
    wanted = list(requests or central)
    plan = interactive.plan_planes(scan, wanted)
    if plan.full is not None:
        progress.note("info", "slice previews: one full reconstruction makes every plane")
    resolved = plan.full or plan.plans[0]
    grid, unit = resolved.scan.grid, unit_of(scan)
    method = resolved.method
    for (axis, index), pixels in zip(plan.requests, interactive.planes_of(plan), strict=True):
        dim = "zyx".index(axis)
        sample = registered_plane(np.asarray(pixels), volume_axes(grid, unit), dim, index, unit)
        name = slice_name(axis, index)
        image = _images.sample_image(sample)
        label = f"{axis} = {index} · {method} reconstruction of the scan"
        _images.save_png(
            _images.labelled(_images.to_uint8(image, _images.limits(image)), label),
            directory / name,
        )
        out.files[name] = str(directory / name)
    for axis, index in wanted:
        out.slice_requests.append(
            {"axis": axis, "index": index, "path": str(directory / slice_name(axis, index))}
        )


def write_sweep(
    scan: tj.Scan,
    source: Path,
    directory: Path,
    centres: Sequence[float],
    plane: tuple[Axis, int] | None,
    out: Previews,
) -> None:
    """Reconstruct ``plane`` at each candidate centre: one image each and a contact sheet."""
    from tomojax import interactive
    from tomojax.cli._results import unit_of

    axis, index = plane or ("z", scan.grid.nz // 2)
    sweep = interactive.resolve_sweep(scan, axis, index, centres)
    count, du = len(centres), scan.detector.du
    unit = unit_of(scan)
    shown_unit = "scan units" if unit == "scan_length" else unit
    recorded = scan.detector.center[0] / du
    full = sweep.full_solves
    progress.note(
        "info",
        f"centre sweep: {count} candidates on the {axis} = {index} plane"
        + (f", {full} of them a full reconstruction each" if full else ""),
    )
    columns = math.ceil(math.sqrt(count))
    cell = max(32, min(256, _images.SHEET_SIDE // columns))
    thumbs: list[np.ndarray] = []
    bounds: list[tuple[float, float]] = []
    for ordinal, (centre, pixels) in enumerate(
        zip(centres, interactive.sweep_planes(sweep), strict=True)
    ):
        image = np.asarray(pixels, np.float32)
        name = centre_name(ordinal, count, centre)
        label = f"centre {centre:+.2f} px ({centre * du:+.4g} {shown_unit}) · {axis} = {index}"
        _images.save_png(_picture(image, label), directory / name)
        out.files[name] = str(directory / name)
        out.sweep.append(
            {
                "ordinal": ordinal,
                "centre_px": centre,
                "centre_length": centre * du,
                "path": str(directory / name),
            }
        )
        rows, cols = image.shape
        scale = cell / max(rows, cols)
        thumbs.append(
            _images.resample(image, (max(1, round(rows * scale)), max(1, round(cols * scale))))
        )
        bounds.append(_images.limits(image))
    shared = (min(low for low, _ in bounds), max(high for _, high in bounds))
    panels = []
    for thumb, entry in zip(thumbs, out.sweep, strict=True):
        centre = cast("float", entry["centre_px"])
        panel = _images.labelled(_images.to_uint8(thumb, shared), f"{centre:+.2f} px")
        panels.append(
            _images.outlined(panel) if math.isclose(centre, recorded, abs_tol=1e-6) else panel
        )
    sheet = _images.sheet(panels, columns=columns)
    if not any(math.isclose(c, recorded, abs_tol=1e-6) for c in centres):
        sheet = _images.labelled(
            np.asarray(sheet), f"recorded centre {recorded:+.2f} px (not swept)"
        )
    _images.save_png(sheet, directory / "centre_sweep.png")
    out.files["centre_sweep.png"] = str(directory / "centre_sweep.png")
    v = scan.detector.center[1]
    out.apply = {
        "command": f"tomojax correct {source} -o centred.nxs --detector-center U {v:g}",
        "u_per_pixel": du,
        "v": v,
        "unit": shown_unit,
        "recorded_px": recorded,
    }


__all__ = [
    "LISTING",
    "Previews",
    "centre_name",
    "destinations",
    "has_saved_volume",
    "is_centres_value",
    "parse_centres",
    "parse_slice",
    "projection_image",
    "saved_plane",
    "slice_name",
    "write_saved_slices",
    "write_scan_slices",
    "write_sweep",
]
