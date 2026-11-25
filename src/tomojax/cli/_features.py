"""Feature masks on a source array's grid: shapes, seeded regions and bands.

Positions and distances are in array indices of the source, the same coordinates
view marks use, so a confirmed mark converts directly into a feature. Every recipe
works on a window around the feature, reads the source through bounded windows and
keeps only the mask's bounding box, so features on volumes far larger than memory
cost memory in proportion to the feature.
"""

from __future__ import annotations

from math import ceil, floor
from typing import TYPE_CHECKING, Any, cast

import msgspec
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from tomojax.cli import execution as ex, operations as op, products as p
from tomojax.cli._handles import stage_array
from tomojax.errors import InvalidParameter
from tomojax.wire.arrays import Array, Coordinates, Hdf5, Member, Npy, Uniform

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from tomojax.wire.arrays import Storage

type Window = tuple[slice, ...]
_GROWTH = 32  # first half-width of a seeded window without max_radius, in voxels


def check_positions(positions: Sequence[Sequence[float]], axes: Sequence[str], field: str) -> None:
    for point in positions:
        if len(point) != len(axes):
            raise InvalidParameter(
                field, f"give {len(axes)} array indices ({', '.join(axes)}) per position"
            )


def check(recipe: op.Recipe, axes: Sequence[str]) -> None:
    """Reject positions or axis names that do not fit the source's axes."""
    if isinstance(recipe, op.Box):
        check_positions(recipe.corners, axes, "recipe.corners")
    elif isinstance(recipe, op.Sphere | op.Cylinder):
        check_positions((recipe.at,), axes, "recipe.at")
    elif isinstance(recipe, op.Prism | op.Seeded):
        field = "points" if isinstance(recipe, op.Prism) else "seeds"
        least = 3 if isinstance(recipe, op.Prism) else 1
        positions = getattr(recipe, field)
        if len(positions) < least:
            raise InvalidParameter(f"recipe.{field}", f"give at least {least} positions")
        check_positions(positions, axes, f"recipe.{field}")
    if isinstance(recipe, op.Cylinder | op.Prism) and recipe.axis not in axes:
        raise InvalidParameter("recipe.axis", f"choose one of {', '.join(axes)}")
    if isinstance(recipe, op.Band) and recipe.outer <= recipe.inner:
        raise InvalidParameter("recipe.outer", "outer must exceed inner")


def clipped(
    lo: Sequence[float] | np.ndarray, hi: Sequence[float] | np.ndarray, shape: Sequence[int]
) -> Window:
    """The voxels from ``lo`` to ``hi`` inclusive, inside the grid."""
    return tuple(
        slice(min(max(floor(a), 0), n), min(max(ceil(b) + 1, 0), n))
        for a, b, n in zip(lo, hi, shape, strict=True)
    )


def grids(window: Window) -> list[np.ndarray]:
    """Open index grids of a window, in the grid's own indices."""
    return list(np.ogrid[window])


def read(data: Any, window: Window) -> np.ndarray:
    """Source values over a window as float32, read without loading the rest."""
    from tomojax.cli._browse import pixels

    return np.asarray(pixels(data, tuple(range(w.start, w.stop) for w in window)), np.float32)


def tight(mask: np.ndarray, window: Window) -> tuple[np.ndarray, Window]:
    """Crop a mask computed over ``window`` to the bounding box of its set voxels."""
    spans = [
        np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != d)))
        for d in range(mask.ndim)
    ]
    if any(s.size == 0 for s in spans):
        return np.zeros((0,) * mask.ndim, dtype=bool), tuple(slice(0, 0) for _ in window)
    local = tuple(slice(int(s[0]), int(s[-1]) + 1) for s in spans)
    box = tuple(
        slice(w.start + c.start, w.start + c.stop) for w, c in zip(window, local, strict=True)
    )
    return np.ascontiguousarray(mask[local]), box


def shape_mask(
    recipe: op.Box | op.Sphere | op.Cylinder | op.Prism, shape: Sequence[int], axes: Sequence[str]
) -> tuple[np.ndarray, Window]:
    """A box, sphere, cylinder or prism, computed over its own bounds only."""
    a = axes.index(recipe.axis) if isinstance(recipe, op.Cylinder | op.Prism) else None
    if isinstance(recipe, op.Box):
        first, second = (np.asarray(c, dtype=float) for c in recipe.corners)
        window = clipped(np.minimum(first, second), np.maximum(first, second), shape)
        g = grids(window)
        mask = np.ones([w.stop - w.start for w in window], dtype=bool)
        for d, (u, v) in enumerate(zip(first, second, strict=True)):
            mask &= (g[d] >= min(u, v)) & (g[d] <= max(u, v))
        return mask, window
    if isinstance(recipe, op.Prism):
        assert a is not None
        rows, cols = (d for d in range(len(shape)) if d != a)
        points = np.asarray(recipe.points, dtype=float)
        lo, hi = points.min(0), points.max(0)
        lo[a], hi[a] = sorted(recipe.extent)
        window = clipped(lo, hi, shape)
        plane = Image.new(
            "1", (window[cols].stop - window[cols].start, window[rows].stop - window[rows].start), 0
        )
        corner = (window[cols].start, window[rows].start)
        ImageDraw.Draw(plane).polygon(
            [(q[cols] - corner[0], q[rows] - corner[1]) for q in recipe.points], fill=1, outline=1
        )
        along = window[a].stop - window[a].start
        return np.repeat(np.expand_dims(np.array(plane, dtype=bool), a), along, axis=a), window
    centre = np.asarray(recipe.at, dtype=float)
    lo, hi = centre - recipe.radius, centre + recipe.radius
    across = [d for d in range(len(shape)) if d != a]
    if a is not None:
        assert isinstance(recipe, op.Cylinder)
        lo[a], hi[a] = sorted(recipe.extent)
    window = clipped(lo, hi, shape)
    g = grids(window)
    mask = sum((g[d] - centre[d]) ** 2 for d in across) <= recipe.radius**2
    if a is not None:
        mask = mask & (g[a] >= lo[a]) & (g[a] <= hi[a])
    return np.broadcast_to(mask, [w.stop - w.start for w in window]).copy(), window


def seed_statistics(recipe: op.Seeded, data: Any, seeds: np.ndarray) -> dict[str, float]:
    """The level at the seeds and, for a tolerance in noise units, the local noise."""
    shape = data.shape
    near = clipped(seeds.min(0) - 8, seeds.max(0) + 8, shape)
    values = read(data, near)
    smooth = ndimage.gaussian_filter(values, recipe.smoothing) if recipe.smoothing else values
    local = np.round(seeds - [w.start for w in near]).astype(int)
    level = float(
        np.mean([np.median(smooth[tuple(slice(max(v - 1, 0), v + 2) for v in s)]) for s in local])
    )
    found = {"level": level}
    if recipe.tolerance_unit == "value":
        found["band"] = recipe.tolerance
        return found
    # Robust noise from the high-pass residual in the seeds' neighbourhood.
    residual = values - ndimage.gaussian_filter(values, 1.0)
    noise = float(1.4826 * np.median(np.abs(residual - np.median(residual))))
    if noise == 0:
        raise InvalidParameter(
            "recipe.tolerance_unit",
            "the source shows no noise near the seeds; give the tolerance in value units",
        )
    return {**found, "noise": noise, "band": recipe.tolerance * noise}


def grow(
    recipe: op.Seeded, data: Any, seeds: np.ndarray, found: dict[str, float], window: Window
) -> np.ndarray:
    """The seeds' connected component of in-band voxels within one window."""
    values = read(data, window)
    if recipe.smoothing:
        values = ndimage.gaussian_filter(values, recipe.smoothing, output=values)
    inside = np.abs(values - found["level"]) <= found["band"]
    del values
    local = np.round(seeds - [w.start for w in window]).astype(int)
    if recipe.max_radius is not None:
        g = grids(window)
        nearest = np.full(inside.shape, np.inf, dtype=np.float32)
        for s in seeds:
            np.minimum(nearest, sum((g[d] - s[d]) ** 2 for d in range(len(s))), out=nearest)
        inside &= nearest <= recipe.max_radius**2
        del nearest
    for s in local:
        inside[tuple(s)] = True
    labels = cast("tuple[np.ndarray, int]", ndimage.label(inside))[0]
    kept = np.unique([labels[tuple(s)] for s in local])
    return np.isin(labels, kept)


def seeded(recipe: op.Seeded, data: Any) -> tuple[np.ndarray, Window, dict[str, float]]:
    """Grow the region in a window that doubles until the region stops at its faces."""
    shape = data.shape
    seeds = np.asarray(recipe.seeds, dtype=float)
    if (seeds < -0.5).any() or (seeds >= np.asarray(shape) - 0.5).any():
        raise InvalidParameter("recipe.seeds", f"give seeds inside the source, shape {shape}")
    found = seed_statistics(recipe, data, seeds)
    half = recipe.max_radius if recipe.max_radius is not None else _GROWTH
    while True:
        window = clipped(seeds.min(0) - half, seeds.max(0) + half, shape)
        mask = grow(recipe, data, seeds, found, window)
        # A region reaching a face that is not the grid's edge may continue beyond it.
        open_faces = any(
            (w.start > 0 and mask.take(0, axis=d).any())
            or (w.stop < n and mask.take(-1, axis=d).any())
            for d, (w, n) in enumerate(zip(window, shape, strict=True))
        )
        if recipe.max_radius is not None or not open_faces:
            return mask, window, found
        half *= 2


def band(recipe: op.Band, base: ex.Feature, shape: Sequence[int]) -> tuple[np.ndarray, Window]:
    """Voxels whose signed distance from the base feature lies in (inner, outer].

    The feature's boundary voxels are at 0 and their outside neighbours at 1. The
    distances are computed over the feature's box widened by the band.
    """
    if base.shape != tuple(shape):
        raise InvalidParameter(
            "recipe.feature", f"the feature covers {base.shape}; the source is {tuple(shape)}"
        )
    if not base.inside.any():
        raise InvalidParameter("recipe.feature", "the feature is empty")
    reach = ceil(max(recipe.outer, -recipe.inner, 0)) + 1
    window = clipped(
        [b.start - reach for b in base.box], [b.stop - 1 + reach for b in base.box], shape
    )
    mask = base[window]
    result = np.zeros(mask.shape, dtype=bool)
    if recipe.outer > 0:  # voxels outside, at distances 1, sqrt(2), ...
        outside = np.asarray(ndimage.distance_transform_edt(~mask))
        result |= ~mask & (outside > recipe.inner) & (outside <= recipe.outer)
        del outside
    if recipe.inner < 0:  # voxels inside, at distances 0, -1, ...
        depth = 1 - np.asarray(ndimage.distance_transform_edt(mask))
        result |= mask & (depth > recipe.inner) & (depth <= recipe.outer)
    return result, window


def owned(
    registration: Array, values: Callable[[Storage], np.ndarray], context: ex.ExecutionContext
) -> Array:
    """The registration with coordinate vectors copied into the feature's own arrays.

    A feature then keeps its coordinates after the handle it was made from closes,
    and never refers to a binding that was not published.
    """
    axes = []
    for axis in registration.axes:
        c = axis.coordinates
        if isinstance(c, Coordinates):
            c = Coordinates(stage_array(context, values(c.values)))
        axes.append(msgspec.structs.replace(axis, coordinates=c))
    return msgspec.structs.replace(registration, axes=tuple(axes))


def feature_handle(feature: ex.Feature, data: Any | None) -> ex.NewHandle:
    """Publish a feature with its mask member and a summary of ``data`` inside it."""
    if len(feature.box) != len(feature.shape) or feature.inside.shape != tuple(
        b.stop - b.start for b in feature.box
    ):
        raise InvalidParameter("mask", "the mask does not fit its box")
    voxels = int(feature.inside.sum())
    bounds = tuple((b.start, b.stop) for b in feature.box) if voxels else ()
    mean = std = None
    if data is not None and voxels:
        inside = read(data, feature.box)[feature.inside]
        inside = inside[np.isfinite(inside)]
        if inside.size:
            mean, std = float(inside.mean(dtype=np.float64)), float(inside.std(dtype=np.float64))

    def bind(identifier: str) -> ex.BoundHandle:
        metadata = msgspec.structs.replace(
            feature.registration,
            storage=Member(identifier, "mask"),
            dtype="|b1",
            unit="1",
            domain="mask",
            columns=(),
        )
        product = p.Feature(
            identifier, feature.name, metadata, voxels, bounds, mean, std, feature.recipe
        )
        return ex.BoundHandle(
            product, {"self": feature, "mask": ex.RegisteredArray(feature, metadata)}
        )

    return ex.NewHandle(feature.inside.nbytes, bind, lambda: None)


def feature_work(operation: op.Feature, context: ex.ExecutionContext) -> Callable[[], ex.NewHandle]:
    """Check the recipe against the source's axes, then build the mask on demand."""
    from tomojax.cli._browse import source_array

    recipe = operation.recipe
    name = operation.name or str(recipe.__struct_config__.tag)
    with source_array(operation.source, context) as (_, meta):
        registration: Array = meta
    axes = [a.name for a in registration.axes]
    shape = tuple(registration.shape)
    check(recipe, axes)
    base = None
    if isinstance(recipe, op.Band):
        base = context.leases[recipe.feature].value
        if not isinstance(base, ex.Feature):
            raise InvalidParameter("recipe.feature", "choose a feature handle")

    def coordinate(ref: Storage) -> np.ndarray:
        with source_array(ref, context) as (values, _):
            return np.asarray(values[:])

    def make() -> ex.NewHandle:
        resolved = msgspec.to_builtins(recipe)
        with source_array(operation.source, context) as (data, _):
            context.control.checkpoint()
            if isinstance(recipe, op.Seeded):
                mask, window, found = seeded(recipe, data)
                resolved.update(found)
            elif isinstance(recipe, op.Band):
                assert base is not None
                mask, window = band(recipe, base, shape)
                resolved["feature"] = base.name
            else:
                mask, window = shape_mask(recipe, shape, axes)
            context.control.checkpoint()
            inside, box = tight(mask, window)
            del mask
            grid = owned(registration, coordinate, context)
            return feature_handle(ex.Feature(name, inside, box, grid, resolved), data)

    return make


FEATURE_GROUP = "/entry/feature"
SAVED_SUFFIXES = (".h5", ".hdf5", ".nxs")


def save_feature(path: Path, feature: ex.Feature) -> None:
    """Write the mask as uint8 on its whole grid, with its registration, name and recipe.

    Only chunks that meet the feature's box are written; the rest read as zero.
    """
    import h5py

    with h5py.File(path, "w") as file:
        group = file.create_group(FEATURE_GROUP)
        mask = group.create_dataset(
            "mask", shape=feature.shape, dtype="u1", chunks=True, compression="gzip", fillvalue=0
        )
        if feature.inside.size:
            mask[feature.box] = feature.inside
        axes = []
        for axis in feature.registration.axes:
            c = axis.coordinates
            if isinstance(c, Coordinates):
                name = f"coordinates/{axis.name}"
                group[name] = _load_storage(c.values)
                c = Coordinates(Hdf5(str(path), f"{FEATURE_GROUP}/{name}"))
            axes.append(msgspec.structs.replace(axis, coordinates=c))
        registration = msgspec.structs.replace(feature.registration, axes=tuple(axes))
        group.attrs["registration"] = msgspec.json.encode(registration).decode()
        group.attrs["box"] = msgspec.json.encode([[b.start, b.stop] for b in feature.box]).decode()
        group.attrs["name"] = feature.name
        group.attrs["recipe"] = msgspec.json.encode(feature.recipe).decode()


def _load_storage(ref: Storage) -> np.ndarray:
    if not isinstance(ref, Npy):
        raise InvalidParameter("source", f"coordinate vector {ref} is not a published array")
    return np.load(ref.path, allow_pickle=False)


def load_feature(path: str, context: ex.ExecutionContext) -> ex.NewHandle:
    """Reopen a saved feature, reading only its box and copying its coordinates."""
    import h5py

    with h5py.File(path, "r") as file:
        group = file.get(FEATURE_GROUP)
        if not isinstance(group, h5py.Group) or "mask" not in group:
            raise InvalidParameter("path", f"{path} holds no saved TomoJAX feature")
        registration = msgspec.json.decode(str(group.attrs["registration"]), type=Array)
        box = tuple(slice(a, b) for a, b in msgspec.json.decode(str(group.attrs["box"])))
        mask = group["mask"]
        assert isinstance(mask, h5py.Dataset)
        inside = np.asarray(mask[box], dtype=bool) if box else np.zeros((0,) * mask.ndim, bool)

        def coordinate(ref: Storage) -> np.ndarray:
            assert isinstance(ref, Hdf5)
            return np.asarray(file[ref.dataset])

        grid = owned(registration, coordinate, context)
        recipe = msgspec.json.decode(str(group.attrs["recipe"]))
        feature = ex.Feature(str(group.attrs["name"]), inside, box, grid, recipe)
    return feature_handle(feature, None)


def export_feature(
    path: Path,
    feature: ex.Feature,
    *,
    dtype: str,
    value_range: tuple[float, float] | None,
    prefix: str,
    source: dict[str, object],
) -> None:
    """Write the mask as 0/1 planes along its first axis, as TIFFs or one raw file."""
    from tomojax._files import write_volume

    if feature.ndim != 3:
        raise InvalidParameter("source", "export needs a three-dimensional feature")
    spacing = [
        abs(a.coordinates.spacing) if isinstance(a.coordinates, Uniform) else 1.0
        for a in feature.registration.axes
    ]
    shape = cast("tuple[int, int, int]", feature.shape)
    write_volume(
        path,
        lambda z: feature[z].astype(np.float32),
        shape,
        (spacing[2], spacing[1], spacing[0]),
        dtype=dtype,
        value_range=value_range,
        prefix=prefix,
        source=source,
    )
