"""Feature masks on a source array's grid: shapes, seeded regions and bands.

Positions and distances are in array indices of the source, the same coordinates
view marks use, so a confirmed mark converts directly into a feature.
"""

from __future__ import annotations

from math import ceil, floor
from typing import TYPE_CHECKING, Any, cast

import msgspec
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from tomojax.cli import execution as ex, operations as op, products as p
from tomojax.errors import InvalidParameter
from tomojax.wire.arrays import Array, Member

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def grids(shape: Sequence[int]) -> list[np.ndarray]:
    """Open index grids that broadcast to ``shape``."""
    return list(np.ogrid[tuple(slice(0, n) for n in shape)])


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


def prism(recipe: op.Prism, shape: Sequence[int], axes: Sequence[str]) -> np.ndarray:
    """Rasterise the polygon across the axis, then extrude it over the extent."""
    a = axes.index(recipe.axis)
    rows, cols = (d for d in range(len(shape)) if d != a)
    plane = Image.new("1", (shape[cols], shape[rows]), 0)
    ImageDraw.Draw(plane).polygon(
        [(point[cols], point[rows]) for point in recipe.points], fill=1, outline=1
    )
    across = np.expand_dims(np.array(plane, dtype=bool), a)
    lo, hi = sorted(recipe.extent)
    along = grids(shape)[a]
    return across & (along >= lo) & (along <= hi)


def shape_mask(
    recipe: op.Box | op.Sphere | op.Cylinder | op.Prism, shape: Sequence[int], axes: Sequence[str]
) -> np.ndarray:
    """Voxels inside a box, sphere, cylinder or prism given in array indices."""
    g = grids(shape)
    if isinstance(recipe, op.Box):
        mask = np.ones(shape, dtype=bool)
        for d, (a, b) in enumerate(zip(*recipe.corners, strict=True)):
            mask &= (g[d] >= min(a, b)) & (g[d] <= max(a, b))
        return mask
    if isinstance(recipe, op.Sphere | op.Cylinder):
        across = range(len(shape))
        if isinstance(recipe, op.Cylinder):
            a = axes.index(recipe.axis)
            across = [d for d in across if d != a]
        mask = sum((g[d] - recipe.at[d]) ** 2 for d in across) <= recipe.radius**2
        if isinstance(recipe, op.Cylinder):
            lo, hi = sorted(recipe.extent)
            mask = mask & (g[a] >= lo) & (g[a] <= hi)
        return np.broadcast_to(mask, shape).copy()
    assert isinstance(recipe, op.Prism)
    return prism(recipe, shape, axes)


def seeded(recipe: op.Seeded, values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Grow the connected region around the seeds whose smoothed value stays in band."""
    shape = values.shape
    seeds = np.asarray(recipe.seeds, dtype=float)
    if (seeds < -0.5).any() or (seeds >= np.asarray(shape) - 0.5).any():
        raise InvalidParameter("recipe.seeds", f"give seeds inside the source, shape {shape}")
    # Work on the box the region may reach, so a small feature in a big volume stays cheap.
    reach = recipe.max_radius
    lo = [0] * len(shape) if reach is None else [max(floor(v - reach), 0) for v in seeds.min(0)]
    hi = (
        list(shape)
        if reach is None
        else [min(ceil(v + reach) + 1, n) for v, n in zip(seeds.max(0), shape, strict=True)]
    )
    window = tuple(slice(a, b) for a, b in zip(lo, hi, strict=True))
    data = np.asarray(values[window], dtype=np.float32)
    local = np.round(seeds - lo).astype(int)
    smooth = ndimage.gaussian_filter(data, recipe.smoothing) if recipe.smoothing else data
    level = float(
        np.mean([np.median(smooth[tuple(slice(max(v - 1, 0), v + 2) for v in s)]) for s in local])
    )
    if recipe.tolerance_unit == "value":
        width = recipe.tolerance
        noise = None
    else:
        # Robust noise from the high-pass residual in a neighbourhood of the seeds.
        residual = data - ndimage.gaussian_filter(data, 1.0)
        near = np.zeros(data.shape, dtype=bool)
        for s in local:
            near[tuple(slice(max(v - 8, 0), v + 9) for v in s)] = True
        r = residual[near]
        noise = float(1.4826 * np.median(np.abs(r - np.median(r))))
        if noise == 0:
            raise InvalidParameter(
                "recipe.tolerance_unit",
                "the source shows no noise near the seeds; give the tolerance in value units",
            )
        width = recipe.tolerance * noise
    inside = np.abs(smooth - level) <= width
    if reach is not None:
        g = grids(data.shape)
        nearest = np.min(
            [sum((g[d] - s[d]) ** 2 for d in range(len(shape))) for s in seeds - lo], axis=0
        )
        inside &= nearest <= reach**2
    for s in local:
        inside[tuple(s)] = True
    labels = cast("tuple[np.ndarray, int]", ndimage.label(inside))[0]
    kept = np.unique([labels[tuple(s)] for s in local])
    mask = np.zeros(shape, dtype=bool)
    mask[window] = np.isin(labels, kept)
    resolved = {"level": level, "band": float(width)}
    if noise is not None:
        resolved["noise"] = noise
    return mask, resolved


def band(recipe: op.Band, base: ex.Feature, shape: Sequence[int]) -> np.ndarray:
    """Voxels whose signed distance from the base feature lies in (inner, outer]."""
    if base.mask.shape != tuple(shape):
        raise InvalidParameter(
            "recipe.feature", f"the feature covers {base.mask.shape}; the source is {tuple(shape)}"
        )
    if not base.mask.any():
        raise InvalidParameter("recipe.feature", "the feature is empty")
    if recipe.outer <= recipe.inner:
        raise InvalidParameter("recipe.outer", "outer must exceed inner")
    inner = np.asarray(ndimage.distance_transform_edt(base.mask))
    outer = np.asarray(ndimage.distance_transform_edt(~base.mask))
    # Boundary voxels inside are at 0, their outside neighbours at 1.
    distance = np.where(base.mask, 1 - inner, outer)
    return (distance > recipe.inner) & (distance <= recipe.outer)


def feature_handle(feature: ex.Feature, values: Any | None) -> ex.NewHandle:
    """Publish a feature with its mask member and a summary of ``values`` inside it."""
    mask = feature.mask
    if mask.shape != feature.registration.shape:
        raise InvalidParameter(
            "mask", f"mask shape {mask.shape} differs from its grid {feature.registration.shape}"
        )
    voxels = int(mask.sum())
    bounds: tuple[tuple[int, int], ...] = ()
    if voxels:
        spans = [
            np.flatnonzero(mask.any(axis=tuple(a for a in range(mask.ndim) if a != d)))
            for d in range(mask.ndim)
        ]
        bounds = tuple((int(s[0]), int(s[-1]) + 1) for s in spans)
    mean = std = None
    if values is not None:
        inside = np.asarray(values)[mask]
        inside = inside[np.isfinite(inside)]
        if inside.size:
            mean, std = float(inside.mean()), float(inside.std())

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
            product, {"self": feature, "mask": ex.RegisteredArray(mask, metadata)}
        )

    return ex.NewHandle(mask.nbytes, bind, lambda: None)


def feature_work(operation: op.Feature, context: ex.ExecutionContext) -> Callable[[], ex.NewHandle]:
    """Check the recipe against the source's axes, then build the mask on demand."""
    from tomojax.cli._browse import source_array

    recipe = operation.recipe
    name = operation.name or str(recipe.__struct_config__.tag)
    with source_array(operation.source, context) as (_, meta):
        registration: Array = meta
    axes = [a.name for a in registration.axes]
    check(recipe, axes)
    base = None
    if isinstance(recipe, op.Band):
        base = context.leases[recipe.feature].value
        if not isinstance(base, ex.Feature):
            raise InvalidParameter("recipe.feature", "choose a feature handle")

    def make() -> ex.NewHandle:
        resolved = msgspec.to_builtins(recipe)
        with source_array(operation.source, context) as (data, _):
            values = np.asarray(data)
        context.control.checkpoint()
        if isinstance(recipe, op.Seeded):
            mask, found = seeded(recipe, values)
            resolved.update(found)
        elif isinstance(recipe, op.Band):
            assert base is not None
            mask = band(recipe, base, registration.shape)
            resolved["feature"] = base.name
        else:
            mask = shape_mask(recipe, registration.shape, axes)
        context.control.checkpoint()
        return feature_handle(ex.Feature(name, mask, registration, resolved), values)

    return make
