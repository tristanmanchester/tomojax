from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .geometry.base import Detector, Geometry, Grid


def _shape_of(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        try:
            shape = np.shape(value)
        except Exception:
            shape = ()
    return tuple(int(s) for s in shape)


def _shape_text(shape: Iterable[object]) -> str:
    items: list[object] = []
    for s in shape:
        try:
            items.append(int(s))
        except Exception:
            items.append(s)
    return str(tuple(items))


def _raise_shape_error(
    context: str,
    name: str,
    *,
    expected: str,
    actual: Iterable[int],
    fix: str,
    kind: str = "incompatible",
) -> None:
    raise ValueError(
        f"{context}: {name} has {kind} shape; expected {expected}, "
        f"actual {_shape_text(actual)}. Likely fix: {fix}"
    )


def _raise_value_error(
    context: str,
    name: str,
    *,
    expected: str,
    actual: object,
    fix: str,
) -> None:
    raise ValueError(
        f"{context}: {name} has invalid values; expected {expected}, "
        f"actual {actual}. Likely fix: {fix}"
    )


def _is_positive_int(value: object) -> bool:
    try:
        ivalue = int(value)  # type: ignore[arg-type]
    except Exception:
        return False
    try:
        return ivalue > 0 and float(value) == float(ivalue)  # type: ignore[arg-type]
    except Exception:
        return False


def _is_positive_finite(value: object) -> bool:
    try:
        fvalue = float(value)  # type: ignore[arg-type]
    except Exception:
        return False
    return math.isfinite(fvalue) and fvalue > 0.0


def _geometry_view_count(geometry: Geometry | None) -> int | None:
    if geometry is None:
        return None
    thetas = getattr(geometry, "thetas_deg", None)
    if thetas is None:
        return None
    try:
        return int(len(thetas))
    except Exception:
        return None


def validate_grid(grid: Grid, context: str) -> tuple[int, int, int]:
    """Validate reconstruction grid metadata and return ``(nx, ny, nz)``."""
    dims = (getattr(grid, "nx", None), getattr(grid, "ny", None), getattr(grid, "nz", None))
    if not all(_is_positive_int(v) for v in dims):
        _raise_shape_error(
            context,
            "reconstruction grid",
            expected="positive integer (nx, ny, nz)",
            actual=dims,
            fix="pass --grid NX NY NZ with all values > 0 or fix grid metadata.",
            kind="invalid",
        )

    spacing = (getattr(grid, "vx", None), getattr(grid, "vy", None), getattr(grid, "vz", None))
    if not all(_is_positive_finite(v) for v in spacing):
        _raise_value_error(
            context,
            "voxel spacing",
            expected="positive finite (vx, vy, vz)",
            actual=spacing,
            fix="fix grid voxel-size metadata so vx, vy, and vz are finite and > 0.",
        )
    return int(grid.nx), int(grid.ny), int(grid.nz)


def validate_detector(detector: Detector, context: str) -> tuple[int, int]:
    """Validate detector metadata and return detector image shape ``(nv, nu)``."""
    shape = (getattr(detector, "nv", None), getattr(detector, "nu", None))
    if not all(_is_positive_int(v) for v in shape):
        _raise_shape_error(
            context,
            "detector",
            expected="positive integer (nv, nu)",
            actual=shape,
            fix="fix detector metadata so nv and nu are both positive integers.",
            kind="invalid",
        )

    spacing = (getattr(detector, "dv", None), getattr(detector, "du", None))
    if not all(_is_positive_finite(v) for v in spacing):
        _raise_value_error(
            context,
            "detector spacing",
            expected="positive finite (dv, du)",
            actual=spacing,
            fix="fix detector metadata so dv and du are finite and > 0.",
        )
    return int(detector.nv), int(detector.nu)


def validate_projection_shape(
    projections_shape: Iterable[int],
    detector: Detector,
    *,
    geometry: Geometry | None = None,
    context: str,
) -> tuple[int, int, int]:
    """Validate a projection-stack shape tuple without requiring an array."""
    nv, nu = validate_detector(detector, context)
    shape = tuple(int(s) for s in projections_shape)
    if len(shape) != 3:
        _raise_shape_error(
            context,
            "projections",
            expected=f"(n_views, nv, nu)=(*, {nv}, {nu}) from detector",
            actual=shape,
            fix="pass projections with shape (number of angles, detector.nv, detector.nu).",
        )

    n_views = int(shape[0])
    if n_views <= 0:
        _raise_shape_error(
            context,
            "projections",
            expected=f"non-empty (n_views, nv, nu)=(*, {nv}, {nu}) from detector",
            actual=shape,
            fix="provide at least one projection view and one matching angle.",
            kind="invalid",
        )

    geometry_n_views = _geometry_view_count(geometry)
    expected_n_views = geometry_n_views if geometry_n_views is not None else n_views
    expected = f"(n_views, nv, nu)=({expected_n_views}, {nv}, {nu})"
    source = "geometry/detector" if geometry_n_views is not None else "detector"
    expected = f"{expected} from {source}"
    if geometry_n_views is not None and n_views != geometry_n_views:
        _raise_shape_error(
            context,
            "projections",
            expected=expected,
            actual=shape,
            fix="use projections generated with the same angle list and detector metadata.",
        )
    if shape[1:] != (nv, nu):
        _raise_shape_error(
            context,
            "projections",
            expected=expected,
            actual=shape,
            fix=(
                "use detector metadata matching the projection detector axes, or reshape/reload "
                "projections as (n_views, detector.nv, detector.nu)."
            ),
        )
    return n_views, nv, nu


def validate_projection_stack(
    projections: Any,
    detector: Detector,
    *,
    geometry: Geometry | None = None,
    context: str,
) -> tuple[int, int, int]:
    """Validate projections with expected shape ``(n_views, detector.nv, detector.nu)``."""
    return validate_projection_shape(
        _shape_of(projections),
        detector,
        geometry=geometry,
        context=context,
    )


def validate_volume(
    volume: Any,
    grid: Grid,
    *,
    context: str,
    name: str = "volume",
) -> tuple[int, int, int]:
    expected_shape = validate_grid(grid, context)
    actual_shape = _shape_of(volume)
    if actual_shape != expected_shape:
        _raise_shape_error(
            context,
            name,
            expected=f"(nx, ny, nz)={expected_shape} from grid",
            actual=actual_shape,
            fix="use a volume/init_x/support array generated for the same reconstruction grid.",
        )
    return expected_shape


def validate_detector_image(
    image: Any,
    detector: Detector,
    *,
    context: str,
    name: str = "image",
) -> tuple[int, int]:
    expected_shape = validate_detector(detector, context)
    actual_shape = _shape_of(image)
    if actual_shape != expected_shape:
        _raise_shape_error(
            context,
            name,
            expected=f"(nv, nu)={expected_shape} from detector",
            actual=actual_shape,
            fix="use an image generated with the same detector metadata.",
        )
    return expected_shape


def validate_pose_matrix(T: Any, *, context: str, name: str = "pose") -> tuple[int, int]:
    actual_shape = _shape_of(T)
    expected_shape = (4, 4)
    if actual_shape != expected_shape:
        _raise_shape_error(
            context,
            name,
            expected="(4, 4) homogeneous world_from_object transform",
            actual=actual_shape,
            fix="return or pass one 4x4 pose matrix for each view.",
        )
    return expected_shape


def validate_pose_stack(
    T_all: Any,
    n_views: int,
    *,
    context: str,
    name: str = "poses",
) -> tuple[int, int, int]:
    expected_shape = (int(n_views), 4, 4)
    actual_shape = _shape_of(T_all)
    if actual_shape != expected_shape:
        _raise_shape_error(
            context,
            name,
            expected=f"(n_views, 4, 4)={expected_shape}",
            actual=actual_shape,
            fix="stack exactly one 4x4 pose matrix per projection view.",
        )
    return expected_shape


def validate_optional_same_shape(
    value: Any | None,
    expected_shape: Iterable[int],
    *,
    context: str,
    name: str,
    fix: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    expected = tuple(int(s) for s in expected_shape)
    actual = _shape_of(value)
    if actual != expected:
        _raise_shape_error(
            context,
            name,
            expected=_shape_text(expected),
            actual=actual,
            fix=fix,
        )
    return expected


def validate_optional_broadcastable_shape(
    value: Any | None,
    expected_shape: Iterable[int],
    *,
    context: str,
    name: str,
    fix: str,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    expected = tuple(int(s) for s in expected_shape)
    actual = _shape_of(value)
    try:
        broadcast = np.broadcast_shapes(actual, expected)
    except ValueError:
        broadcast = None
    if broadcast != expected:
        _raise_shape_error(
            context,
            name,
            expected=f"broadcastable to {_shape_text(expected)}",
            actual=actual,
            fix=fix,
        )
    return expected


def validate_detector_grid(
    det_grid: tuple[Any, Any] | None,
    detector: Detector,
    *,
    context: str,
) -> None:
    if det_grid is None:
        return
    nv, nu = validate_detector(detector, context)
    expected_shape = (nv * nu,)
    try:
        Xr, Zr = det_grid
    except Exception as exc:
        raise ValueError(
            f"{context}: det_grid has invalid values; expected a pair of detector-grid "
            f"vectors, actual {type(det_grid).__name__}. Likely fix: call "
            "get_detector_grid_device(detector) with matching detector metadata."
        ) from exc
    validate_optional_same_shape(
        Xr,
        expected_shape,
        context=context,
        name="det_grid[0]",
        fix="call get_detector_grid_device(detector) with matching detector metadata.",
    )
    validate_optional_same_shape(
        Zr,
        expected_shape,
        context=context,
        name="det_grid[1]",
        fix="call get_detector_grid_device(detector) with matching detector metadata.",
    )
