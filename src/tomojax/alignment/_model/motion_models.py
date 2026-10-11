"""Low-dimensional pose trajectory models for alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from jax import lax
import jax.numpy as jnp
import numpy as np
import scipy.linalg

from .dofs import DOF_INDEX

if TYPE_CHECKING:
    from collections.abc import Sequence


type PoseModelName = Literal["per_view", "polynomial", "spline"]


@dataclass(frozen=True)
class PoseMotionModel:
    """Low-dimensional linear model for per-view pose parameters.

    The per-view model is the identity, so it keeps no basis (``None``): a
    dense views-by-views matrix and its pseudo-inverse would cost quadratic
    memory and cubic time.
    """

    name: PoseModelName
    basis: jnp.ndarray | None
    basis_pinv: jnp.ndarray | None
    active_indices: tuple[int, ...]
    active_names: tuple[str, ...]
    frozen_pose_params: jnp.ndarray

    @property
    def n_views(self) -> int:
        """Return the number of modeled views."""
        return int(self.frozen_pose_params.shape[0])

    @property
    def n_basis(self) -> int:
        """Return the number of basis coefficients per active DOF."""
        return self.n_views if self.basis is None else int(self.basis.shape[1])

    @property
    def n_active(self) -> int:
        """Return the number of active pose DOFs."""
        return len(self.active_indices)

    @property
    def variable_count(self) -> int:
        """Return the number of optimized model coefficients."""
        return self.n_basis * self.n_active

    @property
    def per_view_variable_count(self) -> int:
        """Return equivalent unconstrained per-view variable count."""
        return self.n_views * self.n_active


def scan_coordinate_from_geometry(geometry: object, n_views: int) -> np.ndarray:
    """Return a normalized scan coordinate for pose models.

    Use geometry angles when they are present and monotonic; otherwise fall back
    to the view index. The returned coordinate is always in [-1, 1].
    """
    coord: np.ndarray | None = None
    raw_thetas = getattr(geometry, "angles", None)
    if raw_thetas is not None:
        try:
            arr = np.asarray(raw_thetas, dtype=np.float64)
            if arr.shape == (int(n_views),) and np.all(np.isfinite(arr)):
                diffs = np.diff(arr)
                if arr.size <= 1 or np.all(diffs > 0.0):
                    coord = arr
        except Exception:
            coord = None
    if coord is None:
        coord = np.arange(int(n_views), dtype=np.float64)
    return _normalize_coordinate(coord)


def build_pose_motion_model(
    *,
    pose_model: str,
    n_views: int,
    active_dofs: Sequence[str],
    frozen_pose_params: jnp.ndarray,
    scan_coordinate: np.ndarray | None = None,
    knot_spacing: int = 8,
    degree: int = 3,
) -> PoseMotionModel:
    """Build a linear basis model for alignment parameter trajectories."""
    name = _normalize_pose_model(pose_model)
    if int(n_views) <= 0:
        raise ValueError("n_views must be positive for pose motion models")
    active_names = tuple(str(name) for name in active_dofs)
    active_indices = tuple(DOF_INDEX[name] for name in active_names)
    coord = (
        _normalize_coordinate(np.arange(int(n_views), dtype=np.float64))
        if scan_coordinate is None
        else _normalize_coordinate(np.asarray(scan_coordinate, dtype=np.float64))
    )
    if coord.shape != (int(n_views),):
        raise ValueError(f"scan_coordinate must have shape ({int(n_views)},), got {coord.shape}")

    basis = basis_pinv = None
    if name != "per_view":
        basis_np = build_pose_basis(
            name,
            n_views=int(n_views),
            scan_coordinate=coord,
            knot_spacing=int(knot_spacing),
            degree=int(degree),
        )
        basis = jnp.asarray(basis_np, dtype=jnp.float32)
        basis_pinv = jnp.asarray(_pseudo_inverse(basis_np, name), dtype=jnp.float32)
    return PoseMotionModel(
        name=name,
        basis=basis,
        basis_pinv=basis_pinv,
        active_indices=active_indices,
        active_names=active_names,
        frozen_pose_params=jnp.asarray(frozen_pose_params, dtype=jnp.float32),
    )


def build_pose_basis(
    pose_model: PoseModelName,
    *,
    n_views: int,
    scan_coordinate: np.ndarray,
    knot_spacing: int,
    degree: int,
) -> np.ndarray:
    """Return a basis matrix with shape (n_views, n_basis)."""
    if pose_model == "per_view":
        return np.eye(int(n_views), dtype=np.float32)
    if pose_model == "polynomial":
        if int(degree) < 0:
            raise ValueError("degree must be >= 0 for polynomial pose_model")
        n_basis = min(int(degree) + 1, int(n_views))
        cols = [np.power(scan_coordinate, p) for p in range(n_basis)]
        return np.stack(cols, axis=1).astype(np.float32)
    if pose_model == "spline":
        if int(knot_spacing) < 1:
            raise ValueError("knot_spacing must be >= 1 for spline pose_model")
        if int(degree) not in (1, 2, 3):
            raise ValueError("degree must be one of 1, 2, or 3 for spline pose_model")
        return _spline_interpolation_basis(
            scan_coordinate,
            knot_spacing=int(knot_spacing),
            degree=int(degree),
        )
    raise ValueError(f"Unsupported pose_model: {pose_model!r}")


def _pseudo_inverse(basis: np.ndarray, pose_model: PoseModelName) -> np.ndarray:
    """The least-squares fit matrix of a basis, as float32.

    A spline basis interpolates its knots, so its Gram matrix is small and
    well conditioned, and solving it is far cheaper than an SVD of the tall
    basis.
    """
    values = basis.astype(np.float64)
    if pose_model == "spline":
        return scipy.linalg.solve(values.T @ values, values.T, assume_a="pos").astype(np.float32)
    return np.linalg.pinv(values).astype(np.float32)


def fit_motion_coefficients(model: PoseMotionModel, pose_params: jnp.ndarray) -> jnp.ndarray:
    """Fit active-DOF model coefficients from an expanded per-view array."""
    active_params = pose_params[:, jnp.asarray(model.active_indices, dtype=jnp.int32)]
    if model.basis_pinv is None:
        return active_params
    return jnp.matmul(model.basis_pinv, active_params, precision=lax.Precision.HIGHEST)


def expand_motion_coefficients(model: PoseMotionModel, coeffs: jnp.ndarray) -> jnp.ndarray:
    """Expand coefficient variables back to a per-view (n_views, 5) array."""
    if model.basis is None:
        active_params = coeffs
    else:
        active_params = jnp.matmul(model.basis, coeffs, precision=lax.Precision.HIGHEST)
    pose_params = jnp.asarray(model.frozen_pose_params, dtype=jnp.float32)
    active_indices = jnp.asarray(model.active_indices, dtype=jnp.int32)
    return pose_params.at[:, active_indices].set(active_params)


def _normalize_pose_model(raw: str) -> PoseModelName:
    value = str(raw).strip().lower()
    if value in {"per-view", "perview"}:
        value = "per_view"
    if value not in {"per_view", "polynomial", "spline"}:
        raise ValueError("pose_model must be one of 'per_view', 'polynomial', or 'spline'")
    return cast("PoseModelName", value)


def _normalize_coordinate(coord: np.ndarray) -> np.ndarray:
    arr = np.asarray(coord, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"scan coordinate must be one-dimensional, got {arr.shape}")
    if arr.size == 1:
        return np.zeros_like(arr, dtype=np.float64)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    span = hi - lo
    if span <= 0.0:
        return np.linspace(-1.0, 1.0, arr.size, dtype=np.float64)
    return (2.0 * ((arr - lo) / span) - 1.0).astype(np.float64)


def _spline_interpolation_basis(
    scan_coordinate: np.ndarray,
    *,
    knot_spacing: int,
    degree: int,
) -> np.ndarray:
    n_views = int(scan_coordinate.shape[0])
    if n_views == 1:
        return np.ones((1, 1), dtype=np.float32)

    knot_indices = list(range(0, n_views, int(knot_spacing)))
    if knot_indices[-1] != n_views - 1:
        knot_indices.append(n_views - 1)
    knot_indices_np = np.asarray(knot_indices, dtype=np.int64)
    knot_x = np.asarray(scan_coordinate[knot_indices_np], dtype=np.float64)
    n_knots = int(knot_x.size)
    if n_knots == 1:
        return np.ones((n_views, 1), dtype=np.float32)

    eff_degree = min(int(degree), n_knots - 1)
    if eff_degree <= 1:
        return _linear_interpolation_basis(scan_coordinate, knot_x)

    try:
        from scipy.interpolate import make_interp_spline
    except Exception as exc:
        raise RuntimeError(
            "Failed to build spline pose_model: scipy.interpolate.make_interp_spline "
            "is required for spline degree > 1"
        ) from exc

    # Every cardinal spline shares the knots, so one fit with the identity as
    # its values gives all the columns.
    try:
        spline = make_interp_spline(knot_x, np.eye(n_knots), k=eff_degree)
        return np.asarray(spline(scan_coordinate), dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(
            "Failed to build spline pose_model basis "
            f"(degree={eff_degree}, n_knots={n_knots}, n_views={n_views})"
        ) from exc


def _linear_interpolation_basis(scan_coordinate: np.ndarray, knot_x: np.ndarray) -> np.ndarray:
    """Hat functions on the knots: each view weighs its two neighbouring knots."""
    right = np.clip(np.searchsorted(knot_x, scan_coordinate, side="right"), 1, knot_x.size - 1)
    left = right - 1
    fraction = np.clip((scan_coordinate - knot_x[left]) / (knot_x[right] - knot_x[left]), 0.0, 1.0)
    basis = np.zeros((scan_coordinate.size, knot_x.size), dtype=np.float32)
    rows = np.arange(scan_coordinate.size)
    basis[rows, left] = 1.0 - fraction
    basis[rows, right] = fraction
    return basis


__all__ = [
    "PoseModelName",
    "PoseMotionModel",
    "build_pose_basis",
    "build_pose_motion_model",
    "expand_motion_coefficients",
    "fit_motion_coefficients",
    "scan_coordinate_from_geometry",
]
