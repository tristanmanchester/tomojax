"""One geometric adapter for the joint fit's parallel and cone projectors.

The adapter changes ray geometry, never the optimizer. Every image adjoint is
the exact transpose of its forward map, including padded-view masks. Pose
parameters are object-frame rotations followed by lab x/z translations and,
for a cone beam, lab y translation. ``scales`` converts each numerical parameter
to radians or physical lengths; it may vary by view.
"""

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tomojax.core.backend_policy import cuda_backend
from tomojax.core.cone import (
    cone_backproject,
    cone_pose_columns,
    cone_project,
    frame_coefficients,
)
from tomojax.core.geometry.base import Detector
from tomojax.core.joseph import (
    forward_project_planes,
    plane_coefficients,
    plane_pose_columns,
    plane_weighted_diagonal,
    sum_backproject_planes,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class PackedMask:
    """Finite input-pixel membership, packed little-endian along detector u.

    The original detector shape is static PyTree metadata: batching and freezing
    coefficients transform only the bytes, never unpack a whole scan.
    """

    bits: jax.Array
    shape: tuple[int, int] = field(metadata={"static": True})


class Coefficients(NamedTuple):
    """Ray coefficients, padded-view mask and optional finite-input-pixel mask.

    ``pixel_mask`` retains which input pixels entered a coarsened observation.
    It may be coarser than the native ray lattice when each input pixel was
    already a known block mean. Its entries are never inferred from precision.
    """

    rays: jax.Array
    valid: jax.Array
    pixel_mask: PackedMask | None = None


def ray_counts(binning):
    """Validate explicitly known block-mean ray counts, in (u, v) order."""
    counts = np.asarray(binning)
    if (
        counts.shape != (2,)
        or counts.dtype.kind not in "iuf"
        or not np.isfinite(counts).all()
        or np.any(counts < 1)
        or np.any(counts != np.floor(counts))
    ):
        raise ValueError("ray_binning must contain two positive integer counts (u, v)")
    return tuple(map(int, counts))


def posed(nominal, q, scales):
    """Apply scaled 5/6-DOF corrections to world-from-object nominal poses.

    ``q`` has columns alpha, beta, phi, x, z, and optionally y. Rotations are
    ``nominal.R @ Ry(beta) @ Rx(alpha) @ Rz(phi)``. Translations add in lab
    coordinates, preserving nominal beam-direction translation in 5 DOF.
    """
    p = q * jnp.asarray(scales, q.dtype)
    a, b, f = (p[:, k] for k in range(3))
    ca, sa, cb, sb, cf, sf = jnp.cos(a), jnp.sin(a), jnp.cos(b), jnp.sin(b), jnp.cos(f), jnp.sin(f)
    one, zero = jnp.ones_like(a), jnp.zeros_like(a)
    ry = jnp.stack([cb, zero, sb, zero, one, zero, -sb, zero, cb], -1).reshape(-1, 3, 3)
    rx = jnp.stack([one, zero, zero, zero, ca, -sa, zero, sa, ca], -1).reshape(-1, 3, 3)
    rz = jnp.stack([cf, -sf, zero, sf, cf, zero, zero, zero, one], -1).reshape(-1, 3, 3)
    # These small products define rigid geometry. Reduced-mantissa matrix
    # products can turn even nominal @ identity into a non-rigid transform.
    local = jnp.matmul(ry, rx, precision=jax.lax.Precision.HIGHEST)
    local = jnp.matmul(local, rz, precision=jax.lax.Precision.HIGHEST)
    rotation = jnp.matmul(nominal[:, :3, :3], local, precision=jax.lax.Precision.HIGHEST)
    shift = jnp.stack([p[:, 3], p[:, 5] if p.shape[1] == 6 else zero, p[:, 4]], -1)
    out = nominal.at[:, :3, :3].set(rotation.astype(nominal.dtype))
    return out.at[:, :3, 3].add(shift.astype(nominal.dtype))


def rigid_gauge(nominal, scales):
    """Parameter tangents of the six common rigid object-coordinate changes.

    This is the gauge at zero correction. For a right rigid coordinate change
    the pose tangent is ``-T @ generator``. The invisible lab-y translation
    is omitted in the parallel case. ``scales`` is (views, 5/6), or one row.
    """
    nominal = np.asarray(nominal, np.float64)
    scales = np.broadcast_to(np.asarray(scales, np.float64), (len(nominal), np.shape(scales)[-1]))
    n, k = scales.shape
    out = np.zeros((n, k, 6), np.float64)
    out[:, :3, :3] = -np.eye(3)[None] / scales[:, :3, None]
    rows = [0, 2] if k == 5 else [0, 2, 1]
    out[:, 3:, 3:] = -nominal[:, rows, :3] / scales[:, 3:, None]
    return out.reshape(n * k, 6)


def cone_gauge(nominal, scales, frames):
    """Rigid object gauge plus the cone's unobservable common physical scale.

    With an unknown image and free beam-depth poses, scaling the image support
    by a and its density by 1/a, and setting t' = a*t + (1-a)*source preserves
    every line integral. The seventh column fixes this similarity gauge to
    the declared nominal magnification convention; it does not estimate a
    physical object size from the data.
    """
    nominal = np.asarray(nominal, np.float64)
    frames = np.broadcast_to(np.asarray(frames, np.float64), (len(nominal), 4, 3))
    scales = np.broadcast_to(np.asarray(scales, np.float64), (len(nominal), 6))
    dilation = np.zeros((len(nominal), 6))
    dilation[:, 3:] = (nominal[:, [0, 2, 1], 3] - frames[:, 3, [0, 2, 1]]) / scales[:, 3:]
    return np.column_stack([rigid_gauge(nominal, scales), dilation.reshape(-1)])


def cone_parameter_scales(nominal, frames, grid, native):
    """Six physical increments that each move the volume by about one native pixel.

    The geometry-only scale is the reciprocal RMS detector displacement of a
    unit pose increment on the eight volume-cell corners. This treats source
    distance, magnification, detector roll, unequal pitch and beam-depth motion
    in the same units. It is a coordinate scaling, not a precision estimate.
    Every supplied support corner must lie in front of the source.
    """
    from tomojax.core.geometry.base import grid_volume_origin

    poses = np.asarray(nominal, np.float64)
    frames = np.broadcast_to(np.asarray(frames, np.float64), (len(poses), 4, 3))
    spacing = np.asarray([grid.vx, grid.vy, grid.vz])
    lower = np.asarray(grid_volume_origin(grid)) - spacing / 2
    upper = lower + spacing * np.asarray([grid.nx, grid.ny, grid.nz])
    corners = np.stack(np.meshgrid(*zip(lower, upper, strict=True), indexing="ij"), -1).reshape(
        -1, 3
    )
    lab = np.einsum("nij,kj->nki", poses[:, :3, :3], corners) + poses[:, None, :3, 3]
    centre, u, v, source = (frames[:, k] for k in range(4))
    normal = np.cross(u, v)
    ray = lab - source[:, None]
    denominator = np.einsum("nki,ni->nk", ray, normal)
    distance = np.einsum("ni,ni->n", centre - source, normal)
    if np.any(denominator * distance[:, None] <= 0):
        raise ValueError("cone pose scaling requires the volume support in front of the source")
    ray_derivative = (distance[:, None] / denominator)[..., None, None] * (
        np.eye(3) - ray[..., :, None] * normal[:, None, None, :] / denominator[..., None, None]
    )
    detector_derivative = np.stack([u / native[0], v / native[1]], axis=1)
    detector_derivative = detector_derivative[:, None] @ ray_derivative
    rotation_derivative = np.einsum(
        "nij,kaj->nkia", poses[:, :3, :3], np.cross(np.eye(3)[None], corners[:, None])
    )
    translations = np.broadcast_to(np.eye(3)[:, [0, 2, 1]], rotation_derivative.shape)
    tangent = np.concatenate([rotation_derivative, translations], axis=-1)
    displacement = detector_derivative @ tangent
    rms = np.sqrt(np.mean(np.sum(displacement**2, axis=-2), axis=1))
    if not np.isfinite(rms).all() or np.any(rms <= 0):
        raise ValueError("cone pose scaling has an unobservable physical coordinate")
    return 1 / rms


def cropped_frames(frames, detector, *, high_u, high_v):
    """Cone frames after removing high-index detector columns/rows.

    Binning the remaining complete blocks keeps this physical centre. A roll
    or tilt makes this shift follow the frame's u/v axes, not the lab x/z axes.
    The input frame array is never modified.
    """
    out = np.array(frames, dtype=np.float64, copy=True)
    out[..., 0, :] -= 0.5 * (
        high_u * detector.du * out[..., 1, :] + high_v * detector.dv * out[..., 2, :]
    )
    return out


class Projection:
    """Coefficients, fused pose columns and matched image operators for one geometry.

    ``frames`` supplied to ``coefficients`` or ``tangent`` are cone lab frames
    (views, 4, 3): detector centre, unit u, unit v, source. Parallel calls omit
    them. A cone's sixth parameter changes magnification by moving the object
    along lab y; discarding it would silently fit only five of its rigid DOFs.

    ``ray_binning=(bu, bv)`` declares that each observed pixel is the mean of
    that many original point rays. It describes a known digital operation on
    line integrals, not a sensor PSF or an inferred physical pixel aperture.
    The full ray lattice is evaluated in one kernel, then reduced. Its exact
    transpose distributes each residual with the same normalized weights.
    """

    def __init__(
        self,
        grid,
        detector,
        *,
        cone=False,
        backend=None,
        interpolation="linear",
        ray_binning=(1, 1),
    ) -> None:
        if interpolation not in {"linear", "cubic"}:
            raise ValueError("joint plane interpolation must be 'linear' or 'cubic'")
        if cone and interpolation != "linear":
            raise ValueError("the cone projector uses linear interpolation")
        self.grid, self.detector = grid, detector
        self.ray_binning = ray_counts(ray_binning)
        bu, bv = self.ray_binning
        self.ray_detector = Detector(
            detector.nu * bu,
            detector.nv * bv,
            detector.du / bu,
            detector.dv / bv,
            center=detector.center,
        )
        self.cone = bool(cone)
        self.interpolation = interpolation
        self.dofs = 6 if self.cone else 5
        self.backend = backend or ("cuda" if cuda_backend() else "jax")
        self.diagonal_kind = (
            "normal_row_sum"
            if self.cone
            else "mean_linear_ray_diagonal_proxy"
            if self.ray_binning != (1, 1)
            else "linear_basis_diagonal_proxy"
            if interpolation == "cubic"
            else "normal_diagonal"
        )

    def raw_coefficients(self, poses, frames=None):
        """Unmasked core coefficients; their pose derivative excludes the mask."""
        if self.cone:
            return frame_coefficients(poses, frames, self.grid, self.ray_detector)
        return plane_coefficients(poses, self.grid, self.ray_detector)

    def coefficients(self, poses, valid, frames=None, pixel_mask=None):
        """Keep padding as an explicit mask; cone ray coordinates stay physical."""
        if pixel_mask is not None:
            nv, nu = pixel_mask.shape
            if (
                pixel_mask.bits.shape != (len(poses), nv, (nu + 7) // 8)
                or min(nv, nu) < 1
                or nv % self.detector.nv
                or nu % self.detector.nu
                or self.ray_detector.nv % nv
                or self.ray_detector.nu % nu
            ):
                raise ValueError(
                    "pixel_mask must describe complete input blocks on the ray lattice"
                )
        return Coefficients(
            self.raw_coefficients(poses, frames), jnp.asarray(valid, jnp.bool_), pixel_mask
        )

    def tangent(self, nominal, q, scales, frames=None):
        """Exact coefficient derivatives, (views, dofs, coefficients), at ``q``."""
        scales = jnp.broadcast_to(jnp.asarray(scales, q.dtype), q.shape)

        def one(n, p, scale, frame):
            return self.raw_coefficients(posed(n[None], p[None], scale), frame)[0]

        columns = jax.vmap(
            jax.jacfwd(one, argnums=1), in_axes=(0, 0, 0, None if frames is None else 0)
        )(nominal, q, scales, frames)
        return jnp.swapaxes(columns, 1, 2)

    def _ray_weights(self, cf):
        """Per-ray coefficients of the block mean, summing to one if observed."""
        bu, bv = self.ray_binning
        if cf.pixel_mask is None:
            return jnp.float32(1 / (bu * bv))
        n = cf.pixel_mask.bits.shape[0]
        nv, nu = cf.pixel_mask.shape
        mv, mu = nv // self.detector.nv, nu // self.detector.nu
        mask = jnp.unpackbits(cf.pixel_mask.bits, axis=-1, count=nu, bitorder="little").astype(
            jnp.float32
        )
        blocks = mask.reshape(n, self.detector.nv, mv, self.detector.nu, mu)
        count = jnp.sum(blocks, axis=(2, 4), keepdims=True)
        # A finite input pixel is itself the mean of bu/mu by bv/mv native
        # rays. All those rays get the same weight; a missing input gets zero.
        normalized = blocks / jnp.maximum(count * ((bu // mu) * (bv // mv)), 1)
        normalized = normalized.reshape(mask.shape)
        return jnp.repeat(jnp.repeat(normalized, bv // mv, axis=-2), bu // mu, axis=-1)

    def _reduce(self, image, cf):
        """Average the last two detector axes, allowing a pose-column axis."""
        weights = self._ray_weights(cf)
        if jnp.ndim(weights) and image.ndim == 4:
            weights = weights[:, None]
        weighted = image * weights
        bu, bv = self.ray_binning
        shape = (*image.shape[:-2], self.detector.nv, bv, self.detector.nu, bu)
        return jnp.sum(weighted.reshape(shape), axis=(-3, -1))

    def _expand(self, image, cf, *, diagonal_proxy=False):
        """Transpose of the mean, or sqrt weights for the positive diagonal proxy."""
        bu, bv = self.ray_binning
        expanded = jnp.repeat(jnp.repeat(image, bv, axis=-2), bu, axis=-1)
        weights = self._ray_weights(cf)
        return expanded * (jnp.sqrt(weights) if diagonal_proxy else weights)

    def project(self, cf, x, *, absolute=False):
        """Apply B A, or B |A| for bounds, with the recorded block-mean weights."""
        if self.cone:
            image = cone_project(x, cf.rays, self.grid, self.ray_detector, backend=self.backend)
        else:
            image = forward_project_planes(
                cf.rays,
                x,
                self.grid,
                self.ray_detector,
                backend=self.backend,
                interpolation=self.interpolation,
                absolute_weights=absolute,
            )
        return self._reduce(image, cf) * cf.valid[:, None, None]

    def back(self, accumulator, cf, y, *, absolute=False):
        """Accumulate A.T B.T (or |A|.T B.T), matching every mean and mask."""
        y = self._expand(y * cf.valid[:, None, None], cf)
        if self.cone:
            return cone_backproject(
                y,
                cf.rays,
                self.grid,
                self.ray_detector,
                backend=self.backend,
                accumulate=accumulator,
            )
        return sum_backproject_planes(
            cf.rays,
            y,
            self.grid,
            self.ray_detector,
            backend=self.backend,
            interpolation=self.interpolation,
            absolute_weights=absolute,
            accumulate=accumulator,
        )

    def columns(self, cf, x, tangent):
        """Prediction and its pose columns, (views, 1 + dofs, v, u)."""
        if not self.cone:
            images = plane_pose_columns(
                cf.rays,
                x,
                tangent,
                self.grid,
                self.ray_detector,
                backend=self.backend,
                interpolation=self.interpolation,
            )
        elif self.backend == "cuda":
            images = cone_pose_columns(
                x, cf.rays, tangent=tangent[..., :12], grid=self.grid, detector=self.ray_detector
            )
        else:

            def predict(coeff):
                return cone_project(x, coeff, self.grid, self.ray_detector, backend="jax")

            columns = jax.vmap(
                lambda dc: jax.jvp(predict, (cf.rays,), (dc,))[1], in_axes=1, out_axes=1
            )(tangent)
            images = jnp.concatenate([predict(cf.rays)[:, None], columns], axis=1)
        return self._reduce(images, cf) * cf.valid[:, None, None, None]

    def diagonal(self, cf, sqrtweights, accumulator):
        """Accumulate a positive image preconditioner, preserving the image operator.

        Linear parallel rays use exact ``diag(A.T W A)``. Cubic rays use the
        corresponding linear-basis diagonal as a positive preconditioner proxy.
        Binned rays use the normalized mean of those linear-ray diagonals;
        this omits cross-ray terms and is only a preconditioner, never a bound
        or the exact normal diagonal. Cone rays use a conservative row-sum
        majorizer. Normal matvecs keep all cross terms through the matched
        forward and transpose. ``diagonal_kind`` records the choice.
        """
        weights = sqrtweights * cf.valid[:, None, None]
        if not self.cone:
            return plane_weighted_diagonal(
                cf.rays,
                self._expand(weights, cf, diagonal_proxy=True),
                accumulator,
                self.grid,
                self.ray_detector,
                backend=self.backend,
            )
        return self.majorizer(cf, weights, accumulator)

    def majorizer(self, cf, sqrtweights, accumulator):
        """Accumulate C.T W C 1 for C = B |A|, bounding (B A).T W B A.

        Cubic interpolation has signed lobes. Using A.T W A 1 instead would
        allow cancellation and would not bound the initialization step. The
        positive block-mean map B preserves the bound, including correlations
        between rays in a block. Cone and linear-plane kernels are nonnegative.
        """
        ones = jnp.ones((self.grid.nz, self.grid.ny, self.grid.nx), jnp.float32)
        images = sqrtweights**2 * self.project(cf, ones, absolute=True)
        return self.back(accumulator, cf, images, absolute=True)
