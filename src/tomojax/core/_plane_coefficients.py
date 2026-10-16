"""Affine Joseph plane geometry shared by JAX and host CUDA execution."""

from typing import Any

from .geometry.base import Detector, Grid, grid_volume_origin


def coefficients(poses, grid: Grid, detector: Detector, det_grid=None, *, xp: Any) -> Any:
    voxel = xp.asarray((grid.vx, grid.vy, grid.vz), xp.float32)
    origin = xp.asarray(grid_volume_origin(grid), xp.float32)
    rotation = poses[:, :3, :3]
    if det_grid is None:
        u0 = detector.center[0] - (detector.nu - 1) * detector.du / 2
        v0 = detector.center[1] - (detector.nv - 1) * detector.dv / 2
        first = xp.asarray([u0, 0.0, v0], xp.float32)
        u, v = rotation[:, 0, :] * detector.du / voxel, rotation[:, 2, :] * detector.dv / voxel
    else:
        xs, zs = (xp.asarray(a, xp.float32).ravel() for a in det_grid)
        nu = detector.nu
        first = xp.stack([xs[0], xp.float32(0), zs[0]])
        step_u = (xs[1] - xs[0], zs[1] - zs[0]) if nu > 1 else (detector.du, 0.0)
        step_v = (xs[nu] - xs[0], zs[nu] - zs[0]) if detector.nv > 1 else (0.0, detector.dv)
        # A singleton dimension has no measured step; complete the lattice
        # perpendicular to the other axis so gather coordinates stay invertible.
        if nu == 1 and detector.nv > 1:
            ratio = detector.du / detector.dv
            step_u = (step_v[1] * ratio, -step_v[0] * ratio)
        elif detector.nv == 1 and nu > 1:
            ratio = detector.dv / detector.du
            step_v = (-step_u[1] * ratio, step_u[0] * ratio)
        u = (step_u[0] * rotation[:, 0, :] + step_u[1] * rotation[:, 2, :]) / voxel
        v = (step_v[0] * rotation[:, 0, :] + step_v[1] * rotation[:, 2, :]) / voxel
    base = xp.sum(rotation * (first[None, :] - poses[:, :3, 3])[:, :, None], axis=1)
    base = (base - origin) / voxel
    direction = poses[:, 1, :3] / voxel
    axis = xp.argmax(xp.abs(direction), axis=1)
    b, c = (axis + 1) % 3, (axis + 2) % 3

    def take(values, index):
        return xp.take_along_axis(values, index[:, None], axis=1)[:, 0]

    da = take(direction, axis)
    kb, kc = take(direction, b) / da, take(direction, c) / da
    ub, vb = take(u, b) - kb * take(u, axis), take(v, b) - kb * take(v, axis)
    uc, vc = take(u, c) - kc * take(u, axis), take(v, c) - kc * take(v, axis)
    cb, cc = take(base, b) - kb * take(base, axis), take(base, c) - kc * take(base, axis)
    determinant = ub * vc - vb * uc
    return xp.stack(
        [
            axis.astype(xp.float32),
            ub,
            vb,
            kb,
            cb,
            uc,
            vc,
            kc,
            cc,
            1 / xp.abs(da),
            vc / determinant,
            -vb / determinant,
            -uc / determinant,
            ub / determinant,
        ],
        axis=1,
    )
