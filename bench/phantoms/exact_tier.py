"""The exact tier: TomoJAX's own projector on a voxel truth at the true poses, with seeded noise.

This tier judges the solver and its levels. Here the data are exactly what the model predicts
at the truth (parallel-beam plane models, with the fit's own kernel), so a correct fit recovers
the truth to the noise. Detail a grid cannot represent belongs to the independent tier (gVXR
from meshes, ``corpus.py``), not to this one. The truth must also be gauge-free relative to the
fit's start (:func:`gauge_free`); otherwise its sub-voxel position differs from the one the
fit's gauge holds, and the fit bends its poses to buy sharpness instead of reaching it.

Noise is the corpus's counting model: Poisson at a photon count per pixel, with counts floored
at 0.5 before the log. Each file has its own seed, derived from the draw's seed, the row's name
and the photon count, so no two files share a noise realization.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    import tomojax as tj


def corrections(nominal: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Per-view (alpha, beta, phi, dx, dz, dy) from nominal to ``poses``: ``Scan.with_poses``."""
    local = np.einsum("nji,njk->nik", nominal[:, :3, :3], poses[:, :3, :3])
    table = np.zeros((len(poses), 6))
    table[:, [1, 0, 2]] = Rotation.from_matrix(local).as_euler("YXZ")
    table[:, [3, 5, 4]] = poses[:, :3, 3] - nominal[:, :3, 3]
    return table


def gauge_free(
    poses: np.ndarray,
    start: np.ndarray,
    *,
    pitch: tuple[float, float],
    lever_px: float,
    rotations: bool = False,
) -> np.ndarray:
    """The truth's rigid representative nearest ``start`` in the fit's own scaled coordinates.

    The fit holds its gauge (the rigid object motions the data cannot see) at the start's: no
    component of its correction along those motions. In continuous space the choice is free,
    but on a voxel grid a sub-voxel rigid shift changes how sharply the grid can represent the
    object, so a grid-aligned truth at another gauge cannot be reached. This returns
    ``poses @ T`` for the rigid T minimizing the start-relative correction, translations over
    the pitch and, with ``rotations``, the wobble angles times ``lever_px``. That is the same
    object moved rigidly, and the representative the fit converges to. Rotations are included
    only when the fit fits them; a translations-only fit cannot express a rotated truth.
    """
    from scipy.optimize import least_squares

    def state(candidate: np.ndarray) -> np.ndarray:
        table = corrections(start, candidate)
        columns = [table[:, 3] / pitch[0], table[:, 4] / pitch[1]]
        if rotations:
            columns += [table[:, 0] * lever_px, table[:, 1] * lever_px]
        return np.concatenate(columns)

    def moved(z: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        if rotations:
            transform[:3, :3] = Rotation.from_rotvec(z[3:]).as_matrix()
        transform[:3, 3] = z[:3]
        return poses @ transform

    size = 6 if rotations else 3
    fitted = least_squares(
        lambda z: state(moved(z)), np.zeros(size), xtol=1e-14, ftol=1e-14, gtol=1e-14
    )
    return moved(fitted.x)


def noise_seed(draw_seed: int, row: str, photons: float) -> int:
    """A seed unique to (draw, row, photon count)."""
    key = hashlib.sha256(f"{row}|{photons:.6g}".encode()).digest()
    entropy = [int(draw_seed), *np.frombuffer(key[:16], dtype=np.uint32).tolist()]
    return int(np.random.SeedSequence(entropy).generate_state(1)[0])


def projections(
    grid: tj.Grid,
    detector: tj.Detector,
    truth: np.ndarray,
    poses: np.ndarray,
    *,
    interpolation: Literal["linear", "cubic"] = "linear",
) -> np.ndarray:
    """Noise-free line integrals of ``truth`` (z, y, x) at world-from-object ``poses``.

    ``interpolation`` is the plane kernel, linear or cubic: the tier's data must come from the
    same kernel as the fit that reads them, or the truth no longer reads zero.
    """
    import jax.numpy as jnp

    from tomojax.core.backend_policy import cuda_backend
    from tomojax.forward import project_joseph

    backend = "cuda" if cuda_backend() else "jax"
    out = project_joseph(
        jnp.asarray(truth, jnp.float32),
        jnp.asarray(poses, jnp.float32),
        grid,
        detector,
        backend=backend,
        interpolation=interpolation,
    )
    return np.asarray(out, np.float32)


def counted(line_integrals: np.ndarray, photons: float, seed: int) -> np.ndarray:
    """Poisson counts at ``photons`` per pixel, floored at 0.5, back to line integrals."""
    counts = np.random.default_rng(seed).poisson(
        photons * np.exp(-np.asarray(line_integrals, np.float64))
    )
    return (-np.log(np.maximum(counts, 0.5) / photons)).astype(np.float32)
