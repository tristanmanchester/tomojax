"""The exact tier: TomoJAX's own projector on a voxel truth at the true poses, with seeded noise.

This tier judges the solver and its levels. Here the data are exactly what the model predicts
at the truth (parallel-beam plane models, with the fit's own kernel), so a correct fit recovers
the truth to the noise. Detail a grid cannot represent belongs to the independent tier (gVXR
from meshes, ``corpus.py``), not to this one.

Noise is the corpus's counting model: Poisson at a photon count per pixel, with counts floored
at 0.5 before the log. Each file has its own seed, derived from the draw's seed, the row's name
and the photon count, so no two files share a noise realisation.
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
