"""Active pose coordinates and the shared calibration border.

The optimizer stores local coordinates once per view and global coordinates
once per scan. Projector batches receive rows with the global coordinates
broadcast; scores apply the transpose of that broadcast. Held parameters do
not have columns or entries in the linear system.
"""

import jax
import jax.numpy as jnp
import numpy as np

from ._projection import posed

_NAMES = ("centre", "tilt", "roll", "translations", "rotations")


def parse_fit(fit):
    """Validate the public parameter selection and put it in a stable order."""
    names = (fit,) if isinstance(fit, str) else fit
    if not isinstance(names, tuple) or not names or any(not isinstance(n, str) for n in names):
        raise ValueError("fit must be a name or a nonempty tuple of parameter names")
    if len(set(names)) != len(names) or any(n not in _NAMES for n in names):
        raise ValueError(
            "fit names are centre, tilt, roll, translations and rotations; no duplicates"
        )
    return tuple(n for n in _NAMES if n in names)


class Parameters:
    """One compact state and its per-view projector coordinates.

    Local columns retain the full pose table's order: alpha, beta, phi, x, z,
    and y for a cone. Global columns are centre u/v, tilt and roll, restricted
    to ``fit``. Centre coordinates are native pixels. Global angles use
    ``lever`` pixels per radian, independently of the working resolution.
    """

    def __init__(self, views, fit, *, cone=False, lever=1.0) -> None:
        self.n, self.fit, self.cone, self.lever = views, parse_fit(fit), bool(cone), float(lever)
        self.full_dofs = 6 if cone else 5
        self.local = (tuple(range(3)) if "rotations" in self.fit else ()) + (
            tuple(range(3, self.full_dofs)) if "translations" in self.fit else ()
        )
        self.globals = (
            (("centre_u", "centre_v") if cone else ("centre_u",)) if "centre" in self.fit else ()
        ) + tuple(name for name in ("tilt", "roll") if name in self.fit)
        self.nlocal, self.nglobal = len(self.local), len(self.globals)
        self.k = self.nlocal + self.nglobal
        self.local_size = self.n * self.nlocal
        self.size = self.local_size + self.nglobal

    def expand(self, state):
        """Broadcast the shared border into ``(views, columns)`` projector rows."""
        state = jnp.asarray(state).reshape(self.size)
        local = state[: self.local_size].reshape(self.n, self.nlocal)
        shared = jnp.broadcast_to(state[self.local_size :], (self.n, self.nglobal))
        return jnp.concatenate([local, shared], axis=1)

    def reduce_score(self, rows):
        """The exact transpose of :meth:`expand`, including a sum over shared columns."""
        rows = jnp.asarray(rows).reshape(self.n, self.k)
        return jnp.concatenate(
            [rows[:, : self.nlocal].reshape(-1), jnp.sum(rows[:, self.nlocal :], axis=0)]
        )

    def physical(self, state, scales):
        """Return local poses in radians/lengths and global pixel/degree corrections."""
        state = np.asarray(state, np.float64).reshape(self.size)
        table = np.zeros((self.n, 6), np.float64)
        scale = np.broadcast_to(np.asarray(scales), (self.n, self.full_dofs))
        if self.nlocal:
            table[:, self.local] = (
                state[: self.local_size].reshape(self.n, self.nlocal) * scale[:, self.local]
            )
        shared = dict(zip(self.globals, state[self.local_size :], strict=True))
        corrections = {}
        if "centre_u" in shared:
            corrections["centre"] = (
                (float(shared["centre_u"]), float(shared["centre_v"]))
                if self.cone
                else float(shared["centre_u"])
            )
        for name in ("tilt", "roll"):
            if name in shared:
                corrections[name] = float(np.rad2deg(shared[name] / self.lever))
        return table, corrections

    def geometry(
        self,
        nominal,
        rows,
        scales,
        *,
        detector,
        native,
        frames=None,
        detector_roll_deg=0.0,
        detector_pivot=None,
    ):
        """Map parameter rows to physical poses and detector frames.

        Positive tilt leans the rotation axis towards the beam: a lab-frame
        ``Rx(-tilt)`` applied to the nominal orientation. Positive roll rotates
        detector u towards v. Detector centre increments are lab x/z lengths,
        matching ``Detector.center``; they never move a cone's source.

        Parallel rays use the inverse detector-frame transform on the poses,
        which is exactly the same ray map as a rolled detector lattice. The
        pivot is the original detector centre, including when high-side pixels
        were cropped before binning. Cone frames retain their source and move
        their actual detector centre and pixel directions.
        """
        full = jnp.zeros((len(rows), self.full_dofs), rows.dtype)
        if self.nlocal:
            full = full.at[:, jnp.asarray(self.local)].set(rows[:, : self.nlocal])
        shared = {name: rows[:, self.nlocal + i] for i, name in enumerate(self.globals)}
        zero = jnp.zeros(len(rows), rows.dtype)
        if "tilt" in shared:
            angle = shared["tilt"] / self.lever
            c, s, one = jnp.cos(angle), jnp.sin(angle), jnp.ones_like(angle)
            rx = jnp.stack([one, zero, zero, zero, c, s, zero, -s, c], axis=-1).reshape(-1, 3, 3)
            nominal = nominal.at[:, :3, :3].set(
                jnp.matmul(rx, nominal[:, :3, :3], precision=jax.lax.Precision.HIGHEST)
            )
        poses = posed(nominal, full, scales)
        centre_u = shared.get("centre_u", zero) * native[0]
        centre_v = shared.get("centre_v", zero) * native[1]
        angle = shared.get("roll", zero) / self.lever
        pivot = detector.center if detector_pivot is None else detector_pivot
        if not any(name in shared for name in ("centre_u", "centre_v", "roll")) and (
            frames is not None or detector_roll_deg == 0
        ):
            return poses, frames
        if frames is not None:
            frame = jnp.broadcast_to(frames, (len(rows), 4, 3))
            u, v = frame[:, 1], frame[:, 2]
            c, s = jnp.cos(angle)[:, None], jnp.sin(angle)[:, None]
            new_u, new_v = c * u + s * v, -s * u + c * v
            offset_u, offset_v = detector.center[0] - pivot[0], detector.center[1] - pivot[1]
            centre = frame[:, 0] + (new_u - u) * offset_u + (new_v - v) * offset_v
            centre += jnp.stack([centre_u, zero, centre_v], axis=-1)
            return poses, frame.at[:, 0].set(centre).at[:, 1].set(new_u).at[:, 2].set(new_v)
        angle += jnp.deg2rad(jnp.asarray(detector_roll_deg, rows.dtype))
        c, s, one = jnp.cos(angle), jnp.sin(angle), jnp.ones_like(angle)
        inverse_roll = jnp.stack([c, zero, s, zero, one, zero, -s, zero, c], axis=-1).reshape(
            -1, 3, 3
        )
        centre = jnp.stack([pivot[0] + centre_u, zero, pivot[1] + centre_v], axis=-1)
        translation = jnp.einsum(
            "nij,nj->ni",
            inverse_roll,
            poses[:, :3, 3] - centre,
            precision=jax.lax.Precision.HIGHEST,
        ) + jnp.asarray([pivot[0], 0.0, pivot[1]], rows.dtype)
        poses = poses.at[:, :3, :3].set(
            jnp.matmul(inverse_roll, poses[:, :3, :3], precision=jax.lax.Precision.HIGHEST)
        )
        return poses.at[:, :3, 3].set(translation), None
