"""Geometry with per-view pose updates, for reconstructions inside alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from tomojax.alignment._geometry.parametrizations import (
    PoseTranslationFrame,
    apply_pose_update,
    apply_pose_updates,
)
from tomojax.core.geometry.cone import ConeBeam, beam_of

if TYPE_CHECKING:
    from tomojax.core.geometry import Geometry

_apply_updates = jax.jit(apply_pose_updates, static_argnames=("translation_frame",))


@dataclass(frozen=True, slots=True)
class PoseAdjustedGeometry:
    """Geometry adapter that applies per-view pose offsets during reconstruction."""

    geometry: Geometry
    pose_params: jnp.ndarray
    translation_frame: PoseTranslationFrame = "object"

    def pose_for_view(self, i: int) -> tuple[tuple[jnp.ndarray, ...], ...]:
        """Return the nominal pose composed with the aligned pose update."""
        T_nom = jnp.asarray(self.geometry.pose_for_view(i), dtype=jnp.float32)
        return tuple(
            map(
                tuple,
                apply_pose_update(
                    T_nom, self.pose_params[i], translation_frame=self.translation_frame
                ),
            )
        )

    def stack_poses(self, n_views: int, dtype: jnp.dtype) -> jnp.ndarray:
        """Return the first ``n_views`` poses, updated in one vectorised step."""
        from tomojax.core.geometry.views import stack_view_poses

        nominal = stack_view_poses(self.geometry, n_views, dtype=jnp.float32)
        poses = _apply_updates(
            nominal,
            jnp.asarray(self.pose_params[:n_views]),
            translation_frame=self.translation_frame,
        )
        return poses.astype(dtype)

    def rays_for_view(self, i: int) -> object:
        """Return detector rays for the wrapped geometry view."""
        return self.geometry.rays_for_view(i)

    @property
    def beam(self) -> ConeBeam | None:
        """Return the wrapped geometry's cone beam, if any."""
        return beam_of(self.geometry)
