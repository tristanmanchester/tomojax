from __future__ import annotations

import jax
import jax.numpy as jnp

from tomojax.core.geometry.base import Detector, Grid
from tomojax.core.projector import _projector_traversal_state

from ._pallas_config import (
    _KERNEL_VARIANT_IDS,
    _LAYOUT_VARIANT_IDS,
    PallasForwardProjectorStackTraversalState,
    PallasProjectorTraversalMetadata,
    PallasProjectorUnsupported,
    _normalize_gather_dtype,
    _unsupported,
    _validate_public_sinogram_call,
)


def prepare_forward_project_views_T_pallas_state(
    T_stack: jnp.ndarray,
    grid: Grid,
    detector: Detector,
    *,
    step_size: float | None = None,
    n_steps: int | None = None,
    gather_dtype: str = "fp32",
    det_grid: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    tile_shape: tuple[int, int] = (8, 16),
    num_warps: int = 4,
    kernel_variant: str = "auto",
    layout_variant: str = "detector_vu",
) -> PallasForwardProjectorStackTraversalState:
    """Prepare fixed pose-stack traversal state for repeated Pallas sinogram calls."""
    (
        nx,
        ny,
        nz,
        nv,
        nu,
        n_views,
        _volume_size,
        step_size_value,
        resolved_n_steps,
        effective_n_steps,
        (tile_v, tile_u),
        num_warps_value,
        kernel_variant_id,
        layout_variant_id,
    ) = _validate_public_sinogram_call(
        T_stack,
        grid,
        detector,
        jnp.zeros((grid.nx, grid.ny, grid.nz), dtype=jnp.float32),
        step_size=step_size,
        n_steps=n_steps,
        gather_dtype=gather_dtype,
        det_grid=det_grid,
        interpret=True,
        tile_shape=tile_shape,
        num_warps=num_warps,
        kernel_variant=kernel_variant,
        layout_variant=layout_variant,
        state_mode="cached",
    )
    if layout_variant_id != _LAYOUT_VARIANT_IDS["detector_vu"]:
        raise PallasProjectorUnsupported(
            _unsupported("cached pose-stack traversal state supports detector_vu only")
        )

    T = jnp.asarray(T_stack, dtype=jnp.float32)
    ix_all: list[jnp.ndarray] = []
    iy_all: list[jnp.ndarray] = []
    iz_all: list[jnp.ndarray] = []
    n_steps_ray_all: list[jnp.ndarray] = []
    dix_all: list[jnp.ndarray] = []
    diy_all: list[jnp.ndarray] = []
    diz_all: list[jnp.ndarray] = []
    for view in range(int(n_views)):
        ix0, iy0, iz0, dix, diy, diz, n_steps_ray, _step32, _resolved, n_rays = (
            _projector_traversal_state(
                T[view],
                grid,
                detector,
                step_size=step_size_value,
                n_steps=n_steps,
                det_grid=det_grid,
            )
        )
        expected_rays = int(nv) * int(nu)
        if int(n_rays) != expected_rays:
            raise PallasProjectorUnsupported(
                _unsupported(f"cached traversal state expected {expected_rays} rays; got {n_rays}")
            )
        ix_all.append(jnp.ravel(jnp.asarray(ix0, dtype=jnp.float32), order="C"))
        iy_all.append(jnp.ravel(jnp.asarray(iy0, dtype=jnp.float32), order="C"))
        iz_all.append(jnp.ravel(jnp.asarray(iz0, dtype=jnp.float32), order="C"))
        n_steps_ray_all.append(jnp.ravel(jnp.asarray(n_steps_ray, dtype=jnp.int32), order="C"))
        dix_all.append(jnp.ravel(jnp.asarray(dix, dtype=jnp.float32), order="C")[0])
        diy_all.append(jnp.ravel(jnp.asarray(diy, dtype=jnp.float32), order="C")[0])
        diz_all.append(jnp.ravel(jnp.asarray(diz, dtype=jnp.float32), order="C")[0])

    return PallasForwardProjectorStackTraversalState(
        traversal=PallasProjectorTraversalMetadata(
            ix0=jnp.concatenate(ix_all, axis=0),
            iy0=jnp.concatenate(iy_all, axis=0),
            iz0=jnp.concatenate(iz_all, axis=0),
            n_steps_ray=jnp.concatenate(n_steps_ray_all, axis=0),
            step_size=float(step_size_value),
            n_steps=int(effective_n_steps),
            resolved_n_steps=int(resolved_n_steps),
            nx=int(nx),
            ny=int(ny),
            nz=int(nz),
            nv=int(nv),
            nu=int(nu),
            tile_shape=(int(tile_v), int(tile_u)),
            num_warps=int(num_warps_value),
            kernel_variant="generic"
            if kernel_variant_id == _KERNEL_VARIANT_IDS["generic"]
            else "z_integer4",
            kernel_variant_id=int(kernel_variant_id),
            gather_dtype=_normalize_gather_dtype(gather_dtype),
        ),
        dix=jnp.asarray(dix_all, dtype=jnp.float32),
        diy=jnp.asarray(diy_all, dtype=jnp.float32),
        diz=jnp.asarray(diz_all, dtype=jnp.float32),
        n_views=int(n_views),
    )


def block_forward_project_views_T_pallas_state(
    state: PallasForwardProjectorStackTraversalState,
) -> PallasForwardProjectorStackTraversalState:
    """Block until prepared stack traversal-state arrays are materialized."""
    jax.block_until_ready((state.ix0, state.iy0, state.iz0, state.n_steps_ray, state.dix))
    return state
