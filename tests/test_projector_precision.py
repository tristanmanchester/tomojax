import numpy as np
import jax.numpy as jnp
from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view


def test_bf16_gather_reasonable_accuracy():
    grid = Grid(nx=16, ny=16, nz=16, vx=1.0, vy=1.0, vz=1.0)
    det = Detector(nu=16, nv=16, du=1.0, dv=1.0, det_center=(0.0, 0.0))
    geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=[0.0])
    vol = jnp.ones((16, 16, 16), dtype=jnp.float32)
    p32 = forward_project_view(geom, grid, det, vol, view_index=0, gather_dtype="fp32")
    pb = forward_project_view(geom, grid, det, vol, view_index=0, gather_dtype="bf16")
    # Relative error bounded (bf16 gather but fp32 accumulation)
    num = jnp.linalg.norm((p32 - pb).ravel())
    den = jnp.linalg.norm(p32.ravel()) + 1e-6
    assert float(num / den) < 1e-3

