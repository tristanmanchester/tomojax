# Geometry

TomoJAX models computed tomography as a geometric relationship between
a 3D voxel grid, a 2D detector, and a set of per-view rigid-body
poses. Understanding these components is essential for setting up
simulations, reconstructions, and alignment workflows.

## The grid

The `Grid` dataclass defines the voxel volume:

```python
from tomojax.core.geometry import Grid

grid = Grid(nx=128, ny=128, nz=128, vx=1.0, vy=1.0, vz=1.0)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `nx`, `ny`, `nz` | `int` | Number of voxels along each axis |
| `vx`, `vy`, `vz` | `float` | Voxel spacing (physical units) |
| `vol_origin` | `tuple` or `None` | Physical location of voxel (0,0,0) center; defaults to centered |
| `vol_center` | `tuple` or `None` | Volume center override |

When `vol_origin` is omitted, TomoJAX uses a centered convention:
voxel centers sit at `(i - (n/2 - 0.5)) * v` along each axis.

## The detector

The `Detector` dataclass describes the flat detector plane:

```python
from tomojax.core.geometry import Detector

det = Detector(nu=128, nv=128, du=1.0, dv=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nu` | `int` | — | Number of detector columns (u direction) |
| `nv` | `int` | — | Number of detector rows (v direction) |
| `du` | `float` | — | Pixel spacing in u |
| `dv` | `float` | — | Pixel spacing in v |
| `det_center` | `tuple` | `(0.0, 0.0)` | Center offset `(cx, cz)` |

## Parallel-beam geometry

`ParallelGeometry` is the standard CT setup. The detector lies in the
(x, z) plane and rays travel along +y. Each view rotates the object
around +z by the scan angle.

```python
import jax.numpy as jnp
from tomojax.core.geometry import ParallelGeometry

thetas = jnp.linspace(0, 180, 200, endpoint=False)
geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
```

The default rotation span is 180 degrees. At `theta=0`, the pose
matrix is the identity.

## Laminography geometry

`LaminographyGeometry` models a tilted rotation axis — the beam
direction stays fixed along +y, but the rotation axis tilts away from
+z by `tilt_deg` degrees.

```python
from tomojax.core.geometry import LaminographyGeometry

thetas = jnp.linspace(0, 360, 360, endpoint=False)
geom = LaminographyGeometry(
    grid=grid,
    detector=det,
    thetas_deg=thetas,
    tilt_deg=35.0,
    tilt_about="x",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tilt_deg` | `float` | `30.0` | Tilt angle in degrees |
| `tilt_about` | `str` | `"x"` | Tilt plane: `"x"` leans toward +y, `"z"` leans toward +x |

Laminography defaults to 360-degree rotation. Reconstructions are
saved in the sample frame, where the object's +z axis equals the
nominal rotation axis.

## 5-DOF rigid-body parameterization

TomoJAX parameterizes per-view alignment with five degrees of freedom.
The full transform for a view is:

```
T · x = R_y(β) · R_x(α) · R_z(φ) · x + t
```

where `t = (Δx, 0, Δz)`.

The five parameters are stored in this order:

| Index | Name | Description | Units |
|-------|------|-------------|-------|
| 0 | `alpha` | Tilt around x-axis | radians |
| 1 | `beta` | Tilt around y-axis | radians |
| 2 | `phi` | In-plane rotation around z-axis | radians |
| 3 | `dx` | Horizontal detector translation | world units |
| 4 | `dz` | Vertical detector translation | world units |

Angles are stored internally in radians but specified in degrees on
the CLI. Translations are stored in world units; the CLI accepts
pixels and converts using detector spacing (`du`, `dv`).

## Volume axis conventions

TomoJAX uses two axis orderings:

- **Internal (compute)**: `(nx, ny, nz)` — all Python API functions
  accept and return volumes in this order.
- **On-disk (HDF5)**: `(nz, ny, nx)` — written with the attribute
  `@volume_axes_order="zyx"` so readers know the layout.

The `load_nxtomo()` function always transposes to internal `(nx, ny,
nz)` order automatically. If you open an `.nxs` file in an external
viewer, slices advance along the z-axis.

> [!TIP]
> Set `TOMOJAX_AXES_SILENCE=1` to suppress heuristic load-time
> axis-order warnings when processing large batches.

## Next steps

- [Python API reference](../reference/api.md) — constructor details
  for `Grid`, `Detector`, and geometry classes
- [Alignment concepts](alignment.md) — how per-view poses are
  optimized
- [CLI reference](../cli/index.md) — specifying geometry on the
  command line
