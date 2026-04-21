# Python API reference

The Python API exposes the same algorithms as the CLI tools. Use it
when you need to integrate TomoJAX into scripts, notebooks, or custom
pipelines. All functions are JIT-compatible and support automatic
differentiation via JAX.

## Core geometry

These classes define the physical setup for CT experiments. Import them
from `tomojax.core.geometry`.

### `Grid`

Defines the voxel volume:

```python
from tomojax.core.geometry import Grid

grid = Grid(nx=128, ny=128, nz=128, vx=1.0, vy=1.0, vz=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nx`, `ny`, `nz` | `int` | — | Voxel count per axis |
| `vx`, `vy`, `vz` | `float` | — | Voxel spacing |
| `vol_origin` | `tuple[float,float,float]` | `None` | Physical location of voxel (0,0,0) center |
| `vol_center` | `tuple[float,float,float]` | `None` | Volume center override |

When `vol_origin` is `None`, voxel centers use the default centered
convention.

### `Detector`

Defines the flat detector plane:

```python
from tomojax.core.geometry import Detector

det = Detector(nu=128, nv=128, du=1.0, dv=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nu` | `int` | — | Detector columns (u) |
| `nv` | `int` | — | Detector rows (v) |
| `du` | `float` | — | Pixel spacing in u |
| `dv` | `float` | — | Pixel spacing in v |
| `det_center` | `tuple[float,float]` | `(0.0, 0.0)` | Center offset `(cx, cz)` |

### `ParallelGeometry`

Standard parallel-beam CT geometry. The detector lies in the (x, z)
plane with rays along +y:

```python
from tomojax.core.geometry import ParallelGeometry

thetas = jnp.linspace(0, 180, 200, endpoint=False)
geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `grid` | `Grid` | Volume specification |
| `detector` | `Detector` | Detector specification |
| `thetas_deg` | `Sequence[float]` | Scan angles in degrees |

### `LaminographyGeometry`

Tilted rotation-axis geometry:

```python
from tomojax.core.geometry import LaminographyGeometry

geom = LaminographyGeometry(
    grid=grid, detector=det,
    thetas_deg=jnp.linspace(0, 360, 360, endpoint=False),
    tilt_deg=35.0, tilt_about="x",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tilt_deg` | `float` | `30.0` | Tilt angle in degrees |
| `tilt_about` | `str` | `"x"` | Tilt plane: `"x"` or `"z"` |

See [Geometry concepts](../concepts/geometry.md) for background.

## Projector

Forward and back-projection functions live in
`tomojax.core.projector`.

### `forward_project_view`

Project a volume through a single view using the geometry's pose:

```python
from tomojax.core.projector import forward_project_view

image = forward_project_view(
    geom, grid, det, volume, view_index=0
)
# image shape: (nv, nu), dtype float32
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `geometry` | `Geometry` | — | Geometry instance |
| `grid` | `Grid` | — | Volume grid |
| `detector` | `Detector` | — | Detector spec |
| `volume` | `jnp.ndarray` | — | 3D volume `(nx, ny, nz)` |
| `view_index` | `int` | — | View to project |
| `use_checkpoint` | `bool` | `True` | Gradient checkpointing |
| `gather_dtype` | `str` | `"fp32"` | Mixed-precision gather |

**Returns:** projected image `(nv, nu)` as float32.

### `forward_project_view_T`

Project using an explicit 4x4 pose matrix instead of a geometry
index:

```python
from tomojax.core.projector import forward_project_view_T

image = forward_project_view_T(T, grid, det, volume)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `T` | `jnp.ndarray` | 4x4 world-from-object pose (row-major) |

All other parameters match `forward_project_view`.

### `backproject_view_T`

Explicit adjoint of the forward projector. Smears a detector image
back into the volume:

```python
from tomojax.core.projector import backproject_view_T

vol = backproject_view_T(T, grid, det, image)
# vol shape: (nx, ny, nz), dtype float32
```

### `get_detector_grid_device`

Pre-compute detector coordinate grids as device arrays. Call this
outside JIT/grad contexts and cache the result:

```python
from tomojax.core.projector import get_detector_grid_device

det_grid = get_detector_grid_device(det)
# Returns (X, Z) tuple, each shape (nu*nv,)
```

> [!TIP]
> All projector functions support `jax.grad` and `jax.jit`. The
> trilinear interpolation has analytical derivatives, so gradients
> are exact.

## Reconstruction

### `fbp`

Filtered backprojection:

```python
from tomojax.recon.fbp import fbp

volume = fbp(geom, grid, det, projections, filter_name="ramp")
# volume shape: (nx, ny, nz)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `projections` | `jnp.ndarray` | — | `(n_views, nv, nu)` |
| `filter_name` | `str` | `"ramp"` | Frequency filter |
| `checkpoint_projector` | `bool` | `True` | Gradient checkpointing |
| `gather_dtype` | `str` | `"fp32"` | Mixed-precision gather |

**Returns:** volume `(nx, ny, nz)` as float32.

### `fista_tv`

FISTA with TV regularization:

```python
from tomojax.recon.fista_tv import fista_tv, FistaConfig

config = FistaConfig(iters=50, lambda_tv=0.005)
volume, info = fista_tv(geom, grid, det, projections, config=config)
```

**`FistaConfig` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `iters` | `int` | `50` | FISTA iterations |
| `lambda_tv` | `float` | `0.005` | TV weight |
| `regulariser` | `str` | `"tv"` | `"tv"` or `"huber_tv"` |
| `huber_delta` | `float` | `0.01` | Huber smoothing |
| `L` | `float \| None` | `None` | Data-term Lipschitz constant (auto if None; huber_tv adds extra contribution) |
| `tv_prox_iters` | `int` | `10` | Chambolle-Pock inner steps |
| `recon_rel_tol` | `float \| None` | `None` | Early-stop tolerance |
| `recon_patience` | `int` | `0` | Consecutive below-tol iters |
| `positivity` | `bool` | `False` | Non-negativity constraint |
| `lower_bound` | `float \| None` | `None` | Voxel lower bound |
| `upper_bound` | `float \| None` | `None` | Voxel upper bound |
| `checkpoint_projector` | `bool` | `True` | Gradient checkpointing |
| `gather_dtype` | `str` | `"fp32"` | Mixed-precision gather |

**Returns:** tuple of:
- Volume `(nx, ny, nz)` as float32
- Info dict with keys: `"loss"`, `"L"`, `"effective_iters"`,
  `"early_stop"`, `"regulariser"`

### `spdhg_tv`

Stochastic Primal-Dual Hybrid Gradient:

```python
from tomojax.recon.spdhg_tv import spdhg_tv, SPDHGConfig

config = SPDHGConfig(iters=400, lambda_tv=0.005, views_per_batch=16)
volume, info = spdhg_tv(geom, grid, det, projections, config=config)
```

**`SPDHGConfig` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `iters` | `int` | `400` | SPDHG iterations |
| `lambda_tv` | `float` | `0.005` | TV weight |
| `views_per_batch` | `int` | `16` | Stochastic batch size |
| `theta` | `float` | `1.0` | Extrapolation parameter |
| `seed` | `int` | `0` | Random seed for view order |
| `positivity` | `bool` | `True` | Non-negativity constraint |

**Returns:** same tuple format as `fista_tv`.

## Alignment

### `align`

Joint per-view alignment and reconstruction:

```python
from tomojax.align.pipeline import align, AlignConfig

cfg = AlignConfig(
    outer_iters=4,
    recon_iters=25,
    lambda_tv=0.003,
    opt_method="gn",
    gn_damping=1e-3,
)
volume, params5, info = align(geom, grid, det, projections, cfg=cfg)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cfg` | `AlignConfig` | `AlignConfig()` | Configuration |
| `init_x` | `jnp.ndarray \| None` | `None` | Initial volume |
| `init_params5` | `jnp.ndarray \| None` | `None` | Initial 5-DOF per view `(n_views, 5)` |

**Returns:** tuple of:
- Volume `(nx, ny, nz)` as float32
- Parameters `(n_views, 5)` in `[alpha, beta, phi, dx, dz]` order
- Info dict with `"loss"`, `"outer_stats"`, `"effective_iters"`,
  `"pose_model"`

**Key `AlignConfig` fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `outer_iters` | `int` | `5` | Alternating iterations |
| `recon_iters` | `int` | `10` | Inner reconstruction iterations |
| `lambda_tv` | `float` | `0.005` | TV weight |
| `recon_algo` | `str` | `"fista"` | `"fista"` or `"spdhg"` |
| `opt_method` | `str` | `"gn"` | `"gd"`, `"gn"`, or `"lbfgs"` |
| `gn_damping` | `float` | `1e-6` | GN Levenberg-Marquardt damping |
| `lr_rot` | `float` | `1e-3` | GD rotation learning rate |
| `lr_trans` | `float` | `1e-1` | GD translation learning rate |
| `pose_model` | `str` | `"per_view"` | `"per_view"`, `"spline"`, `"polynomial"` |
| `gauge_fix` | `str` | `"mean_translation"` | `"mean_translation"` or `"none"` |
| `early_stop` | `bool` | `True` | Stop on plateau |
| `early_stop_rel_impr` | `float` | `1e-3` | Minimum relative improvement |
| `early_stop_patience` | `int` | `2` | Patience iterations |
| `optimise_dofs` | `tuple \| None` | `None` | Active DOFs (all if None) |
| `freeze_dofs` | `tuple` | `()` | DOFs to freeze |
| `checkpoint_projector` | `bool` | `True` | Gradient checkpointing |
| `gather_dtype` | `str` | `"fp32"` | Mixed-precision gather |

See [Alignment concepts](../concepts/alignment.md) for algorithm
background and [align CLI](../cli/align.md) for the full list of
fields.

## Data I/O

HDF5 NXtomo read/write functions live in `tomojax.data.io_hdf5`.

### `load_nxtomo`

Load an NXtomo dataset:

```python
from tomojax.data.io_hdf5 import load_nxtomo

result = load_nxtomo("data/scan.nxs")
projections = result.projections  # (n_views, nv, nu)
metadata = result.metadata        # NXTomoMetadata
```

**Returns:** `LoadedNXTomo` with fields:
- `projections` — `np.ndarray (n_views, nv, nu)`
- `metadata` — `NXTomoMetadata` with `thetas_deg`, `grid`,
  `detector`, `geometry_type`, `volume`, `align_params`, and more

Volumes are always returned in internal `(nx, ny, nz)` order,
regardless of on-disk layout.

### `save_nxtomo`

Write a dataset to HDF5:

```python
from tomojax.data.io_hdf5 import save_nxtomo, NXTomoMetadata

metadata = NXTomoMetadata(
    thetas_deg=thetas,
    grid=grid,
    detector=det,
    volume=volume,
)
save_nxtomo("out/result.nxs", projections, metadata=metadata)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Output file path |
| `projections` | `np.ndarray` | — | `(n_views, nv, nu)` |
| `metadata` | `NXTomoMetadata` | `None` | Metadata bundle |
| `compression` | `str` | `"lzf"` | HDF5 compression |
| `overwrite` | `bool` | `True` | Truncate existing file |

### `validate_nxtomo`

Run lightweight schema checks:

```python
from tomojax.data.io_hdf5 import validate_nxtomo

report = validate_nxtomo("data/scan.nxs")
if report["issues"]:
    print("Validation issues:", report["issues"])
```

**Returns:** dict with `"issues"` list (empty if valid).

## Simulation and phantoms

### `simulate`

Generate synthetic CT data:

```python
from tomojax.data.simulate import simulate, SimConfig

cfg = SimConfig(
    nx=128, ny=128, nz=128,
    nu=128, nv=128, n_views=180,
    phantom="random_shapes",
    seed=42,
)
data = simulate(cfg)
projections = data["projections"]
volume = data["volume"]
```

**Returns:** dict with keys: `"projections"`, `"thetas_deg"`,
`"grid"`, `"detector"`, `"geometry_type"`, `"volume"`,
`"geometry_meta"`.

### Phantom constructors

All phantoms return `np.ndarray` of shape `(nx, ny, nz)` as float32:

```python
from tomojax.data.phantoms import (
    sphere,
    cube,
    rotated_centered_cube,
    blobs,
    shepp_logan_3d,
    random_cubes_spheres,
    lamino_disk,
)
```

| Function | Description | Key parameters |
|----------|-------------|---------------|
| `sphere(nx, ny, nz)` | Centered solid sphere | `size` (diameter fraction), `value` |
| `cube(nx, ny, nz)` | Axis-aligned cube | `size` (side fraction), `value` |
| `rotated_centered_cube(...)` | Randomly rotated cube | `size`, `value`, `seed` |
| `blobs(nx, ny, nz)` | Random Gaussian blobs | `n_blobs`, `seed` |
| `shepp_logan_3d(nx, ny, nz)` | 3D Shepp-Logan | — |
| `random_cubes_spheres(...)` | Random cubes + spheres | `n_cubes`, `n_spheres`, `min_size`, `max_size`, `seed` |
| `lamino_disk(...)` | Thin slab for laminography | `thickness_ratio`, `tilt_deg`, `tilt_about` |

## Next steps

- [CLI overview](../cli/index.md) — command-line equivalents
- [Geometry concepts](../concepts/geometry.md) — background on the
  geometry model
- [Data format](data-format.md) — HDF5 schema details
