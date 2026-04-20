# TomoJAX: Differentiable CT Projector with Alignment

TomoJAX is a fully differentiable, memory-efficient parallel-beam CT
projector implemented in JAX. It provides exact gradients for 5-DOF
rigid-body alignment, making it useful for CT reconstruction, per-view
alignment, and deep learning workloads that require data consistency.

#### Alignment demonstration
<img src="images/montage_scroll.gif" width="1000">

Left to right: ground truth phantom, naive reconstructions (misaligned
and noisy), aligned reconstructions.


## Installation

```bash
# Install uv (https://docs.astral.sh/uv/), then:
uv sync --extra cuda12 --group dev
uv run tomojax-test-gpu
```

For CPU-only setups, use `--extra cpu` instead. See the
[full installation guide](docs/installation.md) for prerequisites,
verification utilities, and troubleshooting.


## Key features

- **Differentiable projector** with exact gradients via JAX autodiff,
  trilinear interpolation, and O(detector pixels) memory via `lax.scan`
- **5-DOF rigid-body alignment** (`alpha`, `beta`, `phi`, `dx`, `dz`)
  with Gauss-Newton, gradient descent, or L-BFGS optimisers
- **Multi-resolution alignment** with coarse-to-fine pyramid
  (e.g. 4x, 2x, 1x) and alternating reconstruct/align steps
- **Three reconstruction algorithms**: FBP, FISTA-TV, and SPDHG-TV
  with automatic Lipschitz estimation and TV proximal operators
- **Laminography support** with tilted rotation-axis geometry,
  360-degree scans, and sample-frame reconstructions
- **Complete CLI toolkit** with 11 commands for simulation, alignment,
  reconstruction, inspection, validation, and benchmarking


## Quick start

```bash
# Simulate a small phantom, misalign it, then align and reconstruct
uv run tomojax-simulate \
  --out data/sim.nxs \
  --nx 64 --ny 64 --nz 64 --nu 64 --nv 64 --n-views 60 \
  --phantom random_shapes --seed 42 --progress

uv run tomojax-misalign \
  --data data/sim.nxs --out data/sim_mis.nxs \
  --rot-deg 1.0 --trans-px 5 --seed 0 --progress

uv run tomojax-align \
  --data data/sim_mis.nxs --levels 4 2 1 \
  --outer-iters 4 --recon-iters 15 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --out out/aligned.nxs --progress
```

See the [quickstart guide](docs/quickstart.md) for a full walkthrough
and the [CLI reference](docs/cli/index.md) for all commands.


## Python API

```python
import jax.numpy as jnp
from tomojax.core.geometry import Grid, Detector, ParallelGeometry
from tomojax.core.projector import forward_project_view, forward_project_view_T
from tomojax.recon.fbp import fbp
from tomojax.recon.fista_tv import fista_tv
from tomojax.align.pipeline import align, AlignConfig

# Define grid, detector, geometry
grid = Grid(nx=128, ny=128, nz=128, vx=1.0, vy=1.0, vz=1.0)
det  = Detector(nu=128, nv=128, du=1.0, dv=1.0)
thetas = jnp.linspace(0, 180, 128, endpoint=False)
geom = ParallelGeometry(grid=grid, detector=det, thetas_deg=thetas)

vol = jnp.ones((grid.nx, grid.ny, grid.nz), jnp.float32)

# Single-view projection (pose from geometry)
p0 = forward_project_view(geom, grid, det, vol, view_index=0)

# FBP and FISTA reconstructions
x_fbp = fbp(geom, grid, det, p0[None, ...])
x_fista, info = fista_tv(
    geom,
    grid,
    det,
    p0[None, ...],
    iters=10,
    lambda_tv=0.001,
    recon_rel_tol=1e-4,
    recon_patience=3,
)

# Alignment (toy 1-view example; use many views in practice)
cfg = AlignConfig(
    outer_iters=1,
    recon_iters=5,
    lambda_tv=0.001,
    recon_rel_tol=1e-4,
    recon_patience=3,
)
x_aligned, params5, info = align(geom, grid, det, p0[None, ...], cfg=cfg)
```

See the [API reference](docs/reference/api.md) for the full public
surface and the [data format reference](docs/reference/data-format.md)
for the HDF5/NXtomo schema used by the CLIs.


## Documentation

| Section | Description |
|---------|-------------|
| [Installation](docs/installation.md) | Prerequisites, GPU/CPU setup, verification |
| [Quickstart](docs/quickstart.md) | Five-minute workflow on a small dataset |
| **Tutorials** | |
| [End-to-end](docs/tutorials/end-to-end.md) | Full 256^3 workflow with alignment |
| [Laminography](docs/tutorials/laminography.md) | Tilted rotation-axis geometry |
| [Single sample](docs/tutorials/single-sample.md) | Sphere or cube phantom |
| **Concepts** | |
| [Geometry](docs/concepts/geometry.md) | Grid, detector, 5-DOF parameterisation |
| [Alignment](docs/concepts/alignment.md) | Multi-resolution pipeline and optimisers |
| [Reconstruction](docs/concepts/reconstruction.md) | FBP, FISTA-TV, SPDHG-TV |
| **CLI reference** | |
| [CLI overview](docs/cli/index.md) | All 11 commands, config files, env vars |
| **Reference** | |
| [Data format](docs/reference/data-format.md) | NXtomo HDF5 schema |
| [Loss functions](docs/reference/loss-functions.md) | Available losses and schedules |
| [Config files](docs/reference/config-files.md) | TOML configuration system |
| [API](docs/reference/api.md) | Python API reference |
| [Misalignment modes](docs/reference/misalign-modes.md) | Deterministic perturbation schedules |
| [Troubleshooting](docs/troubleshooting.md) | Common issues and solutions |


## Visual examples

### Basic projector workflow
| Phantom | Projections | Sinogram | Reconstruction |
|---------|-------------|----------|----------------|
| <img src="images/phantom_slice.png" width="200"><br><img src="images/phantom_volume.png" width="200"> | <img src="images/projections.gif" width="200"> | <img src="images/sinogram.png" width="200"> | <img src="images/recon_slice.png" width="200"><br><img src="images/recon_volume.png" width="200"> |
| Top: slice<br>Bottom: volume projection | Animated over 360° | Angle vs detector | Top: slice<br>Bottom: volume projection |

### Alignment and reconstruction workflow

#### Misaligned input data
| Clean Misaligned Projections | Noisy Misaligned Projections |
|------------------------------|------------------------------|
| <img src="images/spin_projections_misaligned.gif" width="300"> | <img src="images/spin_projections_noisy.gif" width="300"> |
| Random rigid-body misalignments | Same misalignments + Poisson noise |

#### Sinogram analysis
| Clean Misaligned Sinogram | Noisy Misaligned Sinogram |
|---------------------------|---------------------------|
| <img src="images/misaligned_sinogram.png" width="300"> | <img src="images/noisy_sinogram.png" width="300"> |
| Clear view-to-view inconsistencies | Inconsistencies + noise artifacts |

#### Multi-resolution alignment process
| Clean Data Alignment | Noisy Data Alignment |
|---------------------|---------------------|
| <img src="images/alignment_process_misaligned.gif" width="300"> | <img src="images/alignment_process_noisy.gif" width="300"> |
| 4x, 2x, 1x resolution refinement | Robust alignment despite noise |
