# Simulate and reconstruct tomography data

Use synthetic data to check your installation and understand the array and
geometry conventions before processing a scan. Follow [installation](installation.md)
first; all commands below run from the checkout root.

## Generate a parallel scan

```bash
uv run --no-sync tomojax simulate --out synthetic.nxs \
  --nx 32 --ny 32 --nz 32 --nu 32 --nv 32 --n-views 60
uv run --no-sync tomojax recon --data synthetic.nxs --out recon.nxs \
  --algo fbp --roi off
uv run --no-sync tomojax validate recon.nxs
uv run --no-sync tomojax slices --data recon.nxs --out quicklooks
```

The default phantom is Shepp–Logan. Simulation writes geometry and projection
metadata into the dataset. Use `tomojax simulate --help` for other phantoms,
noise and detector artifacts, and the explicit random seed.

## Try tilted geometry

```bash
uv run --no-sync tomojax simulate --out tilted.nxs --geometry lamino --tilt-deg 30 \
  --nx 32 --ny 32 --nz 32 --nu 32 --nv 32 --n-views 60
uv run --no-sync tomojax recon --data tilted.nxs --out tilted-recon.nxs \
  --algo fista --iters 50 --lambda-tv 0.005 --positivity --roi off
uv run --no-sync tomojax slices --data tilted-recon.nxs --out tilted-slices
```

The tilt is measured away from the nominal tomography rotation axis. Laminography
has incomplete angular information; changing the solver does not restore that
missing information. Iteration count and TV weight above are starting settings,
not validated choices for every object or noise level.

## Use the Python API

```bash
uv run --no-sync python examples/simulate_and_reconstruct.py
```

The [complete example](../examples/simulate_and_reconstruct.py) constructs a
`Grid`, `Detector` and `ParallelGeometry`, simulates a scan with `tj.project`,
and reconstructs it with `tj.reconstruct(scan, "cgls")`:

```python
import numpy as np
import tomojax as tj
from tomojax.datasets import shepp_logan_3d

grid = tj.Grid(32, 32, 32, 1.0, 1.0, 1.0)
geometry = tj.ParallelGeometry(grid, tj.Detector(32, 32, 1.0, 1.0), np.linspace(0, 180, 60, endpoint=False))
scan = tj.Scan(tj.project(geometry, shepp_logan_3d(32, 32, 32)), geometry)
volume = tj.reconstruct(scan, "cgls", iterations=24).volume
```

Python volumes use `(x, y, z)` and projections `(view, v, u)`. All spacings use
one consistent physical length unit; projected values integrate attenuation
over that length. On-disk volumes may use a different labelled axis order.

This example deliberately uses matched simulation and reconstruction models.
For independent analytic fixtures and failure-inclusive comparisons, see the
[measurement guide](measurements.md). For the README figure and its settings,
see [example reproduction](../examples/README.md).
