# Simulate and reconstruct tomography data

Use synthetic data to check your installation and understand the array and
geometry conventions before processing a scan. Follow [installation](installation.md)
first; all commands below run from the checkout root.

## Generate a parallel scan

```bash
uv run --no-sync tomojax simulate -o synthetic.nxs --size 32 --views 60
uv run --no-sync tomojax recon synthetic.nxs -o recon.nxs
uv run --no-sync tomojax inspect recon.nxs --preview previews
```

`--size 32` makes a 32³ volume and a 32×32 detector. The default phantom is
Shepp–Logan. Simulation writes geometry and projection metadata into the
dataset, and `inspect` writes the central projection and the volume's central
slices to `previews/`. Use `tomojax simulate --help` for other phantoms,
Poisson noise (`--poisson-scale`) and the random seed (`--seed`). Detector
artifacts such as dead and hot pixels, zingers and stripes, and grids or
detectors of other shapes (`grid`, `detector`), are keys of a `--config` TOML
file; `tomojax simulate --config-keys` lists them.

## Try tilted geometry

```bash
uv run --no-sync tomojax simulate -o tilted.nxs --geometry lamino --tilt 30 \
  --size 32 --views 60
uv run --no-sync tomojax recon tilted.nxs -o tilted-recon.nxs \
  --method fista --iterations 50 --tv-weight 0.005 --nonnegative --roi off
uv run --no-sync tomojax inspect tilted-recon.nxs --preview tilted-previews
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

The method and keywords of `tj.reconstruct` match the `tomojax recon` options:
`"cgls"`, `iterations=`, `tv_weight=` and `nonnegative=` are `--method cgls`,
`--iterations`, `--tv-weight` and `--nonnegative`.

Python volumes use `(x, y, z)` and projections `(view, v, u)`. All spacings use
one consistent physical length unit; projected values integrate attenuation
over that length. On-disk volumes may use a different labelled axis order.

This example deliberately uses matched simulation and reconstruction models.
For independent analytic fixtures and failure-inclusive comparisons, see the
[measurement guide](measurements.md). For the README figure and its settings,
see [example reproduction](../examples/README.md).
