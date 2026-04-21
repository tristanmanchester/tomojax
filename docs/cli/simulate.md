# simulate

The `tomojax-simulate` command generates synthetic CT datasets and
saves them as HDF5 NXtomo files. You can create phantoms with various
geometries, add composable noise/artifacts, and produce reproducible
test data for reconstruction and alignment workflows.

```
tomojax-simulate --out <path.nxs> \
  --nx <int> --ny <int> --nz <int> \
  --nu <int> --nv <int> --n-views <int> \
  [options...]
```

## Geometry options

These flags control the scan geometry and angular coverage.

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--geometry` | `parallel`, `lamino` | `parallel` | Beam geometry |
| `--tilt-deg` | float | `30` | Laminography tilt angle in degrees |
| `--tilt-about` | `x`, `z` | `x` | Axis the tilt is applied about |
| `--rotation-deg` | float | see below | Total rotation range in degrees |

The `--rotation-deg` default depends on the geometry: 180 for
`parallel` and 360 for `lamino`. You can override it to simulate
limited-angle or extended scans.

## Phantom types

The `--phantom` flag selects the 3D object placed inside the volume.

| Phantom | Description |
|---------|-------------|
| `shepp` | 3D Shepp-Logan head phantom (default) |
| `cube` | Single centered cube |
| `sphere` | Single centered sphere |
| `blobs` | Gaussian blob field |
| `random_shapes` | Random cubes and spheres |
| `lamino_disk` | Thin slab with random shapes (for laminography) |

### Single-object controls

These flags apply when `--phantom` is `cube` or `sphere`.

| Flag | Default | Description |
|------|---------|-------------|
| `--single-size` | `0.5` | Relative fraction of the smallest volume dimension (side length for cube, diameter for sphere) |
| `--single-value` | `1.0` | Intensity value of the object |
| `--no-single-rotate` | off | Disable the random 3D rotation applied to cubes (spheres are unaffected) |

### Random shapes controls

These flags apply when `--phantom` is `random_shapes` or
`lamino_disk`.

| Flag | Default | Description |
|------|---------|-------------|
| `--n-cubes` | `8` | Number of random cubes |
| `--n-spheres` | `7` | Number of random spheres |
| `--min-size` | `4` | Minimum object size in voxels |
| `--max-size` | `32` | Maximum object size in voxels |
| `--min-value` | `0.1` | Minimum intensity value |
| `--max-value` | `1.0` | Maximum intensity value |
| `--max-rot-deg` | `180` | Maximum random rotation in degrees |

### Lamino disk

The `lamino_disk` phantom generates a thin slab populated with random
shapes. It uses all the random shapes controls above, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--lamino-thickness-ratio` | `0.2` | Relative slab thickness (0--1) |

## Noise and artifacts

Simulation artifacts are optional, composable, and deterministic via
`--seed`. They're applied after the clean forward projection, so the
underlying sinogram is always generated first from the phantom. All
artifact flags can be combined freely in a single run.

> [!NOTE]
> Explicit artifact flags override the legacy `--noise` /
> `--noise-level` pair only when at least one artifact is enabled. If
> all explicit artifact flags are disabled (set to 0 or their default
> values), the legacy options are preserved.

### Photon and read noise

Photon noise simulates counting statistics; read noise adds
detector-level Gaussian fluctuations.

| Flag | Default | Description |
|------|---------|-------------|
| `--poisson-scale` | `0` | Incident intensity scale for Poisson sampling. Larger values produce lower relative noise. Set 0 to disable. |
| `--gaussian-sigma` | `0` | Standard deviation of additive Gaussian noise |

### Detector defects

Dead and hot pixels are fixed spatial locations set to a constant
value across all views.

| Flag | Default | Description |
|------|---------|-------------|
| `--dead-pixel-fraction` | `0` | Fraction of detector pixels stuck at the dead value |
| `--dead-pixel-value` | `0` | Intensity value for dead pixels |
| `--hot-pixel-fraction` | `0` | Fraction of detector pixels stuck at the hot value |
| `--hot-pixel-value` | `1.0` | Intensity value for hot pixels |

### Sparse outliers

Zingers are single-pixel spikes scattered across the full
(view, v, u) volume. Stripes are per-column gain errors that persist
across all views.

| Flag | Default | Description |
|------|---------|-------------|
| `--zinger-fraction` | `0` | Fraction of total sinogram pixels receiving a zinger spike |
| `--zinger-value` | `1.0` | Additive intensity of each zinger |
| `--stripe-fraction` | `0` | Fraction of detector columns with gain errors |
| `--stripe-gain-sigma` | `0` | Standard deviation of the column gain multiplier (mean 1.0) |

### Acquisition effects

These flags simulate instrument-level effects that vary across views
or blur the detector signal.

| Flag | Default | Description |
|------|---------|-------------|
| `--dropped-view-fraction` | `0` | Fraction of views replaced with a fill value |
| `--dropped-view-fill` | `0` | Fill value for dropped views |
| `--detector-blur-sigma` | `0` | Gaussian blur sigma applied to detector axes (pixels) |
| `--intensity-drift-mode` | `none` | Drift model: `none`, `linear`, or `sinusoidal` |
| `--intensity-drift-amplitude` | `0` | Amplitude of the multiplicative drift envelope |

### Legacy aliases

The older `--noise` and `--noise-level` flags still work for backward
compatibility.

| Flag | Values | Description |
|------|--------|-------------|
| `--noise` | `none`, `gaussian`, `poisson` | Legacy noise type |
| `--noise-level` | float | Legacy noise parameter (sigma for Gaussian, scale for Poisson) |

These are mapped internally to `--gaussian-sigma` or
`--poisson-scale`. We recommend using the explicit artifact flags
instead.

## Examples

The examples below use `uv run` to invoke the console script. You
can substitute `python -m tomojax.cli.simulate` if you prefer.

### Basic parallel-beam simulation

This creates a 256-cube Shepp-Logan dataset with 200 views over 180
degrees.

```bash
uv run tomojax-simulate \
  --out data/sim_parallel.nxs \
  --nx 256 --ny 256 --nz 256 \
  --nu 256 --nv 256 --n-views 200 \
  --phantom shepp --seed 42 --progress
```

### Laminography simulation

This generates a laminography dataset with a thin disk phantom,
tilted 30 degrees about the x-axis over a full 360-degree rotation.

```bash
uv run tomojax-simulate \
  --out data/sim_lamino.nxs \
  --nx 256 --ny 256 --nz 64 \
  --nu 256 --nv 256 --n-views 360 \
  --geometry lamino --tilt-deg 30 --tilt-about x \
  --phantom lamino_disk --lamino-thickness-ratio 0.2 \
  --seed 42 --progress
```

### Noisy and artifacted benchmark

This produces a dataset with photon noise, Gaussian read noise,
stripes, zingers, dead/hot pixels, dropped views, detector blur, and
sinusoidal intensity drift -- useful for stress-testing preprocessing
and reconstruction pipelines.

```bash
uv run tomojax-simulate \
  --out data/sim_artifacts.nxs \
  --nx 128 --ny 128 --nz 128 \
  --nu 128 --nv 128 --n-views 180 \
  --phantom random_shapes --n-cubes 30 --n-spheres 30 \
  --poisson-scale 5000 \
  --gaussian-sigma 0.002 \
  --stripe-fraction 0.02 --stripe-gain-sigma 0.05 \
  --zinger-fraction 0.0005 --zinger-value 2.0 \
  --dead-pixel-fraction 0.0002 \
  --hot-pixel-fraction 0.0002 \
  --dropped-view-fraction 0.02 \
  --detector-blur-sigma 0.6 \
  --intensity-drift-mode sinusoidal \
  --intensity-drift-amplitude 0.05 \
  --seed 42
```

---

See also: [data format](../reference/data-format.md),
[CLI overview](index.md).
