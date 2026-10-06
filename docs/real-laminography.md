# Reconstruct a real parallel or laminography scan

Start with measured geometry and corrected absorption projections. This guide
assumes an [installed checkout](installation.md) and uses example dimensions and
spacings that **must be replaced with your acquisition values**. TomoJAX models
parallel rays, including tilted-axis laminography; for lab cone-beam CT, see
[lab cone-beam CT](lab-ct.md).

## Check an existing NeXus dataset

```bash
uv run --no-sync tomojax inspect scan.nxs
```

The report ends with `Valid: yes`, or lists the issues that would stop a
reconstruction. Confirm the number and order of views, angles in degrees, detector dimensions
and pitch, volume grid and pitch, and the geometry type. For laminography, check
`tilt_deg` and `tilt_about` in geometry metadata. `tilt_deg` is a departure from
the nominal tomography axis, not an angle to the beam. A missing laminography
tilt currently defaults to 30° when geometry is built; that is not a measurement
of your scanner.

For raw NXtomo data with sample/flat/dark frame keys, follow
[preprocessing](quickstart.md#correct-raw-detector-frames). Nonstandard HDF5
layouts can set `data_path`, `angles_path`, and `image_key_path` in a TOML
file passed to `preprocess --config`. Structural validation does not establish correct physical geometry.

## Prepare TIFF data

A TIFF directory is read in natural filename order (`view2` before `view10`);
frames within each file keep their stored order. The angle sidecar must match
that order. It can be a one-dimensional `.npy` array or text/CSV with degrees in
the first column. Leading headers, blank lines, and `#` comments are allowed.
After numeric rows begin, malformed rows are errors; empty or nonfinite angles
are rejected. Angles are not sorted automatically.

### Already-corrected absorption projections

For a parallel scan, package an attenuation stack with explicit geometry:

```bash
uv run --no-sync tomojax import ./projections --angles angles.csv \
  --pixel-size 0.65 -o corrected.nxs
```

`import` does not apply flat/dark correction or a logarithm. It records a
centred detector and no grid. The grid is chosen at reconstruction, with
voxels of the detector pitch: `tomojax recon --grid 256 256 128` reconstructs
256 × 256 × 128 voxels of 0.65. `tomojax align --mode cor` estimates a
centre-of-rotation offset. Detector and voxel pitch share a physical length
unit; Python's `Detector.det_center` also uses physical lengths, so an offset
of 2 pixels at pitch 0.65 is a `det_center` of 1.3.

### Raw intensities with flat and dark frames

The Python preprocessing API accepts measured geometry alongside TIFF inputs.
Save the following as a script and adapt the paths and acquisition values:

```python
from tomojax.geometry import Detector, Grid
from tomojax.io import preprocess_tiff_stack

result = preprocess_tiff_stack(
    "projections",
    flats_path="flats",
    darks_path="darks",
    angles_path="angles.csv",
    output_path="corrected.nxs",
    detector=Detector(nu=256, nv=128, du=0.65, dv=0.65),
    grid=Grid(nx=256, ny=256, nz=128, vx=0.65, vy=0.65, vz=0.65),
    geometry_type="lamino",
    geometry_metadata={"tilt_deg": 30.0, "tilt_about": "x"},
)
print(result.output_domain, result.output_shape)
```

This writes absorption projections by default. Use `geometry_type="parallel"`
and omit the tilt metadata for ordinary parallel tomography. `tomojax
preprocess` with TIFF input currently records unit detector spacing and
parallel geometry, so use the Python API above when supplying measured TIFF
geometry. Do not pass already-log-transformed data into this raw-intensity path.

For an already-corrected dataset that needs explicit laminography metadata:

```python
from tomojax.io import load_dataset, save_dataset

scan = load_dataset("corrected.nxs")
scan.geometry_type = "lamino"
scan.geometry_metadata.update(tilt_deg=30.0, tilt_about="x")
save_dataset("corrected-lamino.nxs", scan)
```

Replace 30° with the measured tilt, preserve the original dataset, and check
the new file with `tomojax inspect` before reconstructing it.
`tomojax import --geometry lamino` alone does not record a measured tilt.

## Reconstruct with the recorded geometry

For tilted or irregular-angle data, try an iterative reconstruction:

```bash
uv run --no-sync tomojax recon corrected.nxs -o recon.nxs \
  --method fista --iterations 50 --tv-weight 0.005 --nonnegative --roi off \
  --manifest recon-manifest.json
uv run --no-sync tomojax inspect recon.nxs --preview previews
```

Use `corrected-lamino.nxs` as input if you created that file above. `previews/`
holds PNGs of the central projection and the volume's central slices
(`slice_z.png`, `slice_y.png`, `slice_x.png`). The iteration budget and TV
weight are illustrative. Inspect residuals and image features,
and assess sensitivity to regularization before quantitative use. Positivity
is appropriate only when the object model permits nonnegative attenuation.
Laminography leaves a cone of frequencies around the rotation axis unmeasured.
FBP reconstructs every measured frequency exactly and leaves the cone empty, which
elongates features along the axis. Iterative reconstruction can fill part of the
cone only from the volume bounds or the object prior.

## Reconstruct scans larger than GPU memory

`tomojax recon --method fbp` streams projections from host memory, so only the
volume must fit on the GPU. When the volume does not fit either, use
`fbp_host` from Python with memory-mapped input and output:

```python
import numpy as np

import tomojax as tj
from tomojax.recon import fbp_host

scan = tj.load("corrected-lamino.nxs")
grid = scan.grid
volume = np.lib.format.open_memmap(
    "volume.npy", mode="w+", dtype=np.float32, shape=(grid.nx, grid.ny, grid.nz)
)
fbp_host(scan.geometry, grid, scan.detector, scan.projections, out=volume)
volume.flush()
```

Slabs are sized to the free GPU memory; a 1024³ laminography volume takes
about 40 s on an 8 GB laptop GPU. `tomojax recon --method fista` streams large
projection stacks from host memory, so only its working volumes must fit on
the GPU; CGLS and SPDHG need the projections on the GPU as well.

## Evaluate alignment after reconstruction

If a geometry-checked, corrected scan still shows motion artifacts, follow the
[alignment guide](alignment-guide.md). Save an unaligned reconstruction for
comparison. Check residuals, multiple slice planes, and recovered parameters;
image sharpness alone cannot establish physical calibration. Automatic robust
recovery across real scans has not been established by the current synthetic
pilot.

For a qualitative real-data illustration, see the
[historical DIAD images](../images/README.md#historical-real-data-illustrations).
The raw scan and complete configuration for those images are not bundled.
