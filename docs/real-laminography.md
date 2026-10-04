# Reconstruct a real parallel or laminography scan

Start with measured geometry and corrected absorption projections. This guide
assumes an [installed checkout](installation.md) and uses example dimensions and
spacings that **must be replaced with your acquisition values**. TomoJAX models
parallel rays, including tilted-axis laminography; it does not model cone-beam CT.

## Check an existing NeXus dataset

```bash
uv run --no-sync tomojax inspect scan.nxs
uv run --no-sync tomojax validate scan.nxs
```

Confirm the number and order of views, angles in degrees, detector dimensions
and pitch, volume grid and pitch, and the geometry type. For laminography, check
`tilt_deg` and `tilt_about` in geometry metadata. `tilt_deg` is a departure from
the nominal tomography axis, not an angle to the beam. A missing laminography
tilt currently defaults to 30° when geometry is built; that is not a measurement
of your scanner.

For raw NXtomo data with sample/flat/dark frame keys, follow
[preprocessing](quickstart.md#correct-raw-detector-frames). Nonstandard HDF5
layouts can supply `--data-path`, `--angles-path`, and `--image-key-path` to
`preprocess`. Structural validation does not establish correct physical geometry.

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
uv run --no-sync tomojax ingest ./projections --angles angles.csv \
  --du 0.65 --dv 0.65 --grid 256 256 128 --voxel-size 0.65 0.65 0.65 \
  --out corrected.nxs
```

`ingest` does not apply flat/dark correction or a logarithm. Detector and voxel
pitch must share a physical length unit. `--det-center-u` and `--det-center-v`
are offsets in **detector pixels**; Python's `Detector.det_center` uses physical
lengths. For example, 2 pixels at pitch 0.65 correspond to a physical offset of 1.3.

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
and omit the tilt metadata for ordinary parallel tomography. The CLI's
`preprocess --format tiff-stack` path currently records unit detector spacing and
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

Replace 30° with the measured tilt, preserve the original dataset, and inspect
and validate the new file before reconstructing it. The ingestion CLI's
`--geometry lamino` flag alone does not specify a measured tilt.

## Reconstruct with the recorded geometry

For tilted or irregular-angle data, try an iterative reconstruction:

```bash
uv run --no-sync tomojax recon --data corrected.nxs --out recon.nxs \
  --algo fista --iters 50 --lambda-tv 0.005 --positivity --roi off \
  --save-manifest recon-manifest.json
uv run --no-sync tomojax validate recon.nxs
uv run --no-sync tomojax slices --data recon.nxs --out quicklooks
```

Use `corrected-lamino.nxs` as input if you created that file above. The iteration
budget and TV weight are illustrative. Inspect residuals and image features,
and assess sensitivity to regularization before quantitative use. Positivity
is appropriate only when the object model permits nonnegative attenuation.
Laminography has missing angular information; FBP is an approximate initializer
for that geometry, and iterative reconstruction still depends on the object prior.

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
