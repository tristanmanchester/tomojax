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

For raw NXtomo or Data Exchange frames with flats and darks, follow
[preprocessing](quickstart.md#correct-raw-detector-frames). Nonstandard HDF5
layouts can name their datasets with `--data-path`, `--angles-path` and
`--image-key-path` (or `data_path=`... in `tj.load_frames`). Structural
validation does not establish correct physical geometry.

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
unit; Python's `Detector.center` also uses physical lengths, so an offset
of 2 pixels at pitch 0.65 is a `center` of 1.3.

### Raw intensities with flat and dark frames

`tj.load_frames` reads TIFF frames with their flats, darks and a measured
geometry. Save the following as a script and adapt the paths and acquisition
values:

```python
import tomojax as tj
from tomojax.io.api import load_angles

angles = load_angles("angles.csv")
geometry = tj.LaminographyGeometry(
    tj.Grid(nx=256, ny=256, nz=128, vx=0.65, vy=0.65, vz=0.65),
    tj.Detector(nu=256, nv=128, du=0.65, dv=0.65),
    angles,
    tilt_deg=30.0,
    tilt_about="x",
)
frames = tj.load_frames("projections", flats="flats", darks="darks", geometry=geometry)
scan = frames.corrected()
tj.save("corrected.nxs", scan)
print(scan)
```

This writes line integrals, recording the correction. Use
`tj.ParallelGeometry(grid, detector, angles)` for ordinary parallel
tomography. Do not pass already-log-transformed data into this raw-intensity
path.

On the command line, import the frames with their geometry, record the
measured tilt, then correct them. `tomojax import` has no tilt option, and a
laminography dataset without a recorded tilt gets 30°, so set it before
`tomojax preprocess`:

```bash
uv run --no-sync tomojax import ./projections --angles angles.csv \
  --geometry lamino --pixel-size 0.65 -o raw.nxs
uv run --no-sync python -c "
from tomojax.io import load_dataset, save_dataset
scan = load_dataset('raw.nxs')
scan.geometry_metadata.update(tilt_deg=30.0, tilt_about='x')
save_dataset('raw-tilt.nxs', scan)
"
uv run --no-sync tomojax preprocess raw-tilt.nxs -o corrected-lamino.nxs \
  --flats ./flats --darks ./darks
```

Replace 30° with the measured tilt. To check that it was kept, run
`uv run --no-sync python -c "import tomojax as tj; print(tj.load('corrected-lamino.nxs').geometry.tilt_deg)"`,
which prints the tilt in degrees.

For an already-corrected dataset that needs explicit laminography metadata:

```python
from tomojax.io import load_dataset, save_dataset

scan = load_dataset("corrected.nxs")
scan.geometry_type = "lamino"
scan.geometry_metadata.update(tilt_deg=30.0, tilt_about="x")
save_dataset("corrected-lamino.nxs", scan)
```

Again replace 30° with the measured tilt, keep the original dataset, and check
the new file with `tomojax inspect` before reconstructing it.

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
and assess sensitivity to regularisation before quantitative use. Positivity
is appropriate only when the object model permits nonnegative attenuation.
Laminography leaves a cone of frequencies around the rotation axis unmeasured.
FBP reconstructs every measured frequency exactly and leaves the cone empty, which
elongates features along the axis. Iterative reconstruction can fill part of the
cone only from the volume bounds or the object prior.

## Reconstruct scans larger than GPU memory

FBP streams the projections from host memory, but the volume must fit on the
GPU; for a larger volume, use `fbp_host` from Python. See
[large scans and several GPUs](large-scans.md) for which solvers stream and
how to share a scan among GPUs.

## Evaluate alignment after reconstruction

If a geometry-checked, corrected scan still shows motion artefacts, follow the
[alignment guide](alignment-guide.md). Save an unaligned reconstruction for
comparison. Check residuals, multiple slice planes, and recovered parameters;
image sharpness alone cannot establish physical calibration. Recovery remains
data-dependent, especially for truncated or periodic objects; a lower training residual alone does not establish correct alignment.

## K11 nominal-geometry recipe

The DIAD K11 file labels its rotation field with the wrong units; read its
recorded angles explicitly in degrees. The runnable
[laminography example](../examples/align_laminography.py) starts from the nominal
tilt and centred detector, estimates setup and smooth motion, and chooses a
reconstruction region rather than requiring a hand-selected depth:

```bash
uv run --no-sync python examples/align_laminography.py k11-67208_binned.nxs \
  --angles-path /entry/imaging/smaract_zrot_value_set \
  --image-key-path /entry/tomo_entry/instrument/detector/image_key \
  --tilt 30.9
```

It applies another factor of two detector binning **after** the logarithm.
The recipe excludes internal validation views from setup and pose fitting,
interpolates small pose corrections in acquisition order, and keeps refinement
only if both ordinary and high-pass held-projection errors improve at the final
grid. An uncertain region retains a broad survey and compares its projection-only
motion seed against zero motion using the same held-view checks, without joint
refinement; this is not evidence that broad objects cannot align.
No library defaults change, and no file is written unless `-o result.nxs` is given.

Inspect the correction record, recovered axis, several volume planes and the
reported validation scores. Large nonpositive-transmission counts, a broad/coarse
region or multiple layers still need acquisition-specific investigation. Held
residuals do not establish physical poses or support, and an independent quality
assessment must reserve further views before calling the recipe. The example's
sampling and iteration budgets are not convergence guarantees.

For a qualitative real-data illustration, see the
[historical DIAD images](../images/README.md#historical-real-data-illustrations).
The raw scan and complete configuration for those images are not bundled.
