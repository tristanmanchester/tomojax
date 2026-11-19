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
`tilt` and `tilt_about` in geometry metadata. `tilt` is a departure from
the nominal tomography axis, not an angle to the beam. Missing laminography
tilt requires an explicit `geometry=` with the measured tilt or axis; no 30°
default is inserted.

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

A stack of line integrals is a scan already: build it in Python with its measured
geometry and save it. For a parallel scan:

```python
import tomojax as tj
from tomojax.io.api import load_angles, read_tiff_frames

projections = read_tiff_frames("projections")
nv, nu = projections.shape[1:]
detector = tj.Detector(nu, nv, 0.65, 0.65, center=(0.0, 0.0))  # as measured
grid = tj.Grid(nu, nu, nv, 0.65, 0.65, 0.65)
scan = tj.Scan(projections, tj.ParallelGeometry(grid, detector, load_angles("angles.csv")))
tj.save("corrected.nxs", scan)
```

Nothing applies flat/dark correction or a logarithm to them. The grid is chosen at
reconstruction, with voxels of the detector pitch by default. `tomojax reconstruct --grid 256 256 128`
resamples the selected field of view to 256 × 256 × 128 voxels; their spacing
changes to preserve that physical extent. `tomojax align --fit center` estimates a
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
    tilt=30.0,
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

For already-corrected laminography projections, build the scan the same way with
`tj.LaminographyGeometry(grid, detector, angles, tilt=30.0, tilt_about="x")`, the
measured tilt and centre in place of these, and save it as `corrected-lamino.nxs`.
Use the Python raw-intensity workflow above when the frames still need flat/dark
correction. For raw frames whose file records no geometry, `tomojax load` and
`tomojax correct` declare one with `--geometry lamino --tilt 30 --pixel-size 0.65`.

Replace 30° with the measured tilt. To check that it was kept, run
`uv run --no-sync python -c "import tomojax as tj; print(tj.load('corrected-lamino.nxs').geometry.tilt)"`,
which prints the tilt in degrees.

For an already-corrected dataset that needs explicit laminography metadata:

```python
from tomojax.io import load_dataset, save_dataset

scan = load_dataset("corrected.nxs")
scan.geometry_type = "lamino"
scan.geometry_metadata.update(tilt=30.0, tilt_about="x")
save_dataset("corrected-lamino.nxs", scan)
```

Again replace 30° with the measured tilt, keep the original dataset, and check
the new file with `tomojax inspect` before reconstructing it.

## Reconstruct with the recorded geometry

The default reconstruction of a laminography scan is FBP, as for every
geometry: `tj.reconstruct(scan)` or `tomojax reconstruct corrected.nxs -o recon.nxs`.
It takes seconds (5.5 s on a 1441-view K11 scan with a 704 × 704 × 234 grid).

For a better image, ask for FISTA by name:

```bash
uv run --no-sync tomojax reconstruct corrected.nxs -o recon.nxs \
  --method fista --iterations 100 --nonnegative --region off \
  --manifest recon-manifest.json
uv run --no-sync tomojax inspect recon.nxs --preview previews
```

On two simulated laminography scans with an independent truth, FISTA with its
automatic TV weight halves FBP's error on the frequencies the views measure
(0.079 against 0.165 and 0.163) and fills part of the missing cone, which FBP
leaves empty. It costs its iterations: about 12 s each on that K11 scan, so
about 20 minutes at 100. Its default of 50 iterations is not converged on these
scans (5% and 16% above its converged error), and `iterations` is the setting
to raise. The [measurements](measurements.md#laminography-the-default-fbp-against-fista)
give the table.

Use `corrected-lamino.nxs` as input if you created that file above. `previews/`
holds PNGs of the central projection and the volume's central slices
(`slice_z.png`, `slice_y.png`, `slice_x.png`). Inspect residuals and image features,
and assess sensitivity to regularisation before quantitative use. Positivity
is appropriate only when the object model permits nonnegative attenuation.
Laminography leaves a cone of frequencies around the rotation axis unmeasured.
FBP reconstructs every measured frequency exactly and leaves the cone empty, which
elongates features along the axis. Iterative reconstruction can fill part of the
cone only from the volume bounds or the object prior.

## Reconstruct scans larger than GPU memory

FBP streams projections from host memory and automatically reconstructs in
host slabs when the volume would exceed the device-memory budget. See
[large scans and several GPUs](large-scans.md) for which solvers stream and
how to share a scan among GPUs.

## Evaluate alignment after reconstruction

If a geometry-checked, corrected scan still shows motion artefacts, follow the
[alignment guide](align-a-scan.md). Save an unaligned reconstruction for
comparison. Check residuals, multiple slice planes, and recovered parameters;
image sharpness alone cannot establish physical calibration. Recovery remains
data-dependent, especially for truncated or periodic objects; a lower training residual alone does not establish correct alignment.

## Automatic NeXus laminography

Recognised Diamond/GDA NeXus layouts need no dataset paths, but raw frames do not
record the laminography tilt and often not the detector pitch. Alignment requires
declared geometry, so declare it before fitting, as the
[laminography example](../examples/align_laminography.py) does:

```python
geometry = tj.LaminographyGeometry(grid, tj.Detector(nu, nv, pitch, pitch), angles, tilt=35)
scan = tj.load("scan.nxs", geometry=geometry)
result = tj.align(scan, fit=("translations", "tilt", "roll"))
tj.save("aligned.nxs", result)
```

```bash
uv run --no-sync python examples/align_laminography.py scan.nxs --tilt 35 --pixel 0.00216 -o aligned.nxs
```

`tj.align`'s default fits per-view lab-frame translations jointly with the image,
using a smooth-drift and robust-jump prior. Add `"tilt"` and `"roll"` for shared
setup angles, as above, or `"rotations"` for sample wobble. All settings use the
same solver.

The physical support is preserved, including explicitly placed volume faces. The returned scan
keeps its grid, the original detector samples and the fitted corrections, so reconstructing it
gives the image at the scan's resolution. The alignment image `result.image` is at the fit's
working scale, on `result.grid`.

Pass an explicit `grid=` to `tj.align` or `tj.reconstruct` to fix the physical
region and output sampling. Explicit grids are not coarsened automatically, so
choose a grid that fits the device.

Inspect several slices and the recovered motion: smoothness and lower projection
error do not establish physical pose truth. Reserve held-out views before calling
the workflow for independent evaluation. `fit=` names the geometry parameters being estimated;
reconstruction `method` chooses the image solver. Automatic reconstruction of a posed scan
preserves its grid and corrections.
