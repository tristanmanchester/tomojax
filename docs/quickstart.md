# Reconstruct your first scan

This guide starts from an installed checkout; follow [installation](installation.md)
first. If you do not have data yet, run the [synthetic example](../README.md#first-reconstruction).
Commands use `uv run --no-sync` from the checkout root. Replace example input
paths with your files and use new output paths.

## Inspect the data and geometry

```bash
uv run --no-sync tomojax inspect scan.nxs
```

Check the projection count and shape, angles, detector pitch, voxel pitch, and
geometry type. Recognised Nikon, Zeiss and ESRF acquisitions use the same
readers as loading; inspection reports their raw detector values without applying
corrections. Raw NXtomo, APS Data Exchange and recognised Diamond/GDA layouts
also use loading's acquisition discovery; optional NXtomo labels are not required. `inspect` also checks the acquisition metadata: it ends with
`Valid: yes`, or lists the issues and exits with status 1. That check cannot
prove that metadata matches the instrument. `--json` prints the report as JSON
instead. It also lists the corrections that made the projections from
detector frames, when the file records them.

Reconstruction expects line integrals (log attenuation). Raw detector counts,
normalised transmission, and already-corrected attenuation are different
inputs: establish which you have. TomoJAX corrects raw frames itself when the
file says they are raw: an NXtomo `image_key` marking flats or darks, an APS
Data Exchange file with `data_white` or `data_dark` beside `/exchange/data`,
or a Nikon scan. Missing flats/white level or darks require explicit
`flats=` (frames or one level) and `darks=` in `load_frames`; use zero darks only
when the counts are already dark-corrected.

## Beamline exports

For executable first-slice recipes for ESRF NXtomo, TOMCAT/SYRMEP Data Exchange
and the mixed I13 TIFF folder, start with [beamline exports](beamline-exports.md).
The examples distinguish file metadata from explicitly supplied calibration.

### Raw ESRF acquisitions

`tj.load_frames("scan_folder", detector=..., angles=...)` reads uncompressed
single-image EDF projections with SPEC `.info` or XML acquisition metadata.
`PixelSize` records the effective pitch in micrometres, but the scan range alone
does not establish the starting angle or a calibrated rotation centre.

For EDF acquisitions described by PyHST, pass the parameter file explicitly:
`tj.load_frames("scan.par").corrected()`. It supplies the indexed EDF series,
angles, detector pitch and rotation axis, reconstruction region, and dark/flat
schedule. A directory search does not choose a `.par` file over other acquisition
metadata. This route requires raw, single-image EDFs with fixed axial parallel
geometry; multi-frame, cone, helical, binning and distortion-correction modes
are not supported.

`tj.load_frames("raw_bliss.h5", detector=...)` reads a basic, half-turn or full-turn
Bliss sequence, including its flats/darks and recorded angular positions. Select
among several sequences with `data_path="1.1"`. Effective sample pixels and
recorded distances are retained in millimetres; neither reader substitutes a
centred detector for missing axis calibration. Missing virtual detector pixels
and mismatched EDF dimensions are errors. Helical and multi-volume Bliss
sequences are not yet supported.

`tj.load_frames` keeps HDF5 single-row sinograms stored as `(view, u)` lazy,
presenting them as `(view, 1, u)`. Corrected `Scan` projections retain that
singleton detector-row axis.

## Correct raw detector frames

For an HDF5 scan containing sample, flat, dark frames and measured geometry:

```bash
uv run --no-sync tomojax preprocess raw.nxs -o corrected.nxs
uv run --no-sync tomojax inspect corrected.nxs
```

This writes the line integrals `-log((I - D) / (F - D))` of the sample views,
each view's flat interpolated between the flat sets taken around it, and
records the correction. The frames, `image_key` and angles are found at their
NXtomo paths, at the Data Exchange paths (`/exchange/data` with `data_white`,
`data_dark` and `theta` beside it), or as the file's only datasets of those
names; `--data-path`,
`--image-key-path` and `--angles-path` (expert settings, `--config-keys`) name
them otherwise. Options add corrections: `--zingers`, `--stripes 9`
(rings), `--reject-views` (views that jump from their neighbours, such as a
closed shutter); `--select-views` and `--crop` keep
part of the scan. Already-corrected data should skip this
step; applying the logarithm again changes the data.

Missing detector pitch or an unsuccessful centre estimate requires `detector=`
in `tj.load_frames`, or a
`[detector]` table in the preprocessing TOML config with `nu`, `nv`, `du`, `dv`
and `det_center = [u, v]` in consistent length units. Missing view angles
require `angles=`; missing laminography tilt requires a complete `geometry=`.
Centred unit pixels and a 30° tilt are not assumed.

In Python, `tj.load("raw.nxs")` makes the same correction, and records it in
`scan.corrections`. For other corrections, load the frames and pass steps from
`tomojax.corrections`:

```python
import tomojax as tj
from tomojax.corrections import RejectViews, Stripes

frames = tj.load_frames("raw.nxs")  # or a Nikon .xtekct, or TIFFs with angles=
scan = frames.cropped(slice(100, 900), slice(None)).corrected(RejectViews(), Stripes(9))
print(scan)  # Scan('sample': ..., corrections: flat_dark(...), log(...), reject_views(...), ...)
tj.save("corrected.nxs", scan)
```

HDF5 and TIFF sample, flat and dark stacks are read lazily, as they are
corrected (except sampled rows used for missing-centre estimation): crop or
select first to read less. `np.asarray(frames.counts)` reads
the current selection into memory.

The views are corrected on the GPU a batch at a time, read from the file as
they are needed. `batch_views` must be a positive integer and `epsilon` finite
and positive. `frames.selected(...)` and `scan.selected(...)` keep a nonempty
selection in acquisition order: a forward slice, increasing integer indices,
or a Boolean mask. Selection keeps each view's geometry and its original
position between the flat fields. `scan.corrected(...)` runs line-integral
steps (`Stripes`, `BeamHardening`, `RejectViews`) on a scan that is already
corrected.

For rings that remain after `Stripes(9)`, a detector pixel's response may vary
with the view rather than add a constant offset. In Python, follow the ordinary
stripe correction with `Stripes(51, varying=True)`. Keep densely sampled views
for this correction; selecting widely spaced views first can confuse moving
object edges with defective pixels. Crop detector rows to save memory instead:

```python
import tomojax as tj
from tomojax.corrections import Stripes

frames = tj.load_frames("raw.nxs").cropped(slice(100, 116), slice(None))
scan = frames.corrected(Stripes(9), Stripes(51, varying=True))
# Select fewer views for a quick reconstruction only after correcting them.
preview = scan.selected(slice(None, None, 4))
```

The widths are detector columns, not reconstructed-ring radii. Increase the
ordinary width for wider stripes, and compare images: stationary centred object
features can also be changed. Missing information at dead pixels is not recovered.

`corrected(..., binning=2)` averages 2 x 2 detector blocks of the line integrals,
as `scan.binned(2)` does. With only `Stripes` and the like after the log, it
bins slabs of rows read straight from an uncompressed file, so the unbinned
stack never reaches host memory. The CLI accepts the same factor as an expert
config key:

```bash
printf 'binning = 4\n' > preview.toml
uv run --no-sync tomojax preprocess raw.nxs -o binned.nxs --config preview.toml
```

Binning reduces detector sampling, not the reconstruction grid. In Python,
pass a coarser `grid=` to `tj.reconstruct` when you also want larger voxels;
On the CLI, `--grid NX NY NZ` resamples the field of view selected by `--roi`
to those counts, preserving its physical extent. Python uses
`scan.grid.resampled((NX, NY, NZ))` to do the same without changing any data.

For a small **axial parallel-beam** preview, use one voxel per binned detector
pixel and retain the selected detector rows' height. This explicit grid avoids
upsampling the binned data back onto the original fine grid:

```python
import tomojax as tj

scan = tj.load_frames("raw.nxs").selected(slice(None, None, 4)).corrected(binning=8)
scan = tj.estimate_center(scan)
d = scan.detector
grid = tj.Grid(d.nu, d.nu, d.nv, d.du, d.du, d.dv, vol_center=(0, 0, d.center[1]))
tj.save("preview.nxs", tj.reconstruct(scan, grid=grid))
```

Inspect it with `tomojax inspect preview.nxs --preview previews`. Use enough
views for the binned width; fewer views can create streaks. This grid recipe
is not for cone beams or tilted axes, where magnification or the sample's
extent along the axis also matters.

For calibrated lens distortion, pass
`RadialDistortion(center=(column, row), coefficients=(c0, c1, ...))` from
`tomojax.corrections`. It bilinearly samples a **backward** radial polynomial
on transmission before the log. The centre is in zero-based image pixels,
not the rotation axis; estimate the axis after this correction. Correct
before cropping if the warp needs pixels outside the crop, and convert the
calibration after a crop or binning (see `help(RadialDistortion)`).

A file written by TomoJAX carries its geometry. External NXtomo and Data Exchange
use their format-defined axial parallel-beam model and recorded angles. They
need recorded physical detector pitches; missing pitch requires `detector=` or
`geometry=`, never unit pixels. With pitch available but no rotation centre,
loading estimates detector-u from a few corrected rows and records `inferred`
provenance. This needs usable near-opposite views and contrast; otherwise supply a
calibrated detector, including its centre. The vertical origin is chosen at the
axial detector midplane, not estimated from the projections.

An explicit `detector=tj.Detector(nu, nv, du, dv, center=(u, v))` or full
`geometry=` remains authoritative and is not automatically recalibrated.
For an already loaded unposed axial scan, `tj.estimate_center(scan)` estimates
physical detector-u from row-gradient correlations without a volume solve.
Near-opposite views are resampled in angle before matching. Inspect the result:
repetitive structure, motion and truncation can make the estimate ambiguous.

### Retrieve phase from propagation-based phase contrast

`Paganin` is single-material phase retrieval: it filters each view's
transmission before the logarithm. It takes the material's δ/β, the effective
propagation distance in metres (the sample-to-detector distance, divided by the
magnification in a cone beam), the X-ray energy in keV and the pixel size at
the sample in metres (the detector pixel, divided by the magnification in a
cone beam). The step does not infer these from the scan. The values below are
illustrative:

```python
import tomojax as tj
from tomojax.corrections import Paganin

step = Paganin(delta_beta=100, distance=0.1, energy_kev=25, pixel_size=0.65e-6)
scan = tj.load_frames("raw.nxs").corrected(step)
# Retain a detector window after phase retrieval, not before it.
scan = scan.cropped(slice(100, 900), slice(None))
```

Paganin mixes neighbouring pixels, so cropping raw frames first discards
measured neighbours; correct a larger window and keep its interior if the full
detector is too expensive.

If using calibrated lens correction too, pass `RadialDistortion` before
`Paganin`, so phase retrieval acts on uniformly spaced undistorted pixels.
Put any `Zingers` step before both. Transmission steps run in the order
supplied, and changing their order changes the result.

On the command line, `--paganin` takes the same four values in that order:
`tomojax preprocess raw.nxs -o phase.nxs --paganin 100,0.1,25,0.65e-6`. It is
an expert option, so `--help` does not list it.
`--crop` crops raw frames before correction.

### TIFF frames

For TIFF frames, use the [TIFF and measured-geometry instructions](real-laminography.md#prepare-tiff-data).
`tomojax import` packages a stack; it does not correct frames.

## Reconstruct and inspect slices

Start with the same automatic policy as `tj.reconstruct(scan)`:

```bash
uv run --no-sync tomojax recon corrected.nxs -o recon.nxs \
  --manifest recon-manifest.json
uv run --no-sync tomojax inspect recon.nxs --preview previews
```

The default `--method auto` selects a reconstruction from the geometry and
reports any limitations; use an explicit `--method` for a fixed solver.
`recon.nxs` stores the volume and copies the projections. The manifest records
the selected method, reconstruction settings and automatic outcome. CPU execution
is supported and does not by itself mark the automatic outcome as limited. `previews/`
contains PNGs of the central projection (`projection.png`) and the volume's
three central planes (`slice_z.png`, `slice_y.png`, `slice_x.png`). PNG contrast is scaled for
display; read the stored floating-point volume for quantitative work.
`tomojax export recon.nxs -o slices/` writes the volume as TIFF slices for other
software.
Each z-slice has y rows and x columns, numbered from low to high physical z.
For an in-memory Python reconstruction, write the same orientation explicitly:

```python
from pathlib import Path

import numpy as np
import tifffile
import tomojax as tj

recon = tj.load_reconstruction("recon.nxs")  # Or keep the tj.reconstruct result.
slices = Path("slices")
slices.mkdir()  # Refuse an existing directory rather than overwrite slices.
for z in range(recon.scan.grid.nz):
    plane = np.asarray(recon.volume[:, :, z], dtype=np.float32).T
    tifffile.imwrite(slices / f"slice_{z:04d}.tif", plane)
```

Keep the saved `.nxs` for geometry and correction records; these TIFFs hold only
floating-point pixels. The CLI export also writes a JSON geometry sidecar and
reads one disk slice at a time, avoiding a full-volume load.

Without grid/ROI overrides, automatic mode leaves region selection to the
Python workflow. With an explicit method, the default `--roi auto` may crop
to the detector field of view; `--roi off` preserves the recorded grid.
`--grid NX NY NZ` changes voxel counts and spacing together, preserving the
selected physical extent. Use `--roi off` to retain the whole recorded field of
view, or let `--roi auto` restrict it to the measured detector window first.
Automatic policies that cannot honor an override require an explicit method.
Check both field of view and physical units before comparing reconstructions.
`--roi cyl` also zeroes the volume outside the cylinder every view sees.

For supported parallel scans, `--method fourier` uses the Fourier-slice inverse,
with the volume on the host in bounded slabs. The backend uses matching available
CUDA support from the optional `fourier-cuda12` extra, or NumPy otherwise;
explicit `backend = "cupy"` or `"numpy"` in a config file fixes that choice.
In Python:

```python
from tomojax.recon import FourierConfig

recon = tj.reconstruct(scan, "fourier", config=FourierConfig(backend="cupy"))
```

Fourier is an approximate inverse with its own frequency cutoff, not an FBP
filter selection. It requires built-in parallel geometry without per-view poses;
cone beams and laminography are unsupported. It does not check angular coverage
or correct detector truncation. It refuses `filter`,
iterative-solver options and `warm_start` rather than silently ignoring them.

For FBP, FDK and Fourier volumes larger than host RAM, provide a writable
float32 memory map. The returned volume is that same array, written in bounded
slabs; input and output storage must not overlap. No full output copy is made:

```python
import numpy as np

g = scan.grid
out = np.memmap(
    "volume.dat", mode="w+", dtype=np.float32, shape=(g.nx, g.ny, g.nz), order="F"
)
recon = tj.reconstruct(scan, "fbp", out=out)
out.flush()  # Include this in timings that promise persisted output.
```

Fortran order keeps axial slices together for parallel FBP, FDK and Fourier,
avoiding page-strided file writes. Oblique/posed FBP writes x-slabs instead;
use C order there. Use separate input/output files. A failed reconstruction may
leave partial output. `out` is not supported by the iterative methods.

Expert settings are fields of the method's configuration class
(`tomojax.recon.FBPConfig`, `FourierConfig`, `CGLSConfig`, `FistaConfig` or `SPDHGConfig`),
set in a TOML file passed with `--config` alongside an explicit `--method`;
`tomojax recon --config-keys` lists them. Each replaces that field of the class's
defaults, and a setting the method does not take fails with the ones it does. For FISTA with Huber total
variation:

```bash
printf 'regulariser = "huber_tv"\nhuber_delta = 0.01\n' > huber.toml
uv run --no-sync tomojax recon corrected.nxs -o tv.nxs --method fista --config huber.toml
```

In Python this is
`tj.reconstruct(scan, "fista", config=FistaConfig(regulariser="huber_tv", huber_delta=0.01))`,
with `from tomojax.recon import FistaConfig`.
The keywords, like the command's options, replace the config's fields of the
same names: `iterations=100` wins over the config's `iterations`.

`tomojax align --mode cor` estimates the detector centre (the centre of
rotation). To compare trial centres by eye instead, shift the detector in
Python; the centre is in the geometry's length unit, so pixels times `du`:

```python
from dataclasses import replace

import tomojax as tj

scan = tj.load("corrected.nxs", poses=False)
d = scan.detector
for u_px in (-4, -2, 0, 2, 4):
    detector = replace(d, center=(u_px * d.du, d.center[1]))
    trial = replace(scan, geometry=replace(scan.geometry, detector=detector))
    tj.save(f"cor{u_px:+d}.nxs", tj.reconstruct(trial))
```

If the projections or the volume are larger than GPU memory, see
[large scans and several GPUs](large-scans.md).

For tilted or irregular-angle data, start with the iterative workflow in the
[real scan guide](real-laminography.md#reconstruct-with-the-recorded-geometry).
If motion remains plausible, use the [alignment guide](alignment-guide.md) after
checking geometry and preprocessing. A visually sharper aligned image does not
by itself establish accurate recovered poses.
