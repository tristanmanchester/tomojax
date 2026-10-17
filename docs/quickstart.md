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
geometry type. Raw NXtomo, APS Data Exchange and recognised Diamond/GDA layouts
use the same acquisition discovery as loading; optional NXtomo labels are not
required. `inspect` also checks the acquisition metadata: it ends with
`Valid: yes`, or lists the issues and exits with status 1. That check cannot
prove that metadata matches the instrument. `--json` prints the report as JSON
instead. It also lists the corrections that made the projections from
detector frames, when the file records them.

Reconstruction expects line integrals (log attenuation). Raw detector counts,
normalised transmission, and already-corrected attenuation are different
inputs: establish which you have. TomoJAX corrects raw frames itself when the
file says they are raw: an NXtomo `image_key` marking flats or darks, an APS
Data Exchange file with `data_white` or `data_dark` beside `/exchange/data`,
or a Nikon scan. It refuses integer counts with no flats rather than guess.

## Correct raw detector frames

For an HDF5 scan containing sample, flat, and dark frames:

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
them otherwise. Options add corrections: `--zingers`, `--remove-stripes 9`
(rings), `--reject-outliers` (views that jump from their neighbours, such as a
closed shutter), `--beam-hardening 1,0.05`; `--select-views`, `--reject-views`
and `--crop` keep part of the scan. Already-corrected data should skip this
step; applying the logarithm again changes the data.

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
corrected: crop or select first to read less. `np.asarray(frames.counts)` reads
the current selection into memory.

The views are corrected on the GPU a batch at a time, read from the file as
they are needed. `batch_views` must be a positive integer and `epsilon` finite
and positive. `frames.selected(...)` and `scan.selected(...)` keep a nonempty
selection in acquisition order: a forward slice, increasing integer indices,
or a Boolean mask. Selection keeps each view's geometry and its original
position between the flat fields. `scan.corrected(...)` runs line-integral
steps (`Stripes`, `BeamHardening`, `RejectViews`) on a scan that is already
corrected.

`corrected(..., binning=2)` averages 2 x 2 detector blocks of the line integrals,
as `scan.binned(2)` does. With only `Stripes` and the like after the log, it
bins slabs of rows read straight from an uncompressed file, so the unbinned
stack never reaches host memory.

For calibrated lens distortion, pass
`RadialDistortion(center=(column, row), coefficients=(c0, c1, ...))` from
`tomojax.corrections`. It bilinearly samples a **backward** radial polynomial
on transmission before the log. The centre is in zero-based image pixels,
not the rotation axis; estimate the axis after this correction. Correct
before cropping if the warp needs pixels outside the crop, and convert the
calibration after a crop or binning (see `help(RadialDistortion)`).

A file written by TomoJAX carries its geometry. Other NXtomo and Data Exchange
files get a parallel beam with a centred detector, the file's angles, and the
NXtomo detector's `x_pixel_size` and `y_pixel_size` when it has them. Without
them TomoJAX takes unit pixels, so the reconstruction's length unit is the
pixel. To set the geometry, pass the measured detector,
`tj.load_frames("raw.h5", detector=tj.Detector(nu=2048, nv=2048, du=0.65, dv=0.65))`,
or a whole `geometry=`, with your own detector shape and pitch. Then estimate
the centre of rotation with `tomojax align --mode cor` (see the
[alignment guide](alignment-guide.md)). For an unposed axial parallel scan,
`scan = tj.estimate_center(scan)` estimates physical detector-u without a
volume solve; inspect the result, especially for very compact objects that
can produce false Fourier minima.

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
the selected method, reconstruction settings and automatic outcome. `previews/`
contains PNGs of the central projection (`projection.png`) and the volume's
three central planes (`slice_z.png`, `slice_y.png`, `slice_x.png`). PNG contrast is scaled for
display; read the stored floating-point volume for quantitative work.
`tomojax export recon.nxs -o slices/` writes the volume as TIFF slices for other
software.

Without grid/ROI overrides, automatic mode leaves region selection to the
Python workflow. With an explicit method, the default `--roi auto` may crop
to the detector field of view; `--roi off` preserves the recorded grid.
`--grid NX NY NZ` changes dimensions while keeping the selected voxel spacing.
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
