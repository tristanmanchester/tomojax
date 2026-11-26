# Reconstruct your first scan

This guide walks through inspecting, correcting, reconstructing, and exporting a real scan with the CLI and Python API. If you do not have scan data yet, try the [synthetic example](../README.md#first-reconstruction).

## Inspect the data and geometry

```bash
tomojax inspect scan.nxs
```

`inspect` prints the projection count and shape, angles, detector and voxel pitches, geometry type, and any recorded corrections, then exits 0 with `Valid: yes` (or lists problems and exits 1). Pass `--json` for machine-readable output. It recognizes NXtomo, NeXus scan files (including Diamond/GDA), APS Data Exchange, Nikon `.xtekct`, Zeiss `.txrm`, and ESRF acquisitions and inspects raw frames without applying corrections.

Reconstruction expects line integrals (log attenuation), not raw counts or normalized transmission. TomoJAX applies flat/dark correction and `-log` automatically when the file marks raw frames (NXtomo `image_key`, Data Exchange `data_white`/`data_dark`, Nikon, Zeiss, or ESRF). If flats or darks are absent, supply `flats=` (frames or a single white level) and `darks=` to `tj.load_frames` (pass `darks=0` only when counts are already dark-subtracted).

## Beamline exports

For single-row first-slice recipes covering ESRF NXtomo, PSI TOMCAT, Elettra SYRMEP, PSI cSAXS, and Diamond I13, see [beamline exports](beamline-exports.md).

### Raw ESRF acquisitions

- **SPEC EDF folders:** `tj.load("scan_folder", angles=...)` reads uncompressed single-image EDF projections with `.info` or XML metadata. `PixelSize` gives the effective pitch in micrometers; supply `angles=` because the scan range alone does not fix the starting angle.
- **PyHST `.par` files:** pass the parameter file explicitly with `tj.load_frames("scan.par").corrected()`. This reads the indexed EDF sequence, angles, detector pitch, rotation axis, reconstruction region, and dark/flat schedule for raw single-image axial parallel EDFs (directory discovery does not auto-select `.par`; multi-frame, cone, helical, binning, and distortion modes are unsupported).
- **Raw Bliss HDF5:** `tj.load("raw_bliss.h5")` reads basic, half-turn, and full-turn Bliss step sequences, including flats, darks, and recorded angles (select among multiple sequences with `data_path="1.1"`). Effective pixel sizes and distances are kept in millimeters. Helical and multi-volume sequences are unsupported.

Neither SPEC EDF nor raw Bliss records a rotation center; loading places the center at the detector middle (`detector_center_status="not_recorded"`). Estimate it with `tj.align(scan, fit="center")`. Single-row HDF5 sinograms stored as `(view, u)` stay lazy and appear as `(view, 1, u)`.

## Correct raw detector frames

For an HDF5 scan with sample, flat, and dark frames:

```bash
tomojax correct raw.nxs -o corrected.nxs
tomojax inspect corrected.nxs
```

`correct` computes `-log((I - D) / (F - D))`, interpolating flats between the flat sets bracketing each sample view, and records the processing history in `corrected.nxs`. Dataset paths are discovered automatically for NXtomo, Data Exchange, and NeXus scan files (following the default plot, its `NXdetector` `image_key`, and its plotted angular coordinate); use `--data-path`, `--image-key-path`, and `--angles-path` for other HDF5 layouts. If a scan stopped early and its key or angles have a different length from the frame stack, TomoJAX reads the frames covered by all records from the start and logs a warning with the counts. Add `--zingers`, `--stripes 9`, `--rings`, or `--reject-views` for artifact removal, and `--views` or `--crop` to subset the scan. Skip `correct` if your data are already line integrals.

When an external file omits metadata:

- **Missing pixel pitch (parallel beam):** loads in pixel units (`length_unit: "pixel"`); override with `pixel_size=`, `detector=`, or `--pixel-size`. Cone-beam files require a physical pixel size.
- **Missing rotation center:** defaults to the detector middle (`detector_center_status="not_recorded"`). Supply `detector=` / `--detector-center U V`, or fit it afterwards with `tj.align(scan, fit="center")` (or `method="opposite-views"` for a fast projection-only axial estimate).
- **Missing angles or laminography tilt:** supply `angles=` or a complete `geometry=`. Explicit `detector=` or `geometry=` overrides are always used as given.

In Python, `tj.load("raw.nxs")` runs the default flat/dark/`-log` correction. For cropping, view selection, or extra steps from `tomojax.corrections`, load frames first:

```python
import tomojax as tj
from tomojax.corrections import RejectViews, Stripes

frames = tj.load_frames("raw.nxs")  # also works on TIFFs (with angles=) or Nikon .xtekct
scan = frames.cropped(slice(100, 900), slice(None)).corrected(RejectViews(), Stripes(9))
print(scan)
tj.save("corrected.nxs", scan)
```

HDF5 and TIFF sample, flat, and dark stacks are read lazily during correction, so call `.cropped(...)` or `.selected(...)` before `.corrected(...)` to avoid reading unused pixels (`np.asarray(frames.counts)` materializes the selection). Selection accepts a forward slice, increasing integer indices, or a boolean mask, preserving each view's geometry and its position between flat sets. Call `scan.corrected(...)` to apply line-integral steps (`Stripes`, `Rings`, `BeamHardening`, `RejectViews`) to an already-corrected `Scan`.

### Ring and stripe removal

If rings remain after `Stripes(9)`, a defective pixel's response may vary with intensity rather than by a constant offset. Chain `Stripes(51, varying=True)` after `Stripes(9)` on densely sampled views (crop detector rows to save memory rather than thinning views first, because wide angular gaps can confuse moving sample edges with bad pixels):

```python
import tomojax as tj
from tomojax.corrections import Stripes

frames = tj.load_frames("raw.nxs").cropped(slice(100, 116), slice(None))
scan = frames.corrected(Stripes(9), Stripes(51, varying=True))
preview = scan.selected(slice(None, None, 4))  # thin views only after stripe correction
```

Widths are in detector columns. Use the narrowest width that covers the stripes, since wide windows can alter stationary centered sample features. For full statistical/full-turn detector-response modeling, see `Rings` in [tomojax.corrections](../src/tomojax/corrections/README.md).

![A raw detector frame of a sandstone plug with a band of 16 rows marked, the corrected sinogram of one row, its FBP slice and an enlarged region of the slice.](images/i12-first-slice.png)

Diamond I12 sandstone (Vo et al., Zenodo 3339629; 1801 views of 2560-pixel rows): raw frame with 16 cropped rows marked, middle-row sinogram after `Stripes(9)` and `Stripes(51, varying=True)`, and FBP slice at axis column 1417.5:

```bash
uv run python examples/gallery/first_slice.py full_rows1048-1111.nxs i12-first-slice.png \
  --rows 24 40 --center -138 --crop 1500 1100 --title "Diamond I12, Vo et al."
```

### Binning and fast previews

Pass `binning=N` to `frames.corrected(...)` (or `--binning N` to `tomojax correct`) to average `N × N` blocks of line integrals on the device; when only row-local steps follow `-log`, uncompressed files are binned slab-by-slab without holding the unbinned stack in host RAM.

Binning reduces detector resolution without changing a recorded reconstruction grid. Pair it with a matching coarse grid for fast axial parallel previews:

```python
import tomojax as tj

scan = tj.load_frames("raw.nxs").selected(slice(None, None, 4)).corrected(binning=8)
scan = tj.align(scan, method="opposite-views").scan
d = scan.detector
grid = tj.Grid(d.nu, d.nu, d.nv, d.du, d.du, d.dv, vol_center=(0, 0, d.center[1]))
tj.save("preview.nxs", tj.reconstruct(scan, grid=grid))
```

On the CLI, `--grid NX NY NZ` resamples the chosen field of view to those voxel counts while preserving physical extent (in Python, `scan.grid.resampled(shape=(NZ, NY, NX))`, matching `grid.shape`).

### Lens distortion and Paganin phase retrieval

For calibrated lens distortion, pass `RadialDistortion(center=(col, row), coefficients=(c0, c1, ...))` from `tomojax.corrections`. It applies a backward radial polynomial on transmission before `-log` (`center` is in zero-based image pixels, not the rotation axis; estimate the rotation center afterwards, and apply before cropping if pixels outside the crop are needed).

For propagation-based phase contrast, `Paganin` filters transmission before `-log`. Supply the material's `delta_beta`, effective propagation `distance` in meters (sample-to-detector distance divided by magnification in cone beam), X-ray `energy_kev`, and sample-plane `pixel_size` in meters:

```python
import tomojax as tj
from tomojax.corrections import Paganin

step = Paganin(delta_beta=100, distance=0.1, energy_kev=25, pixel_size=0.65e-6)
scan = tj.load_frames("raw.nxs").corrected(step)
scan = scan.cropped(slice(100, 900), slice(None))  # crop after Paganin, not before
```

Because Paganin convolves neighboring pixels, crop after phase retrieval (or correct a padded window) rather than cropping raw frames first. Transmission steps run in the order given: put `Zingers` first, then `RadialDistortion`, then `Paganin`. On the CLI, pass `--paganin 100,0.1,25,0.65e-6` or a `--config` TOML file to `tomojax correct` (`tomojax correct --config-keys` lists every key).

### TIFF frames

For TIFF stacks with flat/dark folders and angle sidecars, see [preparing TIFF data](real-laminography.md#prepare-tiff-data).

## Reconstruct and inspect slices

```bash
tomojax reconstruct corrected.nxs -o recon.nxs --manifest recon-manifest.json
tomojax inspect recon.nxs --preview-dir previews
```

The default `--method auto` chooses a solver from the geometry:

- Unposed well-sampled axial `ParallelGeometry` -> `fourier` (otherwise `fbp`)
- `LaminographyGeometry` or posed `ParallelGeometry` -> `fbp`
- Single-arrangement `ConeGeometry` -> `fbp` (dispatched to FDK), or `cgls` for an arc shorter than 180° plus the fan angle
- Multi-orbit `ConeSegments` -> `cgls`

Pass `--method` explicitly to fix a solver (`fourier`, `fbp`, `cgls`, `fista`, `spdhg`, `sirt`, `mlem`). Add `--dry-run` to inspect the resolved method, grid, and memory estimate before running, or `--watch-dir DIR` to rewrite central-slice PNGs at each checkpoint of an iterative solve. `recon.nxs` stores the volume, geometry, and record (add `--include-scan` to keep projections too), `--manifest` writes the resolved method and settings to JSON, and `inspect --preview-dir` writes display-scaled PNGs of the central slices (such as `slice_z_64.png`, plus `projection.png` when present). Use the floating-point volume in `recon.nxs` for quantitative analysis.

### Region and grid options

By default, reconstruction keeps the scan's recorded grid (matching `region=None` in Python). Pass `--region auto` (or `bbox`, `cube`, `cyl`) to crop to the detector field of view; `--region cyl` also zeros voxels outside the cylinder seen by every view after solving. `--grid NX NY NZ` changes voxel counts while preserving the selected physical field of view.

### Fourier-slice inverse

`--method fourier` (or `tj.reconstruct(scan, "fourier")`) runs a fast Fourier-slice inverse in host slabs for axial parallel scans without per-view rotational poses. It uses CuPy on CUDA when available (`tomojax[cuda12]` or `tomojax[fourier-cuda12]`) and NumPy otherwise; force a backend with `FourierConfig(backend="cupy")` or `"numpy"`. Use `tj.reconstruct(scan, "fourier", filter="ramp")` or `--method fourier --filter ramp` for the detector response of default FBP. The same option accepts `hann` and `shepp-logan`. For streaming, pass `config=FourierConfig(filter="ramp")` to `tj.reconstruct_slabs`. Omitting the filter keeps the sharper native Fourier response.

### Expert settings and truncated scans

Pass solver-specific settings via config classes from `tomojax.recon` (`FBPConfig`, `FourierConfig`, `CGLSConfig`, `FISTAConfig`, `SPDHGConfig`, `SIRTConfig`, `MLEMConfig`) or a `--config` TOML file alongside an explicit `--method` (`tomojax reconstruct --method fista --config-keys` lists one method's keys):

```bash
printf 'regularizer = "huber_tv"\nhuber_delta = 0.01\n' > huber.toml
tomojax reconstruct corrected.nxs -o tv.nxs --method fista --config huber.toml
```

In Python:

```python
from tomojax.recon import FISTAConfig

recon = tj.reconstruct(scan, "fista", config=FISTAConfig(regularizer="huber_tv", huber_delta=0.01))
```

Direct keywords (such as `iterations=100`) override fields of the same name in `config`.

- **Material outside the field of view:** CGLS, FISTA, and SPDHG fit absolute line integrals by default. Pass `data_term="per_view_contrast"` in the config to profile out one constant offset per view (absolute background level is then undetermined).
- **FISTA TV weight:** `tv_weight=None` estimates detector noise and interpolation uncertainty, backprojects their scale, and accounts for the TV cost of an isolated voxel. Pass a positive number to fix it, or `0` to disable TV.

### Export slices

```bash
tomojax export recon.nxs -o slices/                 # 32-bit TIFF z-slices + JSON sidecar
tomojax export recon.nxs -o recon.raw --dtype uint16
```

Each z-slice has y rows and x columns from low to high physical z. `tomojax export` streams one slice at a time without loading the full volume. In Python:

```python
from pathlib import Path
import numpy as np
import tifffile
import tomojax as tj

recon = tj.load_reconstruction("recon.nxs")
slices = Path("slices")
slices.mkdir()
for z in range(recon.grid.nz):
    tifffile.imwrite(slices / f"slice_{z:04d}.tif", np.asarray(recon.volume[z], dtype=np.float32))
```

### Checking trial rotation centers by eye

Alongside `tomojax align --fit center`, sweep candidate detector centers on one slice and compare them on a single gray scale:

```bash
tomojax inspect corrected.nxs --preview-dir sweep --centers -20:20:2
tomojax correct corrected.nxs -o centered.nxs --detector-center U V
```

`sweep/center_sweep.png` shows every candidate side by side with the recorded center outlined, and `inspect` prints the `U` value to pass to `--detector-center` for each candidate. In Python, `tomojax.interactive.center_sweep(scan, "z", scan.grid.nz // 2, (-4, -2, 0, 2, 4))` returns the same stack of planes directly without writing intermediate volumes.

### Next steps

- **Volumes larger than GPU or host memory** (including writing straight to a memmap with `out=`): [large scans and several GPUs](large-scans.md).
- **Laminography and TIFF stacks:** [real parallel and laminography scans](real-laminography.md).
- **Lab cone-beam CT:** [lab cone-beam CT](lab-ct.md).
- **Motion and setup-angle correction:** [align a scan](align-a-scan.md).

## Watch a Python operation

Use a watch around loading, correction, reconstruction, or alignment. The
handler runs on the calling thread and must return quickly. The innermost
watch receives the events. Each event is an immutable snapshot. On a
`Started` event, use `name` to identify the operation and `label` for display.
For example, a `level` task can have the label `resolution 1/2`, and a
`reconstruct` task can have the label `cgls`.

```python
import tomojax as tj
import tomojax.progress as progress

stop = progress.Control()

def report(event):
    if isinstance(event, progress.Advanced):
        print(event.task, event.done, event.total, dict(event.metrics))

with progress.watch(report, control=stop):
    scan = tj.load("scan.nxs")
    result = tj.reconstruct(scan, "cgls")
```

Call `stop.pause()` to hold work at its next checkpoint and `stop.resume()`
to continue. The task emits `Paused` and `Resumed` on its owner thread.
The ETA on advances uses active elapsed time and excludes the pause.
A `Placed` event names the devices doing the work, in coordinator order.
Call `stop.cancel()` from the handler or another thread to request cancellation.
The operation raises `progress.Canceled` at the next work boundary and closes
its files and workers. A caller-supplied output array can contain completed
slabs. Cancellation requested on the last completed unit preserves the result.
An exception raised by the handler reaches the caller and ends event delivery.

Solver updates count full iterations, including a full sweep for subset methods.
Alignment reports each completed Gauss–Newton step with its diagnostics.
A raw scan can report views read while its correction group reports batches
written. A single-dispatch read or solve can finish with no advances.
For `tj.reconstruct_slabs`, the task spans the `with` block. Each yielded slab
counts once, and leaving the block early closes the task.

`progress.to_json(event)` returns the same dictionary that the CLI writes in
JSON mode. Missing metrics and non-finite numbers become JSON null.
See the [command-line guide](../src/tomojax/cli/README.md#follow-or-cancel-a-command)
and the [record schema](protocol/progress.schema.json).
