# Reconstruct your first scan

This guide walks through inspecting, correcting, reconstructing, and exporting a real scan with the CLI and Python API. If you do not have scan data yet, try the [synthetic example](../README.md#first-reconstruction).

## Inspect the data and geometry

```bash
tomojax inspect scan.nxs
```

`inspect` prints a `Loading` section (whether the file loads with no further arguments or needs one, pixel units when no pixel size is recorded, an unrecorded rotation center or rotation axis, and the frame counts a stopped scan was cut to), followed by the projection count and shape, angles, detector and voxel pitches, geometry type, and any recorded corrections. It exits 0 with `Valid: yes`, or lists problems and exits 1. Pass `--json` for machine-readable output (the same structure `tj.inspect(path)` returns, with `loading`). Given a checkpoint directory from `--checkpoint DIR`, `tomojax inspect DIR` reports the saved run, the command or operation that started it, its save time, and whether a live process holds it. It recognizes NXtomo, NeXus scan files (including Diamond/GDA), APS Data Exchange, Nikon `.xtekct`, Zeiss `.txrm`, and ESRF acquisitions and inspects raw frames without applying corrections.

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

`correct` computes `-log((I - D) / (F - D))`, interpolating flats between the flat sets bracketing each sample view, and records the processing history in `corrected.nxs`. Dataset paths are discovered automatically for NXtomo, Data Exchange, and NeXus scan files (following the default plot, its `NXdetector` `image_key`, and its plotted angular coordinate); use `--data-path`, `--image-key-path`, and `--angles-path` for other HDF5 layouts. If a scan stopped early and its key or angles have a different length from the frame stack, TomoJAX reads the frames covered by all records from the start and logs a warning with the counts. Add `--zingers`, `--rings`, or `--reject-views` for artifact removal, and `--views` or `--crop` to subset the scan. Skip `correct` if your data are already line integrals.

When an external file omits metadata:

- **Missing pixel pitch (parallel beam):** loads in pixel units (`length_unit: "pixel"`); override with `pixel_size=`, `detector=`, or `--pixel-size`. Cone-beam files require a physical pixel size.
- **Missing rotation center:** defaults to the detector middle (`detector_center_status="not_recorded"`). Supply `detector=` / `--detector-center U V`, or fit it afterwards with `tj.align(scan, fit="center")`, which measures the axis directly across the full detector on axial parallel scans without fitting a volume (pass `method="opposite-views"` for a pair-only check).
- **Missing rotation axis:** a file that records no rotation axis or pixel scale logs a warning on load (`loading["geometry"] == "not_recorded"` in `inspect`). Reconstruction and alignment refuse it until geometry is declared when loading: pass `geometry=...` to `tj.load`, `--geometry parallel` (or `lamino` / `cone`) to `tomojax correct`, or `geometry: {"kind": "parallel"}` to worker or MCP `open`.
- **Missing angles or laminography tilt:** supply `angles=` or a complete `geometry=`. Explicit `detector=` or `geometry=` overrides are always used as given.

In Python, `tj.load("raw.nxs")` runs the default flat/dark/`-log` correction. For cropping, view selection, or extra steps from `tomojax.corrections`, load frames first:

```python
import tomojax as tj
from tomojax.corrections import RejectViews, Rings

frames = tj.load_frames("raw.nxs")  # also works on TIFFs (with angles=) or Nikon .xtekct
scan = frames.cropped(slice(100, 900), slice(None)).corrected(Rings(), RejectViews())
print(scan)
tj.save("corrected.nxs", scan)
```

HDF5 and TIFF sample, flat, and dark stacks are read lazily during correction, so call `.cropped(...)` or `.selected(...)` before `.corrected(...)` to avoid reading unused pixels (`np.asarray(frames.counts)` materializes the selection). Selection accepts a forward slice, increasing integer indices, or a boolean mask, preserving each view's geometry and its position between flat sets. Call `scan.corrected(...)` to apply line-integral steps (`Rings`, `BeamHardening`, `RejectViews`) to an already-corrected `Scan`.

### Ring removal

Use `Rings()` in Python or `--rings` on the CLI to remove fine and wide detector
rings. The step needs no width or tuning parameters; `frames.corrected`,
`scan.corrected`, `tj.reconstruct_slabs`, and `tomojax correct --rings` bind the
scan geometry automatically.

```bash
tomojax correct raw.nxs -o corrected.nxs --rings
```

```python
from tomojax.corrections import Rings

frames = tj.load_frames("raw.nxs").cropped(slice(100, 116), slice(None))
scan = frames.corrected(Rings())
```

Crop detector rows for a quick preview so every view still contributes to the
fit. `Rings()` models each column's additive offset and transmission-dependent
response: on half-turn scans it estimates wide offsets from persistent jumps
between columns as the sample rotates, and on stationary parallel-beam full
turns it also compares opposed angular arcs.

`Rings()` corrects stationary column errors only. Artifacts that vary with angle
(bad or drifting flat fields, flux drops, or angular swirls) are not detector
rings and are not removed, and this step does not interpolate dead pixels. A
circular sample feature centered on the rotation axis gives the same
measurements as a symmetric detector offset, so compare corrected and
uncorrected slices near the axis.

![Three rows of slices around the rotation axis, from a half turn with fine rings, a half turn with wide rings and a full turn. Each row shows the slice without a ring step, with Rings(), and what Rings() removed, which is a set of concentric rings. The fine rings and the full-turn rings disappear; on the wide-ring scan most rings disappear, while some broad rings and the structure at the axis remain.](images/ring-removal.png)

One detector row of three real scans, each with its rotation center measured by `tj.align(scan, fit="center")` and reconstructed by the default `tj.reconstruct` without and with `Rings()`. Diamond K11 (DIAD) scan 52567 and setup scan 52510 are half turns of 8000 views on 2560 pixels of 0.54 µm; Diamond I13 scan 159950 is a full turn of 3001 views. Each panel shows 380 pixels around the axis, and the two slices in a row share one gray range.

```bash
uv run python examples/gallery/ring_removal.py k11-52567-row1080.nxs \
  k11-52510-row1080.nxs 159950-row1035.nxs ring-removal.png
```

By default `Rings()` fits on 512 angular bin means, so it adds about 10 s on a
half turn and 20 s on a full turn of an 8,000-view 2160 × 2560 streamed Fourier
volume on one V100S, with ~15 MiB extra GPU memory. See
[ring removal](../src/tomojax/corrections/README.md#ring-removal) and
[ring measurements](measurements.md#ring-correction).

![A raw detector frame of a sandstone plug with a band of 16 rows marked, the corrected sinogram of one row, its FBP slice and an enlarged region of the slice.](images/i12-first-slice.png)

Diamond I12 sandstone (Vo et al., Zenodo 3339629; 1801 views of 2560-pixel rows): raw frame with 16 cropped rows marked, middle-row sinogram after ring correction, and FBP slice at axis column 1417.5:

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
scan = tj.align(scan, fit="center").scan
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

Pass `--method` explicitly to fix a solver (`fourier`, `fbp`, `cgls`, `fista`, `spdhg`, `sirt`, `mlem`). Add `--dry-run` to inspect the resolved method, grid, and memory estimate before running, `--watch-dir DIR` to rewrite central-slice PNGs as an iterative solve progresses, or `--checkpoint DIR` to save state on disk so a canceled or killed run resumes when repeated ([resume a run after a kill](large-scans.md#resume-a-run-after-a-kill)). For iterative solvers, `tomojax reconstruct` ends by printing how many updates ran, why the solve stopped, and any automatic TV weight. `recon.nxs` stores the volume, geometry, and record (add `--include-scan` to keep projections too), `--manifest` writes the resolved method and settings to JSON, and `inspect --preview-dir` writes display-scaled PNGs of the central slices (such as `slice_z_64.png`, plus `projection.png` when present). Use the floating-point volume in `recon.nxs` for quantitative analysis.

### Region and grid options

By default, reconstruction keeps the scan's recorded grid (matching `region=None` in Python). Pass `--region auto` (or `bbox`, `cube`, `cyl`) to crop to the detector field of view; `--region cyl` also zeros voxels outside the cylinder seen by every view after solving. `--grid NX NY NZ` changes voxel counts while preserving the selected physical field of view.

### Fourier-slice inverse

`--method fourier` (or `tj.reconstruct(scan, "fourier")`) runs a fast Fourier-slice inverse in host slabs for axial parallel scans without per-view rotational poses. It uses CuPy on CUDA when available (`tomojax[cuda12]` or `tomojax[fourier-cuda12]`) and NumPy otherwise; force a backend with `FourierConfig(backend="cupy")` or `"numpy"`.

By default (`filter=None`), Fourier uses its native response with a detector-Nyquist disk cutoff: edges stay sharper than in FBP, while high-frequency noise is higher. Pass `filter="ramp"` (`tj.reconstruct(scan, "fourier", filter="ramp")`, `--method fourier --filter ramp`, or `config=FourierConfig(filter="ramp")` in `tj.reconstruct_slabs`) for resolution and noise close to default FBP at Fourier's speed and GPU memory. On a scan with fewer views than detector columns the match is close near the rotation axis and weakens towards the edge of the field of view, as the figure below shows. The same option accepts `"hann"` and `"shepp-logan"` for stronger high-frequency smoothing.

![Three enlargements of one region of a sandstone slice from Diamond I12, reconstructed by FBP, by Fourier with its default response and by Fourier with filter="ramp", above the whole FBP slice and each Fourier slice minus FBP. The default Fourier slice is grainier than FBP and differs from it by strong fine-grained noise; the filter="ramp" slice looks like FBP and its difference is fainter.](images/fourier-response.png)

Detector row 1080 of the Diamond I12 scan of Vo et al. (Zenodo 3339629; 1801 views of 2560 pixels), ring-corrected and reconstructed three ways on one grid. In the enlarged region, about 280 pixels from the rotation axis, the RMS difference from FBP is 0.32 of FBP's standard deviation with the default response and 0.15 with `filter="ramp"`. More than about 700 pixels from the axis, both Fourier slices differ from FBP by similar amounts.

```bash
uv run python examples/gallery/fourier_response.py full_rows1048-1111.nxs fourier-response.png
```

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
- **FISTA TV weight and stopping:** With `tv_weight=None` and `tolerance=None` (the defaults), FISTA estimates detector noise and linear interpolation error from the projections, propagates that scale through the backprojector to set the TV weight, and stops once the proximal gradient reaches the estimated noise level or at `iterations=200`. Check `recon.info` (or `--manifest`, or the closing line of `tomojax reconstruct`) for `resolved_tv_weight` (dimensionless), `effective_tv_weight` (dimensional), `noise_score_rms`, `effective_iterations`, `early_stop`, and `termination` (`"noise_score"`, `"objective_tolerance"`, or `"iteration_limit"`). Pass `tv_weight` explicitly (positive to fix the weight, `0` to disable TV) to bypass calibration and run to `iterations`; pass `FISTAConfig(tolerance=0)` to run an automatic weight for the full budget, or set `iterations` to change the cap. Remove rings before running automatic FISTA, because coherent detector errors survive scalar TV calibration.

![One region of a cathode slice from Diamond K11, reconstructed by FBP and by FISTA with its automatic TV weight, each above an enlarged part. The FBP slice is grainy; the FISTA slice keeps the particles and their edges without the grain.](images/fista-tv-weight.png)

Detector row 1080 of Diamond K11 (DIAD) scan 52567: 8000 views over a half turn on 2560 pixels of 0.54 µm, corrected with `Rings()` and reconstructed by `tj.reconstruct(scan, "fbp")` and `tj.reconstruct(scan, "fista")`. FISTA chose `resolved_tv_weight` 8.2e-6 and ran to the 200-update cap. All four panels share one gray range.

```bash
uv run python examples/gallery/fista_tv_weight.py k11-52567-row1080.nxs fista-tv-weight.png
```

### Iterative solver defaults and stopping

Unregularized iterative solvers (CGLS and SIRT) fit noise and discretization error if run for hundreds of updates, even as their fitted projection residual keeps falling. Stopping near the error level acts as their regularization. When you omit `iterations=` (`--iterations`), each iterative solver follows one of three stopping rules:

| Solver | Default stopping rule | Work cap / budget | Override for a fixed budget or plain path |
| --- | --- | --- | --- |
| `fista` | Proximal gradient falls below one estimated noise-score SD (`tv_weight=None`, `tolerance=None`) | Up to 200 updates | Pass `tv_weight=...`, `FISTAConfig(tolerance=0)`, or `iterations=N` |
| `cgls` | Projection residual RMS falls below estimated detector noise plus sampling error (when unpenalized and `tolerance=None`) | Up to 50 updates | Pass `iterations=N`, explicit tolerance, or `damping` / `gradient_damping` |
| `sirt` (full data, `subsets=1`) | Same projection-discrepancy stop, with FISTA momentum enabled when `relaxation <= 1` | Up to 50 sweeps | Pass `iterations=N` (plain sweeps; add `SIRTConfig(acceleration=True)` for momentum) or `SIRTConfig(acceleration=False)` for plain automatic SIRT |
| `sirt` (`subsets > 1`), `mlem` | Fixed sweep budget (no momentum) | 50 sweeps | Pass `iterations=N` |
| `spdhg` | Fixed stochastic block-update budget | 600 block updates | Pass `iterations=N` |

Check `recon.info`, the JSON `--manifest`, or the closing line of `tomojax reconstruct` after an iterative run:

- `termination` names why the solve ended: `"projection_discrepancy"` (CGLS/SIRT) or `"noise_score"` (FISTA) when the solver stopped at the estimated error level, `"objective_tolerance"` (FISTA with an explicit `tolerance`), `"converged"`, `"roundoff_limit"`, or `"numerical_breakdown"` (CGLS), or `"iteration_limit"` when the budget ran out. `early_stop` is `False` when the budget was exhausted.
- `effective_iterations` gives the completed updates or sweeps, `measurement_noise_rms` gives the estimated detector noise SD, and `projection_error_rms` gives the total discrepancy threshold in projection units.
- CGLS reports the final `residual_rms`; SIRT reports `pre_update_residual_rms` (measured at the point projected for the last sweep, including momentum when accelerated) and the resolved `acceleration` flag. On independent axial SIRT slabs, `slab_stopping` lists each slab's range, sweep count, and residual, while top-level `effective_iterations` is the largest sweep count across slabs.

On real scans, motion, scatter, calibration bias, and correlated detector errors often keep the projection residual above the local noise-and-sampling estimate past 50 updates, so `termination="iteration_limit"` is common. That result means the cap ended the solve without detecting the scan's noise floor. Inspect the slices, pass an explicit `iterations=` if structure is still under-resolved, or switch to `fista` when noise needs explicit TV regularization. To replay an automatic run identically, pass `iterations=recon.info["effective_iterations"]` for CGLS, and pass both `iterations` and `acceleration` from `recon.info` for SIRT.

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

- **Volumes larger than GPU or host memory, several GPUs, and checkpoint/resume for cluster jobs:** [large scans and several GPUs](large-scans.md).
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

Call `stop.pause()` to hold work in memory at the next work boundary (the end of an iteration, slab, or batch) and `stop.resume()` to continue; unlike `checkpoint=`, pausing does not write state to disk or survive a killed process. The task emits `Paused` and `Resumed` on its owner thread.
The ETA on advances uses active elapsed time and excludes the pause.
A `Placed` event names the devices doing the work, in coordinator order.
Call `stop.cancel()` from the handler or another thread to request cancellation.
The operation raises `progress.Canceled` at the next work boundary and closes
its files and workers. When `checkpoint=` is set, a checkpointed loop saves at
its next boundary before raising `progress.Canceled`, whose `checkpoint`
attribute names the directory. A caller-supplied output array can contain completed
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
