# Alignment guide

TomoJAX alignment estimates geometry and per-view pose corrections while it
reconstructs the volume. It is experimental: on every scan, check the
recovered parameters as well as the image.

Start from corrected line integrals and checked physical geometry, following
the [real scan guide](real-laminography.md) or the
[lab cone-beam CT guide](lab-ct.md). Save an unaligned reconstruction to
compare with. Commands run from the checkout root.

## Choose an alignment mode

| Problem | Mode | Command |
| --- | --- | --- |
| Choose a data-dependent workflow and inspect its limitations | `auto` (the default) | `tomojax align scan.nxs -o aligned.nxs` |
| The sample moved from projection to projection | `pose` | `tomojax align scan.nxs -o aligned.nxs --mode pose` |
| The detector centre (centre of rotation) is wrong | `cor` | `tomojax align scan.nxs -o aligned.nxs --mode cor` |
| A detector-centre offset and per-view motion together | `cor-then-pose` | `tomojax align scan.nxs -o aligned.nxs --mode cor-then-pose` |
| Mild setup error (centre, detector roll, axis direction) and motion | `full` | `tomojax align scan.nxs -o aligned.nxs --mode full` |
| A multi-orbit cone-beam scan, or a scan that already carries poses | `pose` | `tomojax align scan.nxs -o aligned.nxs --mode pose` |
| The reference elevation or a detector-v shift is uncertain | None: check it by hand | `det_v_px` is not a reliably recoverable alignment target |

On cone-beam data, `cor` estimates the rotation axis's offset and the detector
roll instead of a detector shift; see
[cone-beam scans](#align-cone-beam-scans-in-six-degrees-of-freedom).

## Run an alignment

```bash
uv run --no-sync tomojax align corrected.nxs -o aligned.nxs --mode pose
uv run --no-sync tomojax inspect aligned.nxs
uv run --no-sync tomojax recon aligned.nxs -o recon.nxs --method cgls
```

Automatic alignment records its selected workflow and any limitations in
`result.info["automatic"]`; the CLI prints limitations without requiring a
manifest. These conditional checks do not certify physical pose truth.

The aligned file holds the corrected scan, its per-view poses and the volume
the alignment converged with, and `tomojax recon` applies the poses.
`tomojax align` corrects on top of any poses saved in its input; with either
command, `--no-poses` starts from the nominal geometry instead.

`tomojax align` is `tj.align` on the command line: its options are
`tj.align`'s keywords (`--mode`, `--quality`, `--levels`, `--freeze`, `--grid`,
`--checkpoint`). `--quality reference` runs slower, more conservative solver
settings, and `--levels 4 2 1` sets the coarse-to-fine binning factors.
`--dry-run` prints the resolved plan for an explicit mode as JSON without
aligning. Automatic mode reports that calibration and region selection are
data-dependent; it does not run them during a dry run.

Expert settings are `tomojax.alignment.AlignConfig` fields: put them in a TOML
file and pass it with `--config FILE` and an explicit `--mode`. Each replaces
that field of the configuration the mode and quality give, which `--dry-run` prints;
`tomojax align --config-keys` lists the fields with the pose profile's values.
The [alignment solver reference](alignment-solver.md) describes them.

In Python:

```python
from dataclasses import replace

import tomojax as tj
from tomojax.alignment import alignment_plan

scan = tj.load("corrected.nxs")
result = tj.align(scan, mode="pose")
tj.save("aligned.nxs", result)
recon = tj.reconstruct(result.scan, "cgls")  # result.scan carries the corrections

# An expert setting: replace one field of the mode's configuration.
config = replace(alignment_plan("pose", scan.grid).config, ray_integrator="exact")
result = tj.align(scan, mode="pose", config=config)
```

`scan.with_poses(table)` attaches externally estimated or interpolated corrections
in `Scan.poses`'s detector frame; it replaces, rather than adds to, existing poses.
`scan.with_poses(None)` removes poses without undoing setup calibration.

`result.poses` is the `(views, 6)` pose table and `result.info` the solver's
record.

### Align large scans at reduced resolution

Pose parameters are physical lengths and angles, so they transfer between
resolutions. For a large scan, stop the coarse-to-fine schedule early and
reconstruct the full data with the recovered poses:

```bash
uv run --no-sync tomojax align corrected.nxs -o aligned.nxs \
  --mode pose --levels 4 2
uv run --no-sync tomojax recon aligned.nxs -o recon.nxs \
  --method fista --nonnegative --tv-weight 0 --iterations 300
```

Laminography leaves a cone of frequencies unmeasured, so the full-resolution
reconstruction needs a prior: positivity, as here, and TV (`--tv-weight`) for
noisy data. On a 256³ laminography scan, stopping at half resolution took a
quarter of the time for slightly less accurate rotations
([measurements](measurements.md#per-view-pose-recovery)). When the detector
samples finer than the grid, binning it also helps; see
[large scans](large-scans.md#bin-detectors-finer-than-the-grid).

### Align on a region or another grid

`grid=` aligns and reconstructs on another grid than the scan's, a region
around the object or coarser voxels, as it does for `tj.reconstruct`. The
returned volume is on that grid and `result.scan` carries it:

```python
region = tj.Grid(192, 192, 128, scan.grid.vx, scan.grid.vy, scan.grid.vz)
result = tj.align(scan, mode="pose", grid=region)
assert result.scan.grid == region
```

On the command line, `--roi` (`auto`, `off`, `cube`, `bbox`, `cyl`) and `--grid`
choose the grid.

### Resume an interrupted alignment

With `checkpoint=`, alignment saves its progress (volume, poses, solver state)
after each outer iteration. Run the same call again after an interruption and
it resumes where the last checkpoint left off, reaching the result an
uninterrupted run would; once the run has finished, the call returns its
result at once:

```python
result = tj.align(scan, mode="pose", checkpoint="align.ckpt")
```

A checkpoint resumes only the alignment that wrote it: the same projections
and geometry (checked by a fingerprint of sixteen evenly spaced views and of
every geometry setting, angles and pose corrections included), mode, levels,
grid and `AlignConfig`, written by the same TomoJAX version. Anything else
raises `ValueError` naming the difference, for example
`config differs in freeze (checkpoint [], current ['dx'])`, and the file is
left as it is: delete it or choose another path to start again. Checkpoints
written by versions before schema 4 do not resume.

`tomojax align --checkpoint PATH` does the same: run the same command again
to resume.

### Use several GPUs

`tj.align(scan, mode="pose", devices=jax.devices())` shares the views of the joint
update and of cone-beam reconstruction steps among the GPUs; the rest runs on
the first. A checkpoint written on some GPUs resumes on others. The command
line uses one GPU. See [large scans and several GPUs](large-scans.md#use-several-gpus).

## Check the result

Alignment can improve the image without every recovered parameter being a
physical calibration: pose-only correction can absorb some setup error, so the
reconstruction can look good while the poses differ from the true geometry.
Neither a lower objective nor a sharper image proves that the estimated
geometry is calibrated.

- Compare the aligned reconstruction with the unaligned one in several slice
  planes, and check the residuals.
- Check recovered setup parameters against what you know of the acquisition.
  To estimate a detector-centre offset explicitly, use `cor` or
  `cor-then-pose` rather than reading it from pose-only results.
- In `pose` mode the offset stays in the per-view `dx`; `tj.align` reports the
  implied offset, in detector pixels, as
  `result.info["implied_detector_u_px"]`, and `tomojax align` logs it and
  writes it to the `--manifest`.

Known failure modes:

- Abrupt jumps may need jump-aware pose handling, and short bursts of bad views
  a robust loss or bad-view detection.
- A per-view shift with a nonzero mean over the scan cannot be told apart from
  a detector-centre offset; `cor-then-pose` assigns it to the offset.
- Detector-v or sample-elevation reference shifts are physically ambiguous and
  are not reliably recoverable.
- Shifts beyond a quarter of the detector, which leave the object partly
  outside the field of view, are not recovered.
- Large combined setup and pose errors can need staged initialisation,
  stronger priors or manual review.

## Correct per-view motion

Explicit `pose` mode optimises one pose per projection: rotations `alpha`,
`beta` and `phi`, and translations `dx` and `dz`. Pose tables carry a sixth
column, `dy`, a translation along the beam. It has no effect on parallel-beam
projections and stays at zero there; cone-beam alignment estimates it. Five-
column tables from earlier versions load with `dy = 0`.

Each Gauss–Newton step updates the volume and the poses together, using
Joseph plane sampling and an unregularised least-squares fit. Up to 30 outer
iterations stop early once the fit stops improving. Scans of 64³ and larger
align coarse to fine by default. Accuracy depends on resolution: coarse grids
leave a discretisation floor on the rotations, even when the solve starts from
the true poses ([measurements](measurements.md#per-view-pose-recovery)).

Large per-view stage shifts are found first by a global shift search (the
`seed_translations` setting, on by default in `pose` mode): reconstruct with
the current shifts removed, reproject, and move each view to its
cross-correlation peak, searching up to a quarter of the detector. Local
solvers alone converge only from shifts of a few pixels.

The setting `gn_coupling = "fixed_volume"` restores the older scheme that
refines poses against a reconstruction held fixed between volume updates. It
accepts other losses, smooth pose models and optimisers, but on the free-voxel
pilot it left rotation errors of 0.1–1° that the coupled solver removes.

### Smooth pose models

The default pose model is `per_view`, which optimises an independent pose for
every projection. Use a smooth model when you expect the motion to change
smoothly over the scan. Smooth models need the alternating solver; put both in
a TOML file, here `spline.toml`:

```toml
gn_coupling = "fixed_volume"
pose_model = "spline"
knot_spacing = 8
```

```bash
uv run --no-sync tomojax align corrected.nxs -o aligned.nxs --mode pose --config spline.toml
```

Smooth models reduce degrees of freedom but can hide abrupt jumps or outlier
views.

## Calibrate the detector centre

For an unposed axial parallel scan covering a half-turn, `tj.estimate_center`
gives a projection-only estimate with no volume solve:

```python
scan = tj.estimate_center(tj.load_frames("raw.nxs").corrected())
recon = tj.reconstruct(scan)
```

It returns a new `Scan` whose `detector.center[0]` is the absolute physical
detector-u coordinate, estimated from up to three detector rows. The Fourier
metric can have false minima for compact objects, so check the reconstruction.

Use `cor` mode when the main problem is a detector-u or centre-of-rotation
offset rather than sample motion:

```bash
uv run --no-sync tomojax align corrected.nxs -o aligned.nxs --mode cor
```

COR mode fits the detector-u offset explicitly. It starts from a one-parameter
search for the offset whose FBP reprojects most consistently, which works for
laminography and partial arcs, then refines it against held-out views. The
search assumes the views are otherwise consistent: under per-view motion it
misjudges the offset, so use `cor-then-pose` when the sample also moves. The
[measurements](measurements.md#detector-centre) give its accuracy.

To compare trial centres by eye instead, see the
[quickstart](quickstart.md#reconstruct-and-inspect-slices).

## Estimate a detector offset together with motion

`cor-then-pose` runs the same solver as `pose` and then reports the constant
part of the recovered detector-u shifts as the detector centre. With
detector-frame translations (the default) a detector-u offset is exactly a
constant `dx`; a rigid object translation instead shifts each view by a
sinusoid of its angle, so a fit over the scan separates the two. The output
geometry carries the offset as `Detector.center`, and the saved `dx` values are
the remaining per-view motion; together they predict the same data as the
alignment, so the volume is unchanged. `full` adds the same constant to the
detector centre from its setup stage.

```bash
uv run --no-sync tomojax align corrected.nxs -o aligned.nxs --mode cor-then-pose
```

In the Python API, `align_multires` with
`AlignConfig(schedule="cor_then_pose", pose_translation_frame="detector")`
returns the offset as `det_u_px` in `info["geometry_calibration_state"]`, and
`implied_detector_offset` performs the separation on any pose table.

## Use mixed setup and pose as expert mode

`full` mode combines setup and pose stages: detector centre, detector roll
and axis direction, then per-view motion, coarse to fine. Because setup and
pose parameters can represent similar image changes, mixed correction has
gauge ambiguity, and each stage handles it with a fixed gauge policy: the pose
stage anchors the mean translation (`anchor_mean`) and the axis-direction stage
is diagnostic only (`diagnose_only`). `--dry-run` prints each stage's policy.

```bash
uv run --no-sync tomojax align corrected.nxs -o aligned.nxs --mode full
```

An expert direct parameter set that mixes setup and pose parameters (the
`optimise_dofs` setting) must choose a policy itself, with the `gauge_policy`
setting, in a file passed with `--config`:

```toml
optimise_dofs = ["det_u_px", "alpha", "beta", "phi", "dx", "dz"]
gauge_policy = "anchor_mean"
```

Gauge policies:

- `anchor_mean`: anchors the mean translation so setup and pose do not drift
  together; this is a convention, not an independent calibration.
- `prior_required`: requires physical setup priors.
- `diagnose_only`: produces diagnostics without treating the result as a
  calibrated correction.
- `reject`: fails instead of running an ambiguous mixed correction.

## Align cone-beam scans in six degrees of freedom

In a cone beam, moving the sample along the beam changes its magnification,
so `tomojax align` on a cone-beam dataset estimates `dy` with the other five
pose parameters (`--freeze dy` keeps it fixed). Everything else is as for
parallel scans: the coupled solver, translation seeding and saved alignments
(a sixth column of the aligned file's poses, applied by `tomojax recon`, and
`dy_world` in exported parameter files).

Two gauges apply. As in parallel beams, moving the whole volume rigidly and
every pose with it predicts the same data. In addition a common `dy` for all
views rescales the image exactly as a larger object would, so the mean `dy`
is always fixed at zero.

A centre-of-rotation offset in a cone beam is a lateral offset of the rotation
axis, not a detector shift: `ConeBeam.axis_offset` places the axis at
`x = axis_offset`. On cone data, `--mode cor` calibrates it together with the
detector roll, and `cor-then-pose` and `full` calibrate both before their pose
stages; the aligned file records the calibrated beam, so `tomojax recon` on it
uses them. The calibration
([`calibrate_cone_axis`](../src/tomojax/recon/cone_axis.py)) reconstructs thin
FDK slabs near the centre, top and bottom of the volume for trial offsets,
coarse to fine on binned data, and keeps the sharpest. A rolled detector makes
the sharpest offset change linearly with height, which gives the roll. Axis
direction stages are skipped for cone data. In Python:

```python
import tomojax as tj

result = tj.align(scan, mode="cor")      # or "cor-then-pose" to align the poses too
recon = tj.reconstruct(result.scan)      # result.scan carries the calibrated beam
```

`tomojax.recon.calibrate_cone_axis` runs the calibration alone on a geometry
and arrays. Of the six pose parameters, `phi` about the rotation axis is the
weakest; the [measurements](measurements.md#cone-beam-scans) give the accuracy
of both the calibration and the pose alignment.

Multi-orbit `ConeSegments` scans and scans that already carry poses (an ASTRA
import, say, or an earlier alignment) are aligned in `pose` mode only, as one
scan on top of their poses; see
[multi-orbit scans](lab-ct.md#multi-orbit-scans).

## Where the aligned volume sits

The projections cannot tell where the object is: rotating or shifting the
whole object, and every view's pose by the opposite motion, predicts exactly
the same data. Left alone, the solver can end anywhere along that motion, and
the volume comes out displaced.

`tj.align`, `align_multires` and `tomojax align` therefore report the estimate
with the least per-view motion: the poses keep no common rotation and no
rigid shift, the volume is moved to match, and `info["gauge"]` records what
was removed. In a parallel beam with detector-frame poses, a schedule that
reports the detector centre also moves the poses' constant u shift into it.
To compare with a known truth, take its least-motion version too:
`least_motion_estimate` in `tomojax.alignment.api` moves any volume and pose
table to it.

The motion is a symmetry only while the object stays inside the grid. If
moving the volume would push more than 1% of it out of the grid, the grid
edge already pins the estimate, which is then returned unchanged.

## Choose the translation frame

`tj.align` and `tomojax align` use detector-frame translations
(`pose_translation_frame = "detector"`). A bare `AlignConfig()` defaults to
`pose_translation_frame="object"`: `T_nominal @ se3_from_pose_params(params)`,
with translations as physical lengths along the object's x and z axes. Near a
90-degree view those two directions project onto nearly the same detector
direction, so this representation cannot express every image-plane
displacement. A constant detector shift, such as a centre-of-rotation offset,
then needs enormous translations near those views. On the free-voxel pilot,
the chip phantom and the analytic 128³ scans, detector-frame recovery was as
accurate as object-frame recovery or better
([measurements](measurements.md#detector-centre)). The setting
`pose_translation_frame = "object"` in a `--config` file restores the older
tables. The aligned file records the frame, and `tomojax recon` applies the
poses in it.

In the detector frame, rotations still compose in the object frame. `dx` and
`dz` add physical lengths along lab x and z after the nominal transform;
nominal translation along the beam is preserved. Multiply pixel displacements
by the detector spacing before supplying initial parameters. If the detector
grid is rolled, these remain lab x and z directions rather than rolled
sensor-column and row directions.

The frame also applies to `align_multires`, pose smoothness models,
reconstruction and setup objectives. Checkpoints carry the frame, and resuming
with a different frame raises an error.

`tj.load("aligned.nxs").poses` reads the poses `tomojax align` saved. To
export them as JSON (`save_alignment_params_csv` writes CSV the same way, with
a frame column for detector-frame poses), pass the same frame:

```python
from tomojax.alignment.api import save_alignment_params_json

result = tj.align(scan, mode="pose")
save_alignment_params_json(
    "poses.json", result.poses, du=scan.detector.du, dv=scan.detector.dv,
    translation_frame="detector",
)
```

## Expert settings for hard cases

`ray_integrator = "exact"` in a `--config` file (in Python,
`AlignConfig(ray_integrator="exact")`) integrates the zero-extended trilinear
voxel interpolant exactly between voxel-centre planes, using two-point Gaussian
quadrature. It is 10–30 times slower than the default Joseph sampling. It
supports rigid parallel-ray poses, including tilted scans, anisotropic voxels,
shifted detector centres and irregular view angles. Reconstruction, pose loss,
setup validation and translation seeds use the same operator. The CLI and
`coupled_pose_config` default to `"joseph"`; `AlignConfig()` itself keeps
`"sampled"`. A checkpoint cannot resume with a different choice. Exact
polynomial quadrature does not model detector pixel area, beam spectrum or
scatter, and does not guarantee joint recovery. On data made with the same
voxel basis, it recovers poses to numerical precision; that is an inverse
crime, not evidence about real data.

With exact integration, the Huber-FISTA alignment path can use CUDA forward and
matched adjoint kernels with changing poses. Pose derivatives and acceptance
scoring use the exact JAX reference on the selected JAX device. The raw exact
CUDA calls do not provide automatic differentiation; differentiable
reconstruction layers use the JAX reference.

The default Jacobian differentiates within the current interpolation cells,
which sharp voxel boundaries can make unsuitable for a pose update. The
[alignment solver reference](alignment-solver.md#gaussnewton-updates-at-interpolation-boundaries)
describes the symmetric numerical columns and coupled-solve settings, and what
they cost.

For calibrated laminography scans, `region = tj.estimate_region(aligned.scan)`
selects an axial region from coarse projection-validation probes; reconstruct
`region.scan` and inspect `region.info` for uncertainty and any memory-driven
coarsening. Ambiguous or multilayer data retain a broad survey, not an assumed
thin slab; the region does not certify physical support or reconstruction accuracy.
