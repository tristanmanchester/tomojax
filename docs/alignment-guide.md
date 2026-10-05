# Alignment guide

TomoJAX alignment estimates geometry or pose corrections while reconstructing
the volume. Pose alignment solves the free voxels and every view's 5-DOF pose
together (the coupled solver), using Joseph plane sampling as its forward model.

On analytic 128³ scans of continuous objects with 181 views and ±0.25°/±0.5 px
motion, the default `tomojax align --mode pose` recovers per-view rotations to
0.0088° (parallel) and 0.0026° (30° laminography) and translations to 0.002
pixels, in 41 and 45 s on an RTX 4070 Laptop GPU, with volume errors of 0.005
and 0.048. A 256³, 361-view laminography
scan recovers to 0.0030° in 3.1 minutes within 8 GB of GPU memory; bin larger
scans for alignment and reconstruct the full data with the recovered poses. Accuracy depends on
resolution: at 32³ the same objects leave a 0.1–0.5° rotation floor from
discretisation, even when started from the true poses, while reconstructions
still match a true-pose reconstruction. Scans of 64³ and larger align coarse to
fine by default, which halved a 256³ alignment's time and improved its volume.
At 64³, ±3° and ±10 px motion is recovered in parallel and laminography scans
(to 0.02°, the discretisation floor at that size), but the anisotropic test scan
with unequal voxels and an offset detector diverges; ±1° is recovered in all three.

Large per-view stage shifts are found first by a global shift search
(`--seed-translations`, on by default for pose mode, `--no-seed-translations`
to disable): reconstruct with the current shifts removed, reproject, and move
each view to its cross-correlation peak, searching up to a quarter of the
detector. Local solvers alone converge only from shifts of a few pixels. With
±0.5° tilts and ±15 px shifts (23% of a 64-pixel detector), rotations go from
7.7–12.6° wrong without the search to 0.012–0.034° with it in parallel and
laminography scans; at 128³ with ±8 or ±15 px all three geometries reach
0.003–0.018°, unchanged by the search. Shifts beyond a quarter of the detector,
which leave the object partly outside the field of view, are not recovered.

`--ray-integrator exact` integrates the trilinear voxel basis exactly. It is
10–30× slower. On the free-voxel pilot, whose measurements use that same basis,
it recovers clean cells to numerical precision, an inverse crime rather than
evidence about real data.

Start from corrected absorption data and checked physical geometry, following
the [real scan guide](real-laminography.md). Save an unaligned reconstruction,
choose the mode matching your problem, and assess both image quality and
recovered parameters. Run commands below from an installed checkout.

## Choose an alignment mode

`tomojax align` has several modes. Use `pose` for per-projection sample motion,
`cor` for detector-centre calibration, `cor_then_pose` for detector-centre
followed by pose correction, and `auto` for the full setup+pose workflow.

| Problem | Recommended mode | Typical command |
| --- | --- | --- |
| Sample or object motion changes from projection to projection | `pose` | `tomojax align --data scan.nxs --mode pose --out aligned.nxs` |
| Detector centre or centre-of-rotation is wrong | `cor` | `tomojax align --data scan.nxs --mode cor --out aligned.nxs` |
| Detector-centre then per-view pose correction | `cor_then_pose` | `tomojax align --data scan.nxs --mode cor_then_pose --out aligned.nxs` |
| Mild setup error and pose motion are both plausible | `auto` | `tomojax align --data scan.nxs --mode auto --gauge-policy anchor_mean --out aligned.nxs` |
| Reference elevation or detector-v shift is uncertain | Inspect manually | `det_v_px` is not a reliably recoverable alignment target. |

## Use 5-DOF pose correction first

The default `pose` mode optimizes one 5-DOF pose vector per projection:
`alpha`, `beta`, `phi`, `dx`, and `dz`. Use this for scans where the sample moved during acquisition.

Each Gauss–Newton step updates the volume and the poses together, using
Joseph plane sampling and an unregularised least-squares fit; up to 30 outer
iterations stop early once the fit stops improving.
`--pose-solver alternating` restores the older scheme that refines poses
against a reconstruction held fixed between volume updates. It accepts other
losses, smooth pose models and optimizers, but in the pilot it left rotation
errors of 0.1–1° that the coupled solver removes.

```bash
uv run --no-sync tomojax align \
  --data corrected.nxs \
  --mode pose \
  --out aligned.nxs
```

Use `--quality reference` for a slower, higher-fidelity solve. Use explicit
levels when you want a specific coarse-to-fine schedule:

```bash
uv run --no-sync tomojax align \
  --data corrected.nxs \
  --mode pose \
  --quality reference \
  --levels 4 2 1 \
  --out aligned.nxs
```

The aligned dataset stores the reconstruction and recovered parameters.
Inspect it with:

```bash
uv run --no-sync tomojax inspect aligned.nxs
```

### Align large scans at reduced resolution

Pose parameters are physical lengths and angles, so they transfer between
resolutions. For a large scan, stop the coarse-to-fine schedule early and
reconstruct the full data with the recovered poses:

```bash
uv run --no-sync tomojax align --data corrected.nxs --mode pose \
  --levels 4 2 --out aligned.nxs
uv run --no-sync tomojax recon --data aligned.nxs --apply-saved-alignment \
  --algo fista --positivity --lambda-tv 0 --iters 300 --out recon.nxs
```

On the analytic 256³, 361-view laminography scan, stopping at half resolution
takes 57 s instead of 188 s, with rotations recovered to 0.0051° instead
of 0.0030°. Laminography leaves a cone of frequencies unmeasured, so the
full-resolution solve needs a prior: unregularised CGLS (`--algo cgls`) reaches
0.27 relative error, positivity-constrained FISTA 0.12 after 100 and 0.091
after 300 iterations (109 s), against 0.059 for the volume of the full
alignment. Add TV (`--lambda-tv`) for noisy data.

## Correction quality vs physical calibration

Pose-only correction can absorb some global setup errors and still produce a
good reconstruction, but that doesn't mean the recovered pose parameters are
a calibrated description of the machine.

- For per-projection motion, use `--mode pose`.
- To estimate detector-centre correction explicitly, use `--mode cor` and
  check the estimate against acquisition knowledge.
- For both setup and pose correction, use `--mode auto` with an explicit gauge
  policy.

## Use COR mode for detector-centre calibration

Use `cor` mode when the main problem is a detector-u or centre-of-rotation
offset rather than sample motion.

```bash
uv run --no-sync tomojax align \
  --data corrected.nxs \
  --mode cor \
  --out aligned.nxs
```

COR mode fits detector-u offsets explicitly. Pose-only correction may absorb
some of that error into sample motion. Neither a lower objective nor a sharper
image proves that the estimated geometry is physically calibrated.

## Use mixed setup and pose as expert mode

`auto` mode combines setup and pose stages. Because setup and pose parameters
can represent similar image changes, mixed correction has gauge ambiguity.
You must choose how to handle that ambiguity.

```bash
uv run --no-sync tomojax align \
  --data corrected.nxs \
  --mode auto \
  --gauge-policy anchor_mean \
  --out aligned.nxs
```

Gauge policies:

- `anchor_mean`: Anchors mean translation so setup and pose don't drift
  together; this is a convention, not independent calibration.
- `prior_required`: Requires physical setup priors.
- `diagnose_only`: Produces diagnostics without treating the result as a
  calibrated correction.
- `reject`: Fails instead of running an ambiguous mixed correction.

## Smooth pose models

The default pose model is `per_view`, which optimizes an independent 5-DOF
vector for every projection. Use a smooth model when you expect the motion to
change smoothly over the scan.

```bash
uv run --no-sync tomojax align \
  --data corrected.nxs \
  --mode pose \
  --pose-model spline \
  --knot-spacing 8 \
  --out aligned.nxs
```

Smooth models reduce degrees of freedom but can hide abrupt jumps or outlier
views.

## Choose the translation frame in the Python API

Existing pose tables and the CLI use `pose_translation_frame="object"`:
`T_nominal @ se3_from_5d(params)`. Translations are physical lengths along the
object's x/z axes. Near a 90-degree view, these two directions project onto
nearly the same detector direction. This representation cannot express every
image-plane displacement, even with a well-conditioned object.

Given a geometry, grid, detector, and corrected projection stack, the Python
API can use two observable image-plane translations at every view:

```python
from tomojax.align import AlignConfig, align
from tomojax.align.api import apply_pose_updates, save_alignment_params_json
from tomojax.geometry import stack_view_poses

config = AlignConfig(pose_translation_frame="detector", gauge_fix="none")
volume, params, info = align(geometry, grid, detector, projections, config=config)
poses = apply_pose_updates(
    stack_view_poses(geometry, len(params)), params,
    translation_frame=config.pose_translation_frame,
)
save_alignment_params_json(
    "poses.json", params, du=detector.du, dv=detector.dv,
    translation_frame=config.pose_translation_frame,
)
```

In this mode rotations still compose in the object frame. `dx` and `dz` add
physical lengths along lab x/z after the nominal transform; nominal translation
along the beam is preserved. Multiply pixel displacements by detector spacing
before supplying initial parameters. If the detector grid is rolled, these
remain lab x/z directions rather than rolled sensor-column/row directions.

The mode also applies to `align_multires`, pose smoothness models, reconstruction
and setup objectives. Checkpoints carry the frame, and resuming with a different
frame raises an error. Older checkpoints retain object-frame meaning. Pass the
same `translation_frame` when exporting JSON or CSV; detector-frame CSV output
adds a frame column.

`gauge_fix="none"` is required for detector-frame poses. Subtracting mean image
shifts would constrain observable motion and is not a common object-frame
translation gauge. Joint reconstruction still has a shared rigid-frame
ambiguity: compare recovered geometry and volume in one consistent object
frame. The option fixes representation; it does not establish successful
free-voxel recovery or the performance and robustness targets.

## Gauss–Newton updates at interpolation boundaries

The default Jacobian differentiates within the current interpolation cells.
Sharp voxel boundaries can make that derivative unsuitable for a pose update.
The Python API offers symmetric numerical columns and opt-in coupled solves.
See the [alignment solver reference](alignment-solver.md) for the cost, supported
constraints, and actual recovery evidence before changing these settings.

## Known hard cases

For voxel-basis data that need accurate line integration, the Python API accepts
`AlignConfig(ray_integrator="exact")`. This integrates the zero-extended trilinear
interpolant between voxel-centre planes using two-point Gaussian quadrature.
It supports rigid parallel-ray poses, including tilted scans, anisotropic voxels,
shifted detector centres and irregular view angles. Reconstruction, pose loss,
setup validation and translation seeds use the same selected operator. The
default remains `"sampled"`; a checkpoint cannot resume with a different choice.

With exact integration, the Huber-FISTA alignment path can use CUDA forward and
matched adjoint kernels with changing poses. Pose derivatives and acceptance
scoring use the exact JAX reference on the selected JAX device. The raw exact
CUDA calls do not provide automatic differentiation; differentiable reconstruction
layers use the JAX reference. Exact polynomial quadrature does not model detector
pixel area, beam spectrum or scatter, and does not guarantee joint recovery.

Known failure modes include:

- Abrupt jumps may need jump-aware pose handling.
- Short bursts of bad views may need robust loss or bad-view detection.
- Large combined setup and pose errors may need staged initialization.
- Detector-v or sample-elevation reference shifts are physically ambiguous and
  are not reliably recoverable.

## Evidence from the 128^3 sweep

Older project material described a 128³ visual sweep. It has no complete
reproduction record established in the current user guide, so its visual
ratings are not used as a current recovery claim. Historical images are
[catalogued separately](../images/README.md#other-historical-assets).

Use the [six-cell public comparison](research/public-free-voxel-schur-2026-10-04.md)
for quantified free-voxel recovery, cold/warm time, and sampled process GPU
memory. It retains the failed noisy anisotropic case and does not establish
large-motion capture or 99% robustness.

## Next steps

After alignment, inspect the output and compare against the naive
reconstruction. See [`support-matrix.md`](support-matrix.md) for supported
workflows and [`known-limitations.md`](known-limitations.md) for hard cases.
