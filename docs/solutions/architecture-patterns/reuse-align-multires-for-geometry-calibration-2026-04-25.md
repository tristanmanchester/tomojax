---
title: Reuse the alignment loss path for geometry calibration blocks
date: 2026-04-25
last_updated: 2026-04-26
category: architecture-patterns
module: TomoJAX alignment geometry calibration
problem_type: architecture_pattern
component: tooling
severity: high
applies_when:
  - Adding detector or instrument geometry optimization to TomoJAX alignment
  - A new calibration feature starts duplicating reconstruction or multiresolution control flow
  - A solver needs the same memory, checkpoint, preview, and CLI behavior as alignment
  - Geometry calibration demos need to distinguish solver convergence from acquisition conditioning
tags: [tomojax, alignment, geometry-calibration, dry, multires, gn, diagnostics, theta-span]
---

# Reuse the alignment loss path for geometry calibration blocks

## Context

The first geometry-calibration implementation solved the right category of
problem, but it solved it in the wrong place. Detector centre, detector roll,
and axis direction were implemented as a standalone calibration command and
separate solver modules instead of being staged inside the existing
`align_multires` pipeline.

That split created immediate friction:

- The new path duplicated reconstruction and calibration orchestration already
  present in `src/tomojax/align/pipeline.py`.
- It introduced a separate CLI surface, `tomojax-calibrate-geometry`, even
  though users naturally expected calibration to happen inside
  `tomojax-align`.
- It bypassed the normal multiresolution alignment pattern, especially the
  `8 4 2 1` pyramid that keeps 128^3 alignment memory under control.
- It encouraged special-purpose runner scripts and before/after generation
  paths, which made OOM failures and inconsistent preview behavior more likely.
- It risked semantic drift: pose alignment, detector-centre calibration,
  detector roll, and axis tilt would each own slightly different ideas of
  geometry, checkpointing, diagnostics, and metadata.
- It made demo/test failures ambiguous: an axis-direction stress case could
  look like a solver regression when the real issue was a known weak
  acquisition setup.

The first architectural correction was to keep calibration as a distinct
parameter namespace, but not as a distinct solver product. The April 26 follow-up
tightened that further: geometry calibration must also use the same configured
alignment loss path as pose alignment. A geometry block is not allowed to quietly
switch to a private normalized-L2 reprojection objective when the user requested
`l2_otsu` or a loss schedule.

## Guidance

Geometry calibration should be implemented as scoped alignment DOFs inside
`align_multires`.

The durable shape is:

```text
raw projections
  -> align_multires level 8/4/2/1
      -> detector/instrument geometry blocks using the configured LossAdapter
      -> reconstruction with calibrated geometry
      -> residual pose/motion blocks
  -> final reconstruction
```

The boundary is parameter meaning, not execution machinery:

- Instrument geometry parameters describe static scanner geometry:
  `det_u_px`, `det_v_px`, `detector_roll_deg`, `axis_rot_x_deg`,
  `axis_rot_y_deg`, and user-facing aliases such as `tilt_deg`.
- Pose parameters describe residual per-view object motion:
  `alpha`, `beta`, `phi`, `dx`, and `dz`.
- Both groups should share the same multiresolution schedule, reconstruction
  loop, checkpointing, metadata, logging, CLI command, and configured loss
  adapter.

The public DOF namespace is unified, but not every active set is identifiable:

- `--optimise-dofs det_u_px` is centre/ray-grid geometry calibration with pose
  frozen by omission.
- `--optimise-dofs dx,dz` is pose-only translation alignment.
- `--optimise-dofs det_u_px,dx,dz` is rejected today because detector-centre and
  per-view detector-plane translation are gauge-coupled. Run detector-centre
  calibration first, then residual pose alignment.
- `--optimise-geometry` remains a transitional compatibility alias that is
  normalized into the same active geometry DOF set.

Internally the scopes remain separate. Pose DOFs still update per-view
object-frame residual parameters. Geometry DOFs update static detector/scan
state carried by `GeometryCalibrationState`.

The implemented DRY structure is:

```python
GEOMETRY_BLOCKS = (
    ("detector_center", ("det_u_px", "det_v_px")),
    ("detector_roll", ("detector_roll_deg",)),
    ("axis_direction", ("axis_rot_x_deg", "axis_rot_y_deg")),
)
```

`src/tomojax/align/geometry_blocks.py` owns the reusable geometry-block
abstractions:

- `GeometryCalibrationState` stores native-resolution instrument state across
  pyramid levels.
- `normalize_geometry_dofs` remains the geometry-specific compatibility parser.
  The public alignment parser lives in `src/tomojax/align/dofs.py` and resolves
  scoped pose/geometry active sets.
- `level_detector_grid` applies detector-centre and detector-roll state through
  detector-grid overrides rather than projection rewrites.
- `geometry_with_axis_state` applies axis-direction state as instrument
  geometry before residual pose alignment.
- `optimize_geometry_blocks_for_level` stages active geometry blocks using the
  current level's `LossAdapter`. Detector-u centre discovery uses a
  train/held-out reprojection objective; detector roll and axis-direction blocks
  still use the fixed-volume GN path.
- `summarize_geometry_calibration_stats` turns raw per-update GN stats into a
  compact diagnostic contract: accepted updates, total movement, final step,
  gradient norm, loss change, and a block status.
- `add_geometry_acquisition_diagnostics` annotates the diagnostic contract with
  acquisition-conditioning context. In the current implementation, active
  axis-direction solves below the diagnostic threshold of 270 degrees are
  reported as ill-conditioned rather than silently treated as ordinary
  convergence cases.

`src/tomojax/align/pipeline.py` owns orchestration:

- `AlignConfig.optimise_dofs` accepts both pose and geometry names.
- `AlignConfig.geometry_dofs` is retained as a compatibility input and is merged
  into the scoped active DOF set.
- `align_multires` runs geometry blocks before pose blocks at each level.
- `align` accepts an optional `det_grid_override`, so detector-centre and
  detector-roll calibration reuse the existing projector and reconstruction
  code instead of forking it.
- Multires checkpoints include `geometry_calibration_state`, so resumed runs
  continue from the calibrated instrument state rather than silently resetting
  to nominal geometry.
- The returned `info` includes `geometry_calibration_diagnostics`, so callers
  do not have to reverse-engineer convergence or conditioning from raw
  `outer_stats`.

The public surface is the existing alignment CLI:

```bash
tomojax-align --data data/scan.nxs \
  --levels 8 4 2 1 \
  --optimise-dofs det_u_px,detector_roll_deg \
  --out out/geometry_calibrated.nxs
```

Geometry calibration uses the configured alignment loss:

```bash
tomojax-align --data data/scan.nxs \
  --levels 8 4 2 1 \
  --optimise-dofs det_u_px \
  --loss l2_otsu \
  --out out/detector_center_calibrated.nxs
```

For `det_u_px`, the objective name in stats is `heldout_reprojection`. The block
uses a projection-domain detector-u seed to choose a candidate window, rebuilds
train-view TV reconstructions for candidate detector centres, and scores held-out
projections with the configured loss adapter. This breaks the self-consistency
failure where `loss(A(det_u_candidate) x_nominal_recon, y)` can prefer nominal
geometry because `x_nominal_recon` already absorbed the detector-centre error.

The standalone path was removed:

- Removed `tomojax-calibrate-geometry` from `pyproject.toml`.
- Deleted `src/tomojax/cli/calibrate_geometry.py`.
- Deleted standalone solver modules:
  `src/tomojax/calibration/center.py`,
  `src/tomojax/calibration/roll.py`, and
  `src/tomojax/calibration/axis.py`.
- Kept `src/tomojax/calibration/` for shared schema, gauge, detector-grid,
  axis-geometry, objective-card, and manifest helpers.

## Why This Matters

The original implementation was not just extra code. It created a second
alignment product with its own control flow. That made it easy to forget critical
behavior that already existed in `align_multires`, especially:

- pyramid scheduling,
- low-memory reconstruction behavior,
- checkpoint and resume semantics,
- CLI configuration compatibility,
- output metadata,
- diagnostics and stats,
- acquisition-conditioning diagnostics for global axis-direction blocks,
- pose gauge handling.

The DRY version makes the extension point explicit. Adding a geometry parameter
now means adding a block and the state transformations it needs, not inventing a
new command, solver loop, checkpoint format, and test family.

This also preserves the mental model users actually need: "run alignment with
instrument geometry blocks first, then residual pose blocks." Users do not have
to choose between two competing commands that both reconstruct and both claim to
fix geometry.

## Diagnostics and acquisition conditioning

Keeping geometry calibration inside `align_multires` is necessary but not
sufficient. The demos, manifests, and diagnostics also need to encode the
conditions that make a geometry solve meaningful.

The April 26 diagnostic hardening fixed a misleading failure mode in the
geometry-block stress demos:

- Visual-stress axis pitch/yaw scenarios were labelled as parallel CT and
  inherited a 180-degree acquisition from `geometry_type="parallel"`.
- Once a hidden axis pitch/yaw is introduced, the scan is no longer an ordinary
  180-degree parallel CT case. The current visual-stress evidence path uses
  360-degree coverage for these arbitrary-axis cases to avoid the known
  180-degree conditioning failure.
- The docs profile had drifted to `outer_iters=6` with `early_stop=False`,
  even though the historical evidence path used 12 outers with early stopping.
- Raw geometry-block stats existed, but generated artifacts did not summarize
  whether a block had converged, was still improving, or was ill-conditioned.

The corrected evidence path makes those assumptions explicit:

```python
@dataclass(frozen=True)
class Scenario:
    ...
    geometry_type: str
    geometry_dofs: tuple[str, ...]
    theta_span_deg: float | None = None
```

The canonical visual-stress cases now separate nominal geometry family from
acquisition span:

- detector roll stress: 180 degrees,
- arbitrary-axis pitch/yaw stress: 360 degrees,
- laminography tilt stress: 360 degrees.

The docs profile is again the quality profile, not a smoke test:

```python
RunProfile(
    name="docs_128",
    size=128,
    views=128,
    levels=(8, 4, 2, 1),
    outer_iters=16,
    recon_iters=20,
    tv_prox_iters=12,
    views_per_batch=1,
    gather_dtype="bf16",
    early_stop=True,
    early_stop_rel_impr=1e-3,
    early_stop_patience=2,
)
```

`align_multires` now reports block-level diagnostics:

```python
{
    "geometry_calibration_diagnostics": {
        "schema_version": 1,
        "overall_status": "underconverged",
        "blocks": [
            {
                "geometry_block": "axis_direction",
                "geometry_active_dofs": "axis_rot_x_deg",
                "attempted_updates": 24,
                "accepted_updates": 24,
                "total_step_norm": 0.879,
                "final_step_norm": 0.0014,
                "final_gradient_norm": 0.000711,
                "status": "underconverged",
            }
        ],
    }
}
```

When an axis-direction block is active and inferred coverage is below the
diagnostic threshold, currently 270 degrees, the diagnostic layer marks the
block as ill-conditioned and adds the warning
`axis_direction_sub_full_rotation_acquisition`. That is a deliberate guardrail:
180-degree arbitrary-axis stress cases should not be reported as ordinary
successful or failed geometry calibration.

Detector-centre calibration now has a separate diagnostic contract from
fixed-volume GN: stats include `geometry_objective=heldout_reprojection`, the
configured loss name, the projection-domain seed, train/held-out view counts, and
candidate count. Fixed-volume GN stats are grouped by level and objective so
diagnostics do not compare unrelated loss scales across pyramid levels.

This does not turn the supported product into a standalone grid-search
calibration command. It keeps detector-centre discovery inside `align_multires`
and reuses the same state, detector-grid transforms, reconstruction code, and
loss adapter. Under some acquisition setups, fixed-volume objectives can still
appear self-consistent without recovering the intended global geometry. That is a
conditioning/solver-design fact the metadata should expose, not a reason to fork
another calibration pipeline.

## When to Apply

- Use this pattern when adding a static scanner/instrument parameter that should
  be optimized before residual pose motion.
- Use it when a calibration feature needs the same multiresolution behavior as
  pose alignment.
- Use it when the proposed implementation starts copying reconstruction loops,
  runner scripts, checkpoint writing, preview generation, or CLI metadata.
- Use explicit acquisition metadata whenever a synthetic or documentation
  example changes the geometry conditioning story. Do not let `geometry_type`
  silently choose the angular span for arbitrary-axis stress cases.
- Treat 180-degree axis-direction calibration as a known weak setup unless
  future solver work proves a stronger claim. The manifest should say that
  directly.
- Do not use a standalone calibration command unless it is explicitly an
  experimental diagnostic and does not become the supported product path.

## Examples

### Before: separate calibration product

The first implementation added a standalone command and parallel solver modules:

```text
tomojax-calibrate-geometry
src/tomojax/cli/calibrate_geometry.py
src/tomojax/calibration/center.py
src/tomojax/calibration/roll.py
src/tomojax/calibration/axis.py
```

That looked modular, but the modules duplicated the alignment pipeline's real
responsibilities. A calibration run could now diverge from `tomojax-align` in
memory behavior, checkpointing, metadata, previews, and multires scheduling.

### After: geometry blocks inside align_multires

The corrected implementation keeps one product path:

```text
tomojax-align
  -> align_multires
      -> GeometryCalibrationState
      -> GEOMETRY_BLOCKS
      -> optimize_geometry_blocks_for_level
          -> detector_center: held-out reprojection for det_u_px
          -> other blocks: fixed-volume GN diagnostics
      -> align pose blocks
```

The CLI expresses geometry calibration as an option on alignment:

```bash
tomojax-align --data data/scan.nxs \
  --levels 8 4 2 1 \
  --optimise-dofs det_u_px,axis_rot_x_deg,axis_rot_y_deg \
  --out out/aligned.nxs
```

Geometry-only calibration is still available, but it uses the same pipeline:

```bash
tomojax-align --data data/scan.nxs \
  --levels 8 4 2 1 \
  --optimise-dofs det_u_px \
  --out out/detector_center_calibrated.nxs
```

### Test shape

The key regression test is not a unit test of a standalone solver. It proves the
integrated behavior:

```python
_, params5, info = align_multires(
    geom_nom,
    grid,
    det_nom,
    projs,
    factors=[2, 1],
    cfg=AlignConfig(
        outer_iters=2,
        recon_iters=2,
        lambda_tv=0.0,
        optimise_dofs=("det_u_px",),
        loss=L2OtsuLossSpec(),
        early_stop=False,
        gather_dtype="fp32",
        checkpoint_projector=False,
        views_per_batch=1,
        gn_damping=1e-3,
    ),
    checkpoint_callback=checkpoints.append,
)
```

The assertions check the behavior that matters:

- the hidden detector-centre offset is recovered,
- all pose parameters stay zero when pose DOFs are frozen,
- geometry block stats are emitted through normal alignment info,
- detector-centre stats report `geometry_objective="heldout_reprojection"`,
- checkpoint metadata carries `geometry_calibration_state`,
- geometry diagnostics are emitted through normal alignment info,
- geometry stats report the configured loss name, such as `l2_otsu`, rather than
  a private `geometry_calibration` objective,
- axis-direction calibration below the current 270-degree diagnostic threshold
  is reported as ill-conditioned.

The generator tests also lock down the evidence recipe:

- canonical before/after demos use the selected random-shapes phantom path, not
  an ad hoc Shepp-Logan phantom,
- docs runs use 128 views, levels `8 4 2 1`, 16 outers, and early stopping,
- visual-stress axis pitch/yaw scenarios record 360-degree acquisition span,
- summary rows and manifests include acquisition span and geometry diagnostics.

### Failed approaches worth remembering

Several earlier attempts were useful but incomplete (session history):

- Fixed-volume detector-centre GN used the right loss adapter after cleanup, but
  still failed because a volume reconstructed under wrong detector centre can
  absorb the error. True-volume `l2_otsu` loss minimizes near the hidden offset;
  wrong-geometry fixed-volume loss can minimize at nominal. Detector-centre
  discovery now uses held-out reprojection to break that loop.
- The first geometry-block GN memory shape materialized too much detector-stack
  residual/JVP data and hit a 128^3 OOM. The correct memory discipline is
  chunked normal-equation accumulation.
- 180-degree arbitrary-axis stress cases under-recovered even after the
  zero-Jacobian axis-pose bug was fixed. Full-rotation visual-stress acquisition
  and explicit diagnostics are needed to interpret those examples correctly.

## Related

- `docs/brainstorms/geometry-calibration-solver-requirements.md` records the
  product decision: calibration is conceptually distinct from pose alignment but
  operationally staged inside `align_multires`.
- `docs/cli/align.md` documents geometry DOFs through `--optimise-dofs`, with
  `--optimise-geometry` retained as a compatibility alias.
- `scripts/generate_alignment_before_after_128.py` owns the current
  before/after taxonomy and visual-stress evidence path.
- `tests/test_geometry_block_taxonomy_generator.py` protects the demo profile,
  phantom choice, acquisition span, and manifest schema.
- `tests/test_align_quick.py` protects the integrated geometry diagnostics.
- Verify changes with the focused geometry-calibration tests and the repository
  lint/test commands before using the generated visuals as documentation
  evidence.
