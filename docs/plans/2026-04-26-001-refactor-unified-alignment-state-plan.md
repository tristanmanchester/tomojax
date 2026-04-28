---
title: Refactor Geometry Calibration Into Unified Alignment State
type: refactor
status: completed
date: 2026-04-26
origin: docs/brainstorms/geometry-calibration-solver-requirements.md
---

# Refactor Geometry Calibration Into Unified Alignment State

## Overview

Refactor the current `feat/geometry-calibration-phase0` branch so geometry calibration is no
longer a separate mini-solver with a private normalized-L2 objective. Geometry DOFs and per-view
pose DOFs become scoped variables in one alignment system, selected through one active/frozen DOF
model, using the same configured loss adapter, defaulting to `l2_otsu`.

The plan preserves the useful branch work: calibration state, detector-grid transforms, axis
geometry, manifests, diagnostics, phantom/demo infrastructure, and CLI persistence. It removes the
architectural mistake: a private geometry objective and a separate geometry outer loop that bypasses
the existing loss system.

## Problem Frame

TomoJAX already has a strong per-view pose alignment system: it alternates reconstruction with
projection-domain alignment, runs through the multiresolution pyramid, and uses the configured
alignment loss, especially `l2_otsu`, which has been the best-performing loss in prior experiments.

The current branch added useful geometry plumbing, but geometry calibration became a separate staged
block optimizer inside `align_multires`, with a private normalized L2 reprojection objective. The new
direction is to treat geometry calibration as first-class alignment DOFs inside one unified alignment
system.

## Requirements Trace

- R1. One alignment DOF namespace covers both per-view pose DOFs and global/instrument geometry DOFs.
- R2. Scoped variables remain separate internally.
- R3. COR-only alignment is `det_u_px` active and no pose DOFs active.
- R4. Existing pose DOF semantics remain object-frame right-multiplied residuals.
- R5. Full coupled optimization is allowed only as explicit expert use.
- R6-R7. `--optimise-dofs` and `--freeze-dofs` apply to pose and geometry names.
- R8-R13. Geometry updates use the configured loss adapter and report the real loss name.
- R14-R18. Geometry and pose share multires loop semantics while using block-coordinate staging.
- R19-R23. Geometry variables keep explicit gauge and frame semantics.
- R24-R27. Demos, manifests, and metadata prove the public solver path.
- R28-R31. Existing pose behavior stays compatible and duplicate/private solver code is removed.

Origin actors: A1 TomoJAX user, A2 Alignment engine, A3 Planner/implementer, A4 Documentation/demo
generator

Origin flows: F1 COR-only detector-centre alignment, F2 Pose-only alignment, F3 Staged geometry plus
pose alignment, F4 Demo/evidence generation

Origin acceptance examples: AE1 detector-centre uses `l2_otsu`, AE2 pose-only compatibility,
AE3 staged masks, AE4 public demo path, AE5 gauge diagnostics, AE6 geometry-state reporting

## Scope Boundaries

- Do not tune or preserve the private normalized-L2 geometry objective as the product path.
- Do not introduce grid search, multistart, or another standalone calibration pipeline.
- Do not make full 10-DOF optimization the recommended default.
- Do not redefine existing `dx`, `dz`, or `phi` semantics.
- Do not claim physical centre-of-rotation separation when the implemented gauge estimates
  detector/ray-grid centre.
- Do not start detector pitch/yaw, cone-beam, arbitrary ray bundles, or angle schedule calibration in
  this plan.
- Do not let demos use paths that real users cannot run.

## Context & Research

### Relevant Code and Patterns

- `src/tomojax/align/pipeline.py` already owns loss adapters, multires execution, reconstruction
  cadence, checkpoints, observer handling, and pose updates.
- `src/tomojax/align/losses.py` exposes `LossAdapter` and `gauss_newton_weights`;
  `L2OtsuLossSpec` reports `l2_otsu`.
- `src/tomojax/align/geometry_blocks.py` currently owns useful geometry state helpers but also
  contains the private normalized-L2 objective and duplicate reconstruction loop.
- `src/tomojax/align/dofs.py` is the existing source of truth for pose DOF parsing and masks.
- `src/tomojax/cli/align.py` currently exposes separate `--optimise-dofs` and
  `--optimise-geometry`.
- `scripts/generate_alignment_before_after_128.py` still assumes geometry-specific scenario fields
  and stale `geometry_calibration` loss markers in places.
- Tests already exist around geometry quick recovery, CLI entrypoints, loss schedules, geometry
  taxonomy generation, axis geometry, metadata roundtrip, and loss logic.

### Institutional Learnings

- `docs/solutions/architecture-patterns/reuse-align-multires-for-geometry-calibration-2026-04-25.md`
  correctly warns against standalone calibration pipelines, but it now needs updating because it
  still describes the staged geometry-block objective as the architectural answer.

## Key Technical Decisions

- Use one active/frozen DOF resolver that returns scoped pose and geometry active sets.
- Keep block-coordinate geometry updates for conditioning; do not build a dense joint Hessian over
  all pose and geometry variables.
- Make geometry updates consume the existing `LossAdapter`, including `l2_otsu` masks and GN
  weights.
- Keep `--optimise-geometry` as a temporary alias, but normalize it into the unified active DOF set.
- Preserve calibration metadata and geometry helpers; remove only the private objective and
  duplicated solver cadence.
- Keep the first implementation pragmatic: a one-step geometry update helper called from the shared
  multires loop is acceptable before a larger `align()` refactor.

## Open Questions

### Resolved During Planning

- Should geometry and pose become one user-facing DOF system? Yes. Use one public DOF namespace with
  scoped internals.
- Should full 10-DOF optimization be supported? Yes, only when explicitly requested; not as the
  default or recommended docs path.
- Should the private normalized-L2 geometry objective survive? No. Replace it with the configured
  loss adapter path.
- Should existing calibration metadata work be discarded? No. Keep it and align it with the unified
  model.

### Deferred to Implementation

- Exact helper/function names for the scoped DOF resolver: choose names that fit local style once
  editing starts.
- Whether small numerical conditioning remains necessary inside geometry GN after switching to
  `LossAdapter`: verify through tests; if needed, document it as numerical conditioning, not a
  different objective.
- Whether the final implementation fully refactors `align()` into a one-level general engine or
  keeps a pragmatic bridge in `align_multires`: choose the smallest DRY implementation that avoids
  duplicate loss/reconstruction/outer-loop semantics.

## High-Level Technical Design

This illustrates the intended approach and is directional guidance for review, not implementation
specification. The implementing agent should treat it as context, not code to reproduce.

```text
AlignConfig
  optimise_dofs + freeze_dofs + legacy geometry_dofs
    -> resolve scoped active DOFs
       pose: alpha,beta,phi,dx,dz
       geometry: det_u_px,det_v_px,detector_roll_deg,axis_rot_x_deg,axis_rot_y_deg

align_multires level loop
  resolve loss for current level
  build / carry geometry calibration state
  reconstruct with current full state
  if geometry active:
      update active geometry block using LossAdapter / GN weights
  if pose active:
      update active pose block using existing align() path or shared one-step helper
  record one coherent outer-stat stream
  early-stop against configured loss improvement
```

## Implementation Units

### U1. Introduce Scoped Alignment DOF Resolution

Goal: Create one source of truth for active/frozen alignment DOFs across pose and geometry scopes.

Requirements: R1, R2, R3, R4, R5, R6, R7, R28

Dependencies: None

Files:

- Modify: `src/tomojax/align/dofs.py`
- Modify: `src/tomojax/align/geometry_blocks.py`
- Modify: `src/tomojax/align/pipeline.py`
- Modify: `src/tomojax/cli/align.py`
- Test: `tests/test_align_quick.py`
- Test: `tests/test_cli_entrypoints.py`
- Test: `tests/test_calibration_axis_geometry.py`

Approach:

- Keep existing pose-only helpers working where they are still semantically pose-only.
- Add a scoped resolver that accepts pose + geometry names and returns active pose DOFs, active
  geometry DOFs, the existing 5-column pose mask, and frozen geometry names.
- Move geometry DOF constants and normalization into one canonical alignment DOF source or re-export
  them from one location.
- Keep `tilt_deg` as a geometry-context alias.
- Treat legacy `AlignConfig.geometry_dofs` and `--optimise-geometry` as compatibility inputs merged
  into the unified active set.

Patterns to follow:

- `src/tomojax/align/dofs.py`
- `src/tomojax/align/geometry_blocks.py`
- `src/tomojax/cli/align.py`

Test scenarios:

- Happy path: `optimise_dofs=("det_u_px",)` resolves to no active pose DOFs and active geometry
  `det_u_px`.
- Happy path: `optimise_dofs=("det_u_px","dx","dz")` resolves both scopes.
- Happy path: legacy `geometry_dofs=("det_u_px",)` still activates geometry when `optimise_dofs` is
  absent.
- Happy path: `--optimise-geometry det_u_px` and `--optimise-dofs det_u_px` produce equivalent active
  geometry state.
- Edge case: `freeze_dofs=("det_u_px",)` removes detector centre from active geometry.
- Edge case: `tilt_deg` resolves to the correct axis DOF for laminography.
- Error path: unknown names fail with a message listing valid pose and geometry DOFs.
- Covers AE1: COR-only active mask leaves `params5` inactive.

Verification:

- Existing pose-only active mask behavior remains compatible.
- New geometry names work through the primary `optimise_dofs` path.
- No duplicated geometry DOF list remains across CLI, pipeline, and geometry helpers.

### U2. Replace Geometry Private Loss With Shared LossAdapter

Goal: Make geometry updates use the configured alignment loss, especially `l2_otsu`, instead of
normalized projection MSE.

Requirements: R8, R9, R10, R11, R12, R13, R24, R30

Dependencies: U1 can be partial; this unit can also be implemented first if scoped DOF plumbing is
staged carefully.

Files:

- Modify: `src/tomojax/align/geometry_blocks.py`
- Modify: `src/tomojax/align/pipeline.py`
- Modify: `scripts/generate_alignment_before_after_128.py`
- Test: `tests/test_align_quick.py`
- Test: `tests/test_geometry_block_taxonomy_generator.py`
- Test: `tests/test_align_loss_logic.py`

Approach:

- Pass the current level's resolved loss spec or `LossAdapter` into geometry update code.
- Build Otsu masks and GN weights through the existing `build_loss_adapter` path.
- For GN-compatible losses, construct geometry residuals using the same weighted residual convention
  as pose GN.
- Remove the `denom = sqrt(mean(y**2))` custom normalization from the product geometry objective.
- Record `geometry_loss_kind` and `loss_kind` as the configured loss name.
- If a configured loss is not GN-compatible, follow the existing pose behavior where feasible;
  otherwise fail clearly for geometry DOFs rather than silently using a different loss.

Patterns to follow:

- Loss adapter construction in `src/tomojax/align/pipeline.py`
- GN weighting in the existing per-view pose path
- Loss schedule tests in `tests/test_align_quick.py`

Test scenarios:

- Covers AE1: detector-centre geometry alignment with default config records `l2_otsu`.
- Happy path: a loss schedule such as `4:phasecorr,2:ssim,1:l2_otsu` records the level-specific loss
  name for geometry updates.
- Happy path: an instrumented loss adapter proves geometry code requests Otsu masks/weights for
  `l2_otsu`.
- Error path: a non-GN-compatible loss with geometry-only GN emits a clear failure or documented
  fallback rather than using normalized L2.
- Regression: no geometry stat reports `loss_kind="geometry_calibration"` as the primary objective
  name.
- Demo integration: generated `case_manifest.json` and `summary.csv` include the actual loss name.

Verification:

- Geometry update stats prove the configured loss was used.
- `det_u_px` synthetic recovery tests no longer depend on the private normalized-L2 objective.
- Demo visual loss panels consume the new loss/stat names without special-casing the stale
  `geometry_calibration` label.

### U3. Unify Multires Level Loop Semantics

Goal: Stop running geometry as a separate full outer loop before pose. Make geometry and pose blocks
share reconstruction cadence, loss bookkeeping, checkpoints, and early stopping at each level.

Requirements: R14, R15, R16, R17, R18

Dependencies: U2

Files:

- Modify: `src/tomojax/align/pipeline.py`
- Modify: `src/tomojax/align/geometry_blocks.py`
- Test: `tests/test_align_quick.py`
- Test: `tests/test_align_chunking.py`

Approach:

- Replace `optimize_geometry_blocks_for_level(... outer_iters=cfg.outer_iters)` as a standalone
  pre-pass.
- Introduce a one-step geometry update helper that accepts the current volume, current geometry
  state, active geometry DOFs, current loss adapter, and current level context.
- Let `align_multires` own the level/outer loop for geometry-active workflows.
- On each outer iteration: reconstruct with current full state, run active geometry update, run
  active pose update if active, evaluate/record configured loss, apply early stopping.
- Keep a pragmatic bridge to existing `align()` for pose updates if full extraction is too invasive,
  but do not let pose and geometry each get independent `outer_iters` loops in the same level.
- If pose is inactive, still run the unified loop so geometry-only alignment gets the same
  reconstruction cadence and early-stop semantics.

Patterns to follow:

- Existing `align()` outer iteration structure in `src/tomojax/align/pipeline.py`
- Current geometry-state application helpers: `geometry_with_axis_state`, `level_detector_grid`

Test scenarios:

- Happy path: geometry-only run with `outer_iters=2` produces two level-local update attempts, not a
  hidden geometry loop plus a pose loop.
- Happy path: pose-only run still produces the same outer stats and loss behavior as before.
- Happy path: combined `det_u_px,dx,dz` run records geometry and pose updates under the same
  level/global outer progression.
- Edge case: no active pose DOFs does not require the current `_active_dof_mask_for_cfg` exception
  path that fakes a zero pose mask.
- Integration: checkpoint state includes current geometry calibration state and resumes without
  restarting geometry from zero.
- Covers AE3: staged block-coordinate behavior is implemented as active masks inside the shared
  loop, not a separate calibration pipeline.

Verification:

- Outer stats tell a coherent story: one level loop, one active loss, scoped updates.
- Runtime does not double-count `outer_iters` when both geometry and pose are active.
- Existing observer/checkpoint callbacks continue to work.

### U4. Preserve And Harden Geometry State, Gauges, And Metadata

Goal: Keep the useful calibration-state work while aligning it with the unified DOF model and shared
loss semantics.

Requirements: R18, R19, R20, R21, R22, R23, R27, R29

Dependencies: U1, U2

Files:

- Modify: `src/tomojax/align/geometry_blocks.py`
- Modify: `src/tomojax/calibration/`
- Modify: `src/tomojax/cli/align.py`
- Modify: `src/tomojax/data/io_hdf5.py`
- Test: `tests/test_calibration_state.py`
- Test: `tests/test_calibration_gauge.py`
- Test: `tests/test_regression_geometry_io.py`
- Test: `tests/test_cli_entrypoints.py`

Approach:

- Keep `GeometryCalibrationState` or rename it only if that reduces confusion; do not split it into
  per-block state classes.
- Ensure estimated, frozen, supplied, and derived variables reflect the unified active/frozen
  resolver.
- Preserve detector/ray-grid centre semantics for `det_u_px` / `det_v_px`.
- Preserve detector roll as detector-plane geometry and axis direction as scan geometry.
- Keep gauge validation focused on real current conflicts; do not overbuild future world/lab
  residual support before those DOFs exist.
- Persist actual loss names and geometry diagnostics into CLI manifests and NXtomo metadata where
  this branch already writes calibration output.

Patterns to follow:

- `src/tomojax/calibration/state.py`
- `src/tomojax/calibration/gauge.py`
- `tests/test_regression_geometry_io.py`

Test scenarios:

- Happy path: `det_u_px` active appears as estimated; inactive geometry DOFs appear frozen or omitted
  according to existing schema conventions.
- Happy path: `axis_unit_lab` remains derived when axis direction is optimized.
- Happy path: CLI output metadata includes calibrated detector centre, detector roll, axis unit, and
  loss diagnostics.
- Error path: known gauge-coupled active sets are rejected or clearly diagnosed where currently
  representable.
- Covers AE5: detector-centre plus an equivalent static lab residual is not silently accepted once
  that lab residual exists.
- Covers AE6: detector roll and axis direction are reported in geometry state, not pose params.

Verification:

- Calibration metadata remains portable and strict JSON-compatible.
- Existing output roundtrip tests pass.
- The branch retains the useful metadata work rather than discarding it with the private objective.

### U5. Update Demo Generator And Visual Evidence Path

Goal: Ensure synthetic docs/demo evidence uses the same public unified alignment path and records
objective truthfully.

Requirements: R24, R25, R26, R31

Dependencies: U1, U2, U3

Files:

- Modify: `scripts/generate_alignment_before_after_128.py`
- Modify: `tests/test_geometry_block_taxonomy_generator.py`
- Modify: `docs/brainstorms/geometry-calibration-solver-requirements.md`
- Modify: `docs/solutions/architecture-patterns/reuse-align-multires-for-geometry-calibration-2026-04-25.md`

Approach:

- Replace scenario wiring that passes only `geometry_dofs` with unified active DOF selection.
- Keep scenario metadata fields such as `geometry_dofs` if useful for readability, but add or derive
  `active_dofs`, `active_pose_dofs`, and `active_geometry_dofs`.
- Make manifests and CSV summaries include actual loss name, active/frozen DOFs, configured profile,
  acquisition span, hidden truth, estimates, and diagnostics.
- Update loss-panel generation to group by real loss/stat names rather than the stale
  `geometry_calibration` marker.
- Keep rich visual panels and phantom #94 choices unchanged except where metadata field names change.

Patterns to follow:

- Current rich panel generation in `scripts/generate_alignment_before_after_128.py`
- Current scenario catalog and acquisition-span tests

Test scenarios:

- Happy path: dry-run docs profile manifest records `l2_otsu` as the default loss.
- Happy path: a COR-only scenario records `active_dofs=["det_u_px"]`, no active pose DOFs, and active
  geometry `det_u_px`.
- Happy path: visual-stress axis cases still record explicit 360-degree acquisition spans.
- Regression: no dry-run or manifest path reports the stale private objective as the configured
  training loss.
- Covers AE4: a generated manifest is sufficient to verify that public alignment machinery was used.
- Naive-only control still writes reduced rich panels and no fake alignment loss.

Verification:

- Demo artifacts remain readable and comparable.
- The demo generator can no longer silently test a different optimizer from the product CLI path.

### U6. Clean Up DRY Boundaries And Remove Private Solver Duplication

Goal: Reduce code carrying cost after the behavior is corrected.

Requirements: R15, R28, R30, R31

Dependencies: U2, U3, U5

Files:

- Modify: `src/tomojax/align/pipeline.py`
- Modify: `src/tomojax/align/geometry_blocks.py`
- Modify: `src/tomojax/align/dofs.py`
- Test: `tests/test_align_quick.py`
- Test: `tests/test_align_chunking.py`

Approach:

- Remove or shrink `optimize_geometry_blocks_for_level` once one-step geometry updates are used.
- Remove duplicate reconstruction wrapper code from `geometry_blocks.py` if reconstruction is owned
  by `pipeline.py`.
- Keep geometry-specific math helpers in `geometry_blocks.py` only where they are actually
  geometry-specific.
- Keep loss construction, multires cadence, reconstruction, early stopping, observer, checkpointing,
  and configured-loss bookkeeping in shared alignment pipeline code.
- Avoid adding a broad abstract framework unless it removes real duplication in this branch.

Patterns to follow:

- Existing `align()` and `align_multires()` separation
- Existing reconstruction call patterns for FISTA/SPDHG in `pipeline.py`

Test scenarios:

- Regression: pose-only workflows remain unchanged.
- Regression: geometry-only workflows still recover detector centre on small synthetic cases.
- Regression: combined pose + geometry workflows do not run duplicate full outer loops.
- Static check: no product path uses the old custom normalized-L2 residual.
- Test expectation: no visual-only tests needed beyond U5.

Verification:

- Code reads as one alignment pipeline with scoped update helpers.
- Geometry helper module no longer contains a separate full optimizer/reconstruction loop.
- The number of paths that can compute alignment loss is reduced, not increased.

### U7. Validation Pass And Laptop Evidence Rerun

Goal: Prove the corrected system locally and with the 128^3 phantom #94 evidence suite before
trusting visuals.

Requirements: R8, R9, R13, R24, R25, R26, R28

Dependencies: U1-U6

Files:

- Test: `tests/test_align_quick.py`
- Test: `tests/test_align_chunking.py`
- Test: `tests/test_align_loss_logic.py`
- Test: `tests/test_calibration_axis_geometry.py`
- Test: `tests/test_geometry_block_taxonomy_generator.py`
- Test: `tests/test_cli_entrypoints.py`
- Test: `tests/test_regression_geometry_io.py`

Approach:

- Run focused local regression tests first, especially loss-adapter and geometry recovery tests.
- Then run the relevant local subset around alignment, geometry, CLI, metadata, and demo generator.
- Only after local tests pass, rerun the laptop 128^3 phantom #94 suite.
- The laptop run must verify actual commit, phantom #94 metadata, active DOF masks, `l2_otsu` loss,
  levels `8 4 2 1`, current docs/demo profile, and no stale `geometry_calibration` objective label.

Test scenarios:

- Covers AE1: detector-centre +/-4 recovers with `l2_otsu` and improves aligned TV over naive.
- Covers AE2: pose-only path remains compatible.
- Covers AE3: staged geometry + pose run uses active masks and one shared loop.
- Covers AE4: generated visual manifests are auditable.
- Covers AE6: detector roll and axis direction report geometry state.
- Stress: visual-stress detector roll, full-rotation axis pitch/yaw, and laminography tilt all
  produce final inspection panels.

Verification:

- Local tests pass.
- Laptop evidence contains no stale objective labels.
- The generated panels are trusted evidence for the public solver path.

## System-Wide Impact

- CLI/API contracts: `--optimise-dofs` grows to include geometry names; `--optimise-geometry`
  remains temporarily for compatibility.
- Checkpoint/resume: checkpoint state must preserve scoped active DOFs and geometry calibration
  state.
- Metadata/output: manifests and NXtomo output should carry loss name, active/frozen DOFs,
  calibrated geometry, and diagnostics.
- Compatibility: pose-only alignment must remain stable for existing users.
- Performance: the unified loop should avoid doubling `outer_iters` for combined geometry + pose
  runs.
- Diagnostics: underconverged and ill-conditioned statuses remain useful, but must be tied to the
  real configured loss.

## Risks & Dependencies

| Risk | Mitigation |
|------|------------|
| Refactor accidentally changes pose-only behavior | Keep pose-only tests as regression gates and preserve existing object-frame `params5` semantics. |
| Geometry GN with `l2_otsu` needs different numerical scaling | Start with adapter-consistent weighting; if conditioning requires normalization, add it as documented numerical conditioning without changing the named loss semantics. |
| Unified loop becomes too large or hard to review | Implement in phases: objective correctness, scoped DOFs, loop unification, cleanup. |
| `--optimise-geometry` compatibility complicates resolver logic | Treat it as a merge input into the unified active set and document it as transitional. |
| Full coupled optimization is ill-conditioned | Allow explicit activation but steer docs/tests toward safe staged masks and diagnostics. |
| Demo results change substantially after switching loss | Treat old visuals as invalid evidence; rerun phantom #94 only after objective correctness is fixed. |

## Documentation / Operational Notes

- Update the requirements doc only if implementation discovers a product-level mismatch; otherwise
  keep it as the source of truth.
- Update
  `docs/solutions/architecture-patterns/reuse-align-multires-for-geometry-calibration-2026-04-25.md`
  so it no longer endorses a separate objective.
- Do not rerun or publish visual evidence until the shared loss path is in place.
- When laptop runs resume, monitor for explicit `l2_otsu` metadata and stale objective labels.

## Sources & References

- Origin document: `docs/brainstorms/geometry-calibration-solver-requirements.md`
- Related architecture note:
  `docs/solutions/architecture-patterns/reuse-align-multires-for-geometry-calibration-2026-04-25.md`
- Alignment pipeline: `src/tomojax/align/pipeline.py`
- Loss system: `src/tomojax/align/losses.py`
- DOF parsing: `src/tomojax/align/dofs.py`
- Geometry helpers: `src/tomojax/align/geometry_blocks.py`
