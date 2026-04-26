---
date: 2026-04-24
last_updated: 2026-04-26
topic: geometry-calibration-solver
---

# Geometry Calibration Solver

## Current Status

This document started as the requirements and phase plan for geometry calibration.
As of 2026-04-26 it is also the status roadmap for the work already landed on
`feat/geometry-calibration-phase0`.

The central architecture has landed: geometry calibration is implemented as staged
geometry parameter blocks inside `align_multires`, not as a separate calibration
pipeline. The active product path now supports:

- detector/ray-grid centre via `det_u_px` and `det_v_px`
- detector-plane roll via `detector_roll_deg`
- rotation-axis direction via `axis_rot_x_deg` and `axis_rot_y_deg`
- laminography tilt as an alias onto the axis-direction block
- multiresolution levels `8 4 2 1`
- the normal documentation/demo alignment profile: `outer_iters=16`,
  `early_stop=True`, `early_stop_rel_impr=1e-3`, `early_stop_patience=2`
- structured geometry diagnostics returned through `align_multires` and written
  into generated demo manifests

Recent evidence:

- Unit and characterization tests cover dynamic detector grids, calibration state,
  conventions/objectives, axis geometry, taxonomy generation, and quick
  `align_multires` geometry-block workflows.
- Visual-stress demo generation now uses the dedicated `lamino_disk` phantom path,
  explicit acquisition spans, and the same `align_multires` geometry-block path
  used by normal workflows.
- Detector centre, detector roll, and full-rotation axis pitch/yaw have synthetic
  evidence. The current 128^3 laptop stress run is validating the final visual
  set after the acquisition-span and diagnostics fixes.

The remaining work is product hardening and the next calibration family, not
basic proof that differentiable geometry blocks can exist inside TomoJAX.

## Problem Frame

TomoJAX currently has a clean projector convention and a useful object-frame pose
alignment model, but recent real laminography work showed that this is not enough
for scanner/instrument geometry errors. The k11-54014 laminography dataset improved
dramatically when a detector centre / centre-of-rotation offset was manually tuned,
while `tomojax-align` did not recover that offset via the current `dx`, `dz`, or
`phi` pose variables.

The core issue is that TomoJAX currently exposes one main correction family:
per-view object/sample-frame pose residuals. Centre-of-rotation, detector centre,
rotation-axis direction, laminography tilt, angle schedule, and detector orientation
are instrument/lab geometry parameters. They should be calibrated before residual
sample pose alignment tries to explain the remaining error.

The desired product direction is one staged multiresolution alignment solver with
separate parameter namespaces:

```text
raw projections
  -> align_multires level 8/4/2/1
      -> detector/instrument geometry blocks
      -> reconstruction with calibrated geometry
      -> residual pose/motion blocks
  -> final reconstruction
```

This should make TomoJAX capable of solving real scanner geometry problems without
pretending that every error is a per-view sample pose.

The important boundary is parameter meaning, not solver machinery. Detector centre,
detector roll, axis direction, laminography tilt, and residual object pose must not
share names or gauges, but they should share the existing `align_multires`
execution path, memory controls, checkpointing, previews, and reconstruction loop.
Standalone calibration optimizers should remain experimental at most and must not
become the primary product path.

Oracle review of the initial plan agreed with this direction but flagged that
detector-centre calibration is not narrow enough to implement safely without a
foundation phase. Before optimizing `det_u`, TomoJAX needs explicit conventions,
gauges, calibration-state metadata, objective definitions, and dynamic detector-grid
plumbing. This document incorporates that critique.

---

## Existing Context For Oracle / Planning

Use this section as the briefing context when asking an external reviewer or Oracle
to critique the plan.

**Project:** TomoJAX, a Python/JAX tomography and laminography package.

**Core projector convention:**

- `geometry.pose_for_view(i)` returns `T_world_from_object`.
- Detector rays are represented in lab/world coordinates.
- Reconstructed volumes live in object/sample coordinates.
- The projector transforms world-frame rays into object coordinates and samples
  the object-frame volume.

Relevant files:

- `src/tomojax/core/projector.py`
- `src/tomojax/core/geometry/base.py`
- `src/tomojax/core/geometry/parallel.py`
- `src/tomojax/core/geometry/lamino.py`

**Current alignment convention:**

- Alignment optimizes 5-DOF per-view pose residuals:
  `alpha`, `beta`, `phi`, `dx`, `dz`.
- These are currently right-multiplied object/sample-frame deltas:

  ```text
  T_aug = T_nom @ Delta_object
  ```

- Therefore current `dx` is object-frame `dx`. For normal CT it rotates with
  the view angle. It is not a constant lab/detector-side shift.

Relevant files:

- `src/tomojax/align/pipeline.py`
- `src/tomojax/align/parametrizations.py`
- `src/tomojax/align/dofs.py`

**Historical detector-centre baseline:**

- `Detector.det_center` is part of the geometry and is used by the projector and
  reconstruction.
- TomoJAX can reconstruct correctly when the right detector centre is supplied.
- At the time this plan was written, TomoJAX did not estimate detector centre as
  an alignment/calibration variable.
- Current status: `det_u_px` and `det_v_px` are now estimable geometry-block
  variables inside `align_multires`; the remaining gap is product hardening,
  metadata persistence, presets, and real-data validation.

Relevant files:

- `src/tomojax/core/geometry/base.py`
- `src/tomojax/core/projector.py`
- `src/tomojax/recon/multires.py`

**Synthetic centre-of-rotation case:**

The run at `runs/doc-normal-scan-motion-128/01_center_of_rotation_offset` looked
successful, but it did not estimate centre of rotation. Its manifest says:

```json
{
  "correction_mode": "detector_center",
  "active_dofs": [],
  "alignment_run": false,
  "corrected_by": "detector_center",
  "detector_center_offset_px": [5.0, 0.0]
}
```

That run proves TomoJAX can apply a known detector-centre correction. It does not
prove `tomojax-align` can estimate centre of rotation.

Relevant files:

- `runs/doc-normal-scan-motion-128/01_center_of_rotation_offset/case_manifest.json`
- `runs/doc-normal-scan-motion-128/01_center_of_rotation_offset/metrics.json`
- `runs/doc-normal-scan-motion-128/01_center_of_rotation_offset/correction_info.json`
- `scripts/generate_rotary_stage_taxonomy_128.py`

**Real laminography evidence:**

- Dataset: `k11-54014`, glass spheres, preprocessed to 256 cube copies outside
  the repository.
- Correct detector convention found during exploration: `flip_u`.
- Approximate laminography tilt improved from nominal `30°` to around `34.4°`.
- A slab reconstruction around global `z=209`, `nz=48`, made the sphere layer
  tractable.
- Manual detector centre / COR tuning around positive `cor_u` visibly improved
  the reconstruction far more than staged `phi` then `dx/dz` pose alignment.

Experimental scripts from the exploratory work:

- `scripts/sweep_real_lamino_tilt.py`
- `scripts/run_real_lamino_alignment.py`
- `scripts/optimize_real_lamino_cor.py`

The exploratory COR script is useful evidence, but it should not become the final
product shape without proper tests, gauges, metadata, and objective design.

---

## Key Flows

- F1. Calibration foundation before optimization
  - **Trigger:** A new geometry calibration variable is added or an existing
    geometry parameter is made estimable.
  - **Actors:** TomoJAX developer, calibration solver, downstream documentation.
  - **Steps:** TomoJAX defines the variable's frame, units, native-resolution
    reporting convention, gauge policy, manifest representation, objective family,
    and compatibility with multiresolution scaling before any optimizer uses it.
  - **Outcome:** Optimization results are physically interpretable and reproducible
    rather than a dataset-specific numeric fit.
  - **Covered by:** R6, R7, R8, R9, R10, R11, R29, R32.

- F2. Instrument geometry blocks before pose blocks
  - **Trigger:** A scan has reconstruction artifacts that look like wrong scanner
    geometry rather than residual sample motion.
  - **Actors:** TomoJAX user, `align_multires`, geometry parameter blocks,
    reconstructor.
  - **Steps:** The user runs `tomojax-align` with active geometry blocks; each
    multiresolution level estimates a small set of instrument/lab geometry
    variables, reconstructs with calibrated geometry, then runs residual pose
    alignment if requested.
  - **Outcome:** Pose alignment no longer has to compensate for a wrong coordinate
    system.
  - **Covered by:** R1, R2, R3, R4, R5, R12, R16, R18, R20.

- F3. Synthetic recovery test for a geometry parameter
  - **Trigger:** A geometry calibration variable is added or changed.
  - **Actors:** Test suite, calibration solver.
  - **Steps:** A synthetic scan is generated with a known hidden geometry error;
    the solver estimates that parameter; reconstruction quality is compared
    against naive and known-corrected references; manifest output is checked for
    estimated vs supplied values.
  - **Outcome:** The project proves recovery rather than only metadata application.
  - **Covered by:** R14, R15, R31, R32.

- F4. Multiresolution block-coordinate solve
  - **Trigger:** A scan requires more than one class of correction.
  - **Actors:** Calibration solver, alignment solver, reconstructor.
  - **Steps:** At each resolution level, TomoJAX optimizes one block of geometry
    variables until plateau; reconstructs; optimizes the next block; reconstructs;
    then passes the calibrated state to the next finer level.
  - **Outcome:** Global geometry errors are handled before finer residual motion,
    reducing overfitting and parameter ambiguity.
  - **Covered by:** R7, R15, R20, R27, R29, R30, R31.

---

## Requirements

**Conceptual model**

- R1. TomoJAX must distinguish instrument/lab geometry calibration from residual
  object/sample pose alignment as parameter namespaces inside the same
  multiresolution alignment engine.
- R2. The current projector convention must remain stable:
  `pose_for_view(i)` returns `T_world_from_object`, rays are in world coordinates,
  and volumes are reconstructed in object/sample coordinates.
- R3. Current `alpha`, `beta`, `phi`, `dx`, `dz` must remain object/sample-frame
  right-multiplied pose residuals unless explicitly renamed or namespaced.
- R4. TomoJAX must not silently redefine existing `dx` as lab-frame motion.
- R5. Known geometry correction must be labeled as metadata application, not as
  solver recovery.

**Phase 0: conventions, gauges, calibration state, objectives**

- R6. Add a calibration-state model that separates detector calibration, scan
  calibration, object-frame residuals, world/lab-frame residuals, detector-plane
  residuals, and angle residuals.
- R7. Define unit and scaling conventions before optimization: native detector
  pixels, physical detector units, binned-level pixel units, and how each value is
  reported in multiresolution runs.
- R8. Record and, where possible, audit detector and angle conventions before
  calibration: `flip_u`, `flip_v`, detector transpose, and `theta_sign`.
- R9. Add dynamic detector-grid plumbing that can represent detector centre offsets
  as a cached zero-centre base grid plus JAX scalar offsets, with equivalence tests
  against static `Detector.det_center`.
- R10. Define an objective registry/card for calibration results, including
  projection-domain validation, optional feature/physics metrics, image-domain
  proxy metrics, top-k candidates, uncertainty/curvature, and contact sheets.
- R11. Define a manifest schema before implementation that records estimated
  variables, supplied known variables, frozen variables, gauges, objectives,
  uncertainty, native-resolution units, physical units, and final calibrated
  geometry.

**Phase 1: detector/ray-grid centre calibration**

- R12. Add a supported `tomojax-align` geometry block that estimates
  detector/ray-grid horizontal centre offset, initially `det_u_px`; implement
  `det_v_px` in the state model but freeze it by default.
- R13. Detector/ray-grid centre calibration output must write calibrated geometry
  metadata and a manifest, not only shifted projection arrays.
- R14. Synthetic tests must prove recovery of hidden `det_u_px`, not just successful
  reconstruction when the known offset is supplied.
- R15. Geometry blocks must use the existing differentiable projector and
  multiresolution alignment machinery. Coarse grid-search scripts may exist as
  exploratory diagnostics, but they are not the product path.
- R16. The docs and manifests must state that `det_u_px` is the canonical
  detector/ray-grid representation of a static COR-like offset under a chosen
  gauge, not proof that detector centre and physical rotation-axis intercept are
  independently identifiable.

**Phase 1.5: COR / rotation-axis intercept gauge**

- R17. Formalize the relationship between detector centre, rotation-axis detector
  intercept, static lab-frame object translation, volume centre, and COR wording
  before exposing centre calibration as a general user-facing feature.

**Phase 2: axis direction / laminography tilt calibration**

- R18. Add calibration variables for rotation-axis direction using one 2-DOF
  representation, such as an internal lab-frame unit axis perturbation or a
  user-facing tilt/azimuth pair; do not optimize redundant axis parameterizations
  together.
- R19. Laminography tilt should be represented as instrument geometry, not as
  residual sample pose. User-facing `tilt_deg` may be an alias, but the internal
  canonical representation should be the rotation-axis direction block.
- R20. Axis-direction calibration must be staged after detector/ray-grid centre
  calibration by default, with one optional detector-centre refinement pass after
  axis/tilt changes.

**Phase 2b: detector-plane roll calibration**

- R21. Add detector-plane roll as an optional calibration block after centre and
  axis calibration are stable, because it can likely be represented through a
  rotated detector grid without arbitrary ray-bundle support.

**Phase 3: angle schedule calibration**

- R22. Add support for angle-schedule calibration variables such as `theta_sign`,
  `theta0`, `theta_scale`, and later structured angle residuals.
- R23. Angle schedule examples must modify projection angles or angle metadata,
  not fake angle errors as detector translations.
- R24. `theta0` and mean object-frame `phi` must not be freely optimized together
  without an explicit gauge or warning.

**Phase 4: lab-frame, detector-plane, and pose residuals**

- R25. Add explicitly named lab/world-frame residuals later, applied by left
  multiplication:

  ```text
  T_aug = Delta_world @ T_nom @ Delta_object
  ```

- R26. Lab-frame residuals must be namespaced separately from object-frame residuals
  so users can tell whether `dx` means object-frame sample motion or lab-frame
  detector-side motion.
- R27. Free per-view residual models must not be the default for geometry calibration;
  low-dimensional models such as constant, polynomial, spline, or harmonic should
  be preferred.

**Phase 5: advanced detector and ray geometry**

- R28. Full detector pitch/yaw, arbitrary beam direction, cone-beam geometry, and
  general ray-bundle support are deferred until detector/ray-grid centre, axis,
  angle, detector-roll, and residual-pose calibration justify a deeper projector
  refactor.

**Gauges and safety**

- R29. TomoJAX must hard-fail, fix a gauge, or emit a clear warning for
  gauge-coupled variable sets, including `det_u` vs static `world_dx`, `det_v` vs
  static `world_dz`, `theta0` vs mean object `phi`, detector roll vs global object
  orientation, and object mean translation vs volume centre.
- R30. Presets should choose safe block orders and gauges for common scan types,
  rather than requiring users to manually understand every geometry variable.
- R31. Real-data calibration acceptance must include validation beyond one scalar
  sharpness metric, such as held-out projection residuals, cross-slab agreement,
  objective smoothness, top-k candidate review, or sample-specific feature metrics.
- R32. Manifests must distinguish estimated variables, supplied known variables,
  frozen variables, gauges, objective values, uncertainty, native/physical units,
  and final calibrated geometry.

### Requirement Status

**Landed or mostly landed**

- R1-R5: Geometry calibration and residual pose alignment are separated as
  parameter namespaces while sharing the multiresolution engine.
- R9: Dynamic detector-grid plumbing exists and is covered by equivalence tests.
- R12: `det_u_px` and `det_v_px` are available as detector/ray-grid centre
  geometry blocks.
- R14: Synthetic recovery tests now prove estimation of hidden detector-centre
  offsets instead of only applying supplied offsets.
- R15: The product path uses the differentiable projector and `align_multires`.
  The earlier grid-search-style calibration path is no longer the direction.
- R18-R19: Axis direction and laminography tilt are represented through the
  axis-direction geometry block, with tilt as a user-facing alias.
- R21: Detector-plane roll is implemented as a geometry block using the same
  staged machinery.

**Partly landed; still needs hardening**

- R6: Calibration-state objects exist for the implemented geometry classes, but
  world-frame residuals, detector-plane residuals, and angle residuals are not yet
  complete product namespaces.
- R7: Native pixel scaling is handled for current blocks. Physical-unit reporting
  and final user-facing unit summaries still need to be hardened.
- R8: Convention helpers and audit scaffolding exist, but the user-facing
  convention audit is not yet a complete workflow.
- R10: Objective/diagnostic metadata exists for geometry blocks. Rich objective
  cards, top-k review, and real-data confidence reporting are still incomplete.
- R11, R13, R32: Generated demo manifests and `align_multires` info now carry
  calibrated-state and diagnostics data. CLI/NXtomo output persistence still needs
  product work.
- R16-R17: The detector/ray-grid centre gauge is the implemented canonical
  representation, but user-facing COR wording and physical-axis-intercept language
  still need final documentation.
- R20: Geometry blocks are staged in `align_multires`, currently in the order
  detector centre, detector roll, then axis direction. The optional centre-refine
  pass after axis/tilt changes remains a preset/design decision.
- R29-R30: Some gauge and conditioning guardrails exist, but named safe presets
  and the full hard-fail conflict list are not complete.
- R31: Synthetic and demo validation is much stronger than before. Real-data
  validation still needs held-out/cross-slab checks and sample-specific metrics.

**Not started**

- R22-R24: Angle-schedule calibration.
- R25-R27: Lab/world-frame residuals and structured residual motion models.
- R28: Advanced detector/ray geometry, arbitrary ray bundles, cone-beam support,
  and detector pitch/yaw.

---

## Acceptance Examples

- AE1. **Covers R12, R14, R16, R32.** Given a synthetic parallel CT scan with
  hidden `det_u_px = +5`, when `tomojax-align --optimise-geometry det_u_px`
  runs through the normal multiresolution pyramid, it recovers `det_u_px` within
  a defined tolerance, improves reconstruction quality over naive FBP, and
  records `det_u_px` under estimated variables with the detector-centre gauge.

- AE2. **Covers R5, R11, R32.** Given a workflow that reconstructs using a known
  supplied detector centre, when the manifest is written, it records the value as
  supplied rather than estimated and does not mark `alignment_run` as recovering
  COR.

- AE3. **Covers R18, R19, R20.** Given a synthetic laminography scan with hidden
  axis tilt error, when detector/ray-grid centre is already correct and the axis
  calibration block runs, it recovers the axis/tilt parameter and improves
  reconstruction quality without using object-frame pose variables as the primary
  explanation.

- AE4. **Covers R25, R26, R29.** Given a scan where both static `det_u` and static
  `world_dx` are requested as active variables, when the solver is configured, it
  rejects the configuration, fixes a gauge, or emits a clear warning before running.

- AE5. **Covers R3, R4, R26.** Given a user exports alignment parameters from an
  existing object-frame run, when TomoJAX adds lab-frame residual support, existing
  `dx` values remain object-frame values and new lab-frame values use distinct
  names.

- AE6. **Covers R8, R10, R31.** Given a scan with wrong `flip_u`, wrong
  `theta_sign`, or transposed detector axes, when detector/ray-grid centre
  calibration is attempted, TomoJAX either detects the convention mismatch or
  reports low confidence rather than producing a confident but wrong centre.

- AE7. **Covers R9.** Given a nonzero detector offset, when projections are made
  with static `Detector(det_center=(u, v))` and with a cached zero-centre grid plus
  dynamic offsets, the forward and adjoint operators match within numerical
  tolerance.

- AE8. **Covers R7, R15, R32.** Given a multiresolution calibration run, when
  `det_u_px` is estimated at coarser levels, the final manifest reports the value
  in native detector pixels and physical units, and the physical offset remains
  consistent across levels.

---

## Success Criteria

- TomoJAX can estimate detector-centre / centre-of-rotation offset on synthetic
  CT and laminography examples without being given the true offset.
- The real k11-54014 laminography workflow can use an estimated detector centre
  as calibrated geometry rather than a hand-picked correction.
- Phase 0 decisions make every reported calibration value interpretable: frame,
  gauge, units, objective, convention, and provenance are explicit.
- Geometry calibration and residual pose alignment are conceptually separate in
  API, manifests, logs, and docs, while operationally sharing `align_multires`.
- Downstream planning can implement Phase 0 and Phase 1 without inventing the
  product model, naming scheme, objective semantics, manifest schema, or safety
  constraints.
- Later phases have clear scope boundaries and can be implemented without
  invalidating Phase 1 or the current object-frame alignment semantics.

---

## Scope Boundaries

- Do not include centre-of-rotation correction as a headline docs demo until TomoJAX
  can estimate the offset rather than only apply a known value.
- Do not redefine existing `dx`, `dz`, or `phi` semantics.
- Do not build an unconstrained 10-DOF optimizer as the first implementation.
- Do not implement `det_u` recovery before deciding native-pixel/physical-unit
  reporting, manifest semantics, objective cards, and gauge conflicts.
- Do not call `det_u` "the COR" without stating that it is the detector/ray-grid
  centre representation of a static COR-like offset under a chosen gauge.
- Do not optimize instrument geometry, lab-frame residuals, object-frame residuals,
  and volume centre all at once by default.
- Do not start with arbitrary ray bundles or cone-beam geometry; defer that until
  detector-centre, axis, angle, and detector-roll calibration justify it.
- Do not rely on generic image sharpness alone as the final objective for real-data
  calibration; the exploratory COR script showed that sharpness-like metrics can
  prefer artifact edges over visually correct sphere focus.

---

## Key Decisions

- Keep the current projector contract because it is clean and internally consistent.
- Keep current pose residuals as object-frame right-multiplied deltas because they
  correctly model sample-frame motion and changing them would break existing
  behavior.
- Add geometry blocks before adding more pose knobs because the real failure was
  an instrument geometry problem, not residual sample motion.
- Add Phase 0 before `det_u_px` because units, gauges, objectives, and manifests
  define what a recovered geometry parameter means.
- Start estimation with detector/ray-grid centre `det_u_px` because it is the
  smallest useful missing capability and directly matches the real laminography
  failure.
- Use staged multiresolution block-coordinate optimization inside `align_multires`
  because it reduces coupling, keeps gauges manageable, avoids asking one
  optimizer to infer all scanner semantics at once, and preserves the existing
  memory behavior.
- Add lab-frame residuals later, explicitly namespaced, because they are useful
  but gauge-coupled with detector-centre calibration.

---

## Phased Plan

### Phase 0: Conventions, Gauges, Calibration State, And Objectives

Status: partly implemented. The calibration-state model, detector-grid plumbing,
unit helpers, convention helpers, and objective/diagnostic scaffolding exist for
the current geometry-block path. The remaining Phase 0 work is to make these
decisions fully product-facing through CLI output, NXtomo metadata, presets,
real-data diagnostics, and hard gauge-conflict checks.

Goal: define what a calibration result means before any optimizer returns numbers.

Decisions to make:

```text
detector convention:
  flip_u
  flip_v
  transpose_detector

angle convention:
  theta_sign
  theta_zero convention

units:
  native detector pixels
  binned-level pixels
  physical detector units

state:
  detector calibration
  scan calibration
  object-frame residuals
  world-frame residuals
  detector-plane residuals
  angle residuals

metadata:
  estimated variables
  supplied known variables
  frozen variables
  gauges
  objective card
  uncertainty
```

Why this first:

- The meaning of `det_u_px` depends on detector convention, units, and gauge.
- Multiresolution calibration can otherwise report level-pixel values as if they
  were native detector pixels.
- Real-data objectives can prefer artefact edges unless their diagnostics and
  validation rules are explicit.
- The solver needs to reject or constrain gauge-coupled variable sets before
  optimization starts.

Implementation shape:

```text
define calibration state schema
  -> define gauge conflict registry
  -> define native/physical unit reporting
  -> define dynamic detector-grid API
  -> define objective-card schema
  -> define manifest schema
  -> add tests for convention, unit, and gauge behavior
```

Definition of done:

- A calibration state can represent detector, scan, object residual, world
  residual, detector-plane residual, and angle residual variables without adding
  columns to the existing object-frame `params5`.
- `Detector(det_center=(u, v))` projections match a zero-centre detector grid with
  dynamic `(u, v)` offsets.
- Gauge conflicts such as `det_u + static world_dx` are caught before optimization.
- Manifests can represent supplied, estimated, frozen, and rejected variables.

### Phase 1: Detector/Ray-Grid Centre Calibration

Status: implemented in the `align_multires` product path for `det_u_px` and
`det_v_px`. Synthetic recovery exists. Remaining work is output hardening,
real-data validation, and clearer user-facing COR/gauge wording.

Goal: estimate centre-of-rotation / detector-centre offset instead of only applying
a known offset, using detector/ray-grid centre as the canonical first gauge for a
static COR-like shift.

Initial variables:

```text
det_u_px
det_v_px implemented but frozen by default
```

Later variables:

```text
det_v_px active with prior/bounds where needed
```

Implementation shape:

```text
input projections + nominal geometry
  -> run tomojax-align with staged geometry blocks inside align_multires
  -> at each pyramid level, solve fixed-volume GN updates for det_u_px
  -> continue with any enabled pose blocks using the calibrated detector grid
  -> record estimated variables, objective traces, gauges, and calibrated geometry
  -> write final reconstruction using calibrated geometry
```

Why this first:

- It is the exact real-data failure observed in laminography.
- `Detector.det_center` already exists and reconstruction can use it.
- The missing capability is estimation, not application.
- It is lower-risk than changing pose alignment semantics.

Definition of done:

- Synthetic normal CT and laminography tests recover hidden `det_u`.
- Manifests distinguish estimated vs supplied detector/ray-grid centre.
- Real-data workflows can run detector-centre calibration before alignment.
- The solver stores objective curves, top-k previews, and a confidence/uncertainty
  estimate.

### Phase 1.5: COR / Rotation-Axis Intercept Gauge

Status: conceptually decided for the implemented path: static COR-like offsets
are represented as detector/ray-grid centre under the detector-centre gauge.
Remaining work is documentation and stricter conflict handling for physically
ambiguous parameter combinations.

Goal: stop overloading "centre of rotation" and make the chosen gauge explicit.

Conceptual variables to distinguish:

```text
detector.det_u_px
detector.det_v_px
rotation_axis.intercept_u_px
rotation_axis.intercept_v_px
static world_dx
static world_dz
volume_center
```

Why this is separate:

- In the current parallel-ray model, several of these can explain similar data.
- TomoJAX needs one canonical representation for static COR-like offsets.
- Users should not infer that the solver physically separated detector translation
  from axis intercept unless the model and data make that identifiable.

Decision:

```text
Phase 1 represents static COR-like horizontal offsets as detector/ray-grid
centre `det_u_px` under the detector-centre canonical gauge.
```

Definition of done:

- Docs and manifests use "detector/ray-grid centre" for the estimated variable.
- "COR" wording is accompanied by the gauge statement.
- Static `world_dx`, object mean `dx`, volume centre, and `det_u` cannot all vary
  freely.

### Phase 2: Rotation-Axis Direction / Laminography Tilt Calibration

Status: implemented in the geometry-block path as `axis_rot_x_deg` and
`axis_rot_y_deg`, with laminography tilt as an alias. Synthetic and visual-stress
validation is active. The main remaining risk is conditioning: the solver now
diagnoses weak or invalid acquisition setups instead of claiming success.

Goal: solve cases where the rotation axis is not the nominal axis.

Internal representation:

```text
axis_unit_lab
```

This is a 2-DOF perturbation around the nominal axis. User-facing controls can
expose either `lamino_tilt_delta` plus `lamino_axis_azimuth_delta` or an
`axis_delta_x` / `axis_delta_y` pair, but not both as simultaneously active
variables.

Why this second:

- Wrong axis direction is the next largest geometry error after detector centre.
- It explains normal CT axes that lean forward/backward or sideways.
- It explains laminography tilt errors like the observed `30°` vs `34.4°` mismatch.

Implementation shape:

```text
calibrate det_u first
  -> optimize low-dimensional axis geometry
  -> reconstruct
  -> refine det_u once if axis change moves the centre optimum
```

Definition of done:

- Synthetic axis lean and laminography tilt errors are recovered.
- Current hard-coded `tilt_about` workflow has a path toward a true lab-frame
  axis direction.

### Phase 2b: Detector-Plane Roll Calibration

Status: implemented as `detector_roll_deg`. Synthetic evidence is positive, but
the current 128^3 stress demo still needs final review because detector roll can
be marked `underconverged` even when it materially improves the reconstruction.

Goal: handle detector-plane rotation without jumping to arbitrary ray geometry.

Initial variable:

```text
detector_roll
```

Why this fits here:

- Detector roll is a common setup error.
- It can likely be represented by rotating the detector `(X, Z)` grid.
- It should be calibrated after centre and axis direction because it is coupled
  with global object orientation and detector-centre estimates.

Deferred variables:

```text
detector_pitch
detector_yaw
beam direction
arbitrary ray bundles
cone-beam geometry
```

Definition of done:

- Synthetic detector roll is recovered and not misreported as object `phi`.
- Detector centre and detector roll can be calibrated with clear gauges.
- Adjoint tests pass with nonzero detector centre and detector roll.

### Phase 3: Angle Schedule Calibration

Status: not started.

Goal: solve wrong or imperfect projection angles.

Initial variables:

```text
theta_sign
theta0
theta_scale
```

Later variables:

```text
theta_delta_i
sinusoidal encoder ripple
piecewise backlash / hysteresis models
spline or low-order angle residuals
```

Why this third:

- Angle sign, zero, and scale can destroy reconstructions.
- Encoder and backlash-like errors are common, but gauge-coupled with object `phi`.
- The tool needs clear angle metadata correction before letting pose variables
  absorb angle errors.

Gauge decisions:

- If `theta0` is active, constrain mean object-frame `phi`.
- If per-view object `phi` is active, keep `theta0` fixed unless explicitly staged.

Definition of done:

- Wrong sign and global angle offset can be corrected in synthetic tests.
- Structured angle errors are represented as angle schedule errors, not fake
  translations.

### Phase 4: Lab-Frame, Detector-Plane, And Pose Residuals

Status: not started.

Goal: add residual motion models after instrument geometry is calibrated.

Variables:

```text
world_dx
world_dz
world_phi
object_alpha
object_beta
object_phi
object_dx
object_dz
detector_shift_i
```

Composition:

```text
T_aug_i = Delta_world_i @ T_nom_i @ Delta_object_i
```

Why this fourth:

- Lab-frame residuals are useful, especially for detector-side drift and runout.
- They are dangerous before geometry gauges exist because static `world_dx` can
  be gauge-equivalent to `det_u`.
- Residual pose alignment should explain remaining motion after geometry is
  calibrated, not compensate for the wrong scanner model.

Definition of done:

- World-frame and object-frame residuals have distinct names, tests, exports,
  docs, and manifests.
- Existing object-frame `params5` behavior remains backwards-compatible.

### Phase 5: Advanced Detector And Ray Geometry

Status: deferred.

Goal: generalize beyond detector-plane centre/roll once the staged calibration
architecture has proven itself.

Deferred variables and capabilities:

```text
detector_pitch
detector_yaw
detector_pixel_aspect
du / dv calibration
beam direction changes
general ray bundles
cone-beam geometry
```

Why this is last:

- The current projector assumes fixed world `+y` parallel rays and detector
  coordinates in world `x,z`.
- Detector pitch/yaw and arbitrary beam direction change ray directions, not just
  detector-plane coordinates.
- This is a projector architecture project, not a prerequisite for `det_u`, axis,
  angle, detector-roll, or residual-pose calibration.

Definition of done:

- A separate plan exists for arbitrary ray geometry.
- Earlier calibration phases are not blocked on this refactor.

---

## Multiresolution Solver Shape

The long-term solver should use block-coordinate optimization per resolution level:

```text
run Phase 0 convention audit / state validation once before levels

for level in [8, 4, 2, 1]:
    reconstruct or update volume

    optimize detector/ray-grid centre block until plateau
    reconstruct or update volume

    optimize axis / tilt block until plateau
    reconstruct or update volume

    refine detector/ray-grid centre if axis / tilt changed
    reconstruct or update volume

    optimize detector roll if enabled
    reconstruct or update volume

    optimize angle block if enabled
    reconstruct or update volume

    optimize residual pose block if enabled
    reconstruct or update volume

    pass calibrated geometry and residual state to next finer level
```

Current implementation note: the live `GEOMETRY_BLOCKS` order is detector centre,
detector roll, then axis direction. It does not yet include an automatic
detector-centre refinement pass after axis/tilt changes, and named presets are
still pending.

Not every block should run for every dataset. Presets should choose safe defaults.
Coarser levels should generate proposals, not final authority: native-pixel values,
physical values, objective curves, top-k candidates, and confidence should be
recomputed or validated at the final relevant resolution.

Example presets:

```text
parallel_ct_basic:
  convention_audit -> det_u -> theta0/sign -> residual_pose

laminography_basic:
  convention_audit -> det_u -> axis_direction/lamino_tilt -> det_u_refine -> residual_pose

known_geometry_pose_only:
  residual_pose only

stage_motion:
  fixed detector geometry -> lab drift -> object wobble
```

Real-data laminography calibration should validate global geometry across more than
one slab or neighbouring `z` window when possible. If each slab prefers a different
global detector centre or axis direction, the objective is likely fitting missing
wedge artefacts, volume-origin error, sample-layer curvature, or an incomplete
geometry model.

---

## Dependencies / Assumptions

- `Detector.det_center` is already used by the projector and reconstruction paths.
- The first calibration API can be implemented without arbitrary ray-bundle support.
- Real-data objective design is nontrivial; a metric that works on synthetic data
  may prefer artifacts on real data.
- The current alignment code and exported parameters rely on existing object-frame
  semantics.
- Some geometry variables are not identifiable without gauges or external priors.
- `det_u_px` is a detector/ray-grid centre representation under a gauge, not proof
  that detector translation, axis intercept, and static lab translation have been
  physically separated.
- Detector roll is likely compatible with the current detector-grid path; detector
  pitch/yaw and arbitrary beam directions are not.

---

## Open Work

### Resolved By The Current Implementation

- Dynamic detector-grid offsets are threaded through projector/reconstruction
  workflows without mutating cached detector geometry inside JAX-transformed
  functions.
- Detector centre, detector roll, and axis direction share the same staged
  geometry-block machinery inside `align_multires`.
- Axis direction is represented internally as a two-parameter lab-frame axis
  perturbation, with laminography tilt exposed as an alias rather than a separate
  competing parameterization.
- The demo taxonomy now records acquisition span explicitly, so full-rotation
  arbitrary-axis examples are not accidentally mislabeled as ordinary 180-degree
  parallel CT.
- The visual demo path uses the dedicated phantom-generation path and normal
  multiresolution alignment profile.

### Still Open For Product Hardening

- Persist calibrated geometry, calibration state, and geometry diagnostics through
  the public CLI and NXtomo/output metadata, not only through Python return info
  and generated demo manifests.
- Finish native-pixel, binned-pixel, and physical-unit summaries for every
  geometry block in final manifests.
- Promote convention audit scaffolding into a user-facing preflight: `flip_u`,
  `flip_v`, detector transpose, `theta_sign`, and low-confidence warnings.
- Define the exact user-facing COR wording: `det_u_px` is detector/ray-grid centre
  under the detector-centre gauge, not proof that detector translation, physical
  rotation-axis intercept, and static sample translation were separately
  identified.
- Finish the hard-fail or warning list for gauge-coupled variables, especially
  `det_u` vs static `world_dx`, `det_v` vs static `world_dz`, `theta0` vs mean
  object `phi`, detector roll vs global object orientation, and object mean
  translation vs volume centre.
- Add named presets that choose safe geometry-block order and gauges for common
  scan types instead of expecting users to manually assemble every block.
- Decide whether the default staged order needs a detector-centre refinement pass
  after axis/tilt changes.
- Integrate richer objective cards: loss traces, final gradient/step diagnostics,
  top-k previews where relevant, projection-domain validation, and image-domain
  proxy metrics.
- Validate real laminography on more than one slab or neighbouring `z` window so
  the solver does not overfit missing-wedge or sample-layer artefacts.

### Deferred To Later Phases

- Angle-schedule calibration: `theta_sign`, `theta0`, `theta_scale`, encoder
  ripple, backlash, and low-dimensional per-view angle residuals.
- Lab/world-frame residuals and detector-plane residual models with explicit
  left-multiplied composition and namespacing.
- Arbitrary ray geometry, detector pitch/yaw, detector pixel aspect, beam
  direction changes, cone-beam support, and general ray bundles.

---

## Oracle Review Incorporated

The review in `runs/oracle-geometry-solver-plan-review.md` was incorporated into
this revision. The main changes were:

- Added Phase 0 before detector-centre optimization.
- Renamed Phase 1 to detector/ray-grid centre calibration.
- Added explicit COR / rotation-axis intercept gauge language.
- Moved detector-plane roll earlier as optional Phase 2b.
- Replaced redundant axis variables with one 2-DOF axis representation.
- Strengthened objective-card, cross-slab validation, uncertainty, and top-k
  candidate requirements.
- Moved general ray-bundle and detector pitch/yaw work to a deferred advanced
  geometry phase.

---

## Next Steps

The next engineering phase should be product hardening for the geometry-block
calibration path that now exists.

1. Finish the current 128^3 visual-stress laptop validation.
   - Sync the final run output.
   - Review all four scenarios: detector roll, full-rotation axis pitch,
     full-rotation axis yaw, and laminography tilt.
   - Decide whether detector roll needs more iterations, a centre-refine pass, or
     only clearer `underconverged` diagnostics.

2. Harden output and metadata.
   - Persist `geometry_calibration_state` and
     `geometry_calibration_diagnostics` through CLI/NXtomo outputs.
   - Record estimated, supplied, frozen, and rejected variables in final outputs.
   - Report native detector pixels, binned-level pixels, and physical units.
   - Preserve acquisition span, active blocks, hidden synthetic perturbations
     where applicable, and final calibrated geometry.

3. Add safe user-facing presets.
   - `parallel_ct_basic`: convention audit, detector centre, detector roll if
     requested, optional angle schedule later, residual pose.
   - `parallel_ct_arbitrary_axis`: full-rotation or explicitly warned acquisition,
     detector centre, detector roll, axis direction, optional centre refine,
     residual pose.
   - `laminography_basic`: convention audit, detector centre, axis/tilt,
     optional centre refine, detector roll if requested, residual pose.
   - `known_geometry_pose_only`: existing residual-pose alignment behavior.

4. Turn diagnostics into acceptance criteria.
   - Blocks marked `underconverged` should be actionable: more iterations,
     stronger damping policy, centre refine, or a clear note that quality improved
     but the solve did not plateau.
   - Blocks marked `ill_conditioned` should stop demos and product workflows from
     claiming successful calibration.
   - Weak acquisition setups, especially arbitrary-axis calibration from
     180-degree data, should be diagnosed explicitly.

5. Validate on real laminography.
   - Re-run the k11-54014 slab workflow with detector centre plus axis/tilt
     calibration.
   - Compare against the manually tuned `34.4°`/COR explorations.
   - Check agreement across neighbouring slabs or `z` windows.
   - Use the result to decide whether the next phase should be angle schedule,
     centre-refine/default staging, or deeper reconstruction-differentiated axis
     solving.

After this hardening phase, plan Phase 3 angle-schedule calibration. Do not start
advanced ray geometry or broad lab-frame residuals until the current geometry
blocks are reliable in CLI output, metadata, diagnostics, demos, and at least one
real-data workflow.
