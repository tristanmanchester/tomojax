---
date: 2026-04-24
last_updated: 2026-04-26
topic: unified-alignment-state-geometry-calibration
---

# Unified Alignment State And Geometry Calibration

## Problem Frame

TomoJAX already has a strong per-view pose alignment system: it alternates
reconstruction with projection-domain alignment, runs through the multiresolution
pyramid, and uses the configured alignment loss, especially `l2_otsu`, which has
been the best-performing loss in prior experiments.

The current `feat/geometry-calibration-phase0` branch added useful geometry
plumbing, but it took two wrong turns that are now documented requirements:
geometry calibration initially used a private normalized-L2 reprojection
objective, and detector-centre discovery used a fixed-volume same-data objective
that can become self-consistent at nominal geometry. Geometry calibration must
use the configured loss system, and COR-like detector-centre discovery must use
an identifiable objective such as held-out reprojection.

The new direction is to treat geometry calibration as first-class alignment DOFs
inside one unified alignment system:

```text
alignment state
  -> global/instrument geometry DOFs
  -> per-view residual pose DOFs
  -> one forward model
  -> one configured projection loss
  -> one multiresolution outer loop
  -> active/frozen DOF masks define the workflow
```

COR-only alignment should mean "optimize `det_u_px`; freeze everything else".
Pose-only alignment should mean "optimize `alpha,beta,phi,dx,dz`; freeze global
geometry". Full coupled solves should be allowed for expert use, but safe staged
presets should be the normal user path.

This document replaces the stale "geometry block solver" direction with the
requirements for one unified alignment-state system.

Relevant current files:

- `src/tomojax/align/pipeline.py`
- `src/tomojax/align/dofs.py`
- `src/tomojax/align/losses.py`
- `src/tomojax/align/geometry_blocks.py`
- `src/tomojax/calibration/`
- `src/tomojax/core/geometry/axis.py`
- `src/tomojax/calibration/detector_grid.py`
- `scripts/generate_alignment_before_after_128.py`

---

## Actors

- A1. TomoJAX user: runs alignment on synthetic or real tomography/laminography
  data and chooses which DOFs to optimize.
- A2. Alignment engine: reconstructs, projects, evaluates loss, updates active
  DOFs, and records diagnostics.
- A3. Planner/implementer: modifies the current branch without carrying forward
  the split-objective geometry block mistake.
- A4. Documentation/demo generator: produces evidence panels and manifests that
  honestly represent the solver being tested.

---

## Key Flows

- F1. COR-only detector-centre alignment
  - **Trigger:** A user suspects a detector/ray-grid centre or centre-of-rotation
    offset.
  - **Actors:** A1, A2
  - **Steps:** User activates `det_u_px`; all other geometry and pose DOFs remain
    frozen; the multiresolution loop seeds from projection-domain evidence,
    reconstructs deterministic train views, scores held-out projections with the
    configured loss, updates `det_u_px`, and carries the calibrated state forward.
  - **Outcome:** The reported detector/ray-grid centre is estimated using the same
    alignment loss as pose alignment, normally `l2_otsu`.
  - **Covered by:** R1, R2, R3, R6, R7, R12, R19

- F2. Pose-only alignment
  - **Trigger:** Geometry is trusted or supplied, and the user wants residual
    sample/object pose alignment.
  - **Actors:** A1, A2
  - **Steps:** User activates `alpha,beta,phi,dx,dz`; global geometry DOFs are
    frozen; the existing per-view pose path runs with backward-compatible object
    frame semantics.
  - **Outcome:** Existing TomoJAX pose alignment behavior is preserved.
  - **Covered by:** R1, R4, R8, R11

- F3. Staged geometry plus pose alignment
  - **Trigger:** A scan likely has both instrument geometry error and residual
    sample motion.
  - **Actors:** A1, A2
  - **Steps:** A preset or explicit schedule activates small compatible DOF groups
    in sequence, such as detector centre, detector roll, axis direction, centre
    refinement, then residual pose.
  - **Outcome:** The solver avoids asking a highly coupled 10-DOF system to solve
    everything at once while still using one alignment state and one objective
    system.
  - **Covered by:** R2, R5, R6, R10, R13, R14, R15, R16

- F4. Demo/evidence generation
  - **Trigger:** A developer generates synthetic before/after panels for docs or
    regression evidence.
  - **Actors:** A3, A4
  - **Steps:** The generator creates a known-hidden-geometry scan, runs the same
    public alignment machinery with a declared active DOF mask and loss, writes
    inspection panels, loss traces, diagnostics, and manifests.
  - **Outcome:** The artifacts prove the public solver path, not a separate demo
    optimizer.
  - **Covered by:** R17, R18, R19, R20, R23

---

## Requirements

**Unified state and DOF model**

- R1. TomoJAX must expose one alignment DOF namespace for user selection, covering
  both per-view pose DOFs and global/instrument geometry DOFs.
- R2. The unified alignment state must keep scoped variables separate internally:
  per-view object/sample pose DOFs are not stored or interpreted the same way as
  global detector/scan geometry DOFs.
- R3. COR-only alignment must be representable as an active DOF mask containing
  `det_u_px` and no active pose DOFs.
- R4. Existing pose DOF names `alpha`, `beta`, `phi`, `dx`, and `dz` must keep
  their current object-frame right-multiplied semantics.
- R5. Expert full-coupled optimization must be possible by explicitly activating
  many DOFs, but it must not be the default workflow or the recommended docs path.
- R6. `--optimise-dofs` should become the primary CLI/API way to choose both pose
  and geometry DOFs; `--optimise-geometry` may remain only as a compatibility alias
  during transition.
- R7. `--freeze-dofs` must be able to freeze both pose and geometry DOFs.

**Loss and objective semantics**

- R8. Geometry calibration updates must use the same configured alignment loss
  system as per-view pose alignment.
- R9. The default geometry calibration loss for the current docs/demo profile must
  be `l2_otsu`, matching the main alignment default and prior empirical findings.
- R10. Geometry calibration must not use a private custom objective such as
  normalized projection MSE unless the user explicitly selects that named loss
  through the normal loss system.
- R11. Loss scheduling by multiresolution level must apply consistently to all
  active DOF scopes at that level.
- R12. GN-compatible losses such as `l2_otsu` must use the existing loss-adapter
  weighting/precompute path for geometry updates, including Otsu masks.
- R13. Stats and manifests must record the actual configured loss name used for
  each update, not a generic label that hides objective differences.
- R13a. Detector-centre/COR discovery must not rely on a fixed reconstructed
  volume scored against the same projections that produced that volume. It must
  use projection-domain seeding, held-out reprojection scoring, or a future
  reconstruction-differentiated objective that breaks the self-consistency loop.

**Solver loop and staging**

- R14. Geometry and pose updates must share the same multiresolution execution
  semantics: reconstruction cadence, configured projection loss, early stopping,
  checkpointing, and diagnostics.
- R15. Block-coordinate updates are allowed and preferred for conditioning, but
  they must be blocks inside one alignment-state loop, not separate solvers with
  separate objectives.
- R16. Safe presets should be schedules of active/frozen DOF masks, not separate
  command paths or standalone calibration pipelines.
- R17. Staged defaults should optimize small compatible groups before coupled
  refinement, for example detector centre before axis direction and residual pose.
- R18. The solver must diagnose ill-conditioned or gauge-coupled active DOF sets
  before claiming successful calibration.

**Geometry semantics and gauges**

- R19. `det_u_px` and `det_v_px` are the canonical detector/ray-grid centre
  variables in native detector pixels under the detector-centre gauge.
- R20. User-facing COR wording must state that `det_u_px` is a detector/ray-grid
  representation of a static COR-like offset, not proof that detector translation,
  rotation-axis intercept, and static sample translation were physically separated.
- R21. Detector roll must be represented as detector-plane geometry, not as object
  `phi`.
- R22. Axis direction and laminography tilt must be represented as scan/instrument
  geometry, with `tilt_deg` acting as an alias where appropriate, not as residual
  sample pose.
- R23. Gauge-coupled sets such as `det_u_px` plus active per-view `dx/dz`, static
  lab `world_dx`, `theta0` plus mean object `phi`, or detector roll plus global
  object orientation must be rejected, fixed by gauge, or clearly diagnosed.

**Evidence, demos, and metadata**

- R24. Synthetic demo generation must call the same public alignment path as real
  workflows and must not use a special demo optimizer.
- R25. Demo manifests must record active DOFs, frozen DOFs, loss name, acquisition
  span, phantom metadata, hidden synthetic perturbations, estimated values, and
  diagnostics.
- R26. Rich visual panels must be treated as evidence of reconstruction behavior,
  while synthetic truth metrics such as volume NMSE must be labelled as evaluation
  metrics rather than training losses.
- R27. Output metadata must distinguish estimated variables, supplied known values,
  frozen variables, derived values, gauges, units, and final calibrated geometry.

**Compatibility and cleanup**

- R28. Existing pose-only alignment tests and CLI behavior must continue to pass
  without requiring users to learn geometry calibration.
- R29. Current useful calibration infrastructure should be retained where it
  supports the unified model: calibration state, units, detector-grid transforms,
  axis geometry, manifests, and gauge helpers.
- R30. Code whose only purpose is the private geometry-block objective must be
  removed or rewritten to consume the shared loss adapter.
- R31. Requirements and solution docs must be updated so future planning does not
  repeat the separate-objective geometry-block design.

---

## Acceptance Examples

- AE1. **Covers R1, R3, R8, R9, R12, R13.** Given a synthetic parallel CT scan
  with hidden `det_u_px=-4`, when the user runs alignment with only `det_u_px`
  active, the solver uses `l2_otsu`, records that loss in stats and manifests,
  records the `heldout_reprojection` objective, estimates detector centre, and
  leaves per-view pose parameters unchanged.

- AE2. **Covers R4, R8, R28.** Given an existing pose-only alignment workflow,
  when no geometry DOFs are active, the path behaves as it did on main: object-frame
  `alpha,beta,phi,dx,dz` are optimized using the configured loss adapter.

- AE3. **Covers R5, R15, R16, R17.** Given a user selects a safe laminography
  preset, when the solver runs through levels `8 4 2 1`, it stages detector centre,
  axis/tilt, optional centre refinement, and residual pose through active masks
  rather than a separate calibration pipeline.

- AE4. **Covers R10, R13, R24.** Given a demo generator run, when the manifest is
  inspected, it must be possible to verify that the generated panels came from the
  same configured public alignment loss and active DOF mask that a user could run
  from the CLI.

- AE5. **Covers R18, R20, R23.** Given a user tries to activate a gauge-coupled
  set such as detector centre and active per-view translations, when the config
  is validated, TomoJAX rejects it or records a clear gauge decision before
  optimization starts.

- AE6. **Covers R21, R22.** Given hidden detector roll or axis direction errors,
  when the relevant geometry DOF is active, the estimated variable is reported in
  detector/scan geometry state, not as object-frame residual pose.

---

## Success Criteria

- Geometry calibration results improve because they use the same empirically
  successful loss system as pose alignment, especially `l2_otsu`.
- A user can express COR-only, pose-only, detector-roll-only, axis-only, staged,
  or expert-coupled optimization using active/frozen DOFs rather than different
  solver concepts.
- Planning can implement the correction without inventing a new calibration
  product shape.
- Synthetic and laptop demo artifacts become trustworthy evidence for the public
  solver path.
- Existing pose-only workflows remain backward compatible.
- The codebase carries less solver duplication, not more.

---

## Scope Boundaries

- Do not tune or preserve the private normalized-L2 geometry objective as the
  product path.
- Do not introduce grid search, multistart, or another standalone calibration
  pipeline to compensate for the wrong objective.
- Do not make full 10-DOF optimization the recommended default.
- Do not redefine existing `dx`, `dz`, or `phi` semantics.
- Do not claim physical centre-of-rotation separation when the implemented gauge
  estimates detector/ray-grid centre.
- Do not start detector pitch/yaw, cone-beam, arbitrary ray bundles, or angle
  schedule calibration until the unified state/loss path is corrected.
- Do not let demos use paths that real users cannot run.

---

## Key Decisions

- Use one alignment system: geometry and pose are scopes in the same alignment
  state, not separate products.
- Use active/frozen DOF masks as the core workflow primitive.
- Keep block-coordinate staging for stability, but make it share state, loss,
  multires loop semantics, checkpoints, and diagnostics.
- Use held-out reprojection as the first detector-centre discovery objective;
  keep fixed-volume GN as a local/other-block mechanism, not as COR discovery.
- Make `l2_otsu` the default geometry calibration loss because it is already the
  main alignment default and has worked best in prior tests.
- Keep calibration metadata and gauge work; fix the optimizer boundary rather than
  discarding the whole branch.
- Allow expert coupled solves, but steer normal users toward safe staged presets.

---

## Dependencies / Assumptions

- The existing `LossAdapter` and `gauss_newton_weights` path can support geometry
  updates for L2-like losses, including `l2_otsu`.
- Detector-centre and detector-roll effects can continue to be represented through
  dynamic detector grids.
- Axis direction can continue to be represented through the current rotation-axis
  geometry helpers.
- Some geometry-plus-pose combinations are genuinely ill-conditioned and need
  gauge policy rather than optimizer cleverness.
- Full reconstruction-differentiated geometry optimization may be useful later,
  but is not required to correct the current branch's objective mistake.

---

## Outstanding Questions

### Resolve Before Planning

- None.

### Deferred to Planning

- [Affects R14, R15][Technical] Decide whether the first implementation should
  keep a small geometry-update helper called from `align_multires`, or refactor
  `align()` into a one-level unified alignment engine immediately.
- [Affects R6, R7][Technical] Decide the cleanest compatibility path for
  `--optimise-geometry` while moving users toward unified `--optimise-dofs`.
- [Affects R12][Technical] Verify the geometry GN residual scaling should exactly
  match pose GN weighting for `l2_otsu`, including whether any normalization is
  still needed for numerical conditioning.
- [Affects R16, R17][Technical] Define the first safe preset schedules after the
  objective is corrected.

---

## Next Steps

-> `ce-plan` for structured implementation planning.

The first implementation plan should prioritize objective correctness before broad
API cleanup:

1. Keep regression tests proving true-volume detector-centre loss is minimized
   near the hidden offset while wrong-geometry fixed-volume loss can prefer
   nominal geometry.
2. Keep `det_u_px` geometry alignment on the configured `l2_otsu`
   held-out-reprojection path.
3. Continue staged active/frozen DOF workflows and reject ambiguous
   detector-centre plus translation active sets.
4. Re-run the phantom #94 128^3 evidence suite only after smoke validation on
   the Linux laptop passes.
