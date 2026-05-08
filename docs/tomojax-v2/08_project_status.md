# 08 — Project Status After Stopped det_u Gate

This page summarizes the current v2 state after the production stopped det_u
investigation. It is a status document, not a new architecture plan.

## Production-Ready Evidence

- The supported-only stopped det_u path has an honest first production
  milestone at `64^3`: geometry-first bootstrap, stopped reconstruction, pose
  frozen, theta frozen, det_u active only, no nuisance, no weak-view exclusion,
  no fixed-truth volume, and no candidate-refresh production acceptance reaches
  det_u `0.886244 px` from an initial `7.25 px`.
- The production det_u path uses the supported joint Schur solver and the
  `core_trilinear_ray` projector only.
- Reference FISTA now has finite-difference coverage for the explicit gradient
  of the same filtered/masked loss used by the solver, including lowpass
  residuals and TV/center regularisation.

## Oracle-Only Evidence

- Fixed-truth and true-volume runs are oracle diagnostics. They show that the
  Schur/core geometry path is coherent when the volume is already in the right
  gauge.
- Weak-view exclusion is diagnostic. A pass with excluded weak views is not a
  plain production pass.
- Theta recovery in stopped reconstruction is a volume-orientation gauge unless
  calibration mode supplies an orientation anchor such as truth, fiducials,
  known asymmetric support, or a documented calibration convention.

## Not Production-Ready

- The stopped det_u path does not meet the `<0.2 px` stretch gate at `64^3`.
  Single-scale Schur/refresh refinement stalls near `0.876 px`.
- The same minimal path does not scale to `128^3`; the current evidence is
  det_u `14.5 px -> 2.25510 px`.
- A real multiresolution prototype with actual downsampled detector
  projections/volumes and scaled det_u improves `64^3` to `0.692153 px`, but
  it worsens volume NMSE and is not production behavior.
- Candidate-refresh acceptance, hard x-gauge projection, neutral-refresh
  variants, no-FISTA-first preview, extra polish stages, and weak-view
  exclusion are diagnostic tools only unless a future milestone revalidates
  them against a named production gate.

## Current Blocker

The blocker is the stopped reconstruction/geometry handoff. The geometry-first
bootstrap proves the detector-shift gradient exists before FISTA absorbs setup
error, but the current handoff cannot refine to stretch accuracy or scale to
`128^3`.

## Next Architecture Choices

1. Design a production multiresolution stopped det_u handoff, not just residual
   filtering. It must downsample detector data and volumes, scale det_u between
   levels, and preserve volume gauge across level transitions.
2. Add an explicit stopped-volume gauge or reconstruction constraint that
   prevents detector shift from being absorbed by the volume before Schur.
3. Keep theta frozen in stopped production mode until a calibration anchor is
   defined.
4. Revisit broader benchmark cases only after the stopped supported-only det_u
   gate either reaches stretch accuracy or is explicitly declared blocked by
   evidence.
