---
title: Production-Ready Unified Validation-LM Alignment
type: refactor
status: active
date: 2026-04-27
origin: docs/brainstorms/geometry-calibration-solver-requirements.md
---

# Production-Ready Unified Validation-LM Alignment

## Summary

Make the unified setup/pose alignment branch production-ready around the
validation-LM architecture that is now working. The production setup solver is
cross-validated stopped-reconstruction validation-LM: train-fold
reconstructions, held-out validation scoring with the configured
`LossAdapter`/`l2_otsu`, streamed residual/JVP normal equations, and small
damped LM updates in whitened active setup DOFs.

Implicit reconstruction hypergradients are intentionally out of scope for this
production pass. They remain future research. This plan removes stale product
paths that imply candidate detector-centre solvers, reconstruction-heavy
bilevel/L-BFGS setup discovery, or static vertical detector shifts are part of
the public capability story.

## Key Decisions

- Ship validation-LM as production setup alignment.
- Defer implicit reconstruction hypergradients.
- Execute schedules as staged runtime plans rather than flattened DOF unions.
- Make CLI `--schedule` and explicit `--optimise-dofs` mutually exclusive.
- Add `AlignConfig.schedule` for Python API parity.
- Reject gauge-coupled direct setup+pose DOF sets by default.
- Keep `det_v_px` as a low-level DOF, not a headline preset or benchmark.
- Preserve best-effort legacy checkpoint resume where metadata is safely
  compatible; prefer clear rejection over ambiguous staged resume.

## Implementation Units

### U1. Prune Obsolete Setup Solver Code

Remove old setup-alignment code whose only purpose was candidate search,
full reconstruction-differentiated bilevel setup discovery, or scalar L-BFGS
setup optimization. Keep fold construction, stopped train-fold reconstruction,
streamed validation residual/JVP accumulation, geometry state helpers, and
public pose optimization paths.

### U2. Make Schedules Executable Runtime Plans

Add `AlignConfig.schedule` and a shared schedule resolver. Execute
`AlignmentSchedule.stages` inside `align_multires`, carrying state between
setup validation-LM stages and fixed-volume pose stages. Preserve direct DOF
selection as the lower-level surface.

### U3. Integrate Gauge Policy And Setup Bounds

Validate each resolved stage before reconstruction. Extend bounds parsing to
all public DOFs, with setup `*_deg` bounds accepted in degrees and detector
centre bounds accepted in native detector pixels.

### U4. Upgrade Checkpoint And Resume For Staged Alignment

Persist resolved schedule metadata and current stage state in checkpoints as
optional additive schema fields. Maintain best-effort compatibility for legacy
checkpoints that lack stage metadata but match the effective request.

### U5. Unify CLI, Python API, Demo Runner, And Scenario Catalog Resolution

Make CLI, `AlignConfig`, and the 128³ evidence generator use the same schedule
resolver. Manifest and evidence artifacts should record the resolved schedule,
stages, objective provenance, gauge decision, active DOFs, and final setup
calibration state.

### U6. Normalize Metadata, Diagnostics, And Status Vocabulary

Ensure pose and setup stages both emit objective, optimizer, loss, schedule,
stage, gauge, and validation-LM diagnostics. Avoid stale labels that describe
the product setup path as reconstruction-differentiated scalar bilevel/L-BFGS.

### U7. Harden Tests And Laptop Evidence Gates

Add/update focused tests for schedule resolution and execution, gauge rejection,
setup bounds, checkpoint metadata, CLI/config parity, scenario catalog
contracts, generator dry-runs, and validation-LM provenance. GPU/laptop
evidence remains the gate for 64³ and 128³ claims.

### U8. Documentation And Requirements Refresh

After code lands, update the origin brainstorm and user-facing docs to reflect
the actual production architecture: validation-LM setup stages, executable
schedules, CLI/API parity, gauge policy, setup bounds, and canonical evidence
suites.

## Verification Plan

Run focused local tests continuously, then run the broader alignment/CLI test
surface. Heavy evidence remains on the Linux laptop:

1. `smoke_64`
2. `capability_128`
3. `stress_128`
4. `pose_parity_128`
5. `diagnostic_128`
6. `comprehensive_128` only after component suites pass

Acceptance requires no OOM, no product setup calls to the old
reconstruction-heavy L-BFGS/bilevel objective, staged schedule execution,
matching CLI/demo provenance, expected improvements for success/stress cases,
expected weak/rejected diagnostic outcomes, and refreshed docs.
