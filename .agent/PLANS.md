# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 7/8 det_v observability-gating vertical slice
- Goal: make the existing report-only weak-DOF policy evaluate active
  `det_v_px` from its actual Schur setup slot, independent of theta-scale
  activation, so det_v gating evidence is meaningful when det_v is explicitly
  active.

### Scope

- In scope:
  - Use the actual active setup-parameter index for det_v weak-DOF correlation
    evidence instead of assuming a fixed setup ordering.
  - Treat det_v accepted-step evidence as available when det_v is active and
    Schur diagnostics exist, even if theta_scale remains frozen.
  - Add focused tests for det_v-only observability payloads.
  - Update docs/logs and commit the slice.
- Out of scope:
  - New report fields, automatic activation policy, tolerance changes, or
    benchmark reruns.
- Deep module owner: `tomojax.align` for observability/report payload
  semantics.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Correct det_v weak-DOF evidence indexing.
- [x] Correct det_v accepted-step evidence availability.
- [x] Add focused observability payload tests.
- [x] Run focused Ruff/type/tests and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the slice.

### Validation

- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_observability.py -q` passed: 1 test in 0.66 seconds.
- `uv run ruff check src/tomojax/align/_alternating_verification.py
  tests/test_alternating_observability.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py
  tests/test_alternating_observability.py` passed with 0 errors, 0 warnings,
  and 0 notes.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Report-only weak-DOF policy should not mutate schedules or relax tolerances.
- det_v remains optional and must only be evaluated when explicitly active in
  the Schur setup block.

### Completed Previous Slices

- [x] Detector roll supported and committed: `2be6a99`.
- [x] Axis tilt supported and committed with GPU diagnostic pause:
  `ac347d2`.
- [x] Alpha/beta pose supported and committed: `aea525d`.
- [x] Supported geometry update DOFs exposed in `align-auto`: `19dd503`.
- [x] Theta-scale opt-in setup updates committed: `be3d059`.
- [x] Parallel laminography acquisition metadata committed: `7aa086c`.

### Risks

- Risk: report-only observability policy can look green if proxy evidence is
  wired from the wrong setup slot.
- Mitigation: add a det_v-only payload test so the correlation parameter index
  must match the active Schur setup block.
