# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8/9 GPU memory-regression investigation
- Goal: explain why the v2 `core_trilinear_ray` reference path is using much
  more VRAM than old TomoJAX's streamed/chunked 5-DOF alignment path, then
  implement the smallest chunking fix once the allocation source is isolated.

### Scope

- In scope:
  - Inspect v2 projector, FISTA/backprojector, and Schur residual/Jacobian
    accumulation for all-view/all-parameter materialisation.
  - Run focused diagnostic commands at realistic scale on `cuda:0` as needed to
    identify the allocation owner.
  - Implement chunked accumulation for the offending path when isolated.
  - Update docs/logs and commit the memory-regression slice.
- Out of scope:
  - Report wording, criterion aliasing, or observability-field cleanup.
  - Shrinking the benchmark as a substitute for fixing memory behaviour.
- Deep module owner: `tomojax.forward` for projector memory behaviour,
  `tomojax.recon` for reconstruction/backprojection, and `tomojax.align` for
  Schur geometry-update accumulation.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Identify which path owns the high VRAM allocation.
- [ ] Add chunked accumulation or streaming where needed.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md` and commit the diagnostic slice.

### Validation

- `just imports` passed after the diagnostic log update.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Prior Decisions Still Binding

- The only supported v2 operator family is the existing core trilinear ray
  projector/backprojector (`core_trilinear_ray`).
- Do not add a selector between rotate-and-sum and core trilinear ray.
- Existing weak-DOF policy evidence should make policy criteria real when it is
  sufficient, without adding duplicate benchmark fields.

### Completed Previous Slices

- [x] Detector roll supported and committed: `2be6a99`.
- [x] Axis tilt supported and committed with GPU diagnostic pause:
  `ac347d2`.
- [x] Alpha/beta pose supported and committed: `aea525d`.
- [x] Supported geometry update DOFs exposed in `align-auto`: `19dd503`.
- [x] Theta-scale opt-in setup updates committed: `be3d059`.
- [x] Parallel laminography acquisition metadata committed: `7aa086c`.
- [x] det_v observability gating evidence committed: `7c1e0fe`.
- [x] Synthetic unsupported-term classification committed: `28e336f`.
- [x] Benchmark criterion aliases committed: `fe83427`.
- [x] Laminography solver residuals committed: `7002d42`.
- [x] Recovered det_v policy criterion committed: `f6fe3c4`.
- [x] Backend policy criterion evaluation committed: `b040829`.
- [x] Calibrated-grid backend provenance committed: `a0b69db`.
- [x] Missing-policy criterion reasons committed: `9034b91`.
- [x] 128^3 supported-only GPU scale gate committed: `d2fbd5a`.
- [x] Active Schur DOFs in observability committed: `7ab5013`.
- [x] Smoke expectation cleanup committed: `44dda7e`.
- [x] Nuisance-corrected failure gate committed: `f374d58`.

### Risks

- Risk: full active setup/pose updates at `128^3` may expose memory or runtime
  regressions before producing numerical recovery evidence.
- Mitigation: record command, device, peak GPU memory, runtime, and failure
  artifact path per case rather than shrinking the benchmark.
