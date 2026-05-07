# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8/9 pose-only Schur bug fix
- Goal: fix the pose-only joint Schur path exposed by the five-case CUDA pass
  so `synth128_pose_random_extreme` can run with `active_setup_parameters=()`
  and all five per-view pose DOFs active.

### Scope

- In scope:
  - Support `n_setup == 0` in the streamed Schur normal-equation solver.
  - Preserve diagnostics with empty setup Schur eigen/correlation data and
    finite pose diagnostics.
  - Add a focused pose-only Schur regression test.
  - Rerun the failed `synth128_pose_random_extreme` CUDA benchmark case.
- Out of scope:
  - Report wording, criterion aliasing, or observability-field cleanup.
  - Shrinking the benchmark as a substitute for fixing memory behaviour.
  - Reworking report semantics or benchmark criteria.
  - Solving pose-random recovery quality beyond making the benchmark executable.
- Deep module owner: `tomojax.align` for joint Schur behavior.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Implement pose-only Schur normal-equation support.
- [x] Add focused tests and run validation.
- [x] Rerun the pose-random CUDA benchmark case.
- [x] Update `docs/implementation_log.md` and commit the fix.

### Validation

- `just imports` passed after the diagnostic log update.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed: 20 tests in 268.87 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_reference_fista.py
  tests/test_vertical_smoke.py
  tests/test_joint_schur_lm.py::test_schur_step_matches_dense_normal_solve
  tests/test_joint_schur_lm.py::test_joint_schur_streamed_normals_put_pure_setup_error_in_setup_gauge
  -q` passed: 10 tests in 84.38 seconds.
- `uv run ruff check src/tomojax/recon/_backprojection_accumulation.py
  src/tomojax/recon/_reference.py src/tomojax/recon/_fista_reference.py
  src/tomojax/align/_joint_schur_lm.py src/tomojax/forward/_projector.py
  tests/test_joint_schur_lm.py` passed.
- `uv run basedpyright src/tomojax/recon/_backprojection_accumulation.py
  src/tomojax/recon/_reference.py src/tomojax/recon/_fista_reference.py
  src/tomojax/align/_joint_schur_lm.py src/tomojax/forward/_projector.py
  tests/test_joint_schur_lm.py` passed with 0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the streaming changes.
- GPU diagnostic artifacts:
  `.artifacts/phase8_streamed_schur_probe/logs/128_setup_global_full5_fixed_truth_cuda_rerun2/`
  and
  `.artifacts/phase8_streamed_schur_probe/logs/64_setup_global_full5_fixed_truth_cuda/`.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py -q`
  passed after the batched Schur scan: 20 tests in 100.05 seconds.
- `uv run ruff check src/tomojax/align/_joint_schur_lm.py
  src/tomojax/forward/_filters.py tests/test_joint_schur_lm.py` passed.
- `uv run basedpyright src/tomojax/align/_joint_schur_lm.py
  src/tomojax/forward/_filters.py tests/test_joint_schur_lm.py` passed with
  0 errors, 0 warnings, and 0 notes.
- `just imports` passed after the batched Schur update.
- 64^3/64-view full-5DOF fixed-truth CUDA probe completed in about 79 seconds
  with peak sampled GPU memory 735 MiB.
- 128^3/256-view full-5DOF fixed-truth CUDA probe completed in about 253
  seconds with peak sampled GPU memory 1265 MiB.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_geometry_update_policy.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_run_theta_scale_setup_update
  -q` passed: 5 tests in 9.67 seconds.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_alternating_solver_smoke.py::test_alternating_solver_smoke_writes_artifacts
  -q` passed: 1 test in 49.91 seconds.
- `uv run ruff check src/tomojax/align/_alternating_geometry_update.py
  tests/test_alternating_geometry_update_policy.py` passed.
- `uv run basedpyright src/tomojax/align/_alternating_geometry_update.py
  tests/test_alternating_geometry_update_policy.py` passed with 0 errors,
  0 warnings, and 0 notes.
- `just imports` passed after the Schur staging policy update.
- CUDA setup-global fixed-truth rerun with `theta_scale` and all five pose DOFs
  requested resolved to active setup-only Schur updates, passed all 4/4 manifest
  geometry criteria, and peaked at 1259 MiB sampled GPU memory. Artifact:
  `.artifacts/phase8_setup_staging_policy/128_setup_global_policy_filtered_setup_only_fixed_truth_cuda/`.

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
