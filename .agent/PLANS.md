# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 reporting honesty
- Goal: stop reporting the last Schur training loss as an independent final
  projection residual and classify stopped-reconstruction absorption evidence.

### Scope

- In scope:
  - Record `schur_train_loss` separately from independent all-view projection
    losses.
  - Record final-volume/final-geometry, final-volume/true-geometry,
    true-volume/final-geometry, and true-volume/true-geometry all-view losses.
  - Thread those losses into `benchmark_result.json` and `benchmark_report.md`.
  - Add focused tests for the verification and benchmark artifact contract.
- Out of scope:
  - New solver behavior, reconstruction gauge fixes, low-pass Schur changes, or
    benchmark scenario expansion.
  - Legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add independent projection-loss provenance to verification metrics.
- [x] Include loss provenance in synthetic benchmark artifacts and markdown.
- [x] Add focused artifact contract tests.
- [x] Run focused validation and `just imports`.
- [x] Rerun supported-only fixed-truth/stopped-reconstruction diagnostics if the
  artifact shape change needs fresh benchmark files.
- [x] Update docs and commit the refreshed reporting-classification slice.

### Validation

- `uv run ruff format ...` passed for touched reporting source/test files.
- `uv run ruff check src/tomojax/align/_alternating_verification.py
  src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed.
- `uv run basedpyright src/tomojax/align/_alternating_verification.py
  src/tomojax/align/_alternating_artifacts.py tests/test_align_auto_cli.py`
  passed with 0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_ingests_existing_synthetic_dataset_dir
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 2 tests.
- `just imports` passed.
- GPU fixed-truth refresh passed:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000_reporting/`.
- GPU stopped-reconstruction refresh failed as expected:
  `.artifacts/phase8_supported_only_oracle/runs/64_stopped_reconstruction_joint_pose_prior_1000000_reporting/`.
- Compare artifact:
  `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_reporting.md`.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- The previous stopped-reconstruction diagnosis relied on `final_loss` fields
  that can mean Schur training loss. This slice makes the benchmark artifacts
  explicit before further solver changes.
- Fresh GPU artifacts now include the new loss-provenance fields. Fixed-truth is
  independently consistent; stopped-reconstruction is classified as
  `reconstruction_absorbed_geometry`.

### Risks

- Risk: changing `residual_after` semantics may affect older smoke assertions.
- Mitigation: preserve `schur_train_loss` as an explicit metric and update tests
  to assert the new contract.
