# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 anchored preview reconstruction
- Goal: make stopped-reconstruction preview volumes geometry-informative enough
  that supported-only setup error reaches Schur instead of being absorbed into a
  wrong-gauge preview volume.

### Scope

- In scope:
  - Add optional support masks to reference FISTA and project both candidate and
    momentum state into nonnegative support.
  - Add centered cylindrical/spherical support generation in `tomojax.recon`.
  - Thread preview support, small TV weights, residual filters, and initialization
    choices through the alternating preview reconstruction path.
  - Wire setup-only stopped diagnostics with `det_u_px` active and pose,
    `theta_offset_rad`, and `det_v_px` frozen.
  - Run supported-only 64^3/64-view stopped diagnostics until Gate 1 is measured.
- Out of scope:
  - Sidecar/report-field expansion, nominal theta, Schur trust/damping, pose
    priors, Pallas, 128^3, real data, broad benchmark ingestion, or legacy Ruff
    cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Add support projection to reference FISTA.
- [x] Add centered support generation and tests.
- [x] Thread support/TV/residual filters/init choices through alternating
  preview reconstruction.
- [x] Add setup-only Schur active setup parameter wiring.
- [x] Run focused validation and `just imports`.
- [x] Run supported-only 64^3/64-view stopped diagnostics and record Gate 1
  outcome.
- [x] Update implementation log, benchmark docs, and commit.

### Validation

- `uv run ruff format ...` passed for touched source and tests.
- `uv run ruff check ...` passed for touched source and tests.
- `uv run basedpyright ...` passed with 0 errors and 0 warnings for touched
  source and tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_reference_fista.py
  tests/test_joint_schur_lm.py
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  tests/test_align_auto_cli.py::test_align_auto_smoke_command_ingests_existing_synthetic_dataset_dir
  -q` passed: 22 tests.
- `just imports` passed.
- `just check` did not pass due broad pre-existing repository Ruff debt outside
  this slice; unrelated formatter churn from that command was reverted.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Fixed-truth supported-only recovery now passes with filtered Schur. Stopped
  reconstruction remains at `det_u` RMSE 7.25 px with
  `reconstruction_absorbed_geometry`.
- First stopped schedule should freeze pose/theta/det_v and update only
  `det_u_px`; theta recovery is not required without an explicit orientation
  anchor.
- Gate 1 passed with cylindrical support, preview TV, continuation preview
  residual filters, constant preview initialization, and det_u-only stopped
  setup updates: `det_u` RMSE `0.453199` px and true-volume/final-geometry loss
  `0.0167246`.

### Risks

- Risk: support and TV can reduce volume gauge freedom but still leave detector
  shift hidden by detector boundary semantics.
- Mitigation: run a detector boundary diagnostic only after anchored stopped
  reconstruction reaches at least Gate 1.
- Detector-boundary diagnostic found current wrap semantics and stronger
  wrong-geometry penalties under valid-overlap masked zero-fill semantics. Keep
  that as the next slice.
