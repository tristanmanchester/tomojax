# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 nominal theta geometry root fix
- Goal: make nominal acquisition theta first-class in v2 geometry,
  serialization, projection, reconstruction, solver residuals, and recovery
  metrics.

### Scope

- In scope:
  - Add `theta_nominal_rad` to `PoseParameters` with zeros-compatible defaults.
  - Preserve `theta_nominal_rad` in geometry JSON and pose CSV read/write paths.
  - Use `theta_total_i = theta_scale * theta_nominal_rad_i + theta_offset_rad
    + phi_residual_rad_i` in projector, backprojector, pose-only/setup-only,
    joint Schur, and recovery metrics.
  - Generate synthetic sidecars with nominal theta 0..180 as nominal pose
    metadata instead of losing it.
  - Add focused tests proving sidecars preserve nominal theta, true geometry
    projects to near-zero residual, and corrupted geometry is higher loss.
- Out of scope:
  - Schur block trust refactor.
  - Supported-only oracle benchmark pass.
  - New report fields beyond carrying nominal theta through existing artifacts.
  - Tolerance relaxation or nuisance fitting.
  - Further legacy Ruff cleanup.
- Deep module owner: `tomojax.geometry`, `tomojax.forward`, `tomojax.recon`,
  `tomojax.align`, and `tomojax.datasets`.

### Design Sources

- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`
- `docs/tomojax-v2/07_synthetic_generator_pseudocode.md`

### Tasks

- [x] Add nominal theta to geometry state and serialization.
- [x] Wire nominal theta into forward/backprojection and solver residuals.
- [x] Wire nominal theta into sidecar generation/readback and recovery metrics.
- [x] Add focused nominal-theta tests.
- [x] Run focused validation and `just imports`.
- [x] Update `docs/implementation_log.md`.
- [x] Commit the nominal-theta slice.

### Validation

- `uv run ruff check ...` on the touched source/test files passed.
- `uv run basedpyright ...` on the touched source/test files passed with
  0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_geometry_serialization.py
  tests/test_geometry_gauges.py tests/test_synthetic_datasets.py
  tests/test_forward_reference.py tests/test_reference_fista.py
  tests/test_setup_lm.py tests/test_pose_lm.py tests/test_joint_schur_lm.py -q`
  passed: 46 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Treat nominal theta as the root geometry fix before rerunning honest
  alignment benchmarks.
- The 64^3/64-view fixed-truth failure diagnosis now points at a representation
  bug: v2 was comparing only `theta_offset_rad + phi_residual_rad` and had lost
  nominal acquisition theta from the solver/projector path.

### Risks

- Risk: broad `just check` still exposes legacy Ruff debt outside this slice.
- Mitigation: run focused validation plus `just imports`, and avoid legacy Ruff
  cleanup unless explicitly requested.
