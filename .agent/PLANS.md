# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 staged pose activation
- Goal: parameterise joint Schur active pose DOFs and transfer mean pose steps
  into setup so fixed-truth supported setup recovery is not absorbed into pose.

### Scope

- In scope:
  - Extend `JointSchurLMConfig.active_pose_dofs` beyond all-or-none.
  - Thread partial active pose DOFs through pack/split/update and CLI config.
  - Transfer mean active pose steps into setup inside LM candidate construction.
  - Add focused tests for `phi_residual_rad` frozen with detector pose active.
  - Rerun focused Schur/align tests and the supported-only fixed-truth joint GPU
    benchmark with `phi_residual_rad` frozen.
- Out of scope:
  - Full per-DOF unit trust for every supported/unsupported future DOF.
  - Stopped-reconstruction classification.
  - Report-field expansion beyond diagnostics needed for this fix.
  - Legacy Ruff cleanup.
- Deep module owner: `tomojax.align`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`

### Tasks

- [x] Implement partial active pose DOF packing/splitting/updating.
- [x] Expose active pose DOFs through align-auto CLI/config.
- [x] Implement zero-mean pose step gauge projection in LM.
- [x] Add focused regression tests.
- [x] Run focused validation and `just imports`.
- [x] Rerun supported-only fixed-truth joint GPU benchmark with phi frozen.
- [x] Update docs with result/blocker.
- [x] Commit the staged/zero-mean pose slice.

### Validation

- `uv run ruff check ...` on touched staged/zero-mean source/test files passed.
- `uv run basedpyright ...` on touched staged/zero-mean source/test files passed with
  0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py
  tests/test_align_auto_cli.py -q` passed: 22 tests.
- `just imports` passed.
- Best non-hard-prior GPU run:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_zero_mean_no_phi_reference/`
  passes manifest criteria but misses internal det_u gate by about 0.001 px.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Block-wise setup/pose trust removed aggregate clipping, but the unconstrained
  joint candidate still pointed setup in the wrong detector direction. Stage
  pose DOFs next.
- Zero-mean pose step projection nearly resolves fixed-truth joint setup+pose
  without a hard pose prior, but detector pose active still leaves det_u just
  outside the internal 0.2 px gate.

### Risks

- Risk: setup/pose block-wise trust may improve detector recovery but still
  leave theta absorbed into `phi_residual_rad`.
- Mitigation: classify that separately and then stage/freeze `phi` rather than
  broadening scope.
