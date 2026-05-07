# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 staged pose activation
- Goal: parameterise joint Schur active pose DOFs so fixed-truth supported
  setup recovery can freeze gauge-coupled `phi_residual_rad` without freezing
  every pose DOF.

### Scope

- In scope:
  - Extend `JointSchurLMConfig.active_pose_dofs` beyond all-or-none.
  - Thread partial active pose DOFs through pack/split/update and CLI config.
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
- [x] Add focused regression tests.
- [x] Run focused validation and `just imports`.
- [x] Rerun supported-only fixed-truth joint GPU benchmark with phi frozen.
- [x] Update docs with result/blocker.
- [x] Commit the staged pose activation slice.

### Validation

- `uv run ruff check ...` on touched staged-pose source/test files passed.
- `uv run basedpyright ...` on touched staged-pose source/test files passed with
  0 errors and 0 warnings.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_joint_schur_lm.py
  tests/test_align_auto_cli.py -q` passed: 21 tests.
- `just imports` passed.
- GPU staged-pose runs still failed strict criteria:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_staged_pose_level1/`
  and
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_staged_pose_level1_no_phi/`.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- Block-wise setup/pose trust removed aggregate clipping, but the unconstrained
  joint candidate still pointed setup in the wrong detector direction. Stage
  pose DOFs next.
- Staged pose activation is implemented, but final pose-active updates still
  leave setup outside strict criteria. The next fix should anchor/zero-mean pose
  during LM or protect verified setup during final pose refinement.

### Risks

- Risk: setup/pose block-wise trust may improve detector recovery but still
  leave theta absorbed into `phi_residual_rad`.
- Mitigation: classify that separately and then stage/freeze `phi` rather than
  broadening scope.
