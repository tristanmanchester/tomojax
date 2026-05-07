# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 core projector rebaseline
- Goal: retire the rotate-and-sum v2 operational projector path and make v2
  generation, reconstruction preview, Schur updates, verification losses, and
  benchmark artifacts consistently use the existing core trilinear ray
  projector/backprojector.

### Scope

- In scope:
  - Add a typed adapter from supported v2 `GeometryState` to core `Grid`,
    `Detector`, detector grid, and per-view `T_all` pose matrices.
  - Route `tomojax.forward.project_parallel_reference*` through the core
    trilinear ray projector so existing v2 alignment paths use one projector
    family without a backend selector between toy and core.
  - Route preview backprojection through the core explicit adjoint and wire FISTA
    preview through core projection loss/grad or equivalent core calls.
  - Record core operator provenance in run artifacts and dataset manifests.
  - Rebaseline small CPU adapter tests, supported-only 32^3 smoke, 64^3/64-view
    GPU fixed-truth pose-frozen setup recovery, and anchored stopped setup-only
    diagnostics.
- Out of scope:
  - Detector-boundary rotate-sum polishing, laminography/roll/axis/object-drift
    implementation, new long-term projector selectors, threshold relaxation,
    legacy Ruff cleanup, or five-case reruns before the core path is coherent.
- Deep module owners: `tomojax.forward` for the v2-to-core operator adapter,
  `tomojax.recon` for preview reconstruction, `tomojax.align`/`tomojax.datasets`
  for orchestration/artifact provenance.

### Design Sources

- `docs/tomojax-v2/01_high_level_architecture.md`
- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Replace operational v2 forward projection with core trilinear ray calls.
- [x] Add focused CPU tests for nominal theta and detector/pose shifts in the
  `GeometryState` -> core adapter.
- [~] Replace rotate-and-sum preview backprojection and FISTA projection loss
  with core explicit adjoint / core FISTA arrays.
- [x] Make unsupported DOFs explicit errors or `unsupported_dof_not_evaluated`,
  not silently ignored.
- [x] Record `core_trilinear_ray` operator provenance and core grid/detector
  traversal settings in manifests and benchmark artifacts.
- [x] Run focused Ruff/format/typecheck, relevant core/forward/recon/align/CLI
  tests, and `just imports`.
- [x] Run supported-only 32^3 smoke plus 64^3/64-view GPU fixed-truth and stopped
  anchored diagnostics under the core projector.
- [ ] Update implementation log and commit the coherent rebaseline slice.

### Validation

- `uv run ruff format ...` passed for touched Python files.
- `uv run ruff check ...` passed for touched Python files.
- `uv run basedpyright ...` passed with 0 errors and 0 warnings for touched
  source and tests.
- `JAX_PLATFORM_NAME=cpu uv run pytest tests/test_forward_reference.py
  tests/test_reference_fista.py
  tests/test_joint_schur_lm.py::test_joint_schur_lm_can_freeze_pose_dofs_for_setup_oracle
  tests/test_align_auto_cli.py::test_align_auto_generates_supported_only_pose_frozen_oracle
  -q` passed: 18 tests.
- `just imports` passed.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- New decision: the v2 rotate-and-sum projector is no longer an operational
  projector path. The only supported v2 operator family is the existing core
  trilinear ray projector/backprojector (`core_trilinear_ray`).
- Do not add a long-term selector between rotate-and-sum and core trilinear ray.
  Any old rotate-sum behavior may remain only as deleted history or a narrowly
  private test fixture if unavoidable.
- Unsupported DOFs in this slice are explicit: detector roll, axis tilt,
  laminography, alpha/beta, and object drift are not implemented by the adapter
  until their core convention mapping is defined and tested.
- Preview backprojection now uses the core explicit adjoint. The tiny
  `fista_reconstruct_reference` wrapper still uses reverse-mode over the core
  forward projection for its masked robust loss and remains a follow-up before
  the objective is fully complete.
- 64^3/64-view fixed-truth recovery passes under the core operator when the
  oracle Schur path uses raw geometry residuals, disables the level metadata
  prior, and does not early-exit after coarse preview verification:
  `det_u` RMSE `1.43051e-06` px and theta RMSE `1.06805e-07` rad.
- Anchored stopped det_u-only reaches `0.237177` px under the core projector,
  missing the strict `0.2` px criterion. The current blocker is stopped
  reconstruction/volume gauge under the real operator.

### Risks

- Risk: current recovery metrics were calibrated against the toy projector and
  may regress once the physical core ray operator is used.
- Mitigation: rebaseline fixed-truth first; if fixed-truth fails, treat it as an
  adapter/scaling blocker before interpreting stopped reconstruction quality.
