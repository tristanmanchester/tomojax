# TomoJAX Execution Plans

`docs/tomojax-v2/04_phased_implementation_plan.md` is the canonical phased plan.
This file is the living execution plan for the active milestone only.

Update this file before starting a milestone, keep it current while working, and
summarise outcomes in `docs/implementation_log.md` before moving on.

## Active Milestone

### Canonical Phase

- Source plan: `docs/tomojax-v2/04_phased_implementation_plan.md`
- Phase: Phase 8 fixed-truth joint setup+pose diagnostic
- Goal: make the same 64^3/64-view supported-only setup-global sidecar pass
  with fixed_synthetic_truth volume and joint setup+pose Schur enabled, or
  produce a sharp blocker with artifacts.

### Scope

- In scope:
  - Run the fixed-truth joint setup+pose benchmark on
    `.artifacts/phase8_supported_only_oracle/datasets/synth128_setup_global_tomo_64_supported_only`.
  - If setup recovery is absorbed into pose, add the smallest block-wise or
    staged pose trust/prior fix needed for this supported case.
  - Keep the pose-frozen oracle as the baseline and do not judge the full
    five-case suite yet.
  - Record benchmark_result/benchmark_report/compare artifacts and a concise
    benchmark summary.
- Out of scope:
  - Stopped-reconstruction diagnosis until fixed-truth joint is passing or
    sharply classified.
  - Unsupported DOF implementation.
  - New report fields unless required to classify this diagnostic honestly.
  - Legacy Ruff cleanup.
- Deep module owner: `tomojax.align` and `tomojax.cli`.

### Design Sources

- `docs/tomojax-v2/02_loss_and_optimiser_spec.md`
- `docs/tomojax-v2/04_phased_implementation_plan.md`
- `docs/tomojax-v2/05_synthetic_128_benchmark_suite.md`
- `docs/tomojax-v2/06_verification_and_artifact_contract.md`

### Tasks

- [x] Run fixed-truth joint setup+pose benchmark on GPU.
- [x] Inspect setup/pose trace against pose-frozen oracle baseline.
- [x] Implement smallest trust/prior/staging fix if joint absorbs setup.
- [x] Add focused regression tests for the fix.
- [x] Run focused validation and `just imports`.
- [x] Update benchmark summary and implementation log.
- [x] Commit the joint fixed-truth slice.

### Validation

- No new code was required after `79851ea`; the fixed-truth joint pass used the
  existing `--geometry-update-pose-prior-strength` CLI path.
- GPU baseline joint run failed:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_baseline/`.
- GPU strong-pose-prior joint run passed:
  `.artifacts/phase8_supported_only_oracle/runs/64_fixed_truth_joint_pose_prior_1000000/`.
- Compare artifact:
  `.artifacts/phase8_supported_only_oracle/benchmark_comparison_supported_only_fixed_truth.md`.

If `just check` cannot pass, record the exact failing command, current failure,
and proposed next fix before stopping.

### Decisions And Deviations

- The supported-only pose-frozen fixed-truth oracle passed in `79851ea`; use it
  as the baseline for judging joint setup+pose.
- Fixed-truth joint setup+pose passes only when pose is strongly constrained,
  which classifies the unconstrained failure as setup absorption into per-view
  pose.

### Risks

- Risk: joint setup+pose may legitimately absorb global setup shifts into
  per-view pose without stronger priors or staged pose activation.
- Mitigation: keep this slice scoped to supported-only fixed-truth and record
  the trace before changing the broader benchmark path.
