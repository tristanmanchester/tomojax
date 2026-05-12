# 2026-05-13 Production Hardening Status

This is the current morning-facing status for the TomoJAX v2 production
hardening branch.

## Public Surface

Clean user-facing entrypoints now exist for the primary workflows:

```bash
tomojax align --profile staged-lamino --out <run-dir> <scan-or-stack>
tomojax-align-auto --synthetic-case setup-global --size 128 --views 256 --out-dir <run-dir>
tomojax-align-auto --synthetic-case pose-random --size 128 --views 256 --out-dir <run-dir>
```

Public documentation was added or updated in:

- `README.md`
- `docs/quickstart.md`
- `docs/real-laminography.md`
- `docs/synthetic-tomography.md`
- `docs/known-limitations.md`

Historical naming is still present in archived reports and internal regression
fixtures, but the public docs and help surfaces have been cleaned to describe
behavior rather than development history.

## Real Laminography

The clean staged laminography workflow is documented as the production-shaped
real-data path. The preserved k11 evidence remains the validation anchor:

- the retained 2026-05-12 k11 staged real-laminography run artifacts
- the clean staged workflow documentation in `docs/real-laminography.md`

That evidence should be read as a validated staged workflow on the reference
real dataset, not proof that arbitrary laminography scans are turnkey.

## Synthetic Tomography Gates

The mandatory original 128^3 tomography gates were run on CUDA with the
canonical `core_trilinear_ray` backend.

| Case | Status | Artifact |
|---|---|---|
| `synth128_setup_global_tomo` | passed at 128^3/256 views | `.artifacts/production_hardening_synthetic/synth128_setup_global_128_after_loss_cache` |
| `synth128_pose_random_extreme` | passed at 128^3/256 views after pose-only Schur gauge-carry fix and bounded final pose polish | `.artifacts/production_hardening_synthetic/synth128_pose_random_128_pose_gauge_fix_polish` |
| `synth128_lamino_axis_roll_pose` | failed laminography axis/roll recovery at 128^3/256 views | `.artifacts/production_hardening_synthetic/synth128_lamino_axis_roll_pose_128_classification` |
| `synth128_thermal_object_drift` | flagged object motion but failed recovery because object-frame motion solver is not enabled | `.artifacts/production_hardening_synthetic/synth128_thermal_object_drift_128_classification` |
| `synth128_combined_nuisance_jumps` | failed hard-case setup/axis/theta recovery at 128^3/320 views; bad-view and jump-excluded dx/dz diagnostics evaluated | `.artifacts/production_hardening_synthetic/synth128_combined_nuisance_jumps_128_classification` |

Detailed metrics, commands, device provenance, runtime, and memory evidence are
in `docs/benchmark_runs/2026-05-13-synthetic128-production-gates.md`.

## Runtime Fix

The previous setup-global 256-view blocker was compile/orchestration overhead,
not VRAM. The current slice cached streamed Schur normal equations plus scalar
and per-view loss diagnostics within each LM solve. After that change, the full
128^3/256-view setup-global gate completed with about 1.4 GiB peak sampled GPU
memory and passed all setup/COR/roll/axis/theta criteria.

## Pose-Random Fix

The previous pose-random 256-view blocker was a pose-only Schur state-carry bug,
not missing metric wiring, CPU fallback, or reconstruction absorption. Accepted
LM iterations canonicalized mean `dx`/`phi` into setup, then repacked a
pose-only parameter vector that could not preserve those setup gauge values.
After preserving pose-only mean gauges until final canonicalisation and enabling
the bounded final pose-polish stage for the clean pose-random preset, the
128^3/256-view oracle gate passed:

- `dx_dz_rmse_px = 0.040028767293657966`
- `phi_rmse_rad = 0.007029254273335157`
- `alpha_beta_rmse_rad = 0.0017171851316756014`
- `det_u_realized_rmse_px = 0.015837733351848012`
- `theta_realized_rmse_rad = 0.006883447413717324`

## Remaining Release Blockers

- The two mandatory tomography gates now pass at 128^3/256 views.
- Remaining original synthetic scenarios are explicitly classified but not green:
  laminography still needs axis/roll recovery and det-v policy evidence,
  thermal drift needs real object-frame motion recovery, and the combined
  nuisance/jumps case still fails hard setup/axis/theta recovery.
- These remaining red scenarios are research/Phase 8+ capability work, not
  blockers for the setup-global and pose-random tomography production gates.

## Validation Snapshot

Focused validation for the current Schur/loss-cache and pose-gauge slices:

- `ruff check src/tomojax/align/_joint_schur_lm.py --select F821,I001,E501`
- `basedpyright src/tomojax/align/_joint_schur_lm.py`
- focused alternating Schur recovery pytest for truth-volume geometry updates
- targeted align-auto contract tests after updating stale public-profile and
  sidecar expectations; the combined process hit the known JAX CPU compiler
  abort in the existing-dataset test, and that test passed when rerun alone.
- `pytest tests/test_joint_schur_lm.py::test_joint_schur_lm_pose_only_preserves_mean_gauge_until_final_canonicalization tests/test_align_auto_cli.py::test_synthetic_pose_random_case_resolves_bounded_oracle -q`
- `ruff check src/tomojax/align/_joint_schur_lm.py src/tomojax/cli/align_auto.py tests/test_joint_schur_lm.py tests/test_align_auto_cli.py --select F821,I001,E501`
- `basedpyright src/tomojax/align/_joint_schur_lm.py src/tomojax/cli/align_auto.py tests/test_joint_schur_lm.py tests/test_align_auto_cli.py`
- `just imports`

Full `just check` remains to be rerun before a public release tag.
