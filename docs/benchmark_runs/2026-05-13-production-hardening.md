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

- `runs/real_lamino_v2_v1_parity_full_after_fista_fallback_20260512`
- `docs/benchmark_runs/2026-05-12-real-lamino-v2-production-mvp.md`

That evidence should be read as a validated staged workflow on the reference
real dataset, not proof that arbitrary laminography scans are turnkey.

## Synthetic Tomography Gates

The mandatory original 128^3 tomography gates were run on CUDA with the
canonical `core_trilinear_ray` backend.

| Case | Status | Artifact |
|---|---|---|
| `synth128_setup_global_tomo` | passed at 128^3/256 views | `.artifacts/production_hardening_synthetic/synth128_setup_global_128_after_loss_cache` |
| `synth128_pose_random_extreme` | failed pose recovery at 128^3/256 views | `.artifacts/production_hardening_synthetic/synth128_pose_random_128_after_loss_cache` |
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

## Remaining Release Blockers

- `synth128_pose_random_extreme` still fails the oracle fixed-volume pose gate:
  dx/dz recovers, but phi and alpha/beta remain outside the strict manifest
  tolerances.
- The next functional slice should improve native-resolution pose Schur update
  policy/conditioning, not add report fields or rename more aliases.
- Remaining original synthetic scenarios are now explicitly classified, but the
  full production-hardening goal is not complete because pose-random and the
  hard laminography/object-motion/nuisance cases remain red.

## Validation Snapshot

Focused validation for the current Schur/loss-cache slice:

- `ruff check src/tomojax/align/_joint_schur_lm.py --select F821,I001,E501`
- `basedpyright src/tomojax/align/_joint_schur_lm.py`
- `pytest tests/test_alternating_solver_smoke.py::test_alternating_smoke_schur_recovers_supported_dofs_with_truth_volume -q`
- targeted align-auto contract tests after updating stale public-profile and
  sidecar expectations; the combined process hit the known JAX CPU compiler
  abort in the existing-dataset test, and that test passed when rerun alone.
- `just imports`

Full validation should be rerun after the pose-recovery fix lands.
