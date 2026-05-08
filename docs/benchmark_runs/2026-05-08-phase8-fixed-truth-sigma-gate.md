# 2026-05-08 Phase 8 Fixed-Truth Schur Sigma Gate

Ran the fixed-truth `synth128_pose_random_extreme` 128^3/256-view CUDA oracle
after changing fixed-truth geometry updates to use continuation-level sigma
instead of the robust residual scale estimated from the current corrupted
geometry.

```bash
LD_LIBRARY_PATH="$(find .venv/lib/python3.12/site-packages/nvidia -type d \( -path '*/lib' -o -path '*/lib64' \) | paste -sd: -):${LD_LIBRARY_PATH}" \
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
/usr/bin/time -v uv run tomojax-align-auto-smoke \
  --out-dir .artifacts/phase8_fixed_truth_sigma/runs/synth128_pose_random_extreme_fixed_truth_no_nuisance_fit_cuda \
  --profile reference \
  --synthetic-dataset-dir .artifacts/phase8_five_case_128_cuda_oracle/synth128_pose_random_extreme_fixed_truth/datasets/synth128_pose_random_extreme_128 \
  --geometry-update-volume-source fixed_synthetic_truth \
  --geometry-update-active-setup-parameters none \
  --geometry-update-active-pose-dofs alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px \
  --geometry-update-pose-activate-at-level-factor 4 \
  --apply-synthetic-nuisance
```

## Result

| Metric | Baseline fixed-truth | Level-sigma fixed-truth |
|---|---:|---:|
| JAX device | `cuda:0` | `cuda:0` |
| Benchmark status | failed | failed |
| Wall time s | 210.74 | 217.02 |
| Volume NMSE | 3500.044434 | 0.275307 |
| Final residual | 642.871948 | 2.162959 |
| `alpha_beta_rmse_rad` | 0.032216 | 0.036027 |
| `theta_realized_rmse_rad` | 0.105970 | 0.106481 |
| `det_u_realized_rmse_px` | 9.350136 | 9.349360 |
| Coarse effective sigma | 578.393372 | 1.0 |

Artifacts:

- Level-sigma run:
  `.artifacts/phase8_fixed_truth_sigma/runs/synth128_pose_random_extreme_fixed_truth_no_nuisance_fit_cuda/`
- Command log:
  `.artifacts/phase8_fixed_truth_sigma/logs/synth128_pose_random_extreme_fixed_truth_no_nuisance_fit_cuda.log`

## Interpretation

The previous fixed-truth oracle computed Schur sigma from the corrupted-geometry
projection residual, producing effective sigma values around `400-580`. That
suppressed the LM objective enough to leave a pathological final reconstruction
metric. Using the continuation sigma fixes that residual-scaling pathology.

It does not solve all-5 pose recovery: alpha/beta, phi/theta-realized, and
detector-shift realized errors remain outside tolerance. The next pose slice
should address the pose parameterization/acceptance objective itself, not
residual scaling.
