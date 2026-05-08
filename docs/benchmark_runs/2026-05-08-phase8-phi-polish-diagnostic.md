# 2026-05-08 Phase 8 Phi Polish Diagnostic

Ran direct true-volume Schur probes from the staged alpha/beta result:

Base run:
`.artifacts/phase8_alpha_beta_staging/runs/pose_random_fixed_truth_alpha_beta_final_no_trust_cuda/`

The base run recovered translations to about 1 px but left
`theta_realized_rmse_rad = 0.125796`.

## Phi-Only Polish

Configuration:

```text
start geometry = staged alpha/beta final result
active_setup_parameters = ()
active_pose_dofs = phi_residual_rad
pose_trust_radius = none
sigma = 1.0
residual_filters = raw
```

| Iterations | Final loss | alpha/beta RMSE rad | phi RMSE rad | dx RMSE px | dz RMSE px |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.072537 | 0.017550 | 0.089799 | 0.901970 | 0.954342 |
| 8 | 0.065535 | 0.017550 | 0.072270 | 0.901970 | 0.954342 |
| 16 | 0.059268 | 0.017550 | 0.054667 | 0.901970 | 0.954342 |

## Alpha/Beta/Phi Polish

| Iterations | Final loss | alpha/beta RMSE rad | phi RMSE rad | dx RMSE px | dz RMSE px |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.067765 | 0.020715 | 0.089995 | 0.901970 | 0.954342 |
| 8 | 0.059454 | 0.034194 | 0.073068 | 0.901970 | 0.954342 |
| 16 | 0.052687 | 0.043024 | 0.052549 | 0.901970 | 0.954342 |

## Interpretation

Dedicated phi-only polishing improves the remaining theta/phi error while
preserving recovered dx/dz and alpha/beta. Letting alpha/beta move during the
same polish lowers loss slightly more but worsens alpha/beta recovery. The next
implementation slice should add an opt-in final phi-only polish stage rather
than another all-angular update.
