# Benchmark: rich_phantom94_setup_global_tomo

## Summary

| Impl | Profile | Status | Time to verified | Total time | Volume NMSE | Final residual |
|---|---|---|---:|---:|---:|---:|
| reimagined_align_auto_smoke | reference | passed | n/a | 117.769 | 0.637862 | 42.6748 |

## Bootstrap Runtime

- Role: n/a
- Schur passes: n/a
- Executed geometry updates: n/a
- FISTA refresh iterations: n/a
- Final Schur accepted: n/a
- Final Schur loss: n/a

## Dataset

- Name: rich_phantom94_setup_global_tomo
- Artifact directory: runs/rich_phantom_v1_parity_20260508_155829/datasets/rich_phantom94_setup_global_tomo_128_supported_only
- Volume shape: 128, 128, 128
- Projection views: 128

## Benchmark Manifest Criteria

- det_u_error_px_lt: 0.5
- theta_offset_error_deg_lt: 0.1


## Benchmark Manifest Evaluation

- Status: failed
- Passed: 1
- Failed: 1
- Not evaluated: 0

| Criterion | Status | Value | Threshold | Reason |
|---|---|---:|---:|---|
| det_u_error_px_lt | passed | 0.0578642 | 0.5 | evaluated against smoke geometry recovery metric |
| theta_offset_error_deg_lt | failed | 0.0218166 | 0.00174533 | evaluated against smoke geometry recovery metric |


## Geometry Recovery

| Metric | Value |
|---|---:|
| Passed | True |
| Supported DOFs improved | True |
| det_u realised RMSE px | 0.0578642 |
| det_v realised RMSE px | 0 |
| theta realised RMSE rad | 0.0218166 |

## Backend Provenance

- Requested: core_trilinear_ray
- Actual: core_trilinear_ray

## Projection Loss Provenance

| Metric | Value |
|---|---:|
| Schur train loss | 0.15207 |
| Heldout loss | 0.12504 |
| Final volume / final geometry | 42.6748 |
| Final volume / true geometry | 42.679 |
| True volume / final geometry | 0.371321 |
| True volume / true geometry | 0 |
| Classification | independent_projection_losses_consistent |

## Failure Labels

projection_residual_improvement
