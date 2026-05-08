# Benchmark: rich_phantom94_setup_global_tomo

## Summary

| Impl | Profile | Status | Time to verified | Total time | Volume NMSE | Final residual |
|---|---|---|---:|---:|---:|---:|
| reimagined_align_auto_smoke | reference | failed | n/a | 140.184 | 0.710293 | 44.4006 |

## Bootstrap Runtime

- Role: geometry_first_bootstrap
- Schur passes: 2
- Executed geometry updates: 8
- FISTA refresh iterations: 8
- Final Schur accepted: True
- Final Schur loss: 13.2475

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
- Passed: 0
- Failed: 2
- Not evaluated: 0

| Criterion | Status | Value | Threshold | Reason |
|---|---|---:|---:|---|
| det_u_error_px_lt | failed | 4.10057 | 0.5 | evaluated against smoke geometry recovery metric |
| theta_offset_error_deg_lt | failed | 0.0218166 | 0.00174533 | evaluated against smoke geometry recovery metric |


## Geometry Recovery

| Metric | Value |
|---|---:|
| Passed | False |
| Supported DOFs improved | True |
| det_u realised RMSE px | 4.10057 |
| det_v realised RMSE px | 0 |
| theta realised RMSE rad | 0.0218166 |

## Backend Provenance

- Requested: core_trilinear_ray
- Actual: core_trilinear_ray

## Projection Loss Provenance

| Metric | Value |
|---|---:|
| Schur train loss | 10.7033 |
| Heldout loss | 0.153397 |
| Final volume / final geometry | 44.4006 |
| Final volume / true geometry | 53.3689 |
| True volume / final geometry | 24.1107 |
| True volume / true geometry | 0 |
| Classification | reconstruction_absorbed_geometry |

## Failure Labels

projection_residual_improvement
