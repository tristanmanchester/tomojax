# Phase 8/9 Five-Case 128^3 CUDA Benchmark Classification

Run root: `.artifacts/phase8_five_case_128_cuda/`

Comparison report:
`.artifacts/phase8_five_case_128_cuda/comparison_report.md`

Common runtime environment:

```text
JAX_PLATFORMS=cuda
CUDA_VISIBLE_DEVICES=0
XLA_PYTHON_CLIENT_PREALLOCATE=false
profile=reference
size=128
backend=core_trilinear_ray
geometry_update_volume_source=stopped_reconstruction
```

## Results

| Case | Exit | Criteria | Peak MiB | Runtime s | Classification |
|---|---:|---|---:|---:|---|
| `synth128_setup_global_tomo` | 0 | 0 passed, 4 failed | 1317 | 201.19 | stopped-reconstruction/volume-gauge failure; fixed-truth setup-only evidence passes |
| `synth128_pose_random_extreme` | 0 | 0 passed, 1 failed, 2 not evaluated | 1257 | 123.44 | all-5 pose Schur solver/recovery failure; fixed-truth oracle also fails |
| `synth128_lamino_axis_roll_pose` | 0 | 2 passed, 3 failed | 1327 | 230.37 | laminography setup+pose solver/reconstruction failure; backend fallback and det_v policy pass |
| `synth128_thermal_object_drift` | 0 | 0 failed, 2 not evaluated | 1317 | 255.23 | unsupported object-frame drift and theta-scale policy gap |
| `synth128_combined_nuisance_jumps` | 0 | 0 passed, 3 failed, 3 not evaluated | 1327 | 285.75 | combined unsupported bad-view/jump/object/nuisance behavior plus solver failure |

## Evidence Notes

- `synth128_setup_global_tomo` stopped-reconstruction failed all setup-global
  geometry criteria (`det_u`, roll, axis, theta). The prior fixed-truth CUDA
  setup-only run passed all 4/4 criteria, so the current blocker is
  reconstruction/stopped-volume gauge handling rather than the supported setup
  Schur geometry convention.
- `synth128_pose_random_extreme` initially crashed with an empty `(0, 0)` setup
  Schur matrix when run as pose-only. Commit `621cb54` fixed the pose-only
  normal-equation path. The rerun exits 0, but fixed-truth oracle evidence still
  fails (`alpha_beta_rmse_rad=0.0322`, `theta_realized_rmse_rad=0.106`,
  `det_u/det_v` realised errors around 9 px), so this is a real all-5 pose
  solver/recovery failure.
- `synth128_lamino_axis_roll_pose` records the expected calibrated detector-grid
  fallback and passes det_v policy, but axis, roll, and `det_u` recovery fail.
  This points to laminography setup+pose solver/reconstruction behavior, not
  missing backend provenance.
- `synth128_thermal_object_drift` has only object-motion criteria in the
  manifest and both are not evaluated because object-frame motion metrics are
  absent. The active Schur block also freezes `theta_scale`, while the true
  dataset has `theta_scale=1.0008`.
- `synth128_combined_nuisance_jumps` has failed setup/pose criteria and
  not-evaluated bad-view/jump/current-baseline criteria. This remains the final
  combined-capability benchmark, not the next single-slice target.

## Next Slice

The highest-impact next functional slice is stopped-reconstruction gauge
handling for setup-global recovery. It has the cleanest split: fixed-truth
geometry passes while stopped-reconstruction fails on the same 128^3/256-view
case.
