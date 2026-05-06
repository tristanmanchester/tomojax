# 06 — Verification and Artifact Contract

This document defines the artifacts every TomoJAX alignment run should emit. The goal is to make the system auditable, benchmarkable, and debuggable without exposing internal method complexity to the user.

## Run directory structure

Every run should produce:

```text
run_dir/
  config_resolved.toml
  run_manifest.json
  artifact_index.json

  input_summary.json
  projection_stats.json
  mask_summary.json

  geometry_initial.json
  geometry_final.json
  gauge_policy.json
  gauge_report.json
  observability_report.json

  pose_params.csv
  pose_decomposition.csv

  fista_trace.csv
  geometry_trace.csv
  alignment_summary.csv
  residual_metrics.csv

  verification.json
  backend_report.json
  failure_report.json

  final_volume.zarr or final_volume.npy
  preview_slices/
  residual_maps/
  plots/
```

## `run_manifest.json`

Required fields:

```json
{
  "tomojax_version": "...",
  "git_commit": "...",
  "run_id": "...",
  "started_at": "...",
  "finished_at": "...",
  "profile": "lightning",
  "align_mode": "auto",
  "dataset": {
    "source": "...",
    "shape": [256, 160, 160],
    "projection_dtype": "float32"
  },
  "geometry_model": "parallel_laminography",
  "backend_requested": "auto",
  "backend_actual": "...",
  "status": "passed"
}
```

## `artifact_index.json`

A machine-readable list of all artifacts.

```json
{
  "artifacts": [
    {
      "name": "geometry_final",
      "path": "geometry_final.json",
      "type": "json",
      "description": "Final canonical geometry state"
    }
  ]
}
```

## `geometry_initial.json` and `geometry_final.json`

Should contain setup parameters in physical/canonical form.

```json
{
  "setup": {
    "det_u_px": {
      "value": -3.72,
      "unit": "px",
      "active": true,
      "observable": true
    },
    "det_v_px": {
      "value": 0.0,
      "unit": "px",
      "active": false,
      "observable": false,
      "reason": "gauge_coupled_with_mean_dz"
    },
    "detector_roll_deg": {
      "value": 0.137,
      "active": true
    },
    "axis_rot_x_deg": {
      "value": 0.509,
      "active": true
    },
    "axis_rot_y_deg": {
      "value": -0.002,
      "active": true
    },
    "theta_offset_deg": {
      "value": -0.205,
      "active": true
    }
  },
  "gauge": {
    "mean_dx_zero": true,
    "mean_phi_residual_zero": true,
    "mean_dz_zero": true
  }
}
```

## `pose_params.csv`

One row per projection:

```text
view_index
theta_nominal_deg
theta_total_deg
alpha_deg
beta_deg
phi_residual_deg
dx_px
dz_px
valid
weight
```

## `pose_decomposition.csv`

If motion decomposition is active:

```text
view_index
dx_global_component
dx_smooth_component
dx_harmonic_component
dx_jump_component
dx_residual_component
...
```

This is optional early, but the schema should be anticipated.

## `gauge_report.json`

Record all canonicalisation operations.

```json
{
  "operations": [
    {
      "name": "mean_dx_to_det_u",
      "value_transferred_px": 2.314,
      "projection_difference_max": 1.2e-5,
      "exact": true
    },
    {
      "name": "mean_phi_to_theta_offset",
      "value_transferred_deg": -0.205,
      "projection_difference_max": 3.1e-5,
      "exact": true
    }
  ],
  "status": "passed"
}
```

If a gauge operation is not exact:

```json
{
  "name": "mean_dz_to_det_v",
  "exact": false,
  "status": "soft_prior_only",
  "reason": "changes_forward_model_under_current_lamino_geometry"
}
```

## `observability_report.json`

Required for weak DOFs.

```json
{
  "dofs": {
    "det_v_px": {
      "active": false,
      "curvature": 1.2e-8,
      "correlation_max": 0.997,
      "status": "unobservable",
      "reason": "near_gauge_with_mean_dz"
    },
    "theta_scale": {
      "active": true,
      "curvature": 0.031,
      "correlation_max": 0.54,
      "status": "observable"
    }
  },
  "schur_condition_number": 523.1,
  "weak_modes": [
    "det_v_px_mean_dz"
  ]
}
```

## `fista_trace.csv`

Rows:

```text
stage
level
iteration
loss
data_loss
regulariser
step_size
L_estimate
effective_iteration
wall_time_s
backend_actual
pallas_used
fallback_reason
```

If loss is not computed every iteration, include:

```text
loss_computed = false
loss_alias = "not_computed_for_speed"
```

## `geometry_trace.csv`

Rows:

```text
stage
level
outer_iter
lm_iter
loss_before
loss_after
predicted_decrease
actual_decrease
lm_ratio
accepted
damping
step_scale
grad_norm
setup_step_norm
pose_step_norm_mean
pose_step_norm_max
schur_condition_number
wall_time_s
backend_actual
```

## `alignment_summary.csv`

One row per major level/stage:

```text
level
recon_calls
geometry_steps
accepted_steps
loss_start
loss_end
residual_start
residual_end
time_recon_s
time_geometry_s
time_total_s
early_exit_reason
status
```

## `residual_metrics.csv`

At minimum:

```text
level
view_index
rmse
mae
robust_loss
valid_pixel_fraction
outlier_fraction
lowpass_rmse
bandpass_rmse
raw_rmse
```

Optional detector statistics:

```text
mean_residual_by_u
mean_residual_by_v
residual_anisotropy
```

## `backend_report.json`

Required fields:

```json
{
  "requested": "auto",
  "actual_projector": "pallas",
  "actual_backprojector": "pallas",
  "actual_geometry_reductions": "jax_reference",
  "canonical_detector_grid": true,
  "calibrated_detector_grid": false,
  "pallas_eligible": true,
  "fallbacks": [
    {
      "component": "geometry_reductions",
      "requested": "pallas",
      "actual": "jax_reference",
      "reason": "pallas_jtj_reductions_not_implemented"
    }
  ],
  "agreement_tests": [
    {
      "component": "residual",
      "max_abs_error": 1.2e-5,
      "mean_abs_error": 3.1e-7,
      "status": "passed"
    }
  ]
}
```

## `verification.json`

Top-level pass/fail report:

```json
{
  "status": "passed",
  "summary": {
    "projection_residual_improved": true,
    "final_reconstruction_valid": true,
    "gauge_constraints_satisfied": true,
    "backend_provenance_complete": true,
    "weak_dofs_handled": true
  },
  "metrics": {
    "residual_before": 123.4,
    "residual_after": 45.6,
    "relative_improvement": 0.63,
    "final_loss": 6438.16
  },
  "escalation": {
    "level_1_geometry_run": false,
    "reason": "level_2_verification_passed"
  }
}
```

## Failure classifications

Use structured failure labels:

```text
geometry_not_observable
pose_overfit
nuisance_unmodelled
backend_fallback_unexpected
reconstruction_underconverged
motion_model_insufficient
deformation_suspected
bad_input_metadata
nan_or_inf
no_improvement
```

Each failure should include:

```json
{
  "class": "nuisance_unmodelled",
  "severity": "warning",
  "evidence": [
    "residual has strong detector-column structure",
    "stripe_bias_model inactive"
  ],
  "recommended_action": "enable standard nuisance model or inspect flat-field correction"
}
```

## Verification gates

### Gate 1 — Finite outputs

```text
no NaNs/Infs in volume
no NaNs/Infs in geometry
no NaNs/Infs in pose
valid pixel fraction above threshold
```

### Gate 2 — Projection residual improvement

```text
robust residual after < robust residual before
view-wise residual improves for most views
bad views are flagged rather than silently dominating
```

### Gate 3 — Gauge stability

```text
mean(dx) near 0 after canonicalisation
mean(phi_residual) near 0 after canonicalisation
mean(dz) near 0 if relevant
canonicalisation does not change predicted projections beyond tolerance
```

### Gate 4 — Optimiser health

```text
accepted LM steps exist
damping did not explode indefinitely
Schur condition not catastrophic, or weak modes labelled
actual/predicted decrease ratio sane
```

### Gate 5 — Backend provenance

```text
requested/actual backend recorded
fallback reasons recorded
Pallas/JAX agreement tests passed where applicable
calibrated-grid fallback explicitly expected if used
```

### Gate 6 — Final reconstruction quality

For synthetic data:

```text
volume NMSE below threshold
geometry recovery after canonicalisation below threshold
realised forward geometry residual near true geometry residual
```

For real data:

```text
residual reduced
visual previews finite and plausible
edge/sharpness proxy not worse
no obvious residual structure left unclassified
```

## Benchmark comparison report

Every benchmark should produce a markdown report:

```text
benchmark_report.md
```

Template:

```markdown
# Benchmark: synth128_setup_global_tomo

## Summary

| Impl | Profile | Status | Time to verified | Total time | Volume NMSE | Final residual |
|---|---|---|---:|---:|---:|---:|

## Geometry recovery

| Parameter | Truth | Current | Reimagined | Error current | Error reimagined |
|---|---:|---:|---:|---:|---:|

## Pose recovery

| DOF | RMSE current | RMSE reimagined |
|---|---:|---:|

## Backend provenance

...

## Failure/warning labels

...
```

## CI tests

Minimum CI set:

```text
small 32^3 synthetic smoke test
64^3 pose-only recovery
128^3 benchmark nightly
gauge invariance test
pseudo-Huber outlier test
Schur solve algebra test
backend fallback test
artifact schema validation
```

## Schema validation

Use JSON schema or pydantic-style models for:

```text
run_manifest.json
geometry_final.json
backend_report.json
verification.json
observability_report.json
artifact_index.json
```

Runs should fail loudly if required artifacts are missing.
