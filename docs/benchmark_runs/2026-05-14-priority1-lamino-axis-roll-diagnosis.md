# Priority 1 Laminography Axis/Roll Diagnosis (2026-05-14)

## Objective

Diagnose or fix the clean `128^3` `synth128_lamino_axis_roll_pose` failure without weakening criteria. The target gate is an oracle geometry diagnostic: `geometry_update_volume_source=fixed_synthetic_truth`, so reconstruction-gauge absorption is not the active failure mode.

## Verdict

The setup axis/roll/theta Jacobian is capable of recovering the true laminography setup when the per-view pose is fixed to truth. The failure is the joint setup/pose decomposition: the same projection residual can be fitted by multiple mixtures of global axis/roll/theta and per-view pose (`phi`, `dx/dz`, and especially `alpha/beta`). Current joint Schur LM can drive projection loss very low while assigning the correction to the wrong geometry block.

This is therefore an identifiability / staging / observability blocker, not a projector sign bug and not a stopped-volume issue.

## Evidence

All runs were executed on `vivobook-ts` under:

```text
/home/tristan/projects/tomojax-v2-goal/.artifacts/priority1_lamino_axis_roll/
```

The artifacts were also synced locally to:

```text
.artifacts/priority1_lamino_axis_roll/
```

### Baseline failing product-truth rerun

Artifact:

```text
.artifacts/product_truth_20260513/synth128_lamino_axis_roll_pose_detv_policy_rerun/run/
```

Key result:

```text
status: failed
det_u_realized_rmse_px: 0.10245630932626788
axis_error_rad: 0.19178020406063725
detector_roll_error_rad: 0.034477245845192656
theta_realized_rmse_rad: 0.07879071602198197
```

`det_u` and the detector-v policy pass, while axis and detector roll fail.

### Joint 32^3 reproduction

Artifact:

```text
.artifacts/priority1_lamino_axis_roll/joint_32_20260514a/
```

Key result:

```text
status: failed
det_u_realized_rmse_px: 0.07852913655022938
axis_error_rad: 0.5620753114352325
detector_roll_error_rad: 0.10731086585699914
theta_realized_rmse_rad: 0.21181699871946963
alpha_beta_rmse_rad: 0.2177416529023564
```

The coarse joint solve fits projection loss extremely well but assigns the correction to the wrong setup/pose mixture.

### Alpha/beta staging and freezing probes

Artifacts:

```text
.artifacts/priority1_lamino_axis_roll/alpha_beta_at_1_128_20260514a/
.artifacts/priority1_lamino_axis_roll/no_alpha_beta_128_20260514a/
```

Key results:

```text
alpha_beta_at_1_128_20260514a:
  status: failed
  axis_error_rad: 0.06596764402557101
  detector_roll_error_rad: 0.05707491483510138
  theta_realized_rmse_rad: 0.030875108058916884
  alpha_beta_rmse_rad: 0.0030833451982818227

no_alpha_beta_128_20260514a:
  status: failed
  axis_error_rad: 0.06561577532325091
  detector_roll_error_rad: 0.056983302496081606
  theta_realized_rmse_rad: 0.030839821167615675
  alpha_beta_rmse_rad: 0.003146437166314646
```

Delaying or removing `alpha/beta` prevents the catastrophic alpha/beta blow-up, but axis/roll still fail. This shows `alpha/beta` absorption is one symptom, not the full blocker.

### Pose-frozen with nominal/wrong pose

Artifact:

```text
.artifacts/priority1_lamino_axis_roll/pose_frozen_128_20260514a/
```

Key result:

```text
status: failed
det_u_realized_rmse_px: 2.143221590079274
axis_error_rad: 0.07358131078496231
detector_roll_error_rad: 0.07757416289270522
theta_realized_rmse_rad: 0.08989800936824181
dx_dz_rmse_px: 1.8027756414037213
```

Freezing pose while the dataset contains true pose motion makes setup recovery worse. The gate cannot be solved by simply making it setup-only unless pose is already known.

### True-pose setup-only isolation

Diagnostic dataset:

```text
.artifacts/priority1_lamino_axis_roll/datasets/synth128_lamino_axis_roll_pose_128_true_pose_start/
```

This is a copied generated dataset where only `v2_corrupted_pose_params.csv` was replaced with `v2_true_pose_params.csv`, so the solver starts from the true pose while setup remains corrupted.

Run artifact:

```text
.artifacts/priority1_lamino_axis_roll/true_pose_setup_only_128_20260514a/
```

Command shape:

```text
tomojax dev align-auto \
  --synthetic-dataset synth128_lamino_axis_roll_pose \
  --synthetic-dataset-dir .artifacts/priority1_lamino_axis_roll/datasets/synth128_lamino_axis_roll_pose_128_true_pose_start \
  --profile diagnostic-fast \
  --geometry-update-volume-source fixed_synthetic_truth \
  --geometry-update-pose-frozen \
  --geometry-update-active-setup-parameters det_u_px,det_v_px,detector_roll_rad,axis_rot_x_rad,axis_rot_y_rad,theta_offset_rad,theta_scale
```

Key result:

```text
status: passed
det_u_realized_rmse_px: 3.8144876872911455e-06
det_v_realized_rmse_px: 5.338959359154227e-05
axis_error_rad: 6.781190837978561e-05
detector_roll_error_rad: 5.506761664133414e-06
theta_realized_rmse_rad: 7.2271731676580605e-06
alpha_beta_rmse_rad: 0.0
phi_rmse_rad: 2.0742008880553428e-18
dx_dz_rmse_px: 3.458726603442063e-16
```

This cleanly exonerates the fixed-volume setup Jacobian/operator path: with pose fixed to truth, setup recovery is essentially exact.

## Interpretation

The failed production gate is asking for exact truth-label recovery across a geometry family with strong gauge/correlation structure. The optimizer is allowed to fit the projection residual with several near-equivalent parameter decompositions:

```text
global axis/roll/theta
per-view phi
per-view dx/dz
per-view alpha/beta
```

The current joint Schur path can find a low residual geometry without recovering the synthetic instrument labels. The old v1-style staged workflow likely avoided this by imposing an order: setup/COR first, then roll/axis, then pose polish. v2 needs a similarly explicit staged/observable laminography profile, or the benchmark needs to be split into setup-with-known-pose and pose-after-setup gates rather than treated as one all-DOF calibration pass.

## Recommendation

Do not weaken the axis/roll thresholds and do not call the current joint all-DOF run a pass. Mark `synth128_lamino_axis_roll_pose` as a diagnostic-only failed joint-identifiability gate until v2 has a staged laminography profile that proves:

1. setup recovery with pose known/fixed,
2. pose recovery with setup known/recovered,
3. staged setup-then-pose recovery without truth pose.

For MVP production claims, do not block on exact all-DOF truth-label recovery. Use staged real-data reconstruction quality and observability reporting instead.
