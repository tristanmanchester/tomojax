#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:-.artifacts/product_truth_20260513}
mkdir -p "$ROOT"

if [[ -d "$PWD/.venv/lib/python3.12/site-packages/nvidia" ]]; then
  NVLIB=$(find "$PWD/.venv/lib/python3.12/site-packages/nvidia" -type d -name lib | paste -sd: -)
  export LD_LIBRARY_PATH="$NVLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

export UV_CACHE_DIR=${UV_CACHE_DIR:-.uv-cache}
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda,cpu}
export XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE:-false}

run_case() {
  local name=$1
  shift
  local out="$ROOT/$name"
  mkdir -p "$out"
  echo "=== $name ===" | tee "$out/runner.log"
  date --iso-8601=seconds | tee -a "$out/runner.log"
  set +e
  "$@" 2>&1 | tee -a "$out/runner.log"
  local status=${PIPESTATUS[0]}
  set -e
  date --iso-8601=seconds | tee -a "$out/runner.log"
  echo "exit_status=$status" | tee -a "$out/runner.log"
  printf '{"case":"%s","exit_status":%s}\n' "$name" "$status" > "$out/runner_status.json"
}

run_case synth128_lamino_axis_roll_pose \
  uv run tomojax dev align-auto \
    --out-dir "$ROOT/synth128_lamino_axis_roll_pose/run" \
    --synthetic-dataset synth128_lamino_axis_roll_pose \
    --size 128 \
    --views 256 \
    --profile diagnostic-fast \
    --geometry-update-volume-source fixed_synthetic_truth \
    --geometry-update-active-setup-parameters det_u_px,det_v_px,detector_roll_rad,axis_rot_x_rad,axis_rot_y_rad,theta_offset_rad,theta_scale \
    --geometry-update-active-pose-dofs alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px

run_case synth128_thermal_object_drift \
  uv run tomojax dev align-auto \
    --out-dir "$ROOT/synth128_thermal_object_drift/run" \
    --synthetic-dataset synth128_thermal_object_drift \
    --size 128 \
    --views 256 \
    --profile diagnostic-fast \
    --geometry-update-volume-source fixed_synthetic_truth \
    --geometry-update-active-setup-parameters det_u_px,detector_roll_rad,axis_rot_x_rad,axis_rot_y_rad,theta_offset_rad,theta_scale \
    --geometry-update-active-pose-dofs alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px

run_case synth128_combined_nuisance_jumps \
  uv run tomojax dev align-auto \
    --out-dir "$ROOT/synth128_combined_nuisance_jumps/run" \
    --synthetic-dataset synth128_combined_nuisance_jumps \
    --size 128 \
    --views 320 \
    --profile diagnostic-fast \
    --geometry-update-volume-source fixed_synthetic_truth \
    --geometry-update-active-setup-parameters det_u_px,detector_roll_rad,axis_rot_x_rad,axis_rot_y_rad,theta_offset_rad,theta_scale \
    --geometry-update-active-pose-dofs alpha_rad,beta_rad,phi_residual_rad,dx_px,dz_px \
    --apply-synthetic-nuisance \
    --fit-gain-offset-nuisance \
    --fit-background-nuisance

run_case stopped_detu_scout_tangent \
  uv run python tools/run_rich_phantom_v1_parity_gate.py \
    --out-dir "$ROOT/stopped_detu_scout_tangent/run" \
    --views 128 \
    --profile reference \
    --mode stopped_multires \
    --preview-volume-support scout_soft \
    --preview-support-outside-weight 0.05 \
    --preview-low-frequency-anchor-weight 0.02 \
    --preview-det-u-gauge-mode-weight 0.05

echo "product_truth_root=$ROOT"
