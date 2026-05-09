# Differentiable Stopped det_u Diagnosis

Command:

```text
LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -path '*/lib' -type d | paste -sd: -) \
UV_CACHE_DIR=.uv-cache \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python tools/run_rich_phantom_v1_parity_gate.py \
  --out-dir runs/rich_phantom_v1_parity_20260509_detu_diagnostics \
  --views 128 \
  --profile lightning \
  --mode stopped_multires
```

Artifacts:

- `runs/rich_phantom_v1_parity_20260509_detu_diagnostics/summary.md`
- `runs/rich_phantom_v1_parity_20260509_detu_diagnostics/multires_carried_detu_loss_curves.csv`
- `runs/rich_phantom_v1_parity_20260509_detu_diagnostics/multires_carried_detu_summary.json`
- Per-level `mask_provenance.json`, `fista_gradient_checks.json`,
  `adjoint_checks.json`, `geometry_jvp_vjp_checks.json`,
  `detu_loss_curves.csv/png`, `schur_scalar_diagnostics.json/csv`,
  `reduced_objective_probe.csv/png`, `gauge_transfer_diagnostics.json`, and
  `benchmark_report.md`.

## Result

| Level | Shape | det_u RMSE px | Volume NMSE | Classification |
|---:|---:|---:|---:|---|
| f4 | 32^3 | 1.6074667 | 0.7407774 | `training_loss_not_independent` |
| f2 | 64^3 | 1.6753750 | 0.5128121 | `reconstruction_absorbed_geometry` |
| f1 | 128^3 | 2.9541664 | 0.5029598 | `reconstruction_absorbed_geometry` |

The fixed-truth path was already recorded as passing in
`docs/benchmark_runs/2026-05-09-rich-phantom-v1-parity-gate.md`; this stopped
run still fails after the new diagnostics are present.

## Curve Evidence

| Level | true-volume argmin | true-geometry recon argmin | stopped/carry argmin | gauge transfer |
|---:|---:|---:|---:|---|
| f4 | 3.7702917 | 3.7702917 | 2.6162333 | absorbed-like, transfer 0.8813 |
| f2 | 7.4641349 | 6.9084907 | 5.7972021 | absorbed-like, transfer 0.8570 |
| f1 | 14.6623125 | 13.2840469 | 11.9057813 | absorbed-like, transfer 0.8672 |

The f1 true-volume fixed landscape has its minimum at the expected full-scale
det_u basin, while the final stopped/carried volume landscape is shifted to
about 11.9 px and the gauge-transfer diagnostic says about 87% of the fixed
det_u tangent can be absorbed by a volume update.

Schur scalar diagnostics agree with the scalar landscape signs at f1/f2 on the
same fixed-volume objective, so this run does not support a Schur/JTJ/JTr
implementation failure as the primary blocker.

Reduced-objective probes did not recover the true basin at f1: best alignment
candidate was `schur_backtrack_1` at 11.763933 px and best valid-mask candidate
was `current_final` at 11.545834 px.

## Classification

Decisive classification: `biased_fixed_stopped_volume_objective` with
`reconstruction_absorbed_geometry`.

The evidence points to the stopped reconstruction/volume gauge absorbing det_u,
not to a missing COR finder, theta relaxation, pose freedom, nuisance fitting,
or a Schur scalar mismatch. The next functional work should change the
reconstruction/gauge handoff or inner reconstruction model before considering
local reduced-objective acceptance.
