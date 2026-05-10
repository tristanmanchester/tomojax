# 2026-05-10 Stopped Scout/Tangent Gauge Gate

Command:

```bash
LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -path '*/lib' -type d | paste -sd: -) \
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python tools/run_rich_phantom_v1_parity_gate.py \
  --out-dir runs/rich_phantom_v1_parity_20260510_tangent_gauge_stopped \
  --views 128 \
  --profile lightning \
  --mode stopped_multires \
  --preview-volume-support scout_soft \
  --preview-support-outside-weight 0.1 \
  --preview-low-frequency-anchor-weight 0.05 \
  --preview-det-u-gauge-mode-weight 0.2
```

Artifacts:

- `runs/rich_phantom_v1_parity_20260510_tangent_gauge_stopped/summary.json`
- `runs/rich_phantom_v1_parity_20260510_tangent_gauge_stopped/stopped_otsu_l2_multires_f4_32_128v/`
- `runs/rich_phantom_v1_parity_20260510_tangent_gauge_stopped/stopped_otsu_l2_multires_f2_64_128v/`
- `runs/rich_phantom_v1_parity_20260510_tangent_gauge_stopped/stopped_otsu_l2_multires_f1_128_128v/`

The run completed on CUDA without OOM. Sampled GPU memory during the final
`128^3` stage reached about `6074 MiB`.

| Level | Status | Classification | Initial det_u RMSE px | Final det_u RMSE px | Volume NMSE | Schur accepted | Runtime s |
|---|---|---|---:|---:|---:|---|---:|
| `32^3` | failed | `independent_projection_losses_consistent` | 3.625000 | 0.297959 | 0.769341 | true | 233.020 |
| `64^3` | failed | `reconstruction_absorbed_geometry` | 0.595917 | 0.904070 | 0.203639 | true | 387.077 |
| `128^3` | failed | `reconstruction_absorbed_geometry` | 1.808140 | 1.924456 | 0.218229 | true | 546.017 |

Scout provenance is truth-free at all levels:

| Level | `uses_truth` | Support mass fraction | Geometry source | Mask source |
|---|---|---:|---|---|
| `32^3` | false | 0.150660 | `initial_metadata` | `projection_valid_mask` |
| `64^3` | false | 0.128742 | `initial_metadata` | `projection_valid_mask` |
| `128^3` | false | 0.122443 | `initial_metadata` | `projection_valid_mask` |

Gauge-transfer diagnostics remain high in the final stopped volumes:

| Level | Final transfer ratio | Final reduced/fixed curvature ratio | Interpretation |
|---|---:|---:|---|
| `32^3` | 0.719554 | 0.280446 | `mixed` |
| `64^3` | 0.891959 | 0.108041 | `absorbed_like` |
| `128^3` | 0.894676 | 0.105324 | `absorbed_like` |

Comparison with the previous rich PHANTOM94 stopped baseline:

| Level | Baseline det_u RMSE px | Scout/tangent det_u RMSE px | Baseline volume NMSE | Scout/tangent volume NMSE |
|---|---:|---:|---:|---:|
| `32^3` | 1.607467 | 0.297959 | 0.740777 | 0.769341 |
| `64^3` | 1.675375 | 0.904070 | 0.512812 | 0.203639 |
| `128^3` | 2.954166 | 1.924456 | 0.502960 | 0.218229 |

Interpretation:

The scout support plus tangent-gauge preview improves the carried stopped
trajectory materially at `64^3` and `128^3` compared with the previous stopped
baseline, and it sharply improves volume NMSE. It still does not pass the
strict det_u recovery tolerance, and the final gauge-transfer ratios remain
`absorbed_like` at realistic levels. The support/gauge direction is therefore
useful but incomplete; the next functional work should strengthen the production
alignment-volume gauge rather than add report fields or use truth support.
