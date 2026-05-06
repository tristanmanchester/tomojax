# 2026-05-06 Phase 8 Multi-Case 32^3 Synthetic Benchmark Pass

Generated five deterministic 32^3 sidecar datasets from planned synthetic128
manifest scenarios, then ran `tomojax-align-auto-smoke` against each existing
sidecar directory with `--profile smoke32`, `--fit-gain-offset-nuisance`, and
`--fit-background-nuisance`.

Artifacts were generated under
`.artifacts/phase8_multi_case_32_benchmark_pass/`, which is ignored because it
contains `.npy` arrays. The comparison report was rendered by
`tomojax-synthetic-benchmark-compare` from the five `benchmark_result.json`
files after the Schur block-prior wiring slice was committed.

| Benchmark | Status | Criteria | Geometry | Volume NMSE | Final Residual | Time To Verified (s) | Total Time (s) | Recovery Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| `synth128_setup_global_tomo` | failed | failed | failed | 0.693523 | 0 | n/a | 11.5794 | `det_u_realized_rmse_px=3.625`, `det_v_realized_rmse_px=0`, `theta_realized_rmse_rad=0.0218166` |
| `synth128_pose_random_extreme` | failed | partially_evaluated | failed | 0.662409 | 0.331717 | n/a | 13.3407 | `det_u_realized_rmse_px=2.7415`, `det_v_realized_rmse_px=2.5782`, `theta_realized_rmse_rad=0.2019` |
| `synth128_lamino_axis_roll_pose` | failed | failed | failed | 0.635030 | 0.00978141 | n/a | 13.3946 | `det_u_realized_rmse_px=2.2334`, `det_v_realized_rmse_px=0.7336`, `theta_realized_rmse_rad=0.1598` |
| `synth128_thermal_object_drift` | failed | partially_evaluated | failed | 0.608258 | 0.000758991 | n/a | 13.5649 | `det_u_realized_rmse_px=1.4893`, `det_v_realized_rmse_px=0.0512`, `theta_realized_rmse_rad=0.0052336`; failure label: `nuisance_residual_structure` |
| `synth128_combined_nuisance_jumps` | failed | failed | failed | 0.700399 | 0.00567048 | n/a | 13.4880 | `det_u_realized_rmse_px=3.8751`, `det_v_realized_rmse_px=0.9955`, `theta_realized_rmse_rad=0.0309604` |

Commands:

```bash
uv run python - <<'PY'
from pathlib import Path
from tomojax.datasets import generate_synthetic_dataset

root = Path(".artifacts/phase8_multi_case_32_benchmark_pass/datasets")
for name in (
    "synth128_setup_global_tomo",
    "synth128_pose_random_extreme",
    "synth128_lamino_axis_roll_pose",
    "synth128_thermal_object_drift",
    "synth128_combined_nuisance_jumps",
):
    generate_synthetic_dataset(name, root, size=32, clean=False, views=4)
PY
```

```bash
for name in \
  synth128_setup_global_tomo \
  synth128_pose_random_extreme \
  synth128_lamino_axis_roll_pose \
  synth128_thermal_object_drift \
  synth128_combined_nuisance_jumps
do
  JAX_PLATFORM_NAME=cpu uv run tomojax-align-auto-smoke \
    --out-dir ".artifacts/phase8_multi_case_32_benchmark_pass/runs/$name" \
    --profile smoke32 \
    --synthetic-dataset-dir ".artifacts/phase8_multi_case_32_benchmark_pass/datasets/${name}_32" \
    --fit-gain-offset-nuisance \
    --fit-background-nuisance
done
```

```bash
uv run tomojax-synthetic-benchmark-compare \
  .artifacts/phase8_multi_case_32_benchmark_pass/runs/synth128_setup_global_tomo/benchmark_result.json \
  .artifacts/phase8_multi_case_32_benchmark_pass/runs/synth128_pose_random_extreme/benchmark_result.json \
  .artifacts/phase8_multi_case_32_benchmark_pass/runs/synth128_lamino_axis_roll_pose/benchmark_result.json \
  .artifacts/phase8_multi_case_32_benchmark_pass/runs/synth128_thermal_object_drift/benchmark_result.json \
  .artifacts/phase8_multi_case_32_benchmark_pass/runs/synth128_combined_nuisance_jumps/benchmark_result.json \
  --out .artifacts/phase8_multi_case_32_benchmark_pass/benchmark_comparison.md
```

Notes:

- The 32^3 smoke path exercised real sidecar ingestion and benchmark result
  comparison, but did not satisfy the planned synthetic128 recovery criteria.
- All five failing cases report `Time To Verified` as `n/a`.
- JAX emitted CUDA plugin warnings about missing cuSPARSE before falling back to
  CPU. The alignment commands were run with `JAX_PLATFORM_NAME=cpu`.
