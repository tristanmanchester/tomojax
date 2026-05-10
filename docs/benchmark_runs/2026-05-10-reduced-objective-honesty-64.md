# 2026-05-10 reduced-objective honesty diagnostic, 64^3

## Command

```bash
LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -path '*/lib' -type d | paste -sd: -) \
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python tools/run_detu_variable_projection_diagnostic.py \
  --run-dir runs/rich_phantom_v1_parity_20260509_detu_diagnostics/stopped_otsu_l2_multires_f2_64_128v \
  --out-dir runs/detu_variable_projection_20260510_64_honesty \
  --profile lightning \
  --candidate-radius 1 \
  --candidate-step 1 \
  --fista-iterations 2 \
  --reduced-init zero
```

## Result

The diagnostic now records production-style FISTA step size, returned-volume
loss components, projected-gradient stationarity, volume RMS, support mass,
initialisation, mask source, and loss normalisation. The 64^3 reduced families
used `fista_step_size = 50.0` instead of the old tiny diagnostic step size.

| Objective family | Argmin det_u px | Error from truth px | Inner solve | Interpretation |
|---|---:|---:|---|---|
| true_volume_fixed_objective | 7.250000 | 0.000000 | `not_applicable_fixed_volume` | `geometry_information_present` |
| wrong_geometry_recon_fixed_objective | 10.464933 | 3.214933 | `not_applicable_fixed_volume` | `geometry_information_flat_or_ambiguous` |
| final_stopped_volume_fixed_objective | 5.574625 | -1.675375 | `not_applicable_fixed_volume` | `geometry_information_moved_or_absorbed` |
| honest_reduced_objective | 10.464933 | 3.214933 | `recorded` | `geometry_information_flat_or_ambiguous` |
| reduced_nonnegative_only | 10.464933 | 3.214933 | `recorded` | `geometry_information_flat_or_ambiguous` |
| reduced_support_only | 6.250000 | -1.000000 | `recorded` | `geometry_information_moved_or_absorbed` |
| reduced_support_nonnegative | 6.250000 | -1.000000 | `recorded` | `geometry_information_moved_or_absorbed` |
| reduced_support_tv | 6.250000 | -1.000000 | `recorded` | `geometry_information_moved_or_absorbed` |
| reduced_support_tv_center | 6.250000 | -1.000000 | `recorded` | `geometry_information_moved_or_absorbed` |
| reduced_known_phantom_support | 7.250000 | 0.000000 | `recorded` | `geometry_information_present` |

The decision remains
`constraint_restores_geometry_information:reduced_known_phantom_support`.
The important change is evidentiary: the reduced curves now carry enough
metadata to distinguish inner-solve quality from geometry interpretation.

Artifacts:

- `runs/detu_variable_projection_20260510_64_honesty/objective_summary.json`
- `runs/detu_variable_projection_20260510_64_honesty/summary.md`
- Per-family `inner_solve_quality.json`
- Per-family `reconstruction_config.json`
