# 2026-05-10 tangent-gauge det_u diagnostic, 64^3

## Command

```bash
LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -path '*/lib' -type d | paste -sd: -) \
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python tools/run_detu_variable_projection_diagnostic.py \
  --run-dir runs/rich_phantom_v1_parity_20260509_detu_diagnostics/stopped_otsu_l2_multires_f2_64_128v \
  --out-dir runs/detu_variable_projection_20260510_64_tangent_gauge \
  --profile lightning \
  --candidate-radius 1 \
  --candidate-step 1 \
  --fista-iterations 2 \
  --reduced-init zero
```

## Result

The new diagnostic family `reduced_scout_support_tangent_gauge` computes a
truth-free detector-u volume gauge mode from the scout low-frequency anchor,
initial/current alignment geometry, and projection-valid mask. The recorded
mode provenance has `uses_truth: false`; the transfer ratio before the penalty
was `0.9085224261205981`.

| Objective family | Argmin det_u px | Error from truth px | Interpretation |
|---|---:|---:|---|
| honest_reduced_objective | 10.464933 | 3.214933 | `geometry_information_flat_or_ambiguous` |
| reduced_scout_support_anchor | 10.464933 | 3.214933 | `geometry_information_flat_or_ambiguous` |
| reduced_scout_support_tangent_gauge | 9.464933 | 2.214933 | `geometry_information_flat_or_ambiguous` |
| reduced_known_phantom_support | 7.250000 | 0.000000 | `geometry_information_present` |

The first tangent-gauge weight moved the scout family one candidate toward the
true detector-u but did not restore the basin. Stronger weighting, projection
rather than penalty, or recomputing/freezing policy still needs investigation.

## Artifacts

- `runs/detu_variable_projection_20260510_64_tangent_gauge/objective_summary.json`
- `runs/detu_variable_projection_20260510_64_tangent_gauge/reduced_scout_support_tangent_gauge/reconstruction_config.json`
- `runs/detu_variable_projection_20260510_64_tangent_gauge/reduced_scout_support_tangent_gauge/inner_solve_quality.json`
