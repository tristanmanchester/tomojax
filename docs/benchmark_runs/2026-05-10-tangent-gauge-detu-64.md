# 2026-05-10 tangent-gauge det_u diagnostic, 64^3

## Command

```bash
LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -path '*/lib' -type d | paste -sd: -) \
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python tools/run_detu_variable_projection_diagnostic.py \
  --run-dir runs/rich_phantom_v1_parity_20260509_detu_diagnostics/stopped_otsu_l2_multires_f2_64_128v \
  --out-dir runs/detu_variable_projection_20260510_64_tangent_projection \
  --profile lightning \
  --candidate-radius 1 \
  --candidate-step 1 \
  --fista-iterations 2 \
  --reduced-init zero
```

## Result

The diagnostic family `reduced_scout_support_tangent_gauge` computes a
truth-free detector-u volume gauge mode from the scout low-frequency anchor,
initial/current alignment geometry, and projection-valid mask. It now removes
the candidate volume component along that mode and records before/after transfer
ratio evidence. The recorded mode provenance has `uses_truth: false`; transfer
ratio before projection was `0.9085223081268876`, and sample after-projection
ratios are near zero (`6.154042893016667e-08` for the first candidate).

| Objective family | Argmin det_u px | Error from truth px | Interpretation |
|---|---:|---:|---|
| honest_reduced_objective | 10.464933 | 3.214933 | `geometry_information_flat_or_ambiguous` |
| reduced_scout_support_anchor | 10.464933 | 3.214933 | `geometry_information_flat_or_ambiguous` |
| reduced_scout_support_tangent_gauge | 8.250000 | 1.000000 | `geometry_information_flat_or_ambiguous` |
| reduced_known_phantom_support | 7.250000 | 0.000000 | `geometry_information_present` |

The projection removes the measured gauge component and moves the scout family
closer to truth, but still does not satisfy the `<= 0.5 px` 64^3 threshold.
The remaining work is to run the stopped production gate and decide whether the
projection should be applied during each FISTA step rather than as a diagnostic
post-refresh projection.

## Artifacts

- `runs/detu_variable_projection_20260510_64_tangent_projection/objective_summary.json`
- `runs/detu_variable_projection_20260510_64_tangent_projection/reduced_scout_support_tangent_gauge/reconstruction_config.json`
- `runs/detu_variable_projection_20260510_64_tangent_projection/reduced_scout_support_tangent_gauge/inner_solve_quality.json`
