# 2026-05-10 scout-support det_u diagnostic, 64^3

## Command

```bash
LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -path '*/lib' -type d | paste -sd: -) \
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python tools/run_detu_variable_projection_diagnostic.py \
  --run-dir runs/rich_phantom_v1_parity_20260509_detu_diagnostics/stopped_otsu_l2_multires_f2_64_128v \
  --out-dir runs/detu_variable_projection_20260510_64_scout_support \
  --profile lightning \
  --candidate-radius 1 \
  --candidate-step 1 \
  --fista-iterations 2 \
  --reduced-init zero
```

## Result

The diagnostic includes the new truth-free scout families:

- `reduced_scout_soft_support`
- `reduced_scout_lowfreq_anchor`
- `reduced_scout_support_anchor`

Scout provenance records `uses_truth: false`, `geometry_source:
initial_metadata`, `mask_source: projection_valid_mask`, and support mass
fraction `0.12610477209091187`.

| Objective family | Argmin det_u px | Error from truth px | Interpretation |
|---|---:|---:|---|
| honest_reduced_objective | 10.464933 | 3.214933 | `geometry_information_flat_or_ambiguous` |
| reduced_known_phantom_support | 7.250000 | 0.000000 | `geometry_information_present` |
| reduced_scout_soft_support | 10.464933 | 3.214933 | `geometry_information_flat_or_ambiguous` |
| reduced_scout_lowfreq_anchor | 10.464933 | 3.214933 | `geometry_information_flat_or_ambiguous` |
| reduced_scout_support_anchor | 10.464933 | 3.214933 | `geometry_information_flat_or_ambiguous` |

The first scout-support/anchor weights did not restore the 64^3 reduced det_u
basin. That is useful evidence: the infrastructure is now honest and
truth-free, but either the scout gauge is too weak for this mode or the next
tangent-space gauge projection slice is needed.

## Artifacts

- `runs/detu_variable_projection_20260510_64_scout_support/objective_summary.json`
- `runs/detu_variable_projection_20260510_64_scout_support/summary.md`
- Per-family `inner_solve_quality.json`
- Per-family `reconstruction_config.json`
