# 2026-05-09 variable-projection det_u diagnostic, 64^3

## Command

```bash
LD_LIBRARY_PATH=$(find .venv/lib/python3.12/site-packages/nvidia -path '*/lib' -type d | paste -sd: -) \
env UV_CACHE_DIR=.uv-cache JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python tools/run_detu_variable_projection_diagnostic.py \
  --run-dir runs/rich_phantom_v1_parity_20260509_detu_diagnostics/stopped_otsu_l2_multires_f2_64_128v \
  --out-dir runs/detu_variable_projection_20260509_64 \
  --profile lightning \
  --candidate-radius 1 \
  --candidate-step 1 \
  --fista-iterations 2
```

## Result

The diagnostic decision was
`constraint_restores_geometry_information:reduced_known_phantom_support`.

| Objective family | Argmin det_u px | Error from truth px | Interpretation |
|---|---:|---:|---|
| true_volume_fixed_objective | 7.250000 | 0.000000 | `geometry_information_present` |
| wrong_geometry_recon_fixed_objective | 9.464933 | 2.214933 | `geometry_information_flat_or_ambiguous` |
| final_stopped_volume_fixed_objective | 5.574625 | -1.675375 | `geometry_information_moved_or_absorbed` |
| honest_reduced_objective | 6.250000 | -1.000000 | `geometry_information_moved_or_absorbed` |
| reduced_nonnegative_only | 6.250000 | -1.000000 | `geometry_information_moved_or_absorbed` |
| reduced_support_only | 6.250000 | -1.000000 | `geometry_information_moved_or_absorbed` |
| reduced_support_nonnegative | 6.250000 | -1.000000 | `geometry_information_moved_or_absorbed` |
| reduced_support_tv | 6.250000 | -1.000000 | `geometry_information_moved_or_absorbed` |
| reduced_support_tv_center | 6.250000 | -1.000000 | `geometry_information_moved_or_absorbed` |
| reduced_known_phantom_support | 7.250000 | 0.000000 | `geometry_information_present` |

## Interpretation

This confirms that the fixed true-volume objective still contains the correct
det_u minimum, so the detector convention and scalar projection loss are not the
primary failure. The final stopped volume moves the fixed-volume minimum back to
the absorbed geometry. Independently reconstructed neutral-initializer reduced
objectives also prefer a wrong det_u unless the true phantom support is imposed.

The current blocker is therefore not simply too few continuation iterations.
The stopped reconstruction can absorb detector shift unless the x-step is
constrained by sufficiently informative object support or an equivalent gauge
constraint. The next functional slice should address reconstruction/volume
gauge constraints in the production x-step before spending more time on report
or benchmark wording.

## Artifacts

- `runs/detu_variable_projection_20260509_64/objective_summary.json`
- `runs/detu_variable_projection_20260509_64/summary.md`
- Per-family `detu_loss_curves.csv`, `detu_loss_curves.png`,
  `objective_summary.json`, `mask_provenance.json`, and
  `reconstruction_config.json`.
