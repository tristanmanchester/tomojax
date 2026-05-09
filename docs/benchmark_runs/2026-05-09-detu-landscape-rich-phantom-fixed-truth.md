# 2026-05-09 Det-U Landscape Rich Phantom Fixed-Truth Gate

Command:

```text
LD_LIBRARY_PATH=.venv/lib/python3.12/site-packages/nvidia/*/lib paths \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
env UV_CACHE_DIR=.uv-cache \
uv run python tools/run_rich_phantom_v1_parity_gate.py \
  --out-dir runs/detu_landscape_rich_phantom_20260509 \
  --views 128 \
  --profile lightning \
  --mode fixed_truth
```

Run directory:

- `runs/detu_landscape_rich_phantom_20260509/fixed_truth_otsu_l2_lightning_128v/`

Result:

| Source | Shape | Status | det_u RMSE px | Volume NMSE | Schur accepted | Runtime s |
|---|---:|---|---:|---:|---|---:|
| fixed_synthetic_truth | 128^3 / 128 views | failed | 5.842426 | 0.672174 | true | 235.99 |

Det-u fixed-volume curve argmins:

| Volume source | Argmin det_u px | Loss at argmin | Loss near true | Loss near final |
|---|---:|---:|---:|---:|
| true_volume | 14.1875 | 0.086632 | 0.086632 | 17.913084 |
| final_stopped_volume | 8.40625 | 31.964256 | 38.725636 | 31.964256 |
| true_geometry_reconstructed_volume | 13.03125 | 208.936401 | 208.936401 | 208.936707 |
| zero_initial_volume | -2.0 | 208.951904 | 208.951904 | 208.951904 |

Interpretation:

- The true-volume scalar landscape has the expected basin near the synthetic
  detector offset, so the projection convention and detector-u loss signal are
  present in the fixed-volume objective.
- The final stopped-volume landscape is biased toward the absorbed/final
  geometry, consistent with reconstruction gauge absorption.
- The true-geometry reconstruction curve is nearly flat at this short lightning
  reconstruction budget, so the next diagnostic should not tune Schur first. It
  should extend the landscape source set with preview-iteration and refreshed
  volumes, then compare Schur scalar normals against these fixed-volume curves.
