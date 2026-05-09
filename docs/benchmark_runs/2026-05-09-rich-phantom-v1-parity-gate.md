# Rich PHANTOM94 V1-Parity Det-U Gate

Command:

```text
LD_LIBRARY_PATH=<venv nvidia */lib paths> \
JAX_PLATFORMS=cuda \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python tools/run_rich_phantom_v1_parity_gate.py \
  --out-dir runs/rich_phantom_v1_parity_20260509 \
  --views 128 \
  --profile lightning \
  --mode stopped_multires
```

Artifacts are under `runs/rich_phantom_v1_parity_20260509/` and include
`summary.md`, `summary.csv`, `summary.json`, per-level `benchmark_result.json`,
`benchmark_report.md`, `geometry_trace.csv`, residual maps, and volume contact
sheet summaries.

| Level | det_u RMSE px | Volume NMSE | Schur accepted | Classification |
|---:|---:|---:|---|---|
| 32^3 | 1.607477 | 0.740780 | true | `training_loss_not_independent` |
| 64^3 | 1.687027 | 0.512972 | true | `reconstruction_absorbed_geometry` |
| 128^3 | 3.016660 | 0.504049 | true | `reconstruction_absorbed_geometry` |

Result: failed. The mask split and in-process volume-carry pyramid improved
final volume NMSE versus the previous stopped baseline (`0.504049` vs
`0.710293`), but det-u recovery did not pass `<1 px`, `<0.5 px`, or `<0.2 px`.
Theta was zero/frozen and no bad views were excluded, so the remaining blocker
is preview reconstruction/backprojection and volume gauge absorption rather than
theta contamination, Otsu masking, or coarse sidecar inconsistency.
