# Real Laminography Workflow

The supported real-data entrypoint is the staged laminography runner:

```bash
uv run python scripts/real_laminography/run_real_lamino_staged.py \
  --input /path/to/scan.nxs \
  --out runs/real_lamino_staged_run \
  --profile staged-lamino \
  --overwrite
```

Use `--profile diagnostic-fast` for bounded local checks. Use
`--profile reference-regression` only when comparing a run against the retained
reference-regression evidence.

## Stages

The clean staged workflow is:

1. baseline reconstruction,
2. detector-center setup update,
3. COR-only comparator reconstruction,
4. detector roll,
5. axis direction,
6. per-view pose phi,
7. per-view dx/dz,
8. five-DOF pose polish,
9. final reconstruction.

## Artifacts

New runs write:

- `real_lamino_report/real_lamino_summary.json`
- `real_lamino_report/real_lamino_summary.md`
- `real_lamino_report/real_lamino_residual_trace.csv`
- `real_lamino_report/real_lamino_geometry_trace.json`
- `real_lamino_report/publication/*.png`

The current k11 evidence is summarized in the production-readiness report:
[`docs/benchmark_runs/2026-05-13-production-readiness.md`](benchmark_runs/2026-05-13-production-readiness.md).
Some retained historical artifacts predate the final script/report renames.
New runs use the names listed above.
