# Real Laminography Workflow

The package-facing real-data path starts with inspection and uses the public
alignment command for detector-centre/COR correction:

```bash
uv run tomojax inspect /path/to/scan.nxs
uv run tomojax align /path/to/scan.nxs \
  --mode cor \
  --out aligned.nxs
```

The retained staged laminography runner is an evidence-reproduction path for
the current k11 report. It is useful for development and publication evidence,
but it is not the normal user entrypoint.

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

The current k11 evidence runner and its staged artifacts are summarized in the
production-readiness report:
[`docs/benchmark_runs/2026-05-13-production-readiness.md`](benchmark_runs/2026-05-13-production-readiness.md).
Some retained historical artifacts predate the final script/report renames.
New runs use the names listed above.
