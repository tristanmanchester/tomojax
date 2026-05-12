# TomoJAX v2 Quickstart

This branch is the v2 staged tomography and laminography workflow. Use the
commands below as the clean public entrypoints for current evidence runs.

## GPU Setup

The JAX CUDA wheel on this laptop needs the bundled NVIDIA libraries on
`LD_LIBRARY_PATH`.

```bash
CUDA_LIBS=$(python3 - <<'PY'
from pathlib import Path
base = Path('.venv/lib/python3.12/site-packages/nvidia')
print(':'.join(str(p / 'lib') for p in base.iterdir() if (p / 'lib').is_dir()))
PY
)
export LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JAX_PLATFORM_NAME=cuda
export JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

Verify the selected backend:

```bash
uv run tomojax-test-gpu
```

## Real Laminography

```bash
uv run python scripts/real_laminography/run_real_lamino_staged.py \
  --input /path/to/scan.nxs \
  --out runs/real_lamino_staged_run \
  --profile staged-lamino \
  --overwrite
```

The workflow writes `real_lamino_report/real_lamino_summary.json`,
`real_lamino_report/real_lamino_summary.md`, residual and geometry traces, and
publication PNGs.

## Synthetic Tomography

Setup-global gate:

```bash
uv run tomojax-align-auto \
  --out-dir .artifacts/synthetic/setup_global_128 \
  --synthetic-case setup-global \
  --size 128 \
  --views 16
```

Pose-random gate:

```bash
uv run tomojax-align-auto \
  --out-dir .artifacts/synthetic/pose_random_128 \
  --synthetic-case pose-random \
  --size 128 \
  --views 16
```

Both commands write `benchmark_result.json`, `benchmark_report.md`, recovered
geometry, verification summaries, and reconstruction artifacts under the output
directory.

See the current production-readiness report for what passes, what fails, and
which run artifacts back those claims:
[`docs/benchmark_runs/2026-05-13-production-readiness.md`](benchmark_runs/2026-05-13-production-readiness.md).
