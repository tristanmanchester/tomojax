# TomoJAX v2 Quickstart

TomoJAX v2 is a tomography and laminography reconstruction package with one
public CLI: `tomojax`. The normal path is to inspect data, preprocess it when
needed, reconstruct, and optionally run alignment through the package CLI. The
current support matrix is in [`support-matrix.md`](support-matrix.md).

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

Verify the selected backend from Python:

```bash
uv run python - <<'PY'
import jax
print(jax.default_backend())
PY
```

## Real Laminography

```bash
uv run tomojax inspect /path/to/scan.nxs
uv run tomojax align /path/to/scan.nxs \
  --out runs/real_lamino_aligned.nxs \
  --mode cor
```

For TIFF projection stacks, ingest into the standard dataset contract first:

```bash
uv run tomojax ingest ./projections \
  --angles angles.csv \
  --du 0.65 \
  --dv 0.65 \
  --out scan.nxs
```

## Synthetic Tomography

```bash
uv run tomojax simulate \
  --out synthetic_scan.nxs \
  --nx 128 --ny 128 --nz 128 \
  --nu 128 --nv 128 \
  --n-views 128 \
  --phantom random_shapes

uv run tomojax recon synthetic_scan.nxs --out synthetic_recon.nxs
```

The original synthetic alignment evidence commands are kept out of the normal
quickstart. See the current production-readiness report for what passes, what
fails, and which run artifacts back those claims:
[`docs/benchmark_runs/2026-05-13-production-readiness.md`](benchmark_runs/2026-05-13-production-readiness.md).
