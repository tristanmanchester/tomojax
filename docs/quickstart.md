# TomoJAX Quickstart

TomoJAX is published around one public CLI, `tomojax`, and public Python facades
under `tomojax.*`. The normal workflow is to inspect data, optionally preprocess
or ingest it into the standard dataset contract, reconstruct, and run alignment
through the supported CLI/profile path.

## Install and check the CLI

```bash
uv sync --extra cpu --dev
uv run tomojax --help
```

For CUDA hosts, install the CUDA extra instead of the CPU extra and ensure JAX
can see the selected device:

```bash
uv sync --extra cuda12 --dev
uv run python - <<'PY'
import jax
print(jax.default_backend())
PY
```

## Inspect, validate, and preprocess data

```bash
uv run tomojax inspect /path/to/scan.nxs
uv run tomojax validate /path/to/scan.nxs
uv run tomojax preprocess raw.nxs corrected.nxs
```

For TIFF projection stacks, ingest into the standard dataset contract first:

```bash
uv run tomojax ingest ./projections \
  --angles angles.csv \
  --du 0.65 \
  --dv 0.65 \
  --out scan.nxs
```

## Reconstruct

```bash
uv run tomojax recon --data corrected.nxs --out recon.nxs
```

## Align and reconstruct

Use `pose` mode when the sample moves between projections. This is the default
alignment mode and optimizes per-projection 5-DOF pose corrections.

```bash
uv run tomojax align --data corrected.nxs \
  --out aligned.nxs \
  --mode pose
```

Use `cor` mode when the detector centre or centre of rotation is the dominant
problem:

```bash
uv run tomojax align --data corrected.nxs \
  --out aligned_cor.nxs \
  --mode cor
```

Alignment is routed through the product command and its public profile and
schedule API. For a decision guide, see
[`alignment-guide.md`](alignment-guide.md).

## Synthetic smoke workflow

```bash
uv run tomojax simulate \
  --out synthetic_scan.nxs \
  --nx 64 --ny 64 --nz 64 \
  --nu 64 --nv 64 \
  --n-views 64 \
  --phantom random_shapes

uv run tomojax recon --data synthetic_scan.nxs --out synthetic_recon.nxs
```

For a minimal public-Python example, see
[`examples/simulate_and_reconstruct.py`](../examples/simulate_and_reconstruct.py).

## Next steps

After you can inspect, validate, reconstruct, and align a small dataset, use
[`real-laminography.md`](real-laminography.md) for scan data and
[`support-matrix.md`](support-matrix.md) to check which workflows are supported
product entrypoints.
