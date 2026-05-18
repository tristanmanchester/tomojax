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

```bash
uv run tomojax align --data corrected.nxs \
  --out aligned.nxs \
  --mode cor
```

Alignment is routed through the product command and its public profile/schedule
API.

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
