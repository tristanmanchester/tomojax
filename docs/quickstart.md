# TomoJAX Quickstart

The typical workflow is: inspect data, preprocess or ingest it, reconstruct,
and optionally run alignment.

## Install and check the CLI

```bash
uv sync --locked --extra cpu --dev
uv run tomojax --help
```

For CUDA hosts, use the CUDA extra instead:

```bash
uv sync --locked --extra cuda12 --dev
just accelerator-smoke-cuda
```

`just accelerator-smoke-cuda` verifies the optional accelerator projector path.

## Inspect, validate, and preprocess data

```bash
uv run tomojax inspect /path/to/scan.nxs
uv run tomojax validate /path/to/scan.nxs
uv run tomojax preprocess raw.nxs corrected.nxs
```

For TIFF projection stacks, ingest into NXtomo format first:

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

For help choosing between modes, see
[`alignment-guide.md`](alignment-guide.md).

## Synthetic test workflow

```bash
uv run tomojax simulate \
  --out synthetic_scan.nxs \
  --nx 64 --ny 64 --nz 64 \
  --nu 64 --nv 64 \
  --n-views 64 \
  --phantom random_shapes

uv run tomojax recon --data synthetic_scan.nxs --out synthetic_recon.nxs
```

For a Python example, see
[`examples/simulate_and_reconstruct.py`](../examples/simulate_and_reconstruct.py).

## Next steps

After you can inspect, validate, reconstruct, and align a small dataset, see
[`real-laminography.md`](real-laminography.md) for real scan data and
[`support-matrix.md`](support-matrix.md) for a list of supported workflows.
