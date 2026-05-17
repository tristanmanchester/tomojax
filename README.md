# TomoJAX

TomoJAX is a compact tomography and laminography toolbox for projection IO, preprocessing, reconstruction, and alignment.

The supported command-line surface is the grouped `tomojax` command:

```bash
tomojax inspect scan.nxs
tomojax ingest ./projections --angles angles.csv --du 0.65 --dv 0.65 --out scan.nxs
tomojax preprocess raw.nxs corrected.nxs
tomojax recon corrected.nxs --out recon.nxs
tomojax align corrected.nxs --out aligned.nxs --mode cor
tomojax simulate --out synthetic.nxs --nx 64 --ny 64 --nz 64 --nu 64 --nv 64 --n-views 64
```

`tomojax preprocess` writes reconstruction-ready absorption/log-attenuation projections by default. Use `--transmission` only when you intentionally need normalized transmission output.

## Public package surface

The product-facing imports are:

- `tomojax.io` for projection payloads, NXtomo IO, validation, preprocessing, and quicklooks.
- `tomojax.geometry` for geometry metadata, axes, calibration state, and field-of-view helpers.
- `tomojax.forward` for differentiable forward projection and residual helpers.
- `tomojax.recon` for FBP, FISTA-TV, and SPDHG-TV reconstruction.
- `tomojax.align` for `AlignConfig`, `align`, and `align_multires`.
- `tomojax.datasets` for deterministic synthetic datasets.

Historical benchmark harnesses, development logs, v1-parity gates, diagnostic runners, article artifact builders, and one-off scripts have been removed from the publishable tree. The retained tests prove the supported public API, CLI routing, import boundaries, IO/preprocessing workflows, deterministic simulation contracts, and tiny numerical reconstruction workflows.

## Workflow docs

- [`docs/quickstart.md`](docs/quickstart.md)
- [`docs/synthetic-tomography.md`](docs/synthetic-tomography.md)
- [`docs/real-laminography.md`](docs/real-laminography.md)
- [`docs/support-matrix.md`](docs/support-matrix.md)
- [`docs/known-limitations.md`](docs/known-limitations.md)

## Development checks

```bash
just surface-check
just check
```

`just surface-check` is the bounded product feedback loop. It checks formatting/lint configuration, private-import guardrails, and the retained product tests.
