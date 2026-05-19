# TomoJAX

TomoJAX turns tomography and laminography projection data into reconstructed
volumes. It loads NeXus/HDF5 or TIFF stacks, applies dark/flat correction,
reconstructs with FBP or TV solvers, and can estimate detector-centre/COR
alignment from the data.

<img src="images/tomojax-phantom94-orthoslices.png" width="900" alt="Orthogonal slices through the PHANTOM94 synthetic tomography volume used for TomoJAX validation.">

Use it when you have projections plus approximate geometry and want a clean
command-line path from raw detector frames to a reconstruction you can inspect,
save, and reproduce.

## What comes out

TomoJAX writes `.nxs` datasets that keep the projections, reconstructed volume,
geometry metadata, preprocessing provenance, and alignment metadata together.
That makes a run easier to inspect later and easier to move between CLI tools,
Python scripts, and reports.

The same workflow works in three common starting points:

- raw NeXus/HDF5 scans with sample, flat, and dark frames;
- TIFF projection stacks with an angle sidecar;
- synthetic datasets for testing reconstruction and alignment settings.

## What you can do

TomoJAX focuses on a small set of workflows that are useful for real scan data
and easy to test on synthetic data.

| Task | Command |
| --- | --- |
| Inspect a scan | `tomojax inspect scan.nxs` |
| Validate the dataset contract | `tomojax validate scan.nxs` |
| Convert a TIFF stack into a TomoJAX dataset | `tomojax ingest ./projections --angles angles.csv --du 0.65 --dv 0.65 --out scan.nxs` |
| Apply dark/flat correction | `tomojax preprocess raw.nxs corrected.nxs` |
| Reconstruct a volume | `tomojax recon --data corrected.nxs --out recon.nxs` |
| Correct detector-centre/COR geometry | `tomojax align --data corrected.nxs --out aligned.nxs --mode cor` |
| Generate a synthetic test scan | `tomojax simulate --out synthetic.nxs --nx 64 --ny 64 --nz 64 --nu 64 --nv 64 --n-views 64` |

`tomojax preprocess` writes absorption/log-attenuation projections by default.
Use `--transmission` when you need normalized transmission output instead.

## Get started

From a checkout, install the CPU development environment and check that the CLI
is available.

```bash
uv sync --extra cpu --dev
uv run tomojax --help
```

On a Linux CUDA host, use the CUDA extra instead:

```bash
uv sync --extra cuda12 --dev
uv run python - <<'PY'
import jax
print(jax.default_backend())
PY
```

If you do not have scan data yet, generate a small synthetic dataset and
reconstruct it:

```bash
uv run tomojax simulate \
  --out synthetic_scan.nxs \
  --nx 64 --ny 64 --nz 64 \
  --nu 64 --nv 64 \
  --n-views 64 \
  --phantom random_shapes

uv run tomojax recon --data synthetic_scan.nxs --out synthetic_recon.nxs
```

For a real NX/HDF5 scan, start by inspecting and validating it:

```bash
uv run tomojax inspect /path/to/scan.nxs
uv run tomojax validate /path/to/scan.nxs
uv run tomojax preprocess /path/to/raw.nxs corrected.nxs
uv run tomojax recon --data corrected.nxs --out recon.nxs
```

For TIFF projection stacks, ingest the stack first:

```bash
uv run tomojax ingest ./projections \
  --angles angles.csv \
  --du 0.65 \
  --dv 0.65 \
  --out scan.nxs
```

Then preprocess or reconstruct the resulting `.nxs` dataset.

## Examples

These examples use a synthetic phantom so the expected volume is known. Real
scan reports use the same layout, but rely on visual inspection and projection
losses instead of ground-truth metrics.

| Synthetic misalignment set | Alignment before and after |
| --- | --- |
| <img src="images/tomojax-canonical-misalignment-grid.png" width="360" alt="Grid of canonical PHANTOM94 tomography misalignment scenarios."> | <img src="images/tomojax-alignment-before-after.png" width="420" alt="Before and after reconstruction slices for a detector centre and detector roll alignment scenario."> |

Older animated examples remain in `images/` so existing external links keep
working.

## Alignment

TomoJAX exposes detector-centre/COR alignment through the public CLI. Start with
`--mode cor`; broader pose and laminography workflows are still documented as
limited product claims.

```bash
uv run tomojax align \
  --data corrected.nxs \
  --out aligned.nxs \
  --mode cor
```

The aligned dataset stores the reconstruction and alignment metadata together,
so you can inspect it later with `tomojax inspect aligned.nxs`.

## Python API

Most users should import from these modules:

- `tomojax.io` for projection payloads, NXtomo IO, validation,
  preprocessing, and quicklooks.
- `tomojax.geometry` for geometry metadata, axes, calibration state, and
  field-of-view helpers.
- `tomojax.forward` for differentiable forward projection and residual helpers.
- `tomojax.recon` for FBP, FISTA-TV, and SPDHG-TV reconstruction.
- `tomojax.align` for `AlignConfig`, `align`, and `align_multires`.
- `tomojax.datasets` for deterministic synthetic datasets.

The tests cover the public API, CLI routing, import boundaries, IO and
preprocessing, deterministic simulation, and small numerical reconstruction
cases.

## Workflow docs

Start with the guide that matches your data and the support level you need.

- [`docs/quickstart.md`](docs/quickstart.md)
- [`docs/synthetic-tomography.md`](docs/synthetic-tomography.md)
- [`docs/real-laminography.md`](docs/real-laminography.md)
- [`docs/support-matrix.md`](docs/support-matrix.md)
- [`docs/known-limitations.md`](docs/known-limitations.md)

## Current scope

The stable product path covers dataset inspection, validation, TIFF ingest,
NX/HDF5 preprocessing, reconstruction, synthetic data generation, and
detector-centre/COR alignment. Broader truth-free laminography alignment,
object-frame drift recovery, nuisance fitting, and bad-view handling are still
research workflows. See [`docs/known-limitations.md`](docs/known-limitations.md)
before making stronger claims about a scan.

## Development checks

Run these checks before sending changes for review.

```bash
just surface-check
just check
```

`just surface-check` checks formatting and lint configuration, private-import
guardrails, and product tests.
