# TomoJAX

TomoJAX is a compact tomography and laminography toolbox for projection IO, preprocessing, reconstruction, and alignment.

#### Alignment demonstration
<img src="images/montage_scroll.gif" width="1000">

Left to right: ground truth phantom, naive reconstructions with simulated
misalignment/noise, and aligned reconstructions.

The command-line surface is the grouped `tomojax` command:

```bash
tomojax inspect scan.nxs
tomojax ingest ./projections --angles angles.csv --du 0.65 --dv 0.65 --out scan.nxs
tomojax preprocess raw.nxs corrected.nxs
tomojax recon --data corrected.nxs --out recon.nxs
tomojax align --data corrected.nxs --out aligned.nxs --mode cor
tomojax simulate --out synthetic.nxs --nx 64 --ny 64 --nz 64 --nu 64 --nv 64 --n-views 64
```

`tomojax preprocess` writes reconstruction-ready absorption/log-attenuation projections by default. Use `--transmission` only when you intentionally need normalized transmission output.

## Public package surface

Use these package facades:

- `tomojax.io` for projection payloads, NXtomo IO, validation, preprocessing, and quicklooks.
- `tomojax.geometry` for geometry metadata, axes, calibration state, and field-of-view helpers.
- `tomojax.forward` for differentiable forward projection and residual helpers.
- `tomojax.recon` for FBP, FISTA-TV, and SPDHG-TV reconstruction.
- `tomojax.align` for `AlignConfig`, `align`, and `align_multires`.
- `tomojax.datasets` for deterministic synthetic datasets.

The tests cover the public API, CLI routing, import boundaries, IO/preprocessing workflows, deterministic simulation contracts, and tiny numerical reconstruction workflows.

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

`just surface-check` is the bounded product feedback loop. It checks formatting/lint configuration, private-import guardrails, and product tests.

## Visual Examples

### Basic Projector Workflow

| Phantom | Projections | Sinogram | Reconstruction |
|---------|-------------|----------|----------------|
| <img src="images/phantom_slice.png" width="200"><br><img src="images/phantom_volume.png" width="200"> | <img src="images/projections.gif" width="200"> | <img src="images/sinogram.png" width="200"> | <img src="images/recon_slice.png" width="200"><br><img src="images/recon_volume.png" width="200"> |
| Top: slice<br>Bottom: volume projection | Animated over the scan | Angle vs detector | Top: slice<br>Bottom: volume projection |

### Alignment And Reconstruction Workflow

#### Misaligned Input Data

| Clean Misaligned Projections | Noisy Misaligned Projections |
|------------------------------|------------------------------|
| <img src="images/spin_projections_misaligned.gif" width="300"> | <img src="images/spin_projections_noisy.gif" width="300"> |
| Random rigid-body misalignments | Same misalignments plus noise |

#### Sinogram Analysis

| Clean Misaligned Sinogram | Noisy Misaligned Sinogram |
|---------------------------|---------------------------|
| <img src="images/misaligned_sinogram.png" width="300"> | <img src="images/noisy_sinogram.png" width="300"> |
| View-to-view inconsistencies | Inconsistencies plus noise artifacts |

#### Multi-Resolution Alignment Process

| Clean Data Alignment | Noisy Data Alignment |
|---------------------|---------------------|
| <img src="images/alignment_process_misaligned.gif" width="300"> | <img src="images/alignment_process_noisy.gif" width="300"> |
| Coarse-to-fine resolution refinement | Robust alignment despite noise |
