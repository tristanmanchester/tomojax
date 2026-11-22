# Runnable examples

These scripts run from the repository root using public TomoJAX APIs and write no files unless an output path is given.

## Project and reconstruct a volume

```bash
uv run --no-sync python examples/simulate_and_reconstruct.py
```

[simulate_and_reconstruct.py](simulate_and_reconstruct.py) creates a 32³ Shepp–Logan phantom, simulates 60 parallel projections with the Joseph model, runs 24 CGLS iterations, and prints the shapes and full-volume relative L2 error. Prefix with `JAX_PLATFORMS=cpu` to run on CPU. For independent benchmarks against analytic references and real scans, see [measurements](../docs/measurements.md).

## Reproduce the README figures

All three README figures image a 29 mm skeletonised mechanical watch movement (mainspring barrel, going train, club-tooth escapement, screwed balance with hairspring) modelled in Blender as closed per-material meshes ([watch_blender.py](../bench/phantoms/watch_blender.py)). gVXR traces exact ray path lengths through the meshes at 120 keV (with `xraylib` attenuations), averaging 2 × 2 sub-pixel transmissions before Poisson noise and `-log`; the voxel truth is the exact mesh average ([mesh_voxels.py](../bench/phantoms/mesh_voxels.py)).

Running these requires a CUDA GPU, Blender 4.5, and the gVXR environment described in [bench/README.md](../bench/README.md#watch-movement-phantom) (located under `.artifacts` or passed via `--blender` and `--gvxr-python`):

```bash
uv sync --locked --extra cuda12 --group examples --no-dev
uv run --no-sync python examples/reconstruct_watch.py
uv run --no-sync python examples/align_watch.py
uv run --no-sync python examples/render_watch.py
```

Each script writes its figure to `images/` (or `--out`) alongside a JSON record of settings, errors, and runtimes, caching meshes and projections under `.artifacts/figures/watch` (first run builds the meshes in ~40 s). Timings below are on an RTX 4070 Laptop GPU.

### Reconstruct a few-view, noisy scan

[reconstruct_watch.py](reconstruct_watch.py) simulates 200 views over 180° at 10⁴ photons per pixel and reconstructs a 32-slice slab through the hairspring on 38.5 µm voxels with Hann FBP (relative L2 error 0.357) and 200 iterations of nonnegative FISTA-TV (`tv_weight=1e-5`, error 0.124 in 17 s). Total runtime is ~1 minute.

### Align a scan with per-view motion

[align_watch.py](align_watch.py) simulates 400 views on 77 µm voxels at 2 × 10⁴ photons per pixel with a 5.3 px centre offset plus per-view wobble and jitter (up to 0.25° rotations, 2.4 px across the axis, 1.5 px along it). `tj.align(scan, fit=("center", "translations", "rotations"))` runs in 36 s, recovering rotations to ≤ 0.06° RMS and shifts to 0.07 / 0.05 px RMS. FISTA-TV error drops from 0.905 (nominal) to 0.209 (aligned), matching the true-pose reconstruction (0.212). Total runtime is ~2.5 minutes.

Because a global rigid shift is unobservable, truth, poses, and centre are aligned to the same least-motion gauge before comparison (see [check the result](../docs/align-a-scan.md#check-the-result)).

### Volume-render a reconstruction

[render_watch.py](render_watch.py) simulates 600 views at 3 × 10⁴ photons per pixel, reconstructs a 768 × 768 × 164 grid (38.5 µm voxels) with 60 FISTA-TV iterations (`tv_weight=1e-5`, 50 s), and renders the reconstructed volume directly in JAX ([_volume_render.py](_volume_render.py)) using an attenuation transfer function (steel white, brass gold, ruby red, titanium blue glass). Total runtime is ~3 minutes.

## Prepare and align complex ptychography projections

```bash
uv sync --locked --extra cuda12 --extra fourier-cuda12 --no-dev
uv run --no-sync python examples/align_ptychography.py /path/to/porous_glass_dataset.mat
```

[align_ptychography.py](align_ptychography.py) reads either PSI MATLAB file: `porous_glass_dataset.mat` (v7.3, measured and aligned) or `example_data.mat` (v5, simulated with motion), preparing complex transmission in bounded device batches. For a fast CPU preview of the simulation, bin the unwrapped phase (not the complex pixels):

```bash
JAX_PLATFORMS=cpu uv run --no-sync python examples/align_ptychography.py \
  /path/to/example_data.mat --binning 4 -o phase-slices.npz
```

In Python:

```python
from examples.align_ptychography import run_example

volume, poses, report = run_example("example_data.mat", binning=4)
print(volume.shape, report["value_unit"])
```

The underlying public calls are:

```python
import tomojax as tj
from tomojax.corrections import remove_phase_ramp, unwrap_phase

phase = -remove_phase_ramp(unwrap_phase(objects), air_mask=air * abs(objects) ** 2)
scan = tj.Scan(phase, geometry)  # (view, v, u) in pixel units
aligned = tj.align(scan)
reconstruction = tj.reconstruct(aligned.scan, "fourier")
```

Adjust the air mask and border crop for other samples so `air_mask` covers only background air. For the measured file, recorded wavelength and pixel size convert reconstructed phase to refractive-index decrement (`wavelength / (2 * pi * pixel_size)`); the uncalibrated simulation reports `rad/pixel`. Pass `-o slices.npz` to save orthogonal slices, poses, and unit metadata.

To save the prepared phase as a `.nxs` dataset for the CLI (skipping any absorption logarithm):

```python
import numpy as np
import tomojax as tj
from examples.align_ptychography import make_scan, prepare_phase, read_psi_objects

objects, angles, _, _ = read_psi_objects("example_data.mat")
air = np.zeros(objects.shape[-2:], dtype=np.float32)
air[:, :50] = air[:, -50:] = 1
scan = make_scan(prepare_phase(objects, air), angles, binning=4)
aligned = tj.align(scan)
tj.save("aligned-phase.nxs", aligned.scan)
```

```bash
tomojax reconstruct aligned-phase.nxs -o phase-recon.nxs --method fourier
tomojax inspect phase-recon.nxs --preview-dir phase-previews
```

## Preview a real cone-beam orbit on a CPU

[import_walnut.py](import_walnut.py) prepares a single-orbit FIPS walnut scan for CPU FDK or CGLS without ASTRA; follow [try one FIPS walnut orbit on a CPU](../docs/lab-ct.md#try-one-fips-walnut-orbit-on-a-cpu).

## Bring a real multi-orbit scan into register

Download and unzip a walnut from the FIPS collection (Der Sarkissian et al., Scientific Data 6, 215, 2019; [Zenodo 2686726](https://zenodo.org/records/2686726)):

```bash
uv run --no-sync python examples/align_walnut_orbits.py ~/data/walnuts/Walnut1 -o walnut.nxs
```

[align_walnut_orbits.py](align_walnut_orbits.py) loads all three orbits with their uncorrected scanner geometry via `tj.Scan.from_astra`, bins 2 × 2, aligns every 4th view (900 views) with `tj.align`, prints relative orbit heights, and reconstructs 20 iterations of nonnegative least squares. On Walnut 1, `tj.align` recovers orbit heights of −0.385 and −0.753 mm (against the authors' −0.397 and −0.794 mm) in ~100 s on an 8 GB laptop GPU ([orbit registration](../docs/measurements.md#orbit-registration)).

## Reconstruct raw laminography from nominal geometry

[align_laminography.py](align_laminography.py) attaches nominal tilt (`--tilt`) and pixel pitch (`--pixel`) to raw laminography frames and fits translations, tilt, and roll jointly (override with `--fit`). Diamond/GDA layouts are detected automatically; pass `--data-path`, `--angles-path`, and `--image-key-path` for other HDF5 files. See [automatic NeXus laminography](../docs/real-laminography.md#automatic-nexus-laminography).

## Gallery figures

The scripts in [gallery/](gallery) generate the figures shown in [docs/gallery.md](../docs/gallery.md). Each takes an output PNG path and caches generated phantoms under `.artifacts/figures`.

## Train a denoiser through the PyTorch projector

With a CUDA build of PyTorch installed, run [train_denoiser.py](train_denoiser.py) on the Diamond I12 NXtomo row cut (Zenodo 3339629):

```bash
python examples/train_denoiser.py full_rows1048-1111.nxs --center-px -138
```

A small residual CNN trains across two disjoint angle-interleaved view sets (one for FBP input, one for image and projection targets, backpropagating through `tomojax.torch.Projector`'s exact transpose), with a third set held out for projection-error reporting. Defaults bin 8 × 8 over all views for 500 steps; avoid undersampling the splits so the CNN does not learn to blur angular detail. `--center-px` specifies the rotation centre in native pixels (omit to estimate it automatically). Pass `--output result.npz` to save outputs.
