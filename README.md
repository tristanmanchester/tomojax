# TomoJAX

TomoJAX reconstructs and aligns parallel-beam tomography, laminography, and lab cone-beam CT in JAX with CUDA kernels. It provides differentiable forward projectors with exact transposes, fast analytical and iterative reconstruction (Fourier, FBP/FDK, CGLS, SIRT/OS-SART, MLEM/OSEM, FISTA-TV, SPDHG-TV), joint volume-and-pose alignment, a CLI, an MCP server for AI agents, and a native desktop workspace for NXtomo, APS Data Exchange, Nikon `.xtekct`, Zeiss `.txrm`, ESRF, SkyScan, and TIFF data.

Full documentation is at [tomojax.dev](https://tomojax.dev).

![A simulated mechanical watch movement reconstructed by TomoJAX and drawn by direct volume rendering: brass wheels and balance in gold, steel parts white and the titanium plate and bridges as blue glass.](https://github.com/tristanmanchester/tomojax/raw/main/images/watch-render.png)

A 29 mm mechanical watch movement simulated with 600 views of a 120 keV parallel beam in gVXR (independent of TomoJAX's projector), reconstructed with FISTA-TV on 38.5 µm voxels in 50 s on a laptop GPU, and rendered directly in JAX. [Reproduce it](https://github.com/tristanmanchester/tomojax/blob/main/examples/README.md#volume-render-a-reconstruction).

![Truth, filtered backprojection and FISTA-TV slices through the hairspring of the watch movement, with close-ups of the balance on one grey scale.](https://github.com/tristanmanchester/tomojax/raw/main/images/reconstruction-example.png)

From 200 views at 10⁴ photons per pixel, filtered backprojection through the hairspring has relative error 0.357 against the exact voxel average of the meshes; FISTA-TV reaches 0.124 in 17 s. [Reproduce the figure](https://github.com/tristanmanchester/tomojax/blob/main/examples/README.md#reconstruct-a-few-view-noisy-scan).

![Nominal-geometry, aligned and true slices of the watch movement, close-ups of the balance, and recovered against true shifts for every view.](https://github.com/tristanmanchester/tomojax/raw/main/images/alignment-example.png)

`tomojax align` and `tj.align` solve the volume and requested geometry corrections jointly. By default they fit per-view translations; add `"tilt"` and `"roll"` for shared setup angles, or `"rotations"` for per-view wobble. See [align a scan](https://tomojax.dev/align-a-scan/).

## First reconstruction

On Linux, install [uv](https://docs.astral.sh/uv/getting-started/installation/) and install the CPU wheel into a fresh environment:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python "tomojax[cpu]"

.venv/bin/tomojax simulate -o synthetic.nxs --size 32 --n-views 60
.venv/bin/tomojax reconstruct synthetic.nxs -o recon.nxs
.venv/bin/tomojax inspect recon.nxs --preview-dir previews
```

To try unreleased changes from source, [build and install a wheel](https://tomojax.dev/installation/#build-and-install-an-unreleased-wheel) first, then run the same three commands.

For this axial parallel scan, `reconstruct` selects the Fourier-slice inverse automatically and writes the volume, geometry, and processing record to `recon.nxs` (pass `--include-scan` to keep the projections in the same file). `inspect` checks the file (`Valid: yes`) and writes display-scaled central slice PNGs to `previews/`.

Numerical commands that write data take `-o OUTPUT` and refuse to overwrite existing files without `--force` (`inspect` prints a report and optionally writes PNGs to `--preview-dir DIR`; `simulate` takes no input). Run `tomojax <command> --help` for flags, or see the [CLI overview](https://github.com/tristanmanchester/tomojax/blob/main/src/tomojax/cli/README.md). For CUDA 12 installation see [installation](https://tomojax.dev/installation/); for real data start with the [quickstart](https://tomojax.dev/quickstart/).

## Studio

[TomoJAX Studio](https://github.com/tristanmanchester/tomojax/blob/main/studio/README.md) is a native [Rune](https://github.com/anomalyco/rune) desktop workspace for persistent tomography projects. Keep acquisitions, reconstructions, saved recipes, and job records together; queue operations through TomoJAX's numerical engine; and inspect linked orthogonal slices, sinograms, side-by-side comparisons, and 3D GPU volume renderings while jobs run.

![TomoJAX Studio showing the aligned FISTA-TV watch movement in 3D volume view with linked axial, coronal, sagittal, projection, and sinogram thumbnails.](https://github.com/tristanmanchester/tomojax/raw/main/docs/images/studio-explore-volume.png)

Install the prebuilt Linux `x86_64` or `aarch64` wheel alongside TomoJAX (while `1.0.0` is in release candidate, include `--prerelease=allow`, or `--pre` with `pip`):

```bash
uv pip install --prerelease=allow "tomojax[cuda12]" tomojax-studio
tomojax studio
```

Use `cpu` on CPU-only machines, or build from a source checkout with Rust **1.99.1** (`uv run --no-sync tomojax studio --build`; see [install and run](https://github.com/tristanmanchester/tomojax/blob/main/studio/README.md#install-and-run)). Choose **Create a project** and add scans, or **Try a demo project** to simulate a scan and reconstruct it. Reopen a project later with `tomojax studio /path/to/project`.

See the [Studio guide](https://github.com/tristanmanchester/tomojax/blob/main/studio/README.md) for calibration, live previews, measurements, comparison modes, exports, and verification, and [studio/ARCHITECTURE.md](https://github.com/tristanmanchester/tomojax/blob/main/studio/ARCHITECTURE.md) for its persistence and rendering architecture.

## Choose a workflow

| Task | Guide |
| --- | --- |
| Manage persistent projects, queue work and explore 3D results in the native GUI | [TomoJAX Studio](https://github.com/tristanmanchester/tomojax/blob/main/studio/README.md) |
| Load, correct, and reconstruct NXtomo, Data Exchange, Nikon, or TIFF scans | [Quickstart](https://tomojax.dev/quickstart/) |
| Supply measured geometry for parallel or laminography scans, including TIFF stacks | [Real scan guide](https://tomojax.dev/real-laminography/) |
| Import, calibrate, and reconstruct lab cone-beam CT, including multi-orbit scans | [Lab cone-beam CT](https://tomojax.dev/lab-ct/) |
| Simulate data or script the Python API | [Synthetic tomography](https://tomojax.dev/synthetic-tomography/), [examples](https://github.com/tristanmanchester/tomojax/blob/main/examples/README.md) |
| Correct per-view motion, setup angles, or the centre of rotation | [Align a scan](https://tomojax.dev/align-a-scan/) |
| Reconstruct volumes larger than GPU memory, or use several GPUs | [Large scans and several GPUs](https://tomojax.dev/large-scans/) |
| Let an AI agent such as Claude Code drive TomoJAX through MCP | [Use TomoJAX from an AI agent](https://tomojax.dev/mcp/) |
| Check supported formats, models, and limitations | [Support matrix](https://tomojax.dev/support-matrix/), [known limitations](https://tomojax.dev/known-limitations/) |

## Python API

A `Scan` bundles projections with their geometry, and every operation returns a result carrying its geometry and processing record:

```python
import tomojax as tj

scan = tj.load("scan.nxs")
recon = tj.reconstruct(scan)                                # chooses method from geometry
recon = tj.reconstruct(scan, "cgls", iterations=50)
result = tj.align(scan, fit="translations")  # result.scan holds the updated poses
tj.save("recon.nxs", tj.reconstruct(result.scan))
```

CLI options use the same names: `tomojax reconstruct scan.nxs -o recon.nxs --method cgls --iterations 50` and `tomojax align scan.nxs -o aligned.nxs --fit translations`. Pass method-specific settings via config classes from `tomojax.recon` (such as `FISTAConfig(regularizer="huber_tv")`) or a TOML file passed to `--config`. See [align a scan](https://tomojax.dev/align-a-scan/#settings) for alignment settings.

Build a scan from arrays with `tj.Scan(projections, geometry)` and simulate projections with `tj.project(geometry, volume)`. Volumes use `(z, y, x)` and projections use `(view, v, u)`, with detector and voxel spacings in one consistent length unit. Lower-level interfaces live in `tomojax.recon`, `tomojax.alignment`, `tomojax.corrections`, `tomojax.geometry`, `tomojax.io`, `tomojax.forward`, and `tomojax.datasets`. Start with [examples/simulate_and_reconstruct.py](https://github.com/tristanmanchester/tomojax/blob/main/examples/simulate_and_reconstruct.py) or browse [tomojax.dev](https://tomojax.dev/).

## Benchmarks

| Case | TomoJAX | Other tools |
| --- | --- | --- |
| Parallel beam, real: Diamond I12, 1801 views of 64 × 2560 pixels | 0.81 s default, 3.35 s FBP | ASTRA FBP 7.64 s, TomoPy gridrec 2.90 s (CPU) |
| Cone FDK, real: FIPS walnut, 1200 views, 501³ | 1.02 s | ASTRA 1.71 s |
| Cone CGLS, real: FIPS walnut, 10 iterations | 28.2 s, error 0.184 | ASTRA 26.8 s, error 0.197 |
| Laminography CGLS to fixed error, 256³ synthetic (smooth / sharp) | 3.29 / 0.435 s | ASTRA 16.5 / 0.452 s |
| TV reconstruction, 128³ synthetic, 3% noise | 0.32 s, error 0.075 | TIGRE 9.1 s, best error 0.101 |

Warm times on one 8 GB laptop GPU (measured 2026-11-21), with every tool given the same corrected data, geometry, and output grid. On the walnut, TomoJAX reaches lower error at 10 iterations while ASTRA is slightly faster per iteration. See the [headline comparison](https://tomojax.dev/measurements/#headline-comparison) for full tables and protocols.

Volumes larger than GPU memory stream automatically in slabs: SIRT of a whole Diamond I12 scan (1801 views of 2160 × 2560 pixels into a 56.6 GB volume) ran on an 8 GB laptop GPU using 1.4 GB of VRAM at 1.13× the per-voxel time of an in-core scan ([large scans](https://tomojax.dev/large-scans/)).

## Development and licence

See [CONTRIBUTING.md](https://github.com/tristanmanchester/tomojax/blob/main/CONTRIBUTING.md) for development setup, conventions, and test commands, and [CHANGELOG.md](https://github.com/tristanmanchester/tomojax/blob/main/CHANGELOG.md) for release notes.

TomoJAX is licensed under [GPL-3.0-only](https://github.com/tristanmanchester/tomojax/blob/main/LICENSE).
