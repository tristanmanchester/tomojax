# TomoJAX documentation

TomoJAX reconstructs and aligns parallel-beam tomography, laminography, and lab cone-beam CT in JAX with CUDA kernels.

![Four reconstructed slices, one per geometry: a sandstone plug from Diamond I12 in parallel beam, a layer of glass spheres from Diamond K11 laminography, a copper oxide specimen from a Zeiss Xradia cone-beam scan, and a FIPS walnut from three cone-beam orbits.](images/geometry-montage.png)

Four real scans reconstructed at native detector resolution and cropped to the sample: Diamond I12 sandstone (parallel beam, ring-corrected, FBP), Diamond K11 scan 54286 (laminography, aligned with `tj.align(scan)`, FBP at 2.16 µm), Zeiss Xradia CuO_2 tomo-B (cone beam, FDK at 1.13 µm using recorded center and shifts), and FIPS walnut 1 (three cone orbits, CGLS on the authors' 0.1 mm grid).

```bash
uv run python examples/gallery/geometry_montage.py DATA geometry-montage.png
```

## Speed against other libraries

| Library | 1 V100S (s) | 4 V100S (s) | 1 Blackwell (s) | 8 Blackwell (s) |
| --- | ---: | ---: | ---: | ---: |
| TomoJAX | 74.51 Fourier | 41.28 Fourier | 27.87 Fourier | 28.99 FBP |
| tomocupy | 109.8 `fourierrec` | 66.35 `fourierrec` | 69.71 `fourierrec` | 45.77 `linerec` |
| ASTRA FBP | 768.6 | 199.6 | 182.0 | 46.21 |
| TIGRE FBP | 558.3 | 172.3 | 180.5 | 48.91 |
| LEAP FBP | 753.3 | 198.2 | 227.7 | 51.28 |
| TomoPy `gridrec` (CPU) | 430.4 | — | 143.9 | — |

Median seconds of three fresh worker launches reconstructing a full Diamond K11 (DIAD) scan (8,000 views of 2,160 × 2,560 pixels into a 56.62 GB volume) from process startup through final output `fsync` on Tesla V100S 32 GB and RTX PRO 6000 Blackwell 96 GB nodes (40 CPU cores; excludes the 96.0 s one-time input rechunking required by the five comparison libraries above). See [measurements](measurements.md#full-volume-speed-on-a-synchrotron-scan) for all nine libraries, accuracy, scaling, alignment, and where TomoJAX does not win.

## Guides

| Task | Guide |
| --- | --- |
| Install CPU, CUDA 12, MCP, or Studio dependencies | [Installation](installation.md) |
| Manage persistent projects, queue work, and explore 3D results in the native GUI | [TomoJAX Studio](../studio/README.md) |
| Load, correct, and reconstruct NXtomo, Data Exchange, Nikon, or TIFF scans | [Quickstart](quickstart.md) |
| Run first-slice recipes for ESRF, TOMCAT, SYRMEP, cSAXS, Diamond I13, and offset-detector half-acquisition | [Beamline exports](beamline-exports.md) |
| Simulate data or script the Python API | [Synthetic tomography](synthetic-tomography.md), [examples](../examples/README.md) |
| Supply measured geometry for parallel or laminography scans, including TIFF stacks | [Real parallel and laminography scans](real-laminography.md) |
| Import, calibrate, and reconstruct lab cone-beam CT, including multi-orbit scans | [Lab cone-beam CT](lab-ct.md) |
| Correct per-view motion, setup angles, or the center of rotation | [Align a scan](align-a-scan.md) |
| Reconstruct volumes larger than GPU memory, use several GPUs, or resume killed cluster jobs | [Large scans and several GPUs](large-scans.md) |
| Reconstruct on a cluster and view the result on a laptop | [Remote sessions](remote.md) |
| Export calibrated volumes for ImageJ, Fiji, ITK, VTK, or Amira | [Export a reconstruction](export.md) |
| Differentiate through projection in PyTorch | [PyTorch bridge](pytorch.md) |
| Let an AI agent such as Claude Code drive TomoJAX through MCP | [Use TomoJAX from an AI agent](mcp.md) |
| Split observations for held-out validation | [Held-out evaluation](held-out-evaluation.md) |
| Browse reconstructed examples | [Gallery](gallery.md) |
| Check supported formats, models, and limitations | [Support matrix](support-matrix.md), [known limitations](known-limitations.md) |
| Review speed, accuracy, and memory benchmarks | [Measurements](measurements.md) |

## Python reference and modules

Run `just docs` from a development checkout to build the Sphinx API reference at `.artifacts/docs/index.html` from the NumPy-style docstrings, and `just docstrings` to validate parameter coverage.

Import through public package roots or their `.api` facades; modules starting with `_` and `tomojax.core` are internal.

- [Command line](../src/tomojax/cli/README.md): commands, conventions, live progress, and `--config` files.
- [Worker protocol](protocol.md): JSON Lines session protocol (`tomojax serve`) and [progress metrics](progress-metrics.md).
- [Geometry](../src/tomojax/geometry/README.md): grids, detectors, coordinate frames, and poses.
- [Forward projection](../src/tomojax/forward/README.md): Joseph projection and derivatives.
- [Reconstruction](../src/tomojax/recon/README.md): solvers and configuration classes.
- [Corrections](../src/tomojax/corrections/README.md): flat/dark, phase retrieval, ring removal, and beam hardening.
- [IO](../src/tomojax/io/README.md): dataset formats, metadata readers, and TIFF angle files.
- [Alignment](../src/tomojax/alignment/README.md): axial center measurement, joint alignment solver, and convergence criteria.
- [Architecture map](architecture-map.md): subsystem ownership, shared vocabulary, and sources of truth.
- [Benchmarks](../bench/README.md): rerunning comparisons with ASTRA and TIGRE.
- [Contributing](../CONTRIBUTING.md): development setup, checks, and release process.
- [Changelog](../CHANGELOG.md): behavioral changes and migration notes.
