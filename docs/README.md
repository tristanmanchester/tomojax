# TomoJAX documentation

TomoJAX reconstructs and aligns parallel-beam tomography, laminography, and lab cone-beam CT in JAX with CUDA kernels.

![Four reconstructed slices, one per geometry: a sandstone plug from Diamond I12 in parallel beam, a layer of glass spheres from Diamond K11 laminography, a copper oxide specimen from a Zeiss Xradia cone-beam scan, and a FIPS walnut from three cone-beam orbits.](images/geometry-montage.png)

Four real scans reconstructed at native detector resolution and cropped to the sample: Diamond I12 sandstone (parallel beam, `Stripes(9)` + `Stripes(51, varying=True)`, FBP), Diamond K11 scan 54286 (laminography, aligned with `tj.align(scan)`, FBP at 2.16 µm), Zeiss Xradia CuO_2 tomo-B (cone beam, FDK at 1.13 µm using recorded centre and shifts), and FIPS walnut 1 (three cone orbits, CGLS on the authors' 0.1 mm grid).

```bash
uv run python examples/gallery/geometry_montage.py DATA geometry-montage.png
```

## Speed against ASTRA and TIGRE

| Case | TomoJAX | Other tools |
| --- | --- | --- |
| Parallel beam, real: Diamond I12, 1801 views of 64 × 2560 pixels | 0.81 s default, 3.35 s FBP | ASTRA FBP 7.64 s, TomoPy gridrec 2.90 s (CPU) |
| Cone FDK, real: FIPS walnut, 1200 views, 501³ | 1.02 s | ASTRA 1.71 s |
| Cone CGLS, real: FIPS walnut, 10 iterations | 28.2 s, error 0.184 | ASTRA 26.8 s, error 0.197 |
| Laminography CGLS to fixed error, 256³ synthetic (smooth / sharp) | 3.29 / 0.435 s | ASTRA 16.5 / 0.452 s |
| TV reconstruction, 128³ synthetic, 3% noise | 0.32 s, error 0.075 | TIGRE 9.1 s, best error 0.101 |

Warm times on one 8 GB laptop GPU (measured 2026-11-21) on identical corrected data, geometry, and output grids. See the [headline comparison](measurements.md#headline-comparison) for protocols, commands, and where each library wins.

## Guides

| Task | Guide |
| --- | --- |
| Install CPU, CUDA 12, MCP, or Studio dependencies | [Installation](installation.md) |
| Manage persistent projects, queue work, and explore 3D results in the native GUI | [TomoJAX Studio](../studio/README.md) |
| Load, correct, and reconstruct NXtomo, Data Exchange, Nikon, or TIFF scans | [Quickstart](quickstart.md) |
| Run first-slice recipes for ESRF, TOMCAT, SYRMEP, cSAXS, and Diamond I13 | [Beamline exports](beamline-exports.md) |
| Simulate data or script the Python API | [Synthetic tomography](synthetic-tomography.md), [examples](../examples/README.md) |
| Supply measured geometry for parallel or laminography scans, including TIFF stacks | [Real parallel and laminography scans](real-laminography.md) |
| Import, calibrate, and reconstruct lab cone-beam CT, including multi-orbit scans | [Lab cone-beam CT](lab-ct.md) |
| Correct per-view motion, setup angles, or the centre of rotation | [Align a scan](align-a-scan.md) |
| Reconstruct volumes larger than GPU memory, or use several GPUs | [Large scans and several GPUs](large-scans.md) |
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
- [Corrections](../src/tomojax/corrections/README.md): flat/dark, phase retrieval, ring/stripe removal, and beam hardening.
- [IO](../src/tomojax/io/README.md): dataset formats, metadata readers, and TIFF angle files.
- [Alignment](../src/tomojax/alignment/README.md): joint alignment solver and convergence criteria.
- [Architecture map](architecture-map.md): subsystem ownership, shared vocabulary, and sources of truth.
- [Benchmarks](../bench/README.md): rerunning comparisons with ASTRA and TIGRE.
- [Contributing](../CONTRIBUTING.md): development setup, checks, and release process.
- [Changelog](../CHANGELOG.md): behavioural changes and migration notes.
