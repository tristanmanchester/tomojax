# TomoJAX documentation

Start with a small reconstruction, then supply your acquisition's geometry and
preprocessing details. TomoJAX is an early research library for parallel-beam,
laminography and lab cone-beam models; consult the support and limitations
pages before planning a workflow.

![Four reconstructed slices, one per geometry: a sandstone plug from Diamond I12 in parallel beam, a layer of glass spheres from Diamond K11 laminography, a copper oxide specimen from a Zeiss Xradia cone-beam scan, and a FIPS walnut from three cone-beam orbits.](images/geometry-montage.png)

One real scan for each geometry TomoJAX reconstructs, each slice at its
detector's resolution and cropped to the sample: the Diamond I12 scan of Vo et
al. (parallel beam; `Stripes(9)` and `Stripes(51, varying=True)` for its rings,
FBP), Diamond K11
scan 54286 (laminography; aligned by `tj.align(scan)`, FBP at 2.16 µm), Zeiss
Xradia CuO_2 tomo-B (cone beam; the centre and shifts its file records, FDK at
1.13 µm) and FIPS walnut 1 (three cone orbits; CGLS on the authors' 0.1 mm
grid).

```bash
uv run python examples/gallery/geometry_montage.py DATA geometry-montage.png
```

## Speed against ASTRA and TIGRE

| Case | TomoJAX | Other tools |
|---|---|---|
| Parallel beam, real: Diamond I12, 1801 views of 64 × 2560 pixels | 0.81 s default, 3.35 s FBP | ASTRA FBP 7.64 s, TomoPy gridrec 2.90 s (CPU) |
| Cone FDK, real: FIPS walnut, 1200 views, 501³ | 1.02 s | ASTRA 1.71 s |
| Cone CGLS, real: the same walnut, 10 iterations | 28.2 s, error 0.184 | ASTRA 26.8 s, error 0.197 |
| Laminography CGLS to a fixed error, 256³ synthetic, smooth / sharp object | 3.29 / 0.435 s | ASTRA 16.5 / 0.452 s |
| TV reconstruction, 128³ synthetic, 3% noise | 0.32 s, error 0.075 | TIGRE 9.1 s, best error 0.101 |

Warm times on one 8 GB laptop GPU, measured 2026-11-21, every tool given the same
corrected data, geometry and output grid. ASTRA is faster per CGLS iteration on
the walnut, though TomoJAX's error is lower at the same count. The
[headline comparison](measurements.md#headline-comparison) has the full table, the commands and where each
other tool wins.

## Use the library

| You want to… | Guide |
| --- | --- |
| Install CPU or CUDA dependencies | [Installation](installation.md) |
| Load an NXtomo, APS Data Exchange, Nikon or TIFF scan, correct its frames and reconstruct it | [Quickstart](quickstart.md) |
| Simulate data or call the Python API | [Synthetic tomography](synthetic-tomography.md), [examples](../examples/README.md) |
| Supply measured geometry for parallel or laminography scans, including TIFF stacks | [Real scan guide](real-laminography.md) |
| Import, calibrate and reconstruct a lab cone-beam CT scan | [Lab cone-beam CT](lab-ct.md) |
| Differentiate through projection in PyTorch | [PyTorch bridge](pytorch.md) |
| Correct per-view motion or the centre of rotation | [Alignment guide](align-a-scan.md) |
| Reconstruct scans larger than the GPU, or use several GPUs | [Large scans and several GPUs](large-scans.md) |
| Check geometry, units, algorithms, and GPU scope | [Support matrix](support-matrix.md), [known limitations](known-limitations.md) |
| See the measured speed, accuracy and memory behind the guides | [Measurements](measurements.md) |

## Python reference and development

Run `just docs` from a development checkout to build the Python API reference
at `.artifacts/docs/index.html`, directly from the NumPy-style public docstrings.
Run `just docstrings` to validate their structure and parameter coverage.

Import through public module roots or their `api` facades. Files beginning with
`_` and the `core` implementation are not stable application interfaces.

- [Command line](../src/tomojax/cli/README.md): the commands, their conventions
  and `--config` files.
- [Geometry](../src/tomojax/geometry/README.md): grids, detectors, and poses.
- [Forward projection](../src/tomojax/forward/README.md): model and derivative choices.
- [Reconstruction](../src/tomojax/recon/README.md): algorithms and configuration.
- [Corrections](../src/tomojax/corrections/README.md): the steps from detector
  counts to line integrals, and on line integrals.
- [IO](../src/tomojax/io/README.md): datasets, acquisition metadata and TIFF
  angle files.
- [Alignment](../src/tomojax/alignment/README.md): the joint solver and its numerical contracts.
- [Benchmarks](../bench/README.md): how to rerun the comparisons with ASTRA and
  TIGRE.
- [Contributing](../CONTRIBUTING.md): tests, packaging, and repository conventions.
- [Changelog](../CHANGELOG.md): behavioural changes and migration notes.
