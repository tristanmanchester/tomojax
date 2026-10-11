# TomoJAX documentation

Start with a small reconstruction, then supply your acquisition's geometry and
preprocessing details. TomoJAX is an early research library for parallel-beam,
laminography and lab cone-beam models; consult the support and limitations
pages before planning a workflow.

## Use the library

| You want to… | Guide |
| --- | --- |
| Install CPU or CUDA dependencies | [Installation](installation.md) |
| Load an NXtomo, APS Data Exchange, Nikon or TIFF scan, correct its frames and reconstruct it | [Quickstart](quickstart.md) |
| Simulate data or call the Python API | [Synthetic tomography](synthetic-tomography.md), [examples](../examples/README.md) |
| Supply measured geometry for parallel or laminography scans, including TIFF stacks | [Real scan guide](real-laminography.md) |
| Import, calibrate and reconstruct a lab cone-beam CT scan | [Lab cone-beam CT](lab-ct.md) |
| Correct per-view motion or the centre of rotation | [Alignment guide](alignment-guide.md) |
| Reconstruct scans larger than the GPU, or use several GPUs | [Large scans and several GPUs](large-scans.md) |
| Check geometry, units, algorithms, and GPU scope | [Support matrix](support-matrix.md), [known limitations](known-limitations.md) |
| See the measured speed, accuracy and memory behind the guides | [Measurements](measurements.md) |

## Python reference and development

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
- [Alignment](../src/tomojax/alignment/README.md): configurations, plans and
  checkpoints behind `tj.align`.
- [Alignment solver](alignment-solver.md): `AlignConfig` settings, Jacobians,
  coupling and linear-solve diagnostics.
- [Benchmarks](../bench/README.md): how to rerun the comparisons with ASTRA and
  TIGRE.
- [Contributing](../CONTRIBUTING.md): tests, packaging, and repository conventions.
- [Changelog](../CHANGELOG.md): behavioural changes and migration notes.
