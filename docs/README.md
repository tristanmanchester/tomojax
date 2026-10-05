# TomoJAX documentation

Start with a small reconstruction, then supply your acquisition's geometry and
preprocessing details. TomoJAX is an early research library for parallel-ray
models; consult the support and limitations pages before planning a workflow.

## Use the library

| You want to… | Guide |
| --- | --- |
| Install CPU or CUDA dependencies | [Installation](installation.md) |
| Produce a first volume and inspect slices | [Quickstart](quickstart.md) |
| Simulate data or call the Python API | [Synthetic tomography](synthetic-tomography.md), [examples](../examples/README.md) |
| Process TIFF, NeXus, or laminography scans with measured geometry | [Real scan guide](real-laminography.md) |
| Estimate motion or detector-centre corrections | [Alignment guide](alignment-guide.md) |
| Check geometry, units, algorithms, and GPU scope | [Support matrix](support-matrix.md), [known limitations](known-limitations.md) |
| Understand the evidence behind accuracy and speed claims | [Measurements](measurements.md) |

## Python reference and development

Import through public module roots or their `api` facades. Files beginning with
`_` and the `core` implementation are not stable application interfaces.

- [Geometry](../src/tomojax/geometry/README.md): grids, detectors, and poses.
- [Forward projection](../src/tomojax/forward/README.md): model and derivative choices.
- [Reconstruction](../src/tomojax/recon/README.md): algorithms and configuration.
- [IO](../src/tomojax/io/README.md): datasets and preprocessing.
- [Alignment solver](alignment-solver.md): Jacobians, coupling, and linear-solve diagnostics.
- [Contributing](../CONTRIBUTING.md): tests, packaging, and repository conventions.
- [Changelog](../CHANGELOG.md): behavioral changes and migration notes.

## Research records

[`research/`](research/) holds dated measurement and experiment records for
specific source snapshots, including rejected approaches and failed cases. They
are evidence, not user instructions or current defaults. The
[measurement guide](measurements.md) identifies the relevant complete
comparisons and the retained raw evidence. The
[optimization goal](research/optimization-goal.md) remains open.
