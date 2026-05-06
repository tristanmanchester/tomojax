# TomoJAX documentation

TomoJAX is a fully differentiable, memory-efficient parallel-beam CT
projector built on JAX. It provides exact gradients for 5-DOF rigid-body
alignment, making it useful for CT reconstruction, per-view alignment,
and deep learning workloads that require data consistency. If you work
with synchrotron or lab-source micro-CT data and need joint
reconstruction and alignment, TomoJAX gives you a single, composable
toolkit for the entire pipeline.

## Typical workflow

A standard TomoJAX pipeline follows these steps:

```
simulate → misalign → preprocess → reconstruct → align
```

1. **Simulate** a synthetic phantom and forward-project it.
2. **Misalign** the projections with known perturbations (for testing).
3. **Preprocess** raw sample/flat/dark frames into corrected absorption
   data.
4. **Reconstruct** a volume with FBP, FISTA-TV, or SPDHG-TV.
5. **Align** per-view poses jointly with reconstruction using
   multi-resolution coarse-to-fine optimization.

## Documentation sections

### Getting started

- [Installation](installation.md) — prerequisites, GPU and CPU setup,
  verification
- [Quickstart](quickstart.md) — simulate, reconstruct, and align a
  small dataset in five steps

### Tutorials

Step-by-step walkthroughs of complete workflows:

- [End-to-end tutorial](tutorials/end-to-end.md) — full 256^3 pipeline
  from simulation through alignment
- [Laminography tutorial](tutorials/laminography.md) — tilted
  rotation-axis geometry with 360-degree scans
- [Single-sample tutorial](tutorials/single-sample.md) — minimal
  workflow with a centered cube or sphere

### Concepts

Background on the algorithms and data model:

- [Geometry](concepts/geometry.md) — Grid, Detector, 5-DOF
  parameterization, parallel-beam and laminography geometries
- [Alignment](concepts/alignment.md) — alternating optimization,
  multi-resolution pyramids, optimizers, gauge fixing
- [Reconstruction](concepts/reconstruction.md) — FBP, FISTA-TV,
  SPDHG-TV, and when to use each

### CLI reference

Individual pages for every command-line tool:

- [CLI overview](cli/index.md) — common flags, config files,
  environment variables
- [simulate](cli/simulate.md) | [misalign](cli/misalign.md) |
  [preprocess](cli/preprocess.md) | [recon](cli/recon.md) |
  [align](cli/align.md) | [inspect](cli/inspect.md) |
  [validate](cli/validate.md) | [convert](cli/convert.md) |
  [loss-bench](cli/loss-bench.md)

### Reference

Technical specifications and the Python API:

- [Python API reference](reference/api.md) — public classes and
  functions for scripting workflows
- [Data format (NXtomo)](reference/data-format.md) — HDF5 schema and
  conventions
- [Misalignment modes](reference/misalign-modes.md) — deterministic
  schedule shapes and composition
- [Config files](reference/config-files.md) — TOML configuration for
  `recon` and `align`
- [Loss functions](reference/loss-functions.md) — available alignment
  losses and scheduling

### Help

- [Troubleshooting](troubleshooting.md) — common issues with
  installation, memory, convergence, and data I/O
- [Changelog](../CHANGELOG.md) — version history and breaking changes
