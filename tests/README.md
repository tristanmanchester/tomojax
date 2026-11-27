# Tests

Run the CPU suite with `just test`, or the complete local gate with `just ci`.
Use `just test-cuda` on a CUDA host for real-kernel coverage. It runs the CUDA
smoke check, the exclusive `fresh_gpu` tests, and then the rest of the suite on
shared-GPU workers capped at the smaller of the available CPU count and one
worker per 1,300 MiB of the first GPU's memory; failed parallel tests retry in
one process, and a per-user lock serializes concurrent runs. Environment setup
and the full command list are in [CONTRIBUTING.md](../CONTRIBUTING.md).

## Select a focused check

- `surface` marks explicit public API, CLI, IO/preprocessing, simulation, angle
  sidecar, and alignment-fit contract modules. It is not applied to the whole suite.
  `just surface-check` runs this set excluding numerical and GPU-marked cases.
- `numerical` identifies reconstruction/alignment checks that may compile JAX
  kernels. These remain part of the full CPU gate.
- `gpu` requires real accelerator execution; ordinary CPU runs cannot validate it.
- `cuda` and `slow` further describe applicable tests; use the full suite before
  inferring that a focused selection covers an implementation change.

## What the suite checks

Public tests cover module imports and dependency boundaries, CLI routing and
output contracts, NeXus/HDF5/NPZ/TIFF roundtrips, flat/dark correction, acquisition
ordering, physical metadata, and deterministic simulation.

Numerical tests check independent analytic or dense references, forward/adjoint
consistency, physical geometry, interpolation boundaries, gradients, iterative
convergence, active alignment parameters, bordered solves, and geometry replay. GPU tests compare
real CUDA implementations with their references. `test_properties.py` checks the exact transpose,
file round trips and the automatic method choice on geometries Hypothesis draws, and
`test_surface_parity.py` runs one scenario table through `tj.*`, the CLI, `tomojax serve` and MCP. Benchmarks and their retained
failures are described separately in [bench/README.md](../bench/README.md).

The source and installed-wheel smoke checks exercise the same useful CLI path:
simulate, inspect, validate, reconstruct, validate, and export slices. The wheel
check runs in a fresh environment outside the checkout.
