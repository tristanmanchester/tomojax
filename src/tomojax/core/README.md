# tomojax.core

## Purpose

`tomojax.core` owns TomoJAX's low-level numerical foundation. Its public package
root intentionally exposes only shared runtime primitives, while the package also
contains the geometry contracts, projector kernels, validation helpers, backend
policy, and multiresolution utilities used by higher-level domains.

Higher-level packages should treat `tomojax.core` as the implementation owner for
the base math and execution contracts. Product-facing APIs are layered in
`tomojax.geometry`, `tomojax.forward`, `tomojax.recon`, `tomojax.align`, and
`tomojax.io`.

## Public API

The `tomojax.core` package root remains deliberately small:

- `setup_logging(level="INFO")`
- `log_jax_env()`
- `progress_iter(iterable, total=None, desc="")`
- `format_duration(seconds)`

Numerical foundations are imported by their owning modules rather than re-exported
from the root:

- `tomojax.core.geometry`: canonical geometry dataclasses, transforms, axis
  utilities, view construction, and shared grid-origin helpers. Public geometry
  packages build on these contracts.
- `tomojax.core.projector` and `tomojax.core.operators`: reference JAX projection
  and adjoint operators used by forward, reconstruction, simulation, and
  alignment workflows.
- `tomojax.core.pallas`: optional Pallas fast paths and their support checks.
  They are performance implementations, not a separate public domain API.
- `tomojax.core.validation`: shape, dtype, and geometry validation shared across
  numerical modules.
- `tomojax.core.backend_policy`: backend normalization and provenance helpers.
- `tomojax.core.multires`: multiresolution scheduling primitives consumed by
  alignment and reconstruction orchestration.

## Dependencies

`tomojax.core` may depend on standard library modules, NumPy, JAX, and optional
runtime helpers such as tqdm. It must not depend on alignment, reconstruction,
datasets, I/O, or CLI packages. Optional backend discovery belongs in
`tomojax.backends`; core projector code may call those resolvers, but higher
workflow policy stays outside core.

## Invariants

- Public imports go through `tomojax.core`, not private `_logging`.
- Progress reporting is opt-in via `TOMOJAX_PROGRESS`.
- Missing optional dependencies must not prevent execution.
- Geometry and projector contracts are source-level foundations for the rest of
  the package, but the core root should not become a broad facade for them.
- Pallas implementations must be optional fast paths with explicit fallback
  reasons; unexpected kernel errors should not be silently swallowed.
- Domain orchestration belongs outside core. Core modules should expose reusable
  math and validation primitives, not alignment, reconstruction, data-loading, or
  CLI workflows.

## Tests

Core contracts are covered through focused subsystem tests rather than a single
core-only suite:

- `tests/test_geometry_math.py` covers geometry math helpers.
- `tests/test_pallas_kernel_sources.py` covers optional Pallas kernel contracts
  and resolver import behavior.
- `tests/test_numerical_engines.py` and `tests/test_recon_workflows.py` cover
  projector behavior through forward and reconstruction workflows.
- `tests/test_product_surface.py` checks the public root import surface.
