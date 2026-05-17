# tomojax.backends

## Purpose

`tomojax.backends` owns runtime backend policy, memory-budget heuristics, and
accelerator capability probes. Optional accelerator implementations stay behind
this boundary so reconstruction, alignment, and benchmark code do not import
experimental backend modules directly.

## Public API

- `ViewsPerBatchEstimate`
- `device_free_memory_bytes()`
- `estimate_views_per_batch(...)`
- `estimate_views_per_batch_info(...)`
- `default_gather_dtype()`
- `PallasModuleCapability`
- `resolve_pallas_module()`
- `resolve_pallas_callable(...)`
- `run_command(...)`
- `check_output_command(...)`

## Dependencies

This module may probe JAX and platform commands. It must not depend on
alignment, reconstruction, datasets, or CLI modules.

## Invariants

- Public imports go through `tomojax.backends`, not private `_memory` or
  `_subprocesses`.
- Optional Pallas projector access goes through `resolve_pallas_callable(...)`
  rather than direct cross-module imports from `tomojax.core.pallas_projector`.
- Subprocess probes resolve executables and run with `shell=False`.
- Memory estimates are conservative and deterministic when
  `free_bytes_override` is provided.

## Tests

Covered by `tests/test_memory.py`, `tests/test_cli_geometry_build.py`, and
memory-related benchmark support tests.
