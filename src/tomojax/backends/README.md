# tomojax.backends

`tomojax.backends` handles runtime backend selection, memory-budget heuristics,
and accelerator capability probes.

## Public API

- `ViewsPerBatchEstimate`
- `device_free_memory_bytes()`
- `estimate_views_per_batch(...)`
- `estimate_views_per_batch_info(...)`
- `default_gather_dtype()`
- `PallasModuleCapability`
- `resolve_pallas_module()`
- `resolve_pallas_callable(...)`
- `run_resolved_command(...)`
- `check_output_resolved_command(...)`

## Invariants

- Public imports go through `tomojax.backends`, not private `_memory` or
  `_subprocesses`.
- Optional Pallas projector access is owned by `tomojax.core.pallas_resolver`;
  this package re-exports that resolver for backend-facing callers.
- Subprocess probes resolve executables and run with `shell=False`.
- Memory estimates are conservative and deterministic when
  `free_bytes_override` is provided.
