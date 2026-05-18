# tomojax.backends

`tomojax.backends` owns runtime backend policy, memory-budget heuristics, and
accelerator capability probes. Optional accelerator implementations stay behind
this boundary so reconstruction and alignment code do not import experimental
backend modules directly.

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

## Invariants

- Public imports go through `tomojax.backends`, not private `_memory` or
  `_subprocesses`.
- Optional Pallas projector access goes through `resolve_pallas_callable(...)`
  rather than direct cross-module imports from `tomojax.core.pallas_projector`.
- Subprocess probes resolve executables and run with `shell=False`.
- Memory estimates are conservative and deterministic when
  `free_bytes_override` is provided.
