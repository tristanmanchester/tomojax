# tomojax.backends

`tomojax.backends` handles runtime backend selection, memory-budget heuristics,
and accelerator capability probes.

## Public API

- `ViewsPerBatchEstimate`
- `device_free_memory_bytes(device=None)`
- `is_oom_error(exc)`
- `estimate_views_per_batch(...)`
- `estimate_views_per_batch_info(...)`
- `default_gather_dtype()`
- `CudaModuleCapability`
- `resolve_cuda_module()`
- `resolve_cuda_callable(...)`
- `run_resolved_command(...)`
- `check_output_resolved_command(...)`

## Invariants

- Public imports go through `tomojax.backends`, not private `_memory` or
  `_subprocesses`.
- Optional CUDA projector access is owned by `tomojax.core.cuda_projector_resolver`;
  this package re-exports that resolver for backend-facing callers.
- Subprocess probes resolve executables and run with `shell=False`.
- Memory estimates are conservative and deterministic when
  `free_bytes_override` is provided.
