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

Expert callers also use `tomojax.backends.api.is_oom_error(exc)` to identify
JAX allocation failures, including the narrow cuFFT scratch-plan diagnostic.
It does not perform a retry: callers own synchronization, partial-result
rollback, and their bounded retry or fallback policy.

## Invariants

- Public imports go through `tomojax.backends` or its `.api`, not private `_memory` or
  `_subprocesses`.
- Optional Pallas projector access is owned by `tomojax.core.pallas_resolver`;
  this package re-exports that resolver for backend-facing callers.
- Subprocess probes resolve executables and run with `shell=False`.
- Memory estimates are conservative and deterministic when
  `free_bytes_override` is provided.
- The no-argument memory probe follows `jax.default_device` or the default
  backend; CPU placement reports host memory without initializing a GPU.
- Default gather precision follows the same device. Capability and dtype-JIT
  probes are device-scoped, including CUDA-visible device ordering; explicit CPU
  placement retains full precision even when GPUs are available.

`device_free_memory_bytes(device=...)` queries an explicit producer device without
changing the caller context. Allocator limits take precedence over physical free
memory; unavailable accelerator budgets remain unknown, and known zero stays zero.
