# tomojax.backends

`tomojax.backends` handles device memory probes and resolved subprocess calls.

## Public API

- `device_free_memory_bytes(device=None)`
- `is_oom_error(exc)`
- `run_resolved_command(...)`
- `check_output_resolved_command(...)`

## Invariants

- Public imports go through `tomojax.backends`, not private `_memory` or
  `_subprocesses`.
- Subprocess probes resolve executables and run with `shell=False`.
