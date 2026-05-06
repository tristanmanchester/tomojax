# Installation

TomoJAX uses [uv](https://docs.astral.sh/uv/) to manage its Python
environment. You can run it with GPU acceleration (CUDA 12 on Linux) or
in CPU-only mode on any platform.

## Prerequisites

You need the following before installing TomoJAX:

- **Python 3.12** — pinned in the repository's `.python-version` file;
  uv picks this up automatically
- **uv** — install from
  [docs.astral.sh/uv](https://docs.astral.sh/uv/)
- **CUDA 12 runtime and drivers** (GPU only) — verify with
  `nvidia-smi`

## Install with GPU support

Run this from the repository root to install all dependencies including
JAX with CUDA 12:

```bash
uv sync --extra cuda12 --group dev
```

Verify that JAX detects your GPU:

```bash
uv run tomojax-test-gpu
```

Expected output:

```
JAX backend: gpu
Devices: [CudaDevice(id=0)]
```

## Install for CPU only

If you don't have a CUDA-capable GPU, install with the CPU extra
instead:

```bash
uv sync --extra cpu --group dev
```

Verify the CPU backend:

```bash
JAX_PLATFORM_NAME=cpu uv run tomojax-test-cpu
```

Expected output:

```
JAX backend: cpu
Devices: [CpuDevice(id=0)]
```

## Verification utilities

TomoJAX ships two small verification commands registered as console
scripts:

| Command | Purpose |
|---------|---------|
| `tomojax-test-gpu` | Print JAX backend and device list (expects GPU) |
| `tomojax-test-cpu` | Force CPU mode and print backend info |

These run no computation — they only confirm that JAX imported
correctly and detected the expected backend.

## Optional extras

**Benchmark harness** — the benchmark controller under `bench/` needs
additional dependencies:

```bash
uv sync --extra bench --group dev
```

**JAX compilation cache** — JAX compiles XLA kernels on first use,
which makes the initial run slow. Enable a persistent cache to speed up
subsequent runs:

```bash
export TOMOJAX_JAX_CACHE_DIR=~/.cache/tomojax/jax_cache
```

If you don't set this variable, the `align` CLI enables the default
cache at `~/.cache/tomojax/jax_cache` automatically.

**Sphere rasterization** — the `random_shapes` phantom uses an
ROI-bounded pure-Python sphere implementation by default. No extra
setup is needed.

## Troubleshooting installation

If JAX doesn't detect your GPU, or you see `CUDA_ERROR_INVALID_IMAGE`
errors after switching hardware, see the
[Troubleshooting](troubleshooting.md) guide.
