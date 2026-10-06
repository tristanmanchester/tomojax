# Install TomoJAX

TomoJAX currently requires Python 3.12. The commands below install the checked-out
source, including unreleased changes on its current branch. They do not assume
that a package index release contains those changes.

## Install from a checkout

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git, then:

```bash
git clone https://github.com/tristanmanchester/tomojax.git
cd tomojax
uv sync --locked --extra cpu --no-dev
uv run --no-sync tomojax --help
```

Run subsequent `uv` commands from this directory. `--no-sync` uses the environment
you selected without changing its installed extras. You can also activate `.venv`
and run `tomojax` directly. Development tools are optional; see
[contributing](../CONTRIBUTING.md) if you want to change the code.

## Use CUDA on Linux

A working NVIDIA driver is required. Select the CUDA extra instead of the CPU
installation above, then check both JAX device discovery and the real kernels:

```bash
uv sync --locked --extra cuda12 --no-dev
uv run --no-sync python -c "import jax; print(jax.devices())"
TOMOJAX_REQUIRE_CUDA=1 uv run --no-sync python tools/smoke_accelerator.py
```

The smoke check must succeed on a CUDA device; a CPU-only result does not verify
the accelerator implementation. JAX 0.11.2 and an Ada RTX 4070 Laptop GPU are
tested. The Pallas kernels depend on JAX's deprecated Triton backend, so the
package constrains JAX below 0.12. Other GPU families are not yet covered by the
published validation. See [accelerator scope](support-matrix.md#numerical-and-accelerator-scope).

The `cuda12` extra also installs CuPy, which compiles a CUDA C gather for the
Joseph transpose at first use (cached on disk). It is used for volumes of 2²⁴
voxels (256³) and more, where it makes laminography backprojection about 1.6
times faster; smaller volumes keep the Pallas kernel because the CuPy start-up
costs about 0.3 s per process. `TOMOJAX_CUDA_KERNELS=1` forces it for every
size and `0` disables it.

The optional CuPy Fourier inverse also needs its own extra:

```bash
uv sync --locked --extra cuda12 --extra fourier-cuda12 --no-dev
```

This extra does not change the default reconstruction algorithm. The Fourier
inverse supports uniform parallel scans only; it is not a laminography solver.

## Install a wheel

To distribute this checkout as a wheel without an editable source install:

```bash
uv build
uv venv --python 3.12 /tmp/tomojax-user
uv pip install --python /tmp/tomojax-user/bin/python dist/tomojax-0.3.0-py3-none-any.whl
/tmp/tomojax-user/bin/tomojax --help
```

Use a new environment path and the wheel filename produced by your build.
The example paths use POSIX conventions. The wheel's default dependencies
provide the CPU path; request its `cuda12` extra for a Linux CUDA installation.
The repository's installed-wheel check runs a short simulate, inspect and
reconstruct workflow outside the source tree.

## Verify a useful result

Follow the [first reconstruction](../README.md#first-reconstruction) or run:

```bash
uv run --no-sync python tools/smoke_cli_workflow.py
uv run --no-sync python examples/simulate_and_reconstruct.py
```

If the CLI is missing, check that you are using the selected environment. If JAX
reports CPU on an intended CUDA installation, resolve driver/JAX device discovery
before diagnosing TomoJAX kernel performance. The small examples also run with
`JAX_PLATFORMS=cpu` to explicitly select the CPU.
