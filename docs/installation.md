# Install TomoJAX

TomoJAX supports Linux, with CPU or CUDA 12 dependencies, and requires Python
3.12 or newer. Fresh CPU wheel and sdist installations are checked on CPython
3.12, 3.13 and 3.14; fresh CUDA 12 wheel installations are checked on CPython
3.12 and 3.14. These checks use no package cache and run the README reconstruction
outside the checkout. Python 3.15 cannot yet be validated: h5py and CuPy do not
publish Python 3.15 wheels, and building h5py needs a system HDF5 installation.
Both CPU and CUDA installations include imagecodecs for compressed TIFF inputs,
including LZW-compressed beamline images.

## Install a published release

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create
an environment in a new working directory. No Git checkout is needed:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python "tomojax[cpu]"
.venv/bin/tomojax --help
```

These commands select a published package-index release, not unreleased source.
To test changes that have not been published, use the local wheel instructions
below instead. You can also use `python -m venv .venv` followed by
`.venv/bin/python -m pip install "tomojax[cpu]"` with an existing Python.

## Use CUDA 12

A working NVIDIA driver is required. In a fresh environment, select the CUDA
extra instead of `cpu`, and explicitly require the CUDA backend:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python "tomojax[cuda12]"
JAX_PLATFORMS=cuda .venv/bin/python -c "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu'"
```

`JAX_PLATFORMS=cuda` fails if CUDA cannot initialise; a CPU fallback is not a
successful CUDA installation. Keep it set when verifying your first
reconstruction. The CUDA 12 extra includes JAX's CUDA dependencies and CuPy.
It also supplies NVIDIA's CUDA runtime and NVRTC wheels for kernel compilation;
these wheels do not replace the system NVIDIA driver.
TomoJAX's CUDA C kernels compile at first use and are cached on disk. Published
accelerator validation is limited to an Ada RTX 4070 Laptop GPU; see the
[support matrix](support-matrix.md#numerical-and-accelerator-scope).

The optional `fourier-cuda12` extra installs CuPy alone for a CPU installation
that needs the Fourier inverse's CUDA path; it is redundant with `cuda12`.
The Fourier inverse supports uniform parallel scans, not laminography.

## Build and install an unreleased wheel

From the source checkout, build both the sdist and its wheel:

```bash
uv build
```

Use the wheel filename produced by the build. The commands below install that
wheel and its CPU dependencies into a new environment outside the checkout:

```bash
mkdir /tmp/tomojax-user
cd /tmp/tomojax-user
uv venv --python 3.12
uv pip install --python .venv/bin/python "/absolute/path/to/tomojax/dist/tomojax-0.3.0-py3-none-any.whl[cpu]"
.venv/bin/tomojax --help
```

Replace the absolute path with your build's path. For CUDA, replace `[cpu]`
with `[cuda12]`. Installing the generated `.tar.gz` instead of the wheel builds
a wheel in an isolated build environment. Both distributions include the GPL
licence; the wheel includes the CUDA source and typing marker. Development
setup remains separate: see [contributing](../CONTRIBUTING.md).

## Verify a useful result

Run these commands from the directory containing the installed environment:

```bash
.venv/bin/tomojax simulate -o synthetic.nxs --size 32 --views 60
.venv/bin/tomojax recon synthetic.nxs -o recon.nxs
.venv/bin/tomojax inspect recon.nxs --preview previews
```

This is the README's first reconstruction: automatic Fourier inversion of a
small parallel scan, ending with `Valid: yes` and three central slice PNGs.
No external scan data is needed. Existing outputs are refused; use a new
working directory or `--force` to replace them. Set `JAX_PLATFORMS=cpu` to
explicitly select the CPU, or `JAX_PLATFORMS=cuda` to require CUDA for this run.

If the CLI is missing, check the environment path. If CUDA discovery fails,
resolve the NVIDIA driver/JAX installation before diagnosing kernel performance.

## Maintainer package checks

After building, the installed-package smoke tool creates a disposable environment
outside the checkout, installs without a package cache, checks dependencies and
module provenance, requires the selected device, and runs the commands above.
It reads an LZW-compressed TIFF, checks the saved reconstruction against the
simulated volume, and requires finite values and all previews. It does not import
TomoJAX from the source checkout.

```bash
uv run --no-sync twine check --strict dist/*
uv run --no-sync python tools/smoke_installed_wheel.py --python 3.12 --extra cpu
uv run --no-sync python tools/smoke_installed_wheel.py --python 3.12 --extra cuda12
uv run --no-sync python tools/smoke_installed_wheel.py --python 3.12 --artifact sdist
```

Repeat the CPU wheel/sdist checks with `--python 3.13` and `--python 3.14`, and
the CUDA wheel check with `--python 3.14`.
`--find-links /path/to/wheelhouse` runs without network access or a package
cache, using dependency wheels from that directory and an already available
interpreter. An sdist install also needs the build backend there. Without that option, dependencies come from the package
index. The pull-request workflow runs lint, types and the numerical CPU suite
on Python 3.12, plus separate uncached wheel/sdist jobs on Python 3.12–3.14.
CUDA checks run separately on NVIDIA hardware.
