# Install TomoJAX

TomoJAX runs on Linux with Python 3.12–3.14 on CPU or NVIDIA CUDA 12. Both CPU and CUDA installs include `imagecodecs` for compressed TIFFs (including LZW-compressed beamline files). Python 3.15 is not yet supported because `h5py` and `CuPy` do not publish 3.15 wheels.

## Install a published release

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), create a fresh environment, and install the CPU extra:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python "tomojax[cpu]"
.venv/bin/tomojax --help
```

With standard `venv` and `pip`, run `python -m venv .venv` followed by `.venv/bin/python -m pip install "tomojax[cpu]"`.

## Use CUDA 12

A working system NVIDIA driver is required. Install the `cuda12` extra and check that JAX sees the GPU:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python "tomojax[cuda12]"
JAX_PLATFORMS=cuda .venv/bin/python -c "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu'"
```

Setting `JAX_PLATFORMS=cuda` forces an immediate error if CUDA fails to initialise. The `cuda12` extra installs JAX's CUDA runtime, CuPy, and NVIDIA's runtime and NVRTC wheels for compiling TomoJAX's CUDA C kernels on first use (compiled kernels are cached on disk). These wheels do not replace the system NVIDIA driver.

## Optional extras and Studio

Combine extras in one install command, such as `tomojax[cuda12,mcp,spectra]`:

| Extra | Adds |
| --- | --- |
| `cpu` | JAX CPU backend |
| `cuda12` | JAX CUDA 12 runtime, CuPy, and NVIDIA runtime/NVRTC wheels |
| `fourier-cuda12` | CuPy alone for GPU-accelerated Fourier-slice inversion (`method="fourier"`) alongside `cpu` |
| `mcp` | Model Context Protocol server (`tomojax mcp`); see [use TomoJAX from an AI agent](mcp.md) |
| `spectra` | `xraylib` for physical X-ray tube spectra (`corrections.tube_spectrum`) and beam-hardening curves |

To run the native desktop workspace ([TomoJAX Studio](../studio/README.md)) on Linux `x86_64`, install the `tomojax-studio` wheel alongside `tomojax` (`uv pip install "tomojax[cuda12]" tomojax-studio`), or build from a source checkout with Rust **1.99.1** (`uv run --no-sync tomojax studio --build`).

## Build and install an unreleased wheel

From a source checkout, build the sdist and wheel:

```bash
uv build
```

Then install the built wheel from `dist/` into a fresh environment outside the checkout:

```bash
mkdir /tmp/tomojax-user
cd /tmp/tomojax-user
uv venv --python 3.12
uv pip install --python .venv/bin/python "/absolute/path/to/tomojax/dist/tomojax-1.0.0rc2-py3-none-any.whl[cpu]"
.venv/bin/tomojax --help
```

Replace `[cpu]` with `[cuda12]` for CUDA, or point `uv pip install` at the `.tar.gz` sdist to build in an isolated environment. Both distributions include the GPL-3.0 licence; the wheel bundles the CUDA C sources and typing marker. For development setup, see [CONTRIBUTING.md](../CONTRIBUTING.md).

## Verify the installation

Run a small synthetic reconstruction from the environment directory:

```bash
.venv/bin/tomojax simulate -o synthetic.nxs --size 32 --n-views 60
.venv/bin/tomojax reconstruct synthetic.nxs -o recon.nxs
.venv/bin/tomojax inspect recon.nxs --preview-dir previews
```

`inspect` prints `Valid: yes` and writes three central slice previews to `previews/`. Commands refuse to overwrite existing files unless you pass `--force`. Prefix with `JAX_PLATFORMS=cpu` or `JAX_PLATFORMS=cuda` to pin the backend.

If `tomojax` is not found, check your virtual environment path or activate it with `source .venv/bin/activate`. If CUDA fails to initialise, verify your NVIDIA driver and JAX GPU installation first.

## Maintainer package checks

`tools/smoke_installed_wheel.py` creates a disposable environment outside the checkout, installs the built package without a cache, verifies dependencies and module provenance, requires the requested device, reads an LZW-compressed TIFF, and checks the simulated reconstruction and previews:

```bash
uv run --no-sync twine check --strict dist/*
uv run --no-sync python tools/smoke_installed_wheel.py --python 3.12 --extra cpu
uv run --no-sync python tools/smoke_installed_wheel.py --python 3.12 --extra cuda12
uv run --no-sync python tools/smoke_installed_wheel.py --python 3.12 --artifact sdist
```

Check CPU wheels and sdists on Python 3.12, 3.13, and 3.14, and CUDA wheels on Python 3.12 and 3.14. Pass `--find-links /path/to/wheelhouse` for offline verification (an sdist also needs its build backend in the wheelhouse). Pull requests to `main` run lint, type checks, package build, and surface tests on Python 3.12; release workflows run the full CPU suite and uncached wheel/sdist matrix across Python 3.12–3.14, while CUDA checks run separately on NVIDIA hardware.
