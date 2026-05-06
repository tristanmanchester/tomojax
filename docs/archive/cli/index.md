# CLI overview

TomoJAX provides command-line tools for every step of the CT workflow.
This page covers options shared across all commands. Individual
command pages document tool-specific flags and examples.

## Available commands

| Command | Description | Page |
|---------|-------------|------|
| `tomojax-simulate` | Generate synthetic datasets | [simulate](simulate.md) |
| `tomojax-misalign` | Apply misalignments to datasets | [misalign](misalign.md) |
| `tomojax-preprocess` | Correct raw NXtomo frames | [preprocess](preprocess.md) |
| `tomojax-recon` | Reconstruct volumes (FBP/FISTA/SPDHG) | [recon](recon.md) |
| `tomojax-align` | Joint alignment and reconstruction | [align](align.md) |
| `tomojax-inspect` | Inspect NXtomo metadata and statistics | [inspect](inspect.md) |
| `tomojax-validate` | Validate NXtomo file structure | [validate](validate.md) |
| `tomojax-convert` | Convert between `.npz` and `.nxs` | [convert](convert.md) |
| `tomojax-loss-bench` | Benchmark loss function behavior | [loss-bench](loss-bench.md) |

Two additional utilities verify your installation:

| Command | Description |
|---------|-------------|
| `tomojax-test-gpu` | Print JAX backend and device list (expects GPU) |
| `tomojax-test-cpu` | Force CPU mode and print backend info |

## Running commands

You can invoke any command in two ways:

```bash
# Console script (recommended)
uv run tomojax-simulate --help

# Module invocation
python -m tomojax.cli.simulate --help
```

Both forms are equivalent. The console scripts are registered in
`pyproject.toml` and installed automatically by `uv sync`.

## Config files (TOML)

The `recon` and `align` commands accept `--config <path.toml>` to load
defaults from a TOML file. This is useful for long, reproducible
commands.

**Precedence:** built-in defaults < TOML config < explicit CLI flags.

An explicit CLI flag always overrides the corresponding TOML key, so
you can use a config file as a baseline and tweak individual values:

```bash
uv run tomojax-align --config docs/align_config.toml --levels 2 1
```

**Key naming rules:**

- Use underscores, not dashes: `lambda_tv`, not `lambda-tv`
- Lists use TOML arrays: `levels = [4, 2, 1]`
- Booleans use TOML booleans: `checkpoint_projector = true`
- YAML isn't supported (pyyaml is only an optional benchmark
  dependency)

See [Config files reference](../reference/config-files.md) for
annotated example files.

## Common flags

These flags appear on multiple commands:

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--gather-dtype` | `fp32`, `bf16`, `fp16`, `auto` | `fp32` | Mixed-precision gather; `bf16` recommended on modern GPUs |
| `--checkpoint-projector` | flag | enabled | Rematerialize projector to cut activation memory (~10-25% extra compute) |
| `--progress` | flag | off | Show progress bars |
| `--save-manifest` | path | — | Write a JSON reproducibility sidecar |
| `--out` | path | — | Output file path |

> [!TIP]
> Set `TOMOJAX_PROGRESS=1` to enable progress bars globally instead
> of passing `--progress` on every command.

## Environment variables

| Variable | Description |
|----------|-------------|
| `TOMOJAX_PROGRESS=1` | Enable progress bars |
| `TOMOJAX_PROGRESS_LEAVE=1` | Keep progress bars visible after completion |
| `TOMOJAX_JAX_CACHE_DIR=<path>` | Persistent JAX compilation cache directory |
| `TOMOJAX_AXES_SILENCE=1` | Suppress axis-order heuristic warnings |
| `JAX_PLATFORM_NAME=cpu` | Pin JAX to CPU backend |
| `JAX_PLATFORMS=cuda` | Pin JAX to CUDA backend |
| `XLA_PYTHON_CLIENT_PREALLOCATE=false` | Disable device memory preallocation |

## Reproducibility

Pass `--save-manifest <path.json>` to write a JSON sidecar recording:

- Raw `argv` and parsed CLI arguments
- Resolved config (after TOML + CLI merge)
- TomoJAX, Python, and JAX versions
- JAX backend and device list
- UTC timestamp

This manifest isn't embedded in the `.nxs` output — it's a separate
file for automation and record-keeping.
