# Config files

The `tomojax-recon` and `tomojax-align` commands accept a
`--config <path.toml>` flag to load default values from a TOML file.
This is useful for long, reproducible commands where you don't want to
retype every flag.

## How precedence works

Values are resolved in this order (later sources win):

1. Built-in defaults (hardcoded in the CLI)
2. TOML config file (`--config`)
3. Explicit CLI flags

An explicit flag always overrides the matching TOML key:

```bash
uv run tomojax-align --config docs/align_config.toml --levels 2 1
```

Here, `levels` from the TOML file is replaced by `[2, 1]`.

## Key naming rules

TOML keys must match the argparse destination names using underscores:

| CLI flag | TOML key |
|----------|----------|
| `--lambda-tv` | `lambda_tv` |
| `--gather-dtype` | `gather_dtype` |
| `--checkpoint-projector` | `checkpoint_projector` |
| `--save-manifest` | `save_manifest` |

**Type mapping:**

| Python type | TOML syntax | Example |
|-------------|-------------|---------|
| `int` / `float` | number | `lambda_tv = 0.003` |
| `bool` | boolean | `checkpoint_projector = true` |
| `str` | string | `algo = "fbp"` |
| `list[int]` | array | `levels = [4, 2, 1]` |
| `list[str]` | array | `loss_param = ["delta=1.0"]` |

> [!NOTE]
> YAML isn't supported by the CLI. The `pyyaml` package is only an
> optional benchmark dependency. Use TOML (`.toml`) files.

## Reconstruction config example

The example file at `docs/recon_config.toml`:

```toml
# Minimal tomojax-recon config.
# Run with:
#   uv run tomojax-recon --config docs/recon_config.toml

data = "data/sim_misaligned.nxs"
out = "runs/recon_from_config.nxs"

algo = "fbp"
filter = "ramp"
regulariser = "tv"
roi = "auto"
gather_dtype = "auto"
checkpoint_projector = true
progress = true
save_manifest = "runs/recon_from_config.manifest.json"
```

To switch to iterative reconstruction, change `algo` and add the
relevant keys:

```toml
algo = "fista"
iters = 60
lambda_tv = 0.005
tv_prox_iters = 10
```

## Alignment config example

The example file at `docs/align_config.toml`:

```toml
# Minimal tomojax-align config.
# Run with:
#   uv run tomojax-align --config docs/align_config.toml

data = "data/sim_misaligned.nxs"
out = "runs/align_from_config.nxs"

outer_iters = 4
recon_iters = 25
recon_algo = "fista"
lambda_tv = 0.003
regulariser = "tv"
levels = [4, 2, 1]
opt_method = "gn"
gn_damping = 0.001
gauge_fix = "mean_translation"
gather_dtype = "auto"
checkpoint_projector = true
log_summary = true
progress = true
save_params_json = "runs/align_from_config.params.json"
save_params_csv = "runs/align_from_config.params.csv"
save_manifest = "runs/align_from_config.manifest.json"
```

**Common additions:**

```toml
# DOF selection (translation-only 2-DOF)
optimise_dofs = ["dx", "dz"]

# Loss schedule (per pyramid level)
loss_schedule = "4:phasecorr,2:ssim,1:l2_otsu"

# Bounds on active DOFs
# (TOML inline table syntax)
bounds = { dx = [-20, 20], dz = [-20, 20] }

# Smooth pose model
pose_model = "spline"
knot_spacing = 12
degree = 3

# Checkpoint/resume
checkpoint = "runs/align.checkpoint.npz"
checkpoint_every = 1
```

## Next steps

- [recon CLI reference](../cli/recon.md) — all reconstruction flags
- [align CLI reference](../cli/align.md) — all alignment flags
- [CLI overview](../cli/index.md) — environment variables and common
  patterns
