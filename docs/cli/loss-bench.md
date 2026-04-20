# loss-bench

The `tomojax-loss-bench` command benchmarks multiple alignment
losses on a small synthetic misaligned phantom. It generates a
dataset, runs alignment with each specified loss, and produces
comparative metrics in JSON and CSV format.

```
tomojax-loss-bench [options...]
```

## Dataset generation

The command creates a synthetic phantom internally, so you don't
need to prepare input data. These flags control the phantom and
scan geometry:

| Flag | Default | Description |
|------|---------|-------------|
| `--nx` | `128` | Volume x-dimension |
| `--ny` | `128` | Volume y-dimension |
| `--nz` | `1` | Volume z-dimension |
| `--nu` | `128` | Detector columns |
| `--nv` | `128` | Detector rows |
| `--n-views` | `60` | Number of projection views |
| `--geometry` | `parallel` | Geometry type: `parallel` or `lamino` |
| `--seed` | `0` | Random seed for phantom and misalignment |
| `--rot-deg` | `1.0` | Max per-axis rotation misalignment (degrees) |
| `--trans-px` | `5.0` | Max translation misalignment (pixels) |

The misalignment seed is `seed + 1`, so the phantom and the
perturbation are independently reproducible.

## Alignment parameters

These flags control the alignment loop that runs for each loss.

| Flag | Default | Description |
|------|---------|-------------|
| `--outer-iters` | `4` | Outer alignment iterations |
| `--recon-iters` | `10` | Inner reconstruction iterations per outer |
| `--levels` | *(none)* | Multi-resolution pyramid factors (e.g. `4 2 1`) |
| `--progress` | off | Show progress bars |

> [!NOTE]
> For GN-compatible (LS-like) losses, the benchmark automatically
> raises `outer-iters` to at least 8 and `recon-iters` to at least
> 30 to give the optimizer a fair chance.

When you pass `--levels`, the factors apply to every loss in the
run. Without `--levels`, losses that have built-in defaults
(`l2`, `l2_otsu`, `edge_l2`, `pwls`) use a `4 2 1` pyramid; other
losses run single-level.

## Loss selection

The `--losses` flag controls which alignment losses are benchmarked.

```bash
# Default set (GN-only / LS-like losses)
tomojax-loss-bench --losses l2 l2_otsu pwls

# Custom subset
tomojax-loss-bench --losses l2 edge_l2
```

By default, the benchmark tests GN-only (least-squares-like)
losses. Losses that aren't GN-compatible are logged as `skipped`
unless you pair them with an optimizer that supports them (the
benchmark currently runs GN exclusively).

## Metrics

After alignment finishes (or is loaded from a previous run), the
benchmark computes several accuracy metrics against the known
ground-truth misalignment:

- **Absolute metrics** -- rotation RMSE/MSE/MAE (degrees) and
  translation RMSE/MSE/MAE (pixels).
- **Relative-motion metrics** -- RMSE of k-step differences
  between estimated and true parameters. Control the step size
  with `--k-step` (default `1`).
- **Gauge-fixed metrics** -- rotation and translation RMSE after
  removing the global translation gauge ambiguity.
- **Physics-aware metric** -- `--gt-metric mse` (default)
  forward-projects the ground-truth volume with estimated poses
  and reports the MSE against the original projections. Set
  `--gt-metric none` to skip this step.

Use `--metrics-only` to recompute metrics from existing aligned
outputs without re-running alignment. This is useful when you
change `--k-step` or `--gt-metric` after a long run.

```bash
tomojax-loss-bench --metrics-only --k-step 3 --gt-metric none
```

> [!TIP]
> The `--metrics-only` flag skips alignment entirely. If an
> expected output file is missing, that loss is reported with
> status `missing` rather than failing.

## Output files

All outputs go into the directory set by `--expdir` (default:
`runs/loss_experiment`). The benchmark creates the directory and a
`logs/` subdirectory automatically.

| File | Contents |
|------|----------|
| `results.json` | Full config and per-loss metrics |
| `results.csv` | Tabular metrics (one row per loss) |
| `logs/<loss_name>.log` | Per-loss alignment log |
| `align_<loss_name>.nxs` | Aligned output dataset |

> [!WARNING]
> If an `align_<loss_name>.nxs` file already exists, the benchmark
> skips alignment for that loss and recomputes metrics from the
> existing file. Delete old outputs if you want a fresh run.

## Examples

Run the default benchmark with the standard losses:

```bash
uv run tomojax-loss-bench
```

Specify a custom output directory and losses:

```bash
uv run tomojax-loss-bench \
  --expdir runs/custom_bench \
  --losses l2 l2_otsu edge_l2
```

Use multi-resolution alignment:

```bash
uv run tomojax-loss-bench \
  --levels 4 2 1 \
  --outer-iters 6 --recon-iters 20 \
  --progress
```

Recompute metrics with a different k-step, without re-running
alignment:

```bash
uv run tomojax-loss-bench \
  --expdir runs/loss_experiment \
  --metrics-only --k-step 5
```

Benchmark on a laminography geometry with a larger phantom:

```bash
uv run tomojax-loss-bench \
  --geometry lamino \
  --nx 256 --ny 256 --nz 1 \
  --nu 256 --nv 256 --n-views 120 \
  --seed 42
```

## See also

- [Loss functions reference](../reference/loss-functions.md) for
  details on each available loss
- [align](align.md) for the general-purpose alignment CLI
- [CLI overview](index.md) for all available commands
