# misalign

The `tomojax-misalign` command creates misaligned datasets from a
ground-truth `.nxs` file that contains a volume. It re-projects the
volume with per-view 5-DOF perturbations and optionally adds Poisson
noise. You can use random jitter, deterministic schedules, or both.

```
tomojax-misalign --data <in.nxs> --out <out.nxs> \
  [--rot-deg <float>] [--trans-px <float>] \
  [--poisson <float>] [--seed <int>] \
  [--pert dof:shape[:k=v,...]] [--spec <json>] \
  [--with-random] [--progress]
```

## Random perturbations

By default, `tomojax-misalign` applies uniform random jitter to all
five alignment degrees of freedom for every view. The jitter ranges
are controlled by two flags.

| Flag | Default | Description |
|------|---------|-------------|
| `--rot-deg` | `1.0` | Maximum absolute rotation in degrees for each of alpha, beta, and phi |
| `--trans-px` | `10.0` | Maximum absolute translation in detector pixels for dx and dz (converted to world units via detector spacing) |
| `--poisson` | `0` | Incident intensity scale for Poisson noise. Data are treated as intensities and sampled as `Poisson(proj * s) / s`. Larger values produce lower relative noise. Set 0 to disable. |
| `--seed` | `0` | RNG seed for reproducibility. The Poisson noise uses `seed + 1`. |

> [!TIP]
> Start with moderate values like `--rot-deg 1.0 --trans-px 10` for
> alignment benchmarks. Larger values make alignment harder and may
> require more outer iterations or multi-resolution levels.

## Deterministic schedules

Deterministic schedules let you apply specific, reproducible
perturbation patterns to individual degrees of freedom. They're
useful for testing alignment robustness against known failure modes
like linear drift, sudden shifts, or oscillatory motion.

You specify schedules with repeatable `--pert` flags or a JSON file
via `--spec`.

| Flag | Description |
|------|-------------|
| `--pert dof:shape[:k=v,...]` | Add a single schedule (repeatable). DOFs: `angle`, `alpha`, `beta`, `phi`, `dx`, `dz`. Shapes: `linear`, `sin-window`, `step`, `box`. |
| `--spec <path.json>` | Load schedules from a JSON file |
| `--with-random` | Combine random jitter on top of deterministic schedules |

When any `--pert` or `--spec` is present, random jitter is disabled
by default. Use `--with-random` to add random perturbations on top
of the deterministic pattern. In that case, `--rot-deg` and
`--trans-px` control the random ranges as usual.

> [!NOTE]
> Schedules are additive and applied in the order given. Multiple
> `--pert` flags targeting the same DOF accumulate. See the
> [misalignment modes reference](../reference/misalign-modes.md) for
> full documentation of schedule shapes, parameters, windowing, and
> the JSON spec file format.

## Examples

The examples below use `uv run` to invoke the console script. You
can substitute `python -m tomojax.cli.misalign` if you prefer.

### Random clean misalignment

This applies random 5-DOF jitter without noise, suitable for basic
alignment testing.

```bash
uv run tomojax-misalign \
  --data data/sim_aligned.nxs \
  --out data/sim_misaligned.nxs \
  --rot-deg 1.0 --trans-px 10 \
  --seed 0 --progress
```

### Random misalignment with Poisson noise

Adding `--poisson` simulates photon counting statistics on top of
the misaligned projections.

```bash
uv run tomojax-misalign \
  --data data/sim_aligned.nxs \
  --out data/sim_misaligned_noisy.nxs \
  --rot-deg 1.0 --trans-px 10 \
  --poisson 5000 --seed 0 --progress
```

### Deterministic linear drift

This applies a linear angle drift of 0 to +5 degrees across the
full scan, with no random jitter.

```bash
uv run tomojax-misalign \
  --data data/sim_aligned.nxs \
  --out runs/mis_angle_lin.nxs \
  --pert angle:linear:delta=5deg
```

### Deterministic step shift

This applies a sudden +5 pixel horizontal shift at 90 degrees,
held to the end of the scan.

```bash
uv run tomojax-misalign \
  --data data/sim_aligned.nxs \
  --out runs/mis_dx_step.nxs \
  --pert dx:step:at=90deg,to=5px
```

> [!WARNING]
> The input `.nxs` file must contain a ground-truth volume under
> `/entry/processing/tomojax/volume`. Files produced by
> `tomojax-simulate` include this automatically; files from other
> sources may not.

---

See also: [misalignment modes](../reference/misalign-modes.md),
[simulate](simulate.md).
