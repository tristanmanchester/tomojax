# Single-sample tutorial

This tutorial mirrors the
[end-to-end tutorial](end-to-end.md) but uses a single centered
object — a sphere or cube — as the phantom. It's a good starting
point if you want a simpler test case with fewer variables.

> [!NOTE]
> Make sure you've completed the [installation](../installation.md)
> and verified your environment before starting. Set
> `TOMOJAX_PROGRESS=1` for progress bars.

## Before you begin

Run a quick warmup on a small volume:

```bash
uv run tomojax-test-gpu
uv run tomojax-simulate \
  --out data/sim_single_small.nxs \
  --nx 32 --ny 32 --nz 32 --nu 32 --nv 32 --n-views 32 \
  --phantom sphere --single-size 0.6 --single-value 1.0 \
  --seed 1 --progress
uv run tomojax-misalign \
  --data data/sim_single_small.nxs \
  --out data/sim_single_misaligned_small.nxs \
  --rot-deg 1 --trans-px 4 --seed 0 --progress
```

## 1. Simulate a 256^3 single-object phantom

Choose `sphere` or `cube`. The `--single-size` parameter is a
relative fraction of the minimum volume dimension (diameter for
sphere, side length for cube):

```bash
# Sphere
uv run tomojax-simulate \
  --out data/sim_single.nxs \
  --nx 256 --ny 256 --nz 256 \
  --nu 256 --nv 256 --n-views 200 \
  --phantom sphere \
  --single-size 0.3 --single-value 1.0 \
  --seed 42 --progress
```

For a cube with a random 3D rotation (more interesting for alignment
testing):

```bash
uv run tomojax-simulate \
  --out data/sim_single.nxs \
  --nx 256 --ny 256 --nz 256 \
  --nu 256 --nv 256 --n-views 200 \
  --phantom cube \
  --single-size 0.3 --single-value 0.3 \
  --seed 69 --progress
```

> [!TIP]
> Cubes are randomly rotated by default. Add `--no-single-rotate`
> for an axis-aligned cube.

## 2. Create misaligned projections

```bash
uv run tomojax-misalign \
  --data data/sim_single.nxs \
  --out data/sim_single_misaligned.nxs \
  --rot-deg 3.0 --trans-px 5 \
  --seed 0 --progress
```

For deterministic schedules, see
[misalignment modes](../reference/misalign-modes.md):

```bash
uv run tomojax-misalign \
  --data data/sim_single.nxs \
  --out runs/single_mis_angle_lin.nxs \
  --pert angle:linear:delta=5deg
```

## 3. Add noise (optional)

```bash
uv run tomojax-misalign \
  --data data/sim_single.nxs \
  --out data/sim_single_misaligned_poisson.nxs \
  --rot-deg 3.0 --trans-px 5 \
  --poisson 10 \
  --seed 0 --progress
```

## 4. Reconstruct with FBP (naive baseline)

```bash
uv run tomojax-recon \
  --data data/sim_single_misaligned.nxs \
  --algo fbp --filter ramp \
  --gather-dtype bf16 --checkpoint-projector \
  --out out/fbp_single_misaligned.nxs --progress

uv run tomojax-recon \
  --data data/sim_single_misaligned_poisson.nxs \
  --algo fbp --filter ramp \
  --gather-dtype bf16 --checkpoint-projector \
  --out out/fbp_single_misaligned_noisy.nxs --progress
```

## 5. Align and reconstruct

```bash
# Clean misaligned
uv run tomojax-align \
  --data data/sim_single_misaligned.nxs \
  --levels 4 2 1 \
  --outer-iters 10 --recon-iters 15 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --log-summary \
  --out out/align_single_misaligned.nxs --progress

# Noisy + misaligned
uv run tomojax-align \
  --data data/sim_single_misaligned_poisson.nxs \
  --levels 4 2 1 \
  --outer-iters 5 --recon-iters 10 \
  --lambda-tv 0.1 --tv-prox-iters 10 \
  --opt-method gn --gn-damping 1e-3 \
  --gather-dtype bf16 --checkpoint-projector \
  --log-summary \
  --out out/align_single_misaligned_noisy.nxs --progress
```

## 6. Compare results

Compare the four outputs:

- `out/fbp_single_misaligned.nxs` — naive FBP (blurred)
- `out/fbp_single_misaligned_noisy.nxs` — naive FBP (blurred + noisy)
- `out/align_single_misaligned.nxs` — aligned (clean)
- `out/align_single_misaligned_noisy.nxs` — aligned (noisy)

Aligned outputs include per-view alignment parameters at
`/entry/processing/tomojax/align/thetas`.

## Key flags

| Flag | Description |
|------|-------------|
| `--phantom sphere\|cube` | Single centered object |
| `--single-size` | Diameter (sphere) or side (cube) as fraction of min dim |
| `--single-value` | Voxel intensity |
| `--no-single-rotate` | Disable random cube rotation |
| `--opt-method gn\|gd\|lbfgs` | GN is fastest for L2 losses |

## Next steps

- [End-to-end tutorial](end-to-end.md) — full workflow with
  `random_shapes` phantom
- [Laminography tutorial](laminography.md) — tilted rotation-axis
  geometry
- [simulate CLI](../cli/simulate.md) — all phantom types and options
