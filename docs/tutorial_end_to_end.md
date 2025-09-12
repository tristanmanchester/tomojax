# TomoJAX Next — End‑to‑End Alignment + Reconstruction Tutorial

This short guide walks through a complete workflow using the `tomojax` v2 package:

1) simulate a 3D phantom, 2) generate a misaligned forward model, 3) add noise,
4) run naive FBP reconstructions, 5) run iterative alignment + reconstruction
(coarse→fine), and 6) save results for side‑by‑side comparison.

All commands assume you are inside the pixi environment.

- Enter the environment: `pixi shell`
- One-time install of the root package: `pixi run install-root`
- Show progress bars: export `TOMOJAX_PROGRESS=1`


## 0) Quick Overview (What You’ll Use)

- CLI modules:
  - `tomojax.cli.simulate` — create NXtomo datasets (.nxs)
  - `tomojax.cli.recon` — FBP or FISTA reconstructions
  - `tomojax.cli.align` — joint alignment + reconstruction
- Python APIs (for custom steps):
  - `tomojax.data.io_hdf5` — read/write .nxs
  - `tomojax.core.projector` + `align.parametrizations` — misalign + project

### Preflight (optional, fast)

Before running the full 128³ workflow, do a quick small run to warm JAX/XLA and
verify your environment:

```
pixi run test-gpu
pixi run simulate --out data/sim_aligned_small.nxs --nx 32 --ny 32 --nz 32 --nu 32 --nv 32 --n-views 32 \
  --phantom random_shapes --n-cubes 8 --n-spheres 8 --min-size 3 --max-size 8 --min-value 0.01 --max-value 0.1 --seed 1 --progress
pixi run misalign --data data/sim_aligned_small.nxs --out data/sim_misaligned_small.nxs --rot-deg 1 --trans-px 4 --seed 0 --progress
```

Note: the first projector/align call compiles with XLA and takes longer; subsequent
runs are faster. The preflight helps warm the cache.


## 1) Simulate a 128³ Phantom (20 spheres + 20 cubes, values 0.01–0.1)

We use the “random_shapes” phantom with controlled counts and value range.

```
pixi run simulate \
  --out data/sim_aligned.nxs \
  --nx 256 --ny 256 --nz 256 \
  --nu 256 --nv 256 --n-views 200 \
  --phantom random_shapes \
  --n-cubes 40 --n-spheres 40 \
  --min-size 4 --max-size 64 \
  --min-value 0.01 --max-value 0.1 \
  --seed 42 \
  --progress
```

Notes
- Geometry defaults to parallel‑beam; detector matches the volume size.
- Datasets are saved as NeXus (NXtomo) HDF5 with metadata for geometry, grid, etc.


## 2) Create a Misaligned Forward Model (±10 px translations, ±1° rotations)

This step reprojects the ground‑truth volume using per‑view 5‑DOF perturbations
and saves a new dataset. Use the new misalign CLI for a single command:

```
pixi run misalign \
  --data data/sim_aligned.nxs \
  --out data/sim_misaligned.nxs \
  --rot-deg 1.0 --trans-px 10 \
  --seed 0 \
  --progress
```


## 3) Add Poisson Noise (5000 photons/pixel)

To add Poisson noise (e.g., 5000 photons/pixel) directly:

```
pixi run misalign \
  --data data/sim_aligned.nxs \
  --out data/sim_misaligned_poisson5k.nxs \
  --rot-deg 1.0 --trans-px 10 \
  --poisson 100 \
  --seed 0 \
  --progress
```


## 4) Naive FBP Reconstructions

Run FBP on both the misaligned and the noisy+misaligned datasets.

```
# Misaligned
pixi run recon \
  --data data/sim_misaligned.nxs \
  --algo fbp --filter ramp \
  --out out/fbp_misaligned.nxs \
  --progress

# Noisy + misaligned
pixi run recon \
  --data data/sim_misaligned_poisson5k.nxs \
  --algo fbp --filter ramp \
  --out out/fbp_misaligned_noisy.nxs \
  --progress
```


## 5) Iterative Alignment + Reconstruction (Levels [4, 2, 1])

Use multires alignment with Gauss–Newton updates and view batching for speed.

```
# Misaligned (clean)
pixi run align \
  --data data/sim_misaligned.nxs \
  --levels 4 2 1 \
  --outer-iters 4 --recon-iters 25 --lambda-tv 0.003 \
  --opt-method gn --gn-damping 1e-3 \
  --views-per-batch 16 --projector-unroll 4 \
  --log-summary \
  --out out/align_misaligned.nxs \
  --progress

# Noisy + misaligned (stronger TV and a few more iters)
pixi run align \
  --data data/sim_misaligned_poisson5k.nxs \
  --levels 4 2 1 \
  --outer-iters 5 --recon-iters 30 --lambda-tv 0.01 \
  --opt-method gn --gn-damping 1e-3 \
  --views-per-batch 16 --projector-unroll 4 \
  --log-summary \
  --out out/align_misaligned_noisy.nxs \
  --progress
```

Tips
- If memory is tight, reduce `--views-per-batch` (e.g., 8) or `--projector-unroll` (e.g., 2).
- If convergence stalls under GD, switch to GN (`--opt-method gn`) or increase `--outer-iters`.


## 6) Outputs and Comparison

After running the steps above you’ll have four .nxs files to compare:

- Naive FBP:
  - `out/fbp_misaligned.nxs`
  - `out/fbp_misaligned_noisy.nxs`
- Aligned reconstructions:
  - `out/align_misaligned.nxs`
  - `out/align_misaligned_noisy.nxs`

Each aligned .nxs also contains the estimated per‑view alignment parameters.
You can visualize by loading the volumes in your viewer of choice (napari,
ccpi, tomviz) or by writing a small slice/isosurface notebook.


## Appendix — Knobs That Matter

- `--views-per-batch`: trades memory for throughput by chunking views.
- `--projector-unroll`: unroll factor for the projector’s `lax.scan` (4–8 is good on GPUs).
- `--lambda-tv`: increase for noisy data; keep accumulations in fp32.
- `--opt-method`: `gn` is robust and fast; `gd` is simpler but may need more outer iterations.
- Mixed precision gather (advanced): the projector supports `gather_dtype="bf16"` internally; if you want a CLI flag for this, we can expose it easily.

Logging and progress
- Add `--log-summary` to print per‑outer summaries:
  - FISTA: first/last/min objective.
  - Alignment: loss before/after, and either GN mean step magnitudes or GD gradient RMS.
- Suppress backend probe INFO lines from JAX if you prefer:
  - `export JAX_PLATFORM_NAME=gpu` or `export JAX_PLATFORMS=cuda`.


## Appendix — Data I/O Shortcuts

- Validate an .nxs file quickly:

```
python - << 'PY'
from tomojax.data.io_hdf5 import validate_nxtomo
print(validate_nxtomo('data/some_dataset.nxs'))
PY
```

- Convert between `.npz` and `.nxs`:

```
pixi run python -m tomojax.cli.convert --in data/sim_aligned.nxs --out data/sim_aligned.npz
pixi run python -m tomojax.cli.convert --in data/sim_aligned.npz --out data/sim_aligned_back.nxs
```
