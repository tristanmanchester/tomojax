# recon

The `tomojax-recon` command reconstructs a 3D volume from projections
using FBP, FISTA-TV, or SPDHG-TV. It reads an NXtomo `.nxs` file and
writes a new `.nxs` with the reconstructed volume.

```
tomojax-recon [--config <config.toml>] --data <in.nxs> \
  [--algo fbp|fista|spdhg] --out <out.nxs> [options...]
```

## Algorithm selection

You choose the reconstruction algorithm with `--algo`. The default is
`fbp`, which runs a single-pass filtered back-projection. The two
iterative alternatives, `fista` and `spdhg`, add TV regularisation
and produce higher-quality reconstructions at the cost of more
computation.

- `--algo fbp` -- filtered back-projection (default)
- `--algo fista` -- FISTA with TV or Huber-TV regularisation
- `--algo spdhg` -- stochastic primal-dual hybrid gradient with TV

See [reconstruction concepts](../concepts/reconstruction.md) for
algorithm background and guidance on when to use each solver.

## FBP options

FBP applies a frequency-domain filter to the projections before
back-projecting. The filter controls the sharpness/noise trade-off.

- `--filter ramp` -- sharp reconstruction (default). Aliases:
  `ram-lak`, `ramlak`.
- `--filter shepp` -- Shepp-Logan filter, slightly smoother than ramp
- `--filter hann` -- Hann window, strongest smoothing

## FISTA-TV options

FISTA iteratively minimises a data-fidelity term plus a total
variation penalty. These flags control convergence speed, noise
suppression, and physical constraints on the reconstruction.

- `--iters` -- number of FISTA iterations (default: 50)
- `--lambda-tv` -- TV regularisation weight (default: 0.005)
- `--tv-prox-iters` -- inner iterations for the TV proximal
  operator (default: 10). We recommend increasing to 20--30 for
  heavily noisy data.
- `--L` -- data-term Lipschitz constant. When set, FISTA skips the
  power-method estimation at startup. Note: when using `huber_tv`,
  an additional Lipschitz contribution is added to this value.
- `--regulariser tv|huber_tv` -- regulariser type (default: `tv`).
  `huber_tv` is smoother near zero gradients; tune `--huber-delta`
  to control the transition radius.
- `--huber-delta` -- Huber-TV transition radius
  (default: 0.01). Smaller values behave more like standard TV.

**Constraints.** You can project each iterate onto physically
motivated voxel bounds:

- `--positivity` -- enforce non-negative voxels
- `--lower-bound` -- clip voxels below this value
- `--upper-bound` -- clip voxels above this value

`--positivity` and explicit bounds are mutually exclusive with
`--no-positivity`.

## SPDHG-TV options

SPDHG uses stochastic minibatch updates and can scale better than
FISTA for large view counts. Step sizes default to operator-norm
estimates; you can override them when tuning convergence.

- `--iters` -- outer PDHG iterations (default: 50)
- `--lambda-tv` -- TV regularisation weight (default: 0.005)
- `--views-per-batch` -- stochastic block size (default: 16).
  Typical values range from 16 to 64.
- `--theta` -- extrapolation parameter for xbar
  (default: 1.0). Values between 0.5 and 1.0 are typical.
- `--spdhg-seed` -- RNG seed for block order (default: 0)
- `--regulariser tv|huber_tv` -- same as FISTA (default: `tv`)
- `--huber-delta` -- same as FISTA (default: 0.01)

**Step size overrides.** Leave these unset to use automatic
operator-norm-based defaults:

- `--spdhg-tau` -- primal step size
- `--spdhg-sigma-data` -- data dual step size
- `--spdhg-sigma-tv` -- TV dual step size

> [!WARNING]
> SPDHG objective values are minibatch estimates. Compare their
> trend across iterations rather than treating individual values
> as equivalent to FISTA's full-data objective.

## Memory and performance

These flags control GPU memory usage and mixed-precision
behaviour. They apply to all algorithms.

- `--gather-dtype auto|fp32|bf16|fp16` -- projector gather
  precision (default: `auto`, which selects `bf16` on GPU/TPU and
  `fp32` on CPU). We recommend `bf16` on modern GPUs for a good
  speed/accuracy trade-off. Accumulation always stays in fp32.
- `--checkpoint-projector` / `--no-checkpoint-projector` --
  rematerialise the projector to cut activation memory at the cost
  of ~10--25% extra compute (default: on).

> [!TIP]
> Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` to prevent JAX from
> grabbing all device memory at startup. This is especially useful
> when running multiple processes or when memory is tight.

## ROI and masking

By default, `tomojax-recon` crops the reconstruction grid to the
detector field of view so you don't waste memory on voxels outside
the measured region.

- `--roi auto|cube|bbox|off` -- ROI strategy (default: `auto`).
  `auto` uses square x-y slices plus z from detector height when
  the detector is smaller than the grid. `cube` forces a cubic ROI.
  `bbox` uses the rectangular FOV bounding box. `off` keeps the
  original grid unchanged.
- `--mask-vol off|cyl` -- volume mask (default: `off`). `cyl`
  applies a cylindrical x-y mask that zeros voxels outside the
  circular FOV. This mask acts as a support constraint in FISTA
  and SPDHG, and is applied post-hoc for FBP.
- `--grid NX NY NZ` -- override the reconstruction grid
  dimensions. Voxel sizes remain as defined in the input metadata.

## Output options

These flags control what gets written alongside the main `.nxs`
reconstruction.

- `--quicklook PATH` / `--save-preview PATH` -- write a
  percentile-scaled PNG of the central xy slice after
  reconstruction. Both flag names are equivalent.
- `--save-manifest PATH` -- write a JSON reproducibility manifest
  containing raw argv, parsed CLI args, resolved config,
  TomoJAX/Python/JAX versions, backend info, and a UTC timestamp.
- `--frame sample|lab` -- frame label recorded in the output
  (default: `sample`). Use `lab` only for compatibility exports.

## Config files

You can store long command lines in a TOML file and load them with
`--config`. Explicit CLI flags always override TOML values.

```bash
tomojax-recon --config docs/recon_config.toml --gather-dtype bf16
```

See [config files reference](../reference/config-files.md) for
annotated examples and key-naming rules.

## Examples

FBP with bf16 gathers:

```bash
tomojax-recon --data data/scan.nxs \
  --algo fbp --filter ramp --gather-dtype bf16 \
  --checkpoint-projector --out out/fbp.nxs \
  --save-manifest out/fbp.manifest.json --progress
```

FISTA with TV:

```bash
tomojax-recon --data data/scan.nxs \
  --algo fista --iters 60 --lambda-tv 0.005 \
  --gather-dtype bf16 --checkpoint-projector \
  --out out/fista.nxs --progress
```

FISTA with positivity and box constraints:

```bash
tomojax-recon --data data/scan.nxs \
  --algo fista --iters 80 --lambda-tv 0.005 \
  --positivity --lower-bound 0 --upper-bound 1 \
  --out out/fista_bounded.nxs --progress
```

SPDHG-TV with moderate block size:

```bash
tomojax-recon --data data/scan.nxs \
  --algo spdhg --iters 300 --lambda-tv 0.005 \
  --views-per-batch 32 --theta 0.5 \
  --gather-dtype bf16 --checkpoint-projector \
  --roi auto --mask-vol cyl \
  --out out/spdhg.nxs --progress
```

TOML config with one explicit override:

```bash
tomojax-recon --config docs/recon_config.toml --gather-dtype bf16
```

---

See also: [reconstruction concepts](../concepts/reconstruction.md)
| [config files](../reference/config-files.md)
| [CLI overview](index.md)
