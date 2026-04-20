# Reconstruction

TomoJAX provides three reconstruction algorithms. Each takes a set of
projections and a geometry specification and produces a 3D volume. This
page explains how they work and when to use each one. For CLI flags,
see the [recon CLI reference](../cli/recon.md).

## FBP (filtered backprojection)

FBP is a direct (non-iterative) algorithm. It applies a frequency-
domain filter to each projection and backprojects the filtered data
into a volume.

**When to use FBP:**

- Quick sanity checks and previews
- Well-aligned, low-noise data
- Initializing iterative methods

**Available filters:**

| Filter | CLI name | Notes |
|--------|----------|-------|
| Ramp (Ram-Lak) | `ramp` | Sharp, amplifies noise |
| Shepp-Logan | `shepp` | Slight smoothing |
| Hann | `hann` | Stronger smoothing |

```bash
uv run tomojax-recon --data scan.nxs --algo fbp --filter ramp \
  --out recon_fbp.nxs
```

FBP is fast and memory-efficient. It doesn't iterate, so there are no
convergence parameters to tune.

## FISTA-TV

FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) with total
variation regularization is the primary iterative solver. It minimizes
a data-fidelity term plus a TV penalty:

```
minimize  ½‖Ax − b‖² + λ_TV · TV(x)
```

The TV proximal step uses a Chambolle-Pock (primal-dual) sub-solver
controlled by `--tv-prox-iters`.

**When to use FISTA-TV:**

- Noisy data that needs regularization
- Alignment workflows (this is the default inner solver)
- When you need exact gradients through the reconstruction

**Key parameters:**

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| Iterations | `--iters` | 50 | Number of FISTA iterations |
| TV weight | `--lambda-tv` | 0.005 | Regularization strength |
| TV prox iters | `--tv-prox-iters` | 10 | Inner Chambolle-Pock steps |
| Lipschitz | `--L` | auto | Step size (auto-estimated via power method) |

**Constraints:**

FISTA-TV supports element-wise constraints on the reconstructed
volume:

- `--positivity` — enforce non-negative voxels
- `--lower-bound` / `--upper-bound` — clip voxels to a range

**Early stopping:**

Set `--recon-rel-tol` and `--recon-patience` (or the equivalent
`AlignConfig` fields) to stop reconstruction early when the objective
plateaus. This saves time during alignment outer iterations.

> [!TIP]
> For heavy noise, increase both `--lambda-tv` (for example, 0.03)
> and `--tv-prox-iters` (20 to 30) to strengthen the denoising
> effect.

## SPDHG-TV

Stochastic Primal-Dual Hybrid Gradient is an alternative iterative
solver that processes random subsets of views per iteration instead of
the full dataset.

**When to use SPDHG-TV:**

- Large view counts where processing all views per iteration is
  expensive
- When you want stochastic acceleration
- As an alternative inner solver during alignment
  (`--recon-algo spdhg`)

**Key parameters:**

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| Iterations | `--iters` | 400 | Number of SPDHG iterations |
| TV weight | `--lambda-tv` | 0.005 | Regularization strength |
| Batch size | `--views-per-batch` | 16 | Views per stochastic subset |
| Extrapolation | `--theta` | 1.0 | Extrapolation parameter |
| Random seed | `--spdhg-seed` | 0 | Fixes block ordering |

Step sizes (`--spdhg-tau`, `--spdhg-sigma-data`, `--spdhg-sigma-tv`)
are estimated automatically from operator norms. Override them only
if you know appropriate values.

> [!WARNING]
> SPDHG logs a minibatch objective estimate, not the full objective.
> Compare trends rather than absolute values when comparing SPDHG
> and FISTA logs.

## Choosing an algorithm

Use this table as a starting point:

| Scenario | Algorithm | Reason |
|----------|-----------|--------|
| Quick preview | FBP | Fast, no tuning |
| Standard reconstruction | FISTA-TV | Best quality/speed balance |
| Noisy data | FISTA-TV with high λ | TV denoising |
| Very large view count | SPDHG-TV | Stochastic scaling |
| Alignment inner solver | FISTA-TV (default) | Continuity, full objective |
| Alignment with many views | SPDHG-TV | Faster per-iteration |

## Huber-TV regularization

Both FISTA-TV and SPDHG-TV support a smoother Huber-TV penalty as an
alternative to standard TV:

```bash
--regulariser huber_tv --huber-delta 0.01
```

Standard TV penalizes gradients with L1, which preserves sharp edges
but can produce "staircasing" artifacts. Huber-TV transitions smoothly
from quadratic (below `--huber-delta`) to L1 (above it), reducing
staircasing while still preserving edges.

- Smaller `--huber-delta` values behave more like standard TV.
- Larger values produce smoother reconstructions.

## Next steps

- [recon CLI reference](../cli/recon.md) — all flags and examples
- [Alignment concepts](alignment.md) — how reconstruction fits into
  the alignment pipeline
- [Python API reference](../reference/api.md) — `fbp()`,
  `fista_tv()`, and `spdhg_tv()` function signatures
