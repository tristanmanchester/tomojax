# Alignment L-BFGS 64^3 Smoke Benchmark

This note records CPU smoke tests run after adding the Optax L-BFGS pose
optimizer. The goal is to document behavior and tuning signals, not to make a
general performance claim. All runs below optimize pose/alignment parameters
only; the reconstruction volume is fixed unless explicitly noted.

## Environment

- Date: 2026-04-19
- Backend: JAX CPU
- Device: `CpuDevice(id=0)`
- JAX: `0.10.0`
- Optax: `0.2.8`
- SciPy present in environment: `1.17.1`, but not used by L-BFGS
- NumPy: `2.4.3`
- Volume shape: `64 x 64 x 64`
- Detector shape: `64 x 64`
- Alignment DOFs: translation-only `dx,dz` unless otherwise stated
- Bounds, where used: `dx,dz in [-0.5, 0.5]`

## Fixed-Volume L2 Comparison

These runs used a fixed ground-truth volume, 24 views, irregular per-view
translations, active `dx,dz`, and L2 alignment loss. The reported RMSE is
translation RMSE against the synthetic truth.

| Optimizer | L-BFGS maxiter | Initial loss | Final loss | Loss reduction | RMSE | Wall time | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GD | n/a | `403.2953` | `64.9614` | `83.8923%` | `0.0621994` | `2.01s` | Baseline first-order step |
| GN | n/a | `403.2953` | `21.6833` | `94.6235%` | `0.0255727` | `1.89s` | Best speed/quality tradeoff for this L2 case |
| L-BFGS | `2` | `403.2953` | `66.7873` | `83.4396%` | `0.0528224` | `9.64s` | `nit=2`, `nfev=6`, accepted |
| L-BFGS | `4` | `403.2953` | `4.67817` | `98.8400%` | `0.0329723` | `10.95s` | `nit=4`, `nfev=10`, accepted |
| L-BFGS | `8` | `403.2953` | `0.548235` | `99.8641%` | `0.0197217` | `16.63s` | `nit=8`, `nfev=18`, accepted |
| L-BFGS | `12` | `403.2953` | `0.106266` | `99.9737%` | `0.0139801` | `29.43s` | `nit=12`, `nfev=26`, accepted |

Interpretation:

- GN remains the recommended default for L2-like losses when wall time matters.
- L-BFGS needs several line-search evaluations per outer step, so it is slower
  on CPU, but it can drive the fixed-volume objective lower with enough
  iterations.
- For this problem size, `--lbfgs-maxiter 4` is a reasonable exploratory setting
  and `--lbfgs-maxiter 8` is the better quality setting. `12` gave another
  improvement, but with a clear runtime cost.

## Chunk-Size Sensitivity

These runs repeated the 24-view fixed-volume L2 case with L-BFGS
`maxiter=4`.

| `views_per_batch` | Initial loss | Final loss | Loss reduction | RMSE | Wall time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1` | `403.295288` | `4.678166` | `98.8400146%` | `0.03297234` | `11.10s` |
| `4` | `403.295258` | `4.678167` | `98.8400143%` | `0.03297234` | `12.79s` |
| `8` | `403.295319` | `4.678167` | `98.8400145%` | `0.03297234` | `15.72s` |

The result is numerically stable across chunk sizes. On this CPU run, larger
chunks were slower. GPU behavior may differ because larger chunks can improve
device utilization.

## Robust Loss Comparison

These runs used a 16-view fixed-volume case with the Charbonnier robust loss.

| Optimizer | L-BFGS maxiter | Initial loss | Final loss | Loss reduction | RMSE | Wall time | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GD | n/a | `1342.1337` | `553.6176` | `58.7509%` | `0.0574621` | `1.66s` | Baseline |
| L-BFGS | `8` | `1342.1337` | `158.8580` | `88.1638%` | `0.0285062` | `24.39s` | `nit=8`, `nfev=18`, accepted |

L-BFGS is most compelling in this kind of differentiable robust-loss setting:
GN is not the natural optimizer, while L-BFGS can still use gradients and line
search without leaving JAX/Optax.

## Smooth Pose Model

These runs used 20 views with quadratic ground-truth `dx,dz` drift and
`lbfgs_maxiter=8`.

| Pose model | Bounds | Initial loss | Final loss | Loss reduction | RMSE | Wall time |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `per_view` | yes | `270.7649` | `0.353743` | `99.8694%` | `0.0267563` | `23.20s` |
| `polynomial` | no | `270.7649` | `2.97734` | `98.9004%` | `0.0151492` | `22.46s` |
| `polynomial` | yes | `270.7649` | `0.000465897` | `99.9998%` | `0.00012071` | `26.31s` |

The smooth polynomial pose model is useful when the true motion is smooth. The
bounded polynomial path performed best in this synthetic case, which exercises
the code path that optimizes bounded per-view active pose values and then refits
smooth coefficients.

## Memory-Size Sweep

These runs repeated the 24-view fixed-volume L2 case with
`lbfgs_maxiter=8`.

| `lbfgs_memory_size` | Initial loss | Final loss | Loss reduction | RMSE | Wall time | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `3` | `403.2953` | `0.565654` | `99.8597%` | `0.0198410` | `27.82s` | Slightly worse |
| `10` | `403.2953` | `0.548235` | `99.8641%` | `0.0197217` | `21.77s` | Default |
| `20` | `403.2953` | `0.548235` | `99.8641%` | `0.0197217` | `13.59s` | Same accuracy in this run |

Single-run CPU timings are noisy, but accuracy was effectively identical for
memory sizes `10` and `20`. The default `10` is a reasonable compromise. Tune
`--lbfgs-maxiter` before tuning `--lbfgs-memory-size`.

## Alternating-Reconstruction Smoke

This run checked the strict L-BFGS acceptance policy in a harder alternating
alignment setup with a reconstructed volume. It used 16 views, one outer
iteration, two reconstruction iterations, and `recon_L=5000`.

| Optimizer | Initial loss | Final loss | Loss reduction | Param norm | Wall time | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| GD | `172065.5313` | `172187.0313` | `-0.0706%` | `0.45255` | `3.38s` | Moved but worsened |
| GN | `172065.5313` | `172065.5313` | `0.0000%` | `0.00000` | `3.22s` | No accepted movement |
| L-BFGS | `172065.5313` | `172065.5313` | `0.0000%` | `0.00000` | `7.64s` | Rejected non-improving candidate |

The L-BFGS stats reported `optimizer_backend="optax"`, `optimizer_accepted=false`,
`nit=1`, `nfev=4`, `line_search_steps=8`, and `selected_candidate="rejected"`.
This is the intended behavior: L-BFGS accepts only finite candidates that
strictly improve over the baseline objective.

## Practical Guidance

- Keep GN as the default for L2-like alignment losses.
- Use L-BFGS for differentiable robust/similarity losses, smooth pose models, or
  cases where GN is unavailable or too specialized.
- Start with `--lbfgs-maxiter 4` for quick checks and `--lbfgs-maxiter 8` for
  more serious fixed-volume alignment. Increase further only when the added line
  search cost is acceptable.
- Keep `--lbfgs-memory-size 10` unless there is evidence that the limited-memory
  history is constraining convergence.
- Prefer active DOF masks such as `--optimise-dofs dx,dz` when possible. L-BFGS
  works on the active pose vector only, so reducing active variables directly
  reduces optimizer work.
- Bounds are enforced by differentiable reparameterization in the JAX objective,
  not by a Fortran-style active-set L-BFGS-B backend.
- Use `--log-summary` when tuning L-BFGS. The acceptance, objective, evaluation,
  line-search, and fallback fields are the quickest way to tell whether the step
  improved the alignment or was rejected.
