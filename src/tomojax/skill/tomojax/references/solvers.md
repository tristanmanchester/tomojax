# Direct inverses and matched projection

## Choosing an iterative solver and stopping rule

Unpenalized CGLS and SIRT have no image-space regularizer: running them for hundreds of updates continues to lower the fitted residual while fitting measurement noise and discretization error. Stopping early acts as their regularization. With `iterations` omitted, each iterative solver uses one of three stopping behaviors:

| Solver | Default stopping rule | Work cap / budget | Fixed budget or plain path |
| --- | --- | --- | --- |
| `fista` | Proximal-gradient noise-score stop (`tv_weight=None`, `tolerance=None`) | Up to 200 updates | Pass explicit `tv_weight`, `tolerance=0`, or `iterations=N` |
| `cgls` | Projection-discrepancy stop (when unpenalized and `tolerance=None`) | Up to 50 updates | Pass `iterations=N`, explicit tolerance, or `damping` / `gradient_damping` |
| `sirt` (`subsets=1`) | Projection-discrepancy stop; FISTA momentum on when `relaxation <= 1` | Up to 50 sweeps | Pass `iterations=N` (plain sweeps; add `acceleration=True` for momentum) or `SIRTConfig(acceleration=False)` for plain automatic SIRT |
| `sirt` (`subsets > 1`), `mlem` | Fixed sweep budget (no momentum) | 50 sweeps | Pass `iterations=N` |
| `spdhg` | Block budget scales with view count | ceil(600 × √B) updates for B blocks | Pass `iterations=N` |
| `cgls_multires` | Explicit per-level budgets (`(32, 16)`), numerical stopping per level | Per-level budgets | Pass `iterations_per_level=(...)` |

Inspect `info` after an automatic iterative run (MCP results put every scalar field at the top level and include the stopping summary in the text):

- `termination`: `"projection_discrepancy"` (CGLS/SIRT) or `"noise_score"` (FISTA) when the solver stopped at the estimated error level, `"objective_tolerance"` (FISTA with an explicit `tolerance`), `"iteration_limit"` when the budget ended the run, or `"converged"`, `"roundoff_limit"`, or `"numerical_breakdown"` for CGLS. `early_stop` is `False` when the cap was exhausted.
- `effective_iterations`, plus `resolved_tv_weight`, `effective_tv_weight`, and `noise_score_rms` for FISTA (`spdhg` uses the same isolated-voxel noise calibration when `tv_weight=None` and reports `resolved_tv_weight` and `effective_tv_weight`), or `measurement_noise_rms`, `projection_error_rms`, and the residual (`residual_rms` for CGLS; `pre_update_residual_rms` and resolved `acceleration` for SIRT, plus `slab_stopping` on independent axial SIRT slabs).

On real scans, motion, scatter, calibration bias, and correlated detector errors often leave the projection residual above `projection_error_rms` at 50 updates, so `termination="iteration_limit"` is expected. Check the slices before raising `iterations`, or switch to `fista` when noise needs explicit TV regularization. To replay an automatic run, pass `iterations=info["effective_iterations"]` for CGLS, both `iterations` and `acceleration` from `info` for SIRT, and `tv_weight=info["resolved_tv_weight"]` with `iterations=info["effective_iterations"]` for FISTA.

## Truncated and region-of-interest scans

When the object extends past the detector edges, all seven reconstruction methods model the exterior by default (`exterior="cylinder"`). `fourier`, `fbp`, and cone-beam FDK continue each detector row past its outer edges before filtering, as the projection of a uniform cylinder about the rotation axis fitted to the row's edge values (on an offset full turn, an edge covered by opposing views continues as zero). Rows ending at zero continue as zero, so contained objects reconstruct unchanged. On rolled detectors, `fourier`, `fbp`, and FDK filter along physical horizontal detector lines (quarter turns permute axes directly; other roll angles use cubic resampling), and FDK derives angular weights and short-scan coverage from the effective source orbit (including pose-carried axial angles).

In `tj.reconstruct` and `tj.reconstruct_slabs`, iterative solvers (`cgls`, `sirt`, `fista`, `spdhg`, `mlem`) estimate the contribution from outside the reconstruction grid with a coarse 50-iteration fit to the continued rows (skipped when detector edges are zero or every extended ray hits the grid). CGLS, SIRT, FISTA, and SPDHG subtract that contribution once before the fine solve; MLEM adds its nonnegative estimate to the predicted counts in each update. An FBP or FDK warm start uses the same `exterior` setting. Set `exterior="zero"` in the method configuration (`FBPConfig`, `FDKConfig`, `FourierConfig`, `CGLSConfig`, `SIRTConfig`, `FISTAConfig`, `SPDHGConfig`, or `MLEMConfig`, or `config: {"exterior": "zero"}` with an explicit `method` in MCP) to assume zero beyond the detector and fit the original measurements directly. Direct solver calls below (`recon.cgls`, `recon.sirt`, `recon.fista`, `recon.spdhg`, `recon.mlem`) fit the arrays they receive without running the exterior estimate.

Every reconstruction carries `info["truncation"]` (`edge_ratio`, `outside_fraction`, `interior_shift`, `radius`, `field_of_view_radius`, where `field_of_view_radius` is the nearer edge, or the far edge on an offset axial parallel full turn, and the median over sampled horizontal chords on a rolled detector); iterative methods add `outside_grid`, `data_model` (`"continued_exterior"` or `"measured"`), and `background_treatment` (`"subtracted"`, `"additive_poisson"`, or `"none"`). `tj.reconstruct_slabs` attaches per-slab diagnostics to each yielded block and populates the whole-scan record (alongside iterative stopping fields) in `info=` once the iterator finishes, and worker/MCP `reconstruct_slabs` saves and returns that whole-scan record. Compare `interior_shift` with the contrast of interest: analytic continuation preserves fine detail inside the field of view, the absolute level depends on the cylinder model, non-cylindrical exteriors such as a square bar retain partial cupping and rim error, and a shorter discrepancy-stopped iterative fit can recover less weak fine detail.

## Direct solver calls

With a parallel scan in `scan`, supply its projections in (view, v, u) order.
The iterative inverses return diagnostics alongside the image.

```python
from tomojax import recon
args = (scan.geometry, scan.grid, scan.detector, scan.projections)
coarse, history = recon.cgls_multires(*args, factors=(2, 1), iterations_per_level=(3, 3))
tv, info = recon.fista(*args, config=recon.FISTAConfig(iterations=5))
positive, info = recon.mlem(*args, config=recon.MLEMConfig(iterations=5))
relaxed, info = recon.sirt(*args, config=recon.SIRTConfig(iterations=5))
filtered = recon.fbp(*args)
host = recon.fbp_host(*args, config=recon.FBPHostConfig(slices_per_batch=4))
frequency = recon.fourier(*args)
```

For a circular cone-beam scan, build the arguments from that scan and run
`cone_volume = recon.fdk(*args, config=recon.FDKConfig())` or
`cone_host = recon.fdk_host(*args, config=recon.FDKHostConfig(slices_per_batch=4))`.

## Held-out detector columns

Use a solver with observation weights so the image fit never sees the held-out
columns. Freeze the guards before fitting and use unfiltered observations.

```python
from tomojax.evaluation import Split
split = Split.detector(scan.projections.shape, start=2, width=2, period=8, guard=1)
weights = np.broadcast_to(split.train, scan.projections.shape).astype("float32")
x, info = recon.spdhg(*args, weights=weights, config=recon.SPDHGConfig(iterations=10))
operator = tj.Projector(scan.geometry)
prediction = operator(x)
gradient = operator.adjoint(prediction - scan.projections)
score = split.score(prediction, scan.projections)
(score.mse, score.relative_l2)
```

## Differentiable forward model

On an unrolled parallel detector, measure a translation along lab x. The
single pose direction is a derivative of translation per geometry length unit.

```python
import jax.numpy as jnp
from tomojax import forward, geometry
poses = geometry.stack_view_poses(scan.geometry, len(scan.angles))
target = jnp.asarray(scan.projections)
inputs = (x, poses, scan.grid, scan.detector)
predicted = forward.project_joseph(*inputs)
loss, gradients = forward.joseph_l2_value_and_grad(x, poses, target, scan.grid, scan.detector)
directions = jnp.zeros((*poses.shape, 1)).at[:, 0, 3, 0].set(1)
blocks = forward.joseph_pose_normal_equations(x, poses, directions, target, scan.grid, scan.detector)
```
