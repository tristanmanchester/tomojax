# Direct inverses and matched projection

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
