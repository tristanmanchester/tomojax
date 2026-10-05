# tomojax.forward

## Purpose

`tomojax.forward` provides differentiable projection, projection-domain
residuals, and geometry-parameter reductions.

The projector adapts `GeometryState` to core `Grid`, `Detector`, detector
grids, and per-view 4x4 `T_all`, then projects with
`tomojax.core.projector.forward_project_view_T`.

## Public API

- `project_parallel_reference`
- `project_parallel_reference_from_input`
- `ProjectionArrayGeometryInput`
- `ProjectionOperatorName`
- `core_projection_geometry_from_state`
- `core_projection_geometry_from_input`
- `nominal_axis_unit_from_geometry`
- `CoreProjectionGeometry`
- `PROJECTION_OPERATOR`
- `project_joseph`
- `joseph_l2_value_and_grad`
- `joseph_pose_normal_equations`
- `apply_residual_filter`
- `apply_residual_filter_schedule`
- `ResidualFilterConfig`
- `ResidualFilterKind`
- `ResidualFilterResult`
- `masked_whitened_residual`
- `pseudo_huber_loss`
- `pseudo_huber_weights`
- `residual_loss`
- `ResidualResult`

## Plane-sampled projection

`project_joseph(volume, poses, grid, detector, backend="jax")` exposes the same
default linear Joseph discretization the reconstruction solvers use by default.
Set `interpolation="cubic"` on either Joseph function for Keys cubic convolution
with parameter -1/2. The 4-by-4 transverse stencil has continuous first
coordinate derivatives and includes negative weights. It is more expensive and
can overshoot nonnegative data. Both models have matching CUDA volume/pose
derivatives and a matched volume transpose; dominant-axis switches remain
nonsmooth. To reconstruct with the cubic model, use
`CGLSConfig(projector_model="joseph", joseph_interpolation="cubic")`.
By default it samples voxel-centre planes along the dominant ray direction,
bilinearly interpolates the other two coordinates with zero extension, and multiplies by
physical path length between planes. This differs from the default trilinear
ray operator. Supply finite rigid world-from-object matrices `(views, 4, 4)`;
voxel origins and detector centres retain their physical units.

The explicit `backend="pallas"` runs on NVIDIA CUDA and supports first-order
`jax.grad`, `jax.jvp`, `jax.jacfwd`, `jax.jacrev`, `jax.linearize`, transposing
that linearization, and `jax.vmap`. The input volume and geometry are retained
for reverse mode; the kernel recomputes sample derivatives without saving a
ray-by-plane tape. Use the JAX backend for higher derivatives. Interpolation
knots for the linear model and dominant-axis changes are nonsmooth; derivatives follow the branch
selected at the supplied pose. Backend selection does not silently fall back.

For raw least squares, the fused operation avoids a separate projection pass
when computing pose gradients:

```python
import jax
from tomojax.forward import joseph_l2_value_and_grad, project_joseph

project = jax.jit(
    lambda volume, poses: project_joseph(volume, poses, grid, detector, backend="pallas")
)
loss_and_grad = jax.jit(
    lambda volume, poses, data: joseph_l2_value_and_grad(
        volume, poses, data, grid, detector, backend="pallas"
    )
)
prediction = project(volume, poses)
loss, (volume_grad, matrix_grad) = loss_and_grad(volume, poses, data)
```

The loss is `0.5 * sum((prediction - data)**2)`, without mean normalization,
masking, or fitted scale. To optimize rigid pose parameters, pull `matrix_grad`
through the function that builds your pose matrices; matrix entries themselves
are not unconstrained rigid-pose parameters. Other differentiable losses can
compose `project_joseph` with ordinary JAX operations. The existing alignment
pipeline still uses its default trilinear model; these APIs do not change its
discretization automatically.


`joseph_pose_normal_equations(volume, poses, pose_directions, data, grid, detector,
backend="pallas", interpolation="cubic")` computes per-view raw least-squares
loss, parameter gradient, Gauss–Newton matrix and residual images. Supply
`pose_directions` as `(views, 4, 4, P)`, for 1–16 matrix-pose directions in your
parameter units. For a function `make_pose(parameters, nominal_pose)`,
`jax.vmap(jax.jacfwd(make_pose))(parameters, nominal_poses)` provides those
directions. The outputs have shapes `(views,)`, `(views, P)`, `(views, P, P)`
and `(views, nv, nu)`.

The CUDA kernel computes the small normal matrices directly while traversing
planes. It retains a residual for line search, without storing a full projection
Jacobian or a ray-by-plane tape. This returns `J.T @ J`, not the full loss Hessian.
Damping, parameter steps, priors and gauge constraints remain caller choices.
Shared global parameters can sum these per-view contributions. CUDA supplies
first-order information; the ordinary JAX reference supports further AD.

## Dependencies

Allowed: `tomojax.core`, `tomojax.geometry`, `tomojax.motion`,
`tomojax.nuisance`, `tomojax.backends`.

Forbidden: private files from other modules, reconstruction/alignment
orchestration, Pallas fast paths without JAX-reference equivalence tests.

## Invariants

- Projection residuals support masks and robust whitening.
- Backend fast paths report provenance and compare against the reference path.
- Supported DOFs: nominal theta, theta scale/offset, per-view alpha/beta/phi
  residuals, detector u/v shift, detector roll, axis x/y tilt, per-view dx/dz.
- Detector roll applies around the zero-centre detector plane; detector centre
  offsets are independent.
- Axis rotations use the core rotation-axis pose convention: nominal axis from
  acquisition metadata, x/y setup corrections on top, `T_all` built with
  `axis_pose_stack`.
- Alpha/beta pose rotations compose after nominal axis/theta in object
  coordinates.
- Parallel laminography uses a tilted nominal rotation axis with the same
  projector.
- Residual filter policies: `raw`, `lowpass_gaussian`,
  `bandpass_difference_of_gaussians`.

## Tests

- `tests/test_numerical_engines.py` verifies the public grouped-input projection
  path and core numerical invariants used by reconstruction.
- `tests/test_product_surface.py` verifies the public module import surface.
- `tests/test_joseph_forward.py`, `tests/test_joseph_derivatives.py` and
  `tests/test_joseph_projector.py` cover the public plane model, independent
  physical matrices/finite differences, first-order transformations and CUDA
  equivalence.
