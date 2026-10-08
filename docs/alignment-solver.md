# Alignment solver reference

Use this reference when changing `AlignConfig` in Python. For data preparation,
mode selection, and interpreting recovered parameters, start with the
[alignment guide](alignment-guide.md). These options are experimental; the
[public comparison](research/public-free-voxel-schur-2026-10-04.md) still has a failed cell.
The CLI does not expose every Python solver option described here.

## Gauss–Newton updates at interpolation boundaries

The default `gn_jacobian="autodiff"` differentiates the trilinear ray model
within its current interpolation cells. At a voxel-grid boundary this chooses
a one-sided derivative. For sharp objects, an update can enter a wrong local
minimum even for a small detector shift.

Set `gn_jacobian="central"` to form a symmetric finite-difference Jacobian for
the Gauss–Newton step. The forward prediction, acceptance loss, projector AD,
and matched volume adjoint stay unchanged. This option costs ten projection
evaluations per view for the five Jacobian columns, plus the residual and
candidate checks; it is a recovery option, not a derivative-speed claim.
`gn_difference_step=1e-3` is a fraction of the smallest physical voxel pitch.
Rotation increments use that displacement divided by the grid's bounding
sphere radius, so the stencil scales with physical size and multiresolution
level. Both settings are saved in checkpoint configuration. This local stencil
does not guarantee recovery from large initial motion or poor identifiability.

Small rigid-transform products and Gauss–Newton normal equations explicitly
request full FP32 multiplication precision. This prevents reduced global JAX
matrix-multiply settings from quantizing small pose updates on CUDA.

The Python API also has an experimental `gn_coupling="joint"` option for
per-view least-squares alignment. It solves a damped linearized volume-and-pose
problem after each reconstruction refresh, then accepts the updated pair only
after scoring its constrained nonlinear objective. This addresses slow
alternation when free voxels can compensate for pose errors. The default
remains `gn_coupling="fixed_volume"`. In `align_multires`, the implicit schedule
uses a `joint_volume_pose` stage when this option is enabled. Explicitly
fixed-volume schedule stages continue to keep the volume fixed; a custom joint
stage must declare `objective_kind="joint_volume_pose"` and `optimizer="gn"`.

Joint GN requires `opt_method="gn"`, `pose_model="per_view"` and
`gather_dtype="fp32"`. It supports sampled and exact integration, active/frozen
pose parameters, pose bounds, volume masks, nonnegative voxels, pose smoothness,
and Huber-TV or zero volume regularization. It does not support nonsmooth TV
or polynomial/spline pose models. `gn_joint_iters=40` bounds the inner
matrix-free PCG solve; `gn_joint_rtol=1e-4` controls its residual stopping
criterion. `gn_damping` and `gn_volume_damping` damp pose and voxel increments
respectively; they are not priors on the recovered object. The returned
`joint_linear_relative_residual` is recomputed explicitly, so reaching the
iteration limit does not imply an accurate linear solve. The optional pose
column cache is capped at 64 MiB; larger scans recompute columns per view.

Repeated joint calls reuse compiled objectives when array shapes and solver
options match. Measurements, nominal poses, masks, detector coordinates and
weights remain inputs to each call, so a new scan can reuse a program without
reusing the previous scan's values. Changing shapes or program options still
requires compilation; this reuse is separate from the optional JAX persistent
cache for different processes. The [six-cell reuse comparison](research/public-free-voxel-reuse-2026-10-04.md)
measures faster warm calls with unchanged cold startup and process GPU memory;
noisy anisotropic recovery still fails its rotation gate.

The experimental `gn_joint_solver="pose_eliminated"` factors the pose block
and runs PCG on the volume Schur system, then back-substitutes poses. It solves
the same damped linear problem as the default `"stacked"` method, with the
same joint residual threshold and nonlinear acceptance. Without pose smoothness,
the factors are independent 5-by-5 blocks; with smoothness, a block-banded
factorization retains inter-view coupling with storage linear in view count.
Damping and iteration budgets are unchanged. Independent dense, constrained
workflow and CUDA checks pass. The [complete public comparison](research/public-free-voxel-schur-2026-10-04.md)
recovers five of six modest-motion clean/noisy cells, with faster accepted
tilted recovery but the same noisy anisotropic failure as the stacked solve.
Sampled GPU memory rises from 280 to 320 MiB. The option remains experimental
and is not the default; it does not establish complete recovery or the 20× goal.

Joint steps report `objective_kind="joint_volume_pose"`, their actual
projector backend, accepted line-search scale, and linear-solve diagnostics.
The shared central-difference stencil is unchanged. Recovery and performance
claims require the complete [free-voxel pilot](research/public-free-voxel-pilot.md);
passing the numerical tests alone does not establish them.

Alignment's `info["L"]` and checkpoint `L` retain the effective FISTA step
bound, including smooth-TV curvature. Reusing a bound does not add a safety
factor on each outer iteration. If reconstruction falls back to public FISTA,
the adapter removes the existing TV contribution before that solver adds it;
the bound therefore stays consistent across both paths and resume. A fallback
override too small to supply a positive data bound triggers operator-norm
estimation. The public `FistaConfig.lipschitz` itself remains a data-term bound.

Without an override, Huber-FISTA alignment initializes its data bound from
`max(A.T @ (A @ ones))`, with a 20% margin applied once. For the nonnegative
trilinear ray model this row sum bounds the normal operator's largest
eigenvalue. It costs one projection/adjoint pair and accounts for actual
physical sampling, including detector density and oblique rays. The old
dimension-only estimate did not scale with physical length units.
