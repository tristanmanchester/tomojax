# tomojax.recon

`tomojax.recon` provides reconstruction routines:

- `fbp`
- `fbp_host`
- `fourier_reconstruct`
- `cgls`
- `cgls_multires`
- `fista_tv`
- `spdhg_tv`

Built-in parallel `fbp` zero-extends measured detector rows far enough to retain
the ramp filter's tails at every output voxel. Zero raw data outside a detector
do not imply zero filtered data there. This prevents corner bias when the output
grid extends beyond the detector's inscribed field of view, without changing
measured coordinates or fitting output amplitude. It assumes zero unmeasured
attenuation and is not an object-truncation correction. Generic filtered-adjoint
reconstruction, including explicit detector grids, uses the supplied support.

Direct FBP pads, filters and backprojects one view batch at a time. An internal
512 MiB FFT-workspace estimate selects a power-of-two batch for large stacks;
small stacks stay in one batch. This estimate is not a hard process-memory cap:
input projections, output/accumulator volumes, compiler workspaces and runtime
caches remain additional costs. The final partial batch counts each view once.

`fbp_host` reconstructs built-in parallel geometry from a NumPy array or memmap
into host memory, processing axial slabs without putting the full projection
stack or volume on the device. Slabs retain the detector rows needed for linear
interpolation and the same horizontal filter tails as `fbp`:

```python
from tomojax.recon import FBPHostConfig, fbp_host

volume = fbp_host(
    geometry, grid, detector, projections_numpy,
    config=FBPHostConfig(slices_per_batch=16, views_per_batch=32),
    out=output_memmap,  # optional writable FP32 array, shape (grid.nx, grid.ny, grid.nz)
)
```

Each slab uses the same compiled shape, including a final partial slab. Smaller
slabs reduce device storage but increase transfer and dispatch costs. This API
is not differentiable and rejects tilted/custom geometries and JAX input arrays.
Input and output must not overlap, including separate mappings of the same file.
Completed slabs are written immediately, so a later exception can leave partial
output. Callers own memmap flushing. Device workspace and runtime caches remain
additional costs; batch dimensions are not a hard process-memory cap.

`fourier_reconstruct` is an opt-in Fourier-slice inverse for built-in parallel
scans with uniform, unique angles modulo 180 degrees. It accepts reordered,
reversed and opposing views, offset detectors, anisotropic spacing and cropped
output grids. Projection FFTs use six-point Kaiser–Bessel radial interpolation,
linear angular interpolation and a detector-Nyquist disk cutoff, followed by a
padded Cartesian inverse FFT. The padding covers both the output grid and the
acquired detector field to avoid wrapping an object outside a cropped ROI back
into it. Missing detector data are assumed zero; this does not correct truncation.

```python
from tomojax.recon import FourierConfig, fourier_reconstruct

volume = fourier_reconstruct(
    geometry, grid, detector, projections_numpy,
    config=FourierConfig(slices_per_batch=16, backend="cupy"),
    out=output_memmap,  # optional writable FP32 host storage
)
```

Install `tomojax[fourier-cuda12]` for the optional CuPy implementation on a host
with a working CUDA 12 runtime/toolkit and NVIDIA driver. The
default `backend="numpy"` provides a portable, double-precision FFT reference;
CUDA uses single-precision FFTs and a real inverse FFT to store only half the
Cartesian spectrum. Angular interpolation fractions retain double-precision
coordinates before FP32 accumulation. Both return FP32 host arrays and process fixed axial
slabs, including a padded final batch. The same storage lifetime, nonoverlap and
partial-output rules as `fbp_host` apply. No amplitude fitting or clipping is
performed. Large CUDA jobs overlap bounded host work and transfers on reused
device queues. Smaller slabs can reduce memory at a throughput cost; the
measured 512³ case uses 430 MiB at eight slices versus 678 MiB at sixteen.
This API is not differentiable, is not selected by the CLI, and rejects
tilted/custom geometry and nonuniform or repeated half-turn angles. Its angular
interpolation can introduce artifacts on sharp or undersampled scans; use the
independent quality measurements in [performance](../../../docs/performance.md)
to assess the validated scope.

`cgls` solves unconstrained least squares with a matched discrete adjoint and
optional scalar damping and quadratic smoothness. It supports JAX and CUDA Pallas execution. Iteration
budgets are dynamic, so changing the budget reuses compiled code. Use FISTA or
SPDHG for TV regularization and positivity constraints.

```python
from tomojax.recon import CGLSConfig, cgls

volume, info = cgls(
    geometry, grid, detector, projections,
    config=CGLSConfig(iters=50, rtol=1e-6, damping=0.0),
)
```

`info` reports the actual backend, effective iterations, normal-residual norm,
and termination reason. Convergence measures stationarity, not scientific image
quality. Before reporting convergence, the solver recomputes `y - A x` and
its normal gradient. It restarts the direction if recursive residual drift
would otherwise indicate premature convergence. `roundoff_limit` indicates
voxelwise update stagnation or a componentwise FP32 cancellation estimate
before the requested normal-residual tolerance was met. The last valid volume
is retained; the solver does not claim convergence in that case. The public
CGLS routine is not a differentiable reconstruction layer.

With both penalties disabled, zero-start CGLS converges toward the minimum
Euclidean-norm least-squares solution. If you supply `init_x`, its component
that produces zero projection signal is retained, up to floating-point error.
Measurements cannot determine those image features. In sparse or tilted scans,
different initial volumes can therefore give different images with equally
small data residuals. This also applies to the initialization passed between
multiresolution levels.

`CGLSConfig(gradient_damping=...)` adds squared physical first differences to
the objective:
`||A x - y||² + damping² ||x||² + gradient_damping² ||D x||²`.
`D` divides adjacent voxel differences by their physical spacing. Only edges
inside the volume contribute; constant volumes have zero smoothness penalty,
including at the boundary. This option defaults to zero and adds no operator
work when disabled. Both weights act on an unnormalized data sum, so changing
the number of views can require choosing new weights. `info` reports both
weights, and multiresolution applies the same settings on each physical grid.
This quadratic prior suppresses variation; it can bias shapes and pose estimates
and does not itself guarantee better recovery.

`cgls_multires` uses coarse CGLS solutions to initialize progressively finer
solves. The final factor must be 1, so the final solve uses every original
measurement. Explicit per-level budgets override `config.iters`:

```python
from tomojax.recon import cgls_multires

volume, info = cgls_multires(
    geometry, grid, detector, projections,
    factors=(4, 2, 1), iters_per_level=(40, 20, 20),
    config=CGLSConfig(rtol=1e-6),
)
```

`info["levels"]` records every level's physical grid, detector, requested and
effective iterations, and termination. `effective_iters` counts all levels;
`fine_effective_iters` counts only the final level. Coarsening preserves physical
volume faces and selects actual measured rays at uniform detector strides. It
does not average detector pixels or rescale attenuation. This can accelerate
smooth problems but discards coarse-stage measurements and can alias sharp or
noisy data. The default schedule is not an automatic image-quality criterion.
Each new level shape can require compilation. A numerical breakdown at any
level raises `FloatingPointError`.

CGLS, FISTA-TV and SPDHG-TV share one projector choice. The default
`projector_model="auto"` uses bilinear voxel-centre plane integration (Joseph)
with Pallas kernels on CUDA; its matched CUDA transpose gathers into voxels
without scattered atomic writes, and JAX provides an independently
differentiated reference transpose. `projector_model="ray"` selects the
trilinear ray marcher, which is also used automatically with explicit detector
grids and, in FISTA and SPDHG, the exact ray integrator. Both models match
analytic line integrals to the same accuracy; Joseph is several times faster on
CUDA. Joseph requires rigid homogeneous poses, finite grid/detector placement
and canonical detector coordinates. `info["projector_model"]` identifies the CGLS
selection; multiresolution diagnostics retain it per level. The matching public
forward API exposes first-order pose differentiation; the solvers return host
diagnostics and are not differentiable layers. Alignment keeps the ray model
for its internal reconstructions, matching its pose objective.

Set `joseph_interpolation="cubic"` with that model to use Keys cubic convolution
(a=-1/2) and its matched transpose on either backend. The default is `"linear"`.
Cubic uses a 4-by-4 transverse stencil, including negative weights, and costs
more than bilinear sampling. It matches `project_joseph(interpolation="cubic")`;
select the same interpolation in a reconstruction and its pose objective. This
option is rejected with `projector_model="ray"`. It also applies at every level
of `cgls_multires`, with its value retained in each level's diagnostics.


`normal_residual_is_recomputed` distinguishes directly evaluated diagnostics
from the recursively updated norm at an iteration limit. `residual_recomputations`
counts additional checks beyond initialization. Small global updates trigger
a check, but cannot alone stop a solve: a bright, already-correct region must
not set the precision limit for an independent weak region. The cancellation
estimate is `8 * eps32 * (abs(A).T @ (abs(y) + abs(Ax)) + damping**2 * abs(x))`;
for nonnegative linear weights `abs(A)` is `A`. Cubic uses an explicit
absolute-weight transpose so its negative lobes cannot cancel this bound.
With quadratic smoothness, the estimate also includes
`8 * eps32 * gradient_damping**2 * abs(D.T @ D) @ abs(x)`;
it diagnoses numerical stagnation without relaxing `rtol` or `atol`. A residual
replacement resets the conjugate direction. These checks add operator work near
stopping, while ordinary fixed-budget updates retain the existing operator pair.
Near the FP32 noise floor, checks also run every 16 iterations: an atomic JAX
transpose can keep the recurrence moving and evade an update-stagnation check.
The componentwise cancellation test and requested convergence tolerance remain
unchanged; a precision-limited result is still reported as `roundoff_limit`.
