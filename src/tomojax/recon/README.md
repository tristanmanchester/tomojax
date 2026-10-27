# tomojax.recon

`tomojax.recon` provides the reconstruction routines behind
`tj.reconstruct(scan, method)`:

- `fbp` and `fdk` (cone beams), with `fbp_host` and `fdk_host` for volumes
  kept in host memory
- `fourier_reconstruct`
- `cgls`
- `cgls_multires`
- `fista_tv`
- `spdhg_tv`
- `sirt` (SIRT, OS-SART and SART)
- `mlem`, with ordered subsets in `MLEMConfig`
- `calibrate_cone_axis`, the cone-beam axis-offset and roll calibration

Each method's settings are a configuration class: `FBPConfig` (FBP and FDK),
`FourierConfig`, `CGLSConfig`, `FistaConfig`, `SPDHGConfig`, `SIRTConfig` and
`MLEMConfig`. Pass one as
`tj.reconstruct(scan, "fista", config=FistaConfig(regulariser="huber_tv"))`;
keywords such as `iterations=` replace the fields of the same name. The same
fields are the keys of a `tomojax recon --config` TOML file
(`tomojax recon --config-keys` lists them).

SIRT uses the exact matched projector pair with row/column normalization:
`SIRTConfig(iterations=50, subsets=1, lower_bound=0, upper_bound=None, support=None)`.
More subsets select interleaved OS-SART, with one full sweep per iteration;
one view per subset gives SART. Bounds apply after every subset update, and
voxels outside the support stay zero. Ordered subsets can cycle on inconsistent
data; reduce their number or relaxation for final refinement. Large axial
Joseph scans with detector rows on voxel planes run in exact independent
slabs when needed, returning a host NumPy volume.

CUDA cone CGLS, SIRT/OS-SART and MLEM/OSEM keep device-resident iterates in
private z-contiguous storage. Warm starts and support masks enter as public
`(z, y, x)` arrays; a tiled permutation packs them at the solve boundary and
restores the public output without changing the physical grid or the matched
forward/transpose pair. This also applies to CGLS with streamed projections
and a resident volume. Host-slab solvers and CPU execution use public storage;
the private layout is not a user option or a different reconstruction model.

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
Given a NumPy array or memmap, `fbp` streams view batches from host memory, so
only the volume and one batch occupy the device.

`tj.reconstruct` switches `fbp` to `fbp_host`, and FDK to `fdk_host`, when the
volume would take more than 40% of the free device memory on the first device;
`out=` forces it. `fbp_host` keeps the volume in host memory, for scans whose
volume does not fit on the device. Projections already on the device stay
there and are sliced in place. Built-in parallel geometry is processed in axial
slabs that read only the detector rows they need; laminography and other
geometry in x-slabs, each streaming every view batch once (filtering is repeated per slab).
With `slices_per_batch=None`, non-parallel slabs are sized to about half of
the free device memory, so a volume that fits takes one pass:

```python
from tomojax.recon import FBPHostConfig, fbp_host

volume = fbp_host(
    geometry, grid, detector, projections_numpy,
    config=FBPHostConfig(slices_per_batch=16, views_per_batch=32),
    out=output_memmap,  # optional writable FP32 array, shape (grid.nz, grid.ny, grid.nx)
)
```

Each slab uses the same compiled shape, including a final partial slab. Smaller
slabs reduce device storage but increase transfer and dispatch costs. This API
is not differentiable.
Input and output must not overlap, including separate mappings of the same file.
Completed slabs are written immediately, so a later exception can leave partial
output. Callers own memmap flushing. Device workspace and runtime caches remain
additional costs; batch dimensions are not a hard process-memory cap.

`fourier_reconstruct` (`tj.reconstruct(scan, "fourier")`) is an opt-in
Fourier-slice inverse for axial parallel half-turn scans, including
nonuniform encoder angles, rejected views and an inclusive 0 to 180 degree
range; views of one direction share an angular cell and are averaged. It accepts
reordered, reversed and opposing views, offset detectors, anisotropic spacing
and cropped output grids. Per-view translations are honored; rotational pose
corrections and tilted axes are not supported. Projection FFTs use six-point Kaiser–Bessel radial
interpolation, linear interpolation in the measured angle (periodic over a half
turn) and a detector-Nyquist disk cutoff, followed by a
padded Cartesian inverse FFT. The padding covers both the output grid and the
acquired detector field to avoid wrapping an object outside a cropped ROI back
into it. Missing detector data are assumed zero; this does not correct truncation.

```python
from tomojax.recon import FourierConfig, fourier_reconstruct

volume = fourier_reconstruct(
    geometry, grid, detector, projections_numpy,
    config=FourierConfig(),  # backend and slab depth selected automatically
    out=output_memmap,  # optional writable FP32 host storage
)
```

Install `tomojax[fourier-cuda12]` for the optional CuPy implementation on a host
with a working CUDA 12 runtime/toolkit and NVIDIA driver. The default
`backend="auto"` uses CUDA when it matches the selected JAX device, otherwise
NumPy's portable double-precision FFT reference. CUDA uses single-precision
FFTs and a real inverse FFT to store only half the
Cartesian spectrum. Angular interpolation fractions retain double-precision
coordinates before FP32 accumulation. Both return FP32 host arrays and process fixed axial
slabs, including a padded final batch. The same storage lifetime, nonoverlap and
partial-output rules as `fbp_host` apply. No amplitude fitting or clipping is
performed. Large CUDA jobs overlap bounded host work and transfers on reused
device queues. Default CUDA slab depth targets a 256 MiB padded FP32 plane stack (at least one and at most sixteen slices); NumPy defaults to sixteen.
An explicit integer `slices_per_batch` overrides this choice. Additional buffers
and FFT workspaces need more memory, so CUDA exhaustion still halves the depth.
This API is not differentiable and rejects tilted/custom geometry. Large
missing angular sectors are interpolated, not recovered, and the angular
interpolation can introduce artefacts on sharp or undersampled scans; the
[measurements](../../../docs/measurements.md#fourier-inverse) give its memory
and speed, and the [reconstruction comparison](../../../docs/measurements.md#reconstruction-against-astra)
its accuracy-gated results.

`cgls` solves unconstrained least squares with a matched discrete adjoint and
optional scalar damping and quadratic smoothness. It supports JAX and CUDA execution. Iteration
budgets are dynamic, so changing the budget reuses compiled code. Use FISTA or
SPDHG for TV regularization and nonnegativity constraints.

For resident, unmasked parallel or laminographic nonnegative least squares,
`FistaConfig(initial_epochs=8, initial_subsets=16, iterations=10, tv_weight=0,
nonnegative=True)` optionally initializes with ordered subsets before the full-data
iterations; it is off by default, and more subsets can worsen image quality.

```python
from tomojax.recon import CGLSConfig, cgls

volume, info = cgls(
    geometry, grid, detector, projections,
    config=CGLSConfig(iterations=50, rtol=1e-6, damping=0.0),
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
measurement. Explicit per-level budgets override `config.iterations`:

```python
from tomojax.recon import cgls_multires

volume, info = cgls_multires(
    geometry, grid, detector, projections,
    factors=(4, 2, 1), iterations_per_level=(40, 20, 20),
    config=CGLSConfig(rtol=1e-6),
)
```

`info["levels"]` records every level's physical grid, detector, requested and
effective iterations, and termination. `effective_iterations` counts all
levels; `fine_effective_iterations` counts only the final level. Coarsening preserves physical
volume faces and selects actual measured rays at uniform detector strides. It
does not average detector pixels or rescale attenuation. This can accelerate
smooth problems but discards coarse-stage measurements and can alias sharp or
noisy data. The default schedule is not an automatic image-quality criterion.
Each new level shape can require compilation. A numerical breakdown at any
level raises `FloatingPointError`.

CGLS, FISTA-TV and SPDHG-TV share one projector choice. The default
`projector_model="auto"` uses bilinear voxel-centre plane integration (Joseph)
with CUDA kernels on CUDA; its matched CUDA transpose gathers into voxels
without scattered atomic writes, and JAX provides an independently
differentiated reference transpose. `projector_model="ray"` selects the
trilinear ray marcher, which is also used automatically with explicit detector
grids and, in FISTA and SPDHG, the exact ray integrator. Both models match
analytic line integrals to the same accuracy; Joseph is several times faster on
CUDA. Joseph requires rigid homogeneous poses, finite grid/detector placement
and canonical detector coordinates. `info["projector_model"]` identifies the CGLS
selection; multiresolution diagnostics retain it per level. The matching public
forward API exposes first-order pose differentiation; the solvers return host
diagnostics and are not differentiable layers. Alignment's FISTA
reconstruction stages keep the ray model on parallel beams; its coupled pose
solver samples with Joseph by default (`ray_integrator`).

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

Cone-beam CGLS, SIRT and MLEM switch to slab sweeps when the solver's working
set would not fit on the device, or when `out=` is given. Each z slab of the
volume is projected onto, and transposed from, the band of detector rows its
rays can reach, with one set of per-view coefficients, so the slabs sum to the
exact matched projector pair and the results match the device-resident solvers
to FP32 rounding. A sweep projects the slabs in order into a ring of those rows
on the device and transposes each slab as soon as every slab reaching its rows
has been projected; no projection-sized array is stored. CGLS then solves the
normal equations with four host volumes (iterate, normal residual, direction
and its normal image), reading the measurements only for its initial and
recomputed residuals. SIRT, OS-SART and OSEM update one volume a slab at a
time, once per subset: it stays on the device when it fits there beside the
sweep (the result is then a JAX array), and otherwise in host memory. Host
results are C-ordered `(z, y, x)` NumPy volumes, or `out` itself:

```python
import numpy as np
from tomojax.recon import SIRTConfig, sirt

out = np.lib.format.open_memmap(
    "volume.npy", "w+", np.float32, (grid.nz, grid.ny, grid.nx)
)
volume, info = sirt(
    geometry, grid, detector, projections, config=SIRTConfig(subsets=8), out=out
)
```

With a memory-mapped `out`, CGLS keeps its three work volumes in temporary
files in the same directory, and pages of mapped volumes are released after
each slab so they do not stay resident. SIRT from a zero start never reads
what `out` held: its first sweep overwrites it. The ring holds every view of a sweep
over the tallest band of rows a slab reaches; when it cannot fit on the
device even for one-plane slabs, the solve raises `MemoryError` and more SIRT
subsets reduce it. `info` records `slices_per_batch` and the bytes moved to
and from the device.

`mlem` models nonnegative linear Poisson measurements, not raw transmission
counts. `MLEMConfig(subsets=1)` is MLEM; larger values give angle-interleaved
OSEM, with `iterations` counting full passes through the views. OSEM can cycle
near convergence. The boolean `support` mask fixes excluded voxels to zero;
initial zeros are absorbing under multiplicative updates. `warm_start=True`
starts from FBP/FDK clipped to nonnegative values, retaining its zero set.

