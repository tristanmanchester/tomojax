# Large scans and several GPUs

This page says which parts of a reconstruction must fit in GPU memory, how to
reconstruct scans that do not fit, and how to share a scan among several GPUs.
Streaming from host memory applies to projections held as NumPy arrays or
memmaps, which is what `tj.load` returns. A JAX array already on the GPU stays
there.

## GPU allocator

Importing `tomojax` defaults JAX to on-demand allocation rather than an eager
75% GPU reservation, leaving room for cuSolver and CuPy workspaces. The pool
retains memory for reuse; Python keeps JAX's default pool limit, while GPU CLI
commands set a 90% ceiling. Explicit
`XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_CLIENT_MEM_FRACTION` (or JAX's legacy
`XLA_PYTHON_CLIENT_MEM_FRACTION`) and `XLA_PYTHON_CLIENT_ALLOCATOR` take precedence; the align
CLI defaults to the releasing `platform` allocator. These process-wide
settings cannot change an already initialized JAX backend: import TomoJAX or
set the environment before the first JAX operation in a notebook.

## What has to fit on the GPU

| Method | Projections | Volume |
|---|---|---|
| FBP (parallel and laminography) | Streamed from host memory a batch of views at a time | Moves to host slabs automatically when it does not fit |
| Fourier (axial parallel only) | Host arrays or memmaps, read by row slabs | Always returned in host slabs |
| FDK (cone beam) | Read from host memory a slab's detector rows at a time when the volume is large | Moves to host z-slabs automatically when it does not fit |
| CGLS | Streamed when resident data and solver state would exceed free GPU memory | Cone and CUDA linear Joseph parallel/laminography use host slabs automatically when the working state does not fit |
| FISTA-TV | Streamed under the same rule; host-volume sweeps keep a detector-row ring | CUDA Joseph moves unmasked ordinary-TV state to host slabs when it does not fit |
| SPDHG-TV | Streamed under the same rule, with its sinogram-sized dual and any weights also kept in host memory | On the GPU, with about eight volumes |

`stream_projections = true` or `false` in the method's configuration (a
`--config` key) overrides the 40% rule. Streaming needs the batched projection
operators: CGLS streams only with the canonical detector grid, and FISTA-TV and
SPDHG-TV not with an explicit detector grid or the exact ray integrator.
Streamed CGLS solves the equivalent normal equations, which square the
condition number of each step's recurrence. It recomputes the exact gradient
periodically, as in-core CGLS recomputes its residual.

CUDA Joseph FISTA with ordinary TV, no support mask and no ordered-subset
initializer switches to host slabs when its estimated resident state will not
fit. It retains the same global objective, power estimate and finite-iteration
TV proximal; the planes are not reconstructed independently. Three host volumes
hold the iterate, extrapolate and gradient. If they would occupy more than half
of available host RAM, they are anonymous temporary files in the standard
`TMPDIR` directory. Use a fast local disk with room for the complete state;
the returned NumPy memmap owns its unlinked file until its last reference is
released. Passing `out=` to `tj.reconstruct(scan, "fista", out=volume)`
forces this route and fills the supplied float32 array or memmap.
`info["host_slabs"]` identifies this storage route. The exact TV halo
uses `tv_prox_iterations` planes on each side, so very large proximal iteration
counts can still exhaust device workspace. Huber-TV, masks, ordered initialization
and multi-device FISTA still require resident volume state.

Resident CUDA solvers keep their volumes in a private z-contiguous
order and convert once at entry and exit, which needs one volume-sized copy
at those two points. `tj.project`,
`tj.backproject` and `tj.Projector` make the same copy when one more volume fits
in free GPU memory and otherwise read the `(z, y, x)` array in place; the result
is the same to the bit. An axial parallel-beam projection never needs the copy:
it reads 32 planes at a time. The
[memory measurements](measurements.md#memory) give peak use for 512³
laminography on an 8 GB GPU.

## Reconstruct a volume larger than the GPU

### Cone beam

`tj.reconstruct(scan)` and `tomojax recon` (FDK, the default method on cone
data) switch to host z-slabs when the volume would take more than 40% of the
free GPU memory. The volume then comes back as a NumPy array, and
`Reconstruction.info["host_slabs"]` is true. Each slab filters only the
detector rows it projects onto. The `tomojax` commands still hold the whole
projection stack and volume in host RAM; to write the volume straight to disk,
pass a memmap through `out=`:

```python
import numpy as np

import tomojax as tj

scan = tj.load("scan.nxs")
grid = scan.grid
volume = np.lib.format.open_memmap(
    "volume.npy", mode="w+", dtype=np.float32, shape=(grid.nz, grid.ny, grid.nx)
)
recon = tj.reconstruct(scan, "fbp", out=volume)
volume.flush()
```

For explicit slab-depth control, use `fdk_host` with
`FDKHostConfig(slices_per_batch=...)`; by default slabs take about a third of
the free GPU memory. FDK accepts per-view cone geometry, including
`ConeSegments`; noncircular trajectories retain FDK's approximation.

### Parallel beam and laminography

CUDA linear Joseph `tj.reconstruct(scan, "cgls")` automatically uses host slabs
when its working volumes exceed device memory. Slabs are summed before forming
the data residual: this remains one global solve, including for tilted scans.
The full projection stack must fit on the GPU alongside a few slabs. Ray and
cubic projectors, explicit detector lattices, and multi-device solves retain
the resident-volume requirement.

If the four CGLS volumes exceed half of available host RAM, automatic storage
uses unlinked temporary files. Choose a fast disk with enough free space using
the standard `TMPDIR` environment variable. The returned NumPy memmap owns its
file until released. With explicit `out=`, the supplied array is preserved and
work files are placed beside it if it is a memmap.


`tj.reconstruct(scan, "fbp")` and `tomojax recon --method fbp` switch to host
slabs when the volume would take more than 40% of the free device memory.
`Reconstruction.info["host_slabs"]` records that choice. Passing `out=` forces
host slabs even for a small volume and fills the supplied float32 storage
without allocating another full volume:

```python
import numpy as np

import tomojax as tj

scan = tj.load("corrected-lamino.nxs")
grid = scan.grid
volume = np.lib.format.open_memmap(
    "volume.npy", mode="w+", dtype=np.float32, shape=(grid.nz, grid.ny, grid.nx)
)
recon = tj.reconstruct(scan, "fbp", filter="ramp", out=volume)
volume.flush()
```

Built-in parallel scans use balanced axial slabs of at most 16 slices, reading
only the detector rows they need. Host outputs use C-ordered `(z, y, x)`
arrays, so xy-planes are contiguous. Use the default C order for memmaps too.

Laminography and posed scans use x-slabs,
each streaming and refiltering every view; those slabs use about half the free
GPU memory by default. For explicit slab-depth control, call `fbp_host` with
`FBPHostConfig(slices_per_batch=...)`. Host slabs are not differentiable.
Device projections remain resident and are sliced on the device; use NumPy
arrays or memmaps to stream projections from host memory. Axial CUDA FBP may
stage one slab's detector rows (up to 512 MiB and a quarter of free device
memory) to reduce host dispatch; filtering still uses bounded view batches.
Input and output must not overlap, and public `out=` does not support explicit detector grids.
`method="fourier"` also accepts `out=` for axial parallel scans. These storage
choices do not make raw-frame correction out of core: `frames.corrected()`
still returns a complete corrected projection stack in host memory.
Axial output can overlap one slab download/write with the next
computation when spare device memory permits. All writes finish before return;
a failure can leave a partly written volume. A 1024³ laminography volume takes about
[37 s on an 8 GB laptop GPU](measurements.md#scans-larger-than-the-gpu).

### Stream raw frames

For cone-beam FDK, `tj.reconstruct_slabs(frames, "fbp")` corrects only the
detector-row band needed by each output slab, using bounded device batches.
Neither the full corrected stack nor the whole output volume needs to fit in
host memory. One corrected row band and any retained blocks must fit in host
memory. Angular weights use the complete acquisition, not the row crop.

For an unposed axial parallel scan, `tj.reconstruct_slabs` corrects raw rows
and reconstructs them without storing the full corrected stack. Choose `"fbp"`,
`"fourier"`, `"sirt"` or `"mlem"`, with that method's configuration through
`config=` when needed; `warm_start=True` starts SIRT and MLEM from FBP. The whole
Diamond I12 scan, a 56.6 GB SIRT volume, took 24 minutes with
[1.4 GB of GPU memory](measurements.md#scans-larger-than-the-gpu).
For example, write owned blocks directly to plane-contiguous storage:

```python
import numpy as np

import tomojax as tj
from tomojax.corrections import Stripes

frames = tj.load_frames("raw.nxs")
grid = frames.geometry.grid
volume = np.lib.format.open_memmap(
    "volume.npy", mode="w+", dtype=np.float32,
    shape=(grid.nz, grid.ny, grid.nx),
)
with tj.reconstruct_slabs(frames, "fbp", corrections=[Stripes(), Stripes(51, varying=True)]) as slabs:
    for z_slice, block in slabs:
        volume[z_slice] = block.volume
volume.flush()
```

Each block retains its global physical grid, genuinely corrected source rows
(including interpolation neighbors), and correction records. Blocks remain
valid after iteration or context exit, but collecting them retains both their
volumes and projections. The context drains pending work on early exit or
failure. Only whole-row line-integral corrections are supported, not Paganin
or view rejection; keep the raw and calibration inputs unchanged in the context.

## Bin detectors finer than the grid

When the detector samples the rotation axis more finely than the voxels (the
pixel pitch, divided by the magnification in a cone beam, is smaller than a
voxel), the projectors do work the volume cannot use. `scan.binned(factor)`
averages `factor × factor` pixels, which cuts projection cost by `factor²`.
Pixels that do not fill a block at the edges are dropped, and the detector
centre moves to match. Iterative `tj.reconstruct` and `tj.align` warn, with the
factor to use, when the detector is at least twice as fine as the grid.
Binning is a Python step; the commands have no binning option. On the FIPS
walnut, binning 2 × 2 made least squares
[3.5 times faster for no loss of accuracy](measurements.md#fips-walnuts-real-lab-cone-beam-ct).

For alignment, stopping the coarse-to-fine schedule early has a similar
effect. See
[align large scans at reduced resolution](alignment-guide.md#align-large-scans-at-reduced-resolution).

## Use several GPUs

`devices=` shares a scan's views among GPUs in one process. Pass one device or
several; `jax.devices()` passes them all:

```python
import jax

import tomojax as tj

scan = tj.load("scan.nxs")
recon = tj.reconstruct(scan, "fista", nonnegative=True, devices=jax.devices())
result = tj.align(scan, devices=jax.devices())
```

Each GPU projects its own views and holds the whole volume. Their
backprojections are summed on the first GPU, so the result is the one-GPU
result up to the order of that sum. Several GPUs therefore add speed, not room
for a larger volume, with one exception: in `fdk_host` and FDK's automatic
slabs, each GPU reconstructs its own z-slabs (`FDKConfig(devices=...)` in
`fdk_host`'s `FDKHostConfig`).

- `tj.project`, `tj.backproject` and the `fbp` (including FDK), `cgls` and
  `fista` methods of `tj.reconstruct` take `devices=`. SPDHG-TV runs on one
  GPU, because each of its steps would sum a whole volume across the GPUs for
  one block of views, and `fbp_host` also runs on one.
- Shared projections are held in GPU memory, divided among the GPUs, and are
  never streamed from the host; asking for both raises `ValueError`.
- In `tj.align`, only `pose` mode's joint pose-and-volume update and cone-beam
  reconstruction steps are shared. Parallel-beam reconstruction steps, the
  alternating solver of the other modes and setup calibration run on the first
  GPU. A checkpoint written on some GPUs resumes on others.
- The GPUs must belong to one process; runs across several machines are not
  supported.

The `tomojax` commands have no device option and run on JAX's default device,
the first visible GPU (`CUDA_VISIBLE_DEVICES` chooses which). Call the Python
API to use several. On four H100 GPUs, 20 FISTA-TV iterations on the walnut ran
[3.3 times faster than on one](measurements.md#several-gpus).
