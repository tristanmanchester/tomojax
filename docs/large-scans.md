# Large scans and several GPUs

This page says which parts of a reconstruction must fit in GPU memory, how to
reconstruct scans that do not fit, and how to share a scan among several GPUs.
Streaming from host memory applies to projections held as NumPy arrays or
memmaps, which is what `tj.load` returns. A JAX array already on the GPU stays
there.

## What has to fit on the GPU

| Method | Projections | Volume |
|---|---|---|
| FBP (parallel and laminography) | Streamed from host memory a batch of views at a time | Moves to host slabs automatically when it does not fit |
| Fourier (axial parallel only) | Host arrays or memmaps, read by row slabs | Always returned in host slabs |
| FDK (cone beam) | Read from host memory a slab's detector rows at a time when the volume is large | Moves to host z-slabs automatically when it does not fit |
| CGLS | Streamed when they would take more than 40% of free GPU memory | On the GPU, with about five volume-sized work arrays |
| FISTA-TV | Streamed under the same rule | On the GPU, with about seven volumes |
| SPDHG-TV | Streamed under the same rule, with its sinogram-sized dual and any weights also kept in host memory | On the GPU, with about eight volumes |

`stream_projections = true` or `false` in the method's configuration (a
`--config` key) overrides the 40% rule. Streaming needs the batched projection
operators: CGLS streams only with the canonical detector grid, and FISTA-TV and
SPDHG-TV not with an explicit detector grid or the exact ray integrator.
Streamed CGLS solves the equivalent normal equations, which square the
condition number of each step's recurrence. It recomputes the exact gradient
periodically, as in-core CGLS recomputes its residual.

The iterative solvers need the whole volume and their work arrays on the GPU;
a volume larger than the GPU cannot be reconstructed iteratively. The
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
    "volume.npy", mode="w+", dtype=np.float32, shape=(grid.nx, grid.ny, grid.nz)
)
recon = tj.reconstruct(scan, "fbp", out=volume)
volume.flush()
```

For explicit slab-depth control, use `fdk_host` with
`FDKHostConfig(slices_per_batch=...)`; by default slabs take about a third of
the free GPU memory. FDK refuses multi-orbit
`ConeSegments` scans; reconstruct those with an iterative method.

### Parallel beam and laminography

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
    "volume.npy", mode="w+", dtype=np.float32, shape=(grid.nx, grid.ny, grid.nz)
)
recon = tj.reconstruct(scan, "fbp", filter="ramp", out=volume)
volume.flush()
```

Built-in parallel scans use balanced axial slabs of at most 16 slices, reading
only the detector rows they need. For large disk-backed axial FBP output, pass
`fortran_order=True` to `open_memmap`: contiguous xy-planes avoid repeatedly
dirtying pages across the whole file as successive z-slabs are written. This
changes storage order, not the volume's `(x, y, z)` shape or coordinates.
Keep the default C order for laminography's x-slabs.

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
Axial F-order output can overlap one slab download/write with the next
computation when spare device memory permits. All writes finish before return;
a failure can leave a partly written volume. A 1024³ laminography volume takes about
[37 s on an 8 GB laptop GPU](measurements.md#scans-larger-than-the-gpu).

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
