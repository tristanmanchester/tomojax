# Large scans and several GPUs

This page says which parts of a reconstruction must fit in GPU memory, how to
reconstruct scans that do not fit, and how to share a scan among several GPUs.
Streaming from host memory applies to projections held as NumPy arrays or
memmaps, which is what `tj.load` returns. A JAX array already on the GPU stays
there.

## What has to fit on the GPU

| Method | Projections | Volume |
|---|---|---|
| FBP (parallel and laminography) | Streamed from host memory a batch of views at a time | On the GPU; use `fbp_host` when it does not fit |
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
call `fdk_host` with a memmap:

```python
import numpy as np

import tomojax as tj
from tomojax.recon import fdk_host

scan = tj.load("scan.nxs")
grid = scan.grid
volume = np.lib.format.open_memmap(
    "volume.npy", mode="w+", dtype=np.float32, shape=(grid.nx, grid.ny, grid.nz)
)
fdk_host(scan.geometry, grid, scan.detector, np.asarray(scan.projections, np.float32), out=volume)
volume.flush()
```

`FDKHostConfig(slices_per_batch=...)` sets the slab depth; by default slabs
take about a third of the free GPU memory. FDK refuses multi-orbit
`ConeSegments` scans; reconstruct those with an iterative method.

### Parallel beam and laminography

FBP does not switch to slabs on its own, and the command line has no slab
option. From Python, `fbp_host` reconstructs from host or memmapped
projections into host or memmapped output:

```python
import numpy as np

import tomojax as tj
from tomojax.recon import fbp_host

scan = tj.load("corrected-lamino.nxs")
grid = scan.grid
volume = np.lib.format.open_memmap(
    "volume.npy", mode="w+", dtype=np.float32, shape=(grid.nx, grid.ny, grid.nz)
)
fbp_host(scan.geometry, grid, scan.detector, scan.projections, out=volume)
volume.flush()
```

Parallel scans are processed in axial slabs that read only the detector rows
they need. Laminography and posed scans are processed in x-slabs, each
streaming and refiltering every view, so smaller slabs save GPU memory at the
cost of repeated filtering. By default slabs take about half the free GPU
memory, so a volume that fits is reconstructed in one pass;
`FBPHostConfig(slices_per_batch=...)` sets the depth. `fbp_host` is not
differentiable and rejects JAX arrays. Input and output must not overlap.
Completed slabs are written immediately, so a failure can leave a partly
written volume. A 1024³ laminography volume takes about
[37 s on an 8 GB laptop GPU](measurements.md#scans-larger-than-the-gpu).

### Avoid a full corrected stack for row-local parallel scans

Host reconstruction bounds accelerator storage, but `frames.corrected(...)`
normally still creates the complete line-integral stack in host memory. For
canonical z-axis parallel scans, you can instead correct blocks of detector
rows and reconstruct each block immediately. This recipe uses one output
plane per measured row, retains the full detector width and uses the same
globally selected views for every block. Establish the angles and rotation
axis first; it preserves the detector geometry in `frames` and does not
estimate the centre. See [setting the geometry](quickstart.md#correct-raw-detector-frames).

```python
from dataclasses import replace

import numpy as np

import tomojax as tj
from tomojax.corrections import Stripes
from tomojax.geometry import grid_volume_origin
from tomojax.recon import FBPHostConfig, fbp_host

frames = tj.load_frames("raw.nxs")  # use calibrated angles and detector geometry
if type(frames.geometry) is not tj.ParallelGeometry:
    raise ValueError("this recipe requires canonical z-axis parallel geometry")
d = frames.geometry.detector
grid = tj.Grid(d.nu, d.nu, d.nv, d.du, d.du, d.dv, vol_center=(0, 0, d.center[1]))
frames = replace(frames, geometry=replace(frames.geometry, grid=grid))
ox, oy, oz = grid_volume_origin(grid)
volume = np.lib.format.open_memmap(
    "volume.npy", mode="w+", dtype=np.float32,
    shape=(grid.nx, grid.ny, grid.nz), fortran_order=True,
)
config = FBPHostConfig(slices_per_batch=16)
correction_rows = 64  # raw correction block; independent of FBP's compute slabs
for start in range(0, grid.nz, correction_rows):
    stop = min(start + correction_rows, grid.nz)
    block = frames.cropped(slice(start, stop), slice(None))
    scan = block.corrected(Stripes(width=9))  # omit Stripes if it is not needed
    local_grid = tj.Grid(
        grid.nx, grid.ny, stop - start, grid.vx, grid.vy, grid.vz,
        vol_origin=(ox, oy, oz + start * grid.vz),
    )
    geometry = replace(scan.geometry, grid=local_grid)
    fbp_host(
        geometry, local_grid, scan.detector, scan.projections,
        config=config, out=volume[:, :, start:stop],
    )
    del scan, block, geometry
volume.flush()
```

`Frames.cropped` moves the detector but deliberately retains the original
volume grid. The explicit slab origin above is therefore essential: it keeps
each output plane at its original physical row instead of recentring every
block. `Stripes` needs all retained views of each row, not adjacent rows; it
can also change genuine sample structure, so compare with an uncorrected
reconstruction. Do not independently run `RejectViews` in each block. Apply
one global selection with `frames.selected(...)` before the loop. A selector
must precede `Stripes` in the intended correction chain: moving a later
selector earlier changes the ranks used by `Stripes`. Lens
correction and Paganin phase retrieval are not row-local and cannot simply
be inserted here: they require additional measured support or a different
storage plan. Posed, tilted and fractional-row output grids also need a
different dependency plan.

The corrected intermediate now holds at most one block of all views, rather
than all measured rows. Larger correction blocks can reduce repeated setup
and file-read overhead at a memory cost; smaller blocks are not necessarily
faster. Raw arrays already in RAM stay there, while lazy HDF5 inputs read
only the requested block. Runtime caches, transfers and filesystem caching
still add memory; a memmap is not a hard RSS cap. Flushing and writing the
complete volume cost time, and failures can leave partial output.

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
