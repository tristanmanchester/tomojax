# Large scans and several GPUs

TomoJAX automatically streams projections from host memory and reconstructs volumes in slabs when a scan exceeds GPU memory, and spills large host arrays to disk when they exceed half of available RAM. Streaming applies to projections held as NumPy arrays or memmaps (what `tj.load` returns); JAX arrays already on the GPU stay resident. For long runs on a cluster, pass `checkpoint=` in Python, `--checkpoint DIR` on the CLI, or `checkpoint` to the worker or MCP server to save state to disk and [resume after a kill](#resume-a-run-after-a-kill).

## GPU allocator

Importing `tomojax` sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` unless you have set it yourself, so JAX allocates GPU memory on demand instead of reserving 75% up front. This leaves room for cuSolver, CuPy, and CUDA kernel workspaces. The JAX pool still caps at 75% of VRAM and retains freed blocks for reuse. Explicit `XLA_PYTHON_CLIENT_PREALLOCATE`, `XLA_CLIENT_MEM_FRACTION` (or `XLA_PYTHON_CLIENT_MEM_FRACTION`), and `XLA_PYTHON_CLIENT_ALLOCATOR` settings take precedence. Import `tomojax` or set these variables before the first JAX operation.

## What has to fit in host memory

When a large host array would exceed half of available host RAM, TomoJAX backs it with an unlinked temporary file in `TMPDIR` and returns a NumPy memmap:

| Array | Spills to a temporary file when |
| --- | --- |
| Corrected projection stack from `tj.load` or `Frames.corrected` | Stack exceeds half of available RAM |
| Host-slab volume (FBP, FDK, Fourier, MLEM, SIRT) | Volume exceeds half of available RAM |
| Multi-volume solver state (host-slab CGLS, FISTA-TV, accelerated SIRT) | Combined output and work volumes exceed half of available RAM |

During raw-frame slab reconstruction (`tj.reconstruct_slabs`), raw row blocks target 1 MiB per view (at least 1 GiB per block, capped at 1/24 of available host RAM). Up to three blocks stay live at once (current, prefetched next, and previous while slabs still reference it), taking at most 1/8 of available host RAM (or 1 GiB per block on smaller hosts).

`tj.load` logs the spill location; `tj.reconstruct` logs it and records the directory in `info["temporary_file"]`. Because the temporary file is unlinked, it disappears once the array is garbage-collected. Save it with `tj.save`, or pass your own writable memmap as `out=` so the volume is written straight to a named file (an explicit `out=` array is never moved, and solver work files are placed beside it when `out=` is a memmap).

A full-size synchrotron scan easily hits this limit (Diamond I13 159815 is 73 GB corrected). If `TMPDIR` lacks enough free space, TomoJAX raises `MemoryError` listing the required size and free space. On Linux systems where `/tmp` is `tmpfs` (RAM-backed), point `TMPDIR` to a fast local disk before loading, or reduce size with `binning=` or `Frames.selected(...)`.

## What has to fit on the GPU

| Method | Projections | Volume |
| --- | --- | --- |
| FBP (parallel and laminography) | Streamed in view batches from host memory | Switches to host slabs above 40% of free VRAM with an unrolled detector |
| Fourier (axial parallel only) | Read from host arrays or memmaps by row slab | Always built in host slabs |
| FDK (cone beam) | Read from host memory by detector-row band when volume is in host slabs | Switches to host z-slabs above 40% of free VRAM |
| CGLS | Streamed when resident data and solver state exceed free VRAM | Switches to host slabs automatically for cone beam and CUDA linear Joseph parallel/laminography |
| FISTA-TV | Streamed under the same rule; host-volume sweeps keep a detector-row ring | CUDA Joseph switches unmasked ordinary TV to host slabs when resident state does not fit |
| SPDHG-TV | Streamed under the same rule, with its sinogram dual and weights in host memory | Five volumes on the GPU (iterate, data backprojection, three TV duals) plus one view block and small workspace |

Set `stream = true` or `false` in the method's configuration (`--config` TOML or Python `*Config`) to override automatic streaming. Streaming requires the standard batched operators (canonical detector grid; linear/cubic Joseph rather than exact ray integration).

What the large-volume paths hold in memory:

- **Streamed CGLS** solves the equivalent normal equations and periodically recomputes the exact gradient to prevent drift. Host-slab CGLS keeps four host volumes (iterate, normal residual, search direction, normal image) and requires the full projection stack on the GPU for parallel/laminography; cone-beam slab sweeps stream the projections through a detector-row ring instead.
- **Host-slab FISTA-TV** solves the same global objective, automatic noise-score calibration, stopping rule, and finite-iteration TV proximal as resident FISTA-TV (not independent slices), keeping three host volumes (iterate, extrapolate, gradient). Its exact TV halo adds `tv_prox_iterations` planes per side on the GPU, so very high `tv_prox_iterations` can still exhaust VRAM. Huber-TV, support masks, and multi-GPU FISTA-TV require a device-resident volume.
- **Private z-contiguous layout:** resident CUDA solvers convert the volume once at entry and exit, requiring one temporary volume copy. `tj.project`, `tj.backproject`, and `tj.Projector` make that copy if it fits in free VRAM and read `(z, y, x)` in place otherwise (with identical results). Axial parallel projection streams 32 planes at a time and needs no copy.

See [memory measurements](measurements.md#memory) for peak VRAM across solvers on an 8 GB GPU.

## Plan and watch a long run

`--dry-run` resolves a run without starting it. It prints the method and why it was chosen, the grid, and the estimated memory against the memory available:

```bash
tomojax reconstruct scan.nxs --method cgls --dry-run
```

A run prints the same plan in one line as it starts. If it then runs out of memory, the error says what the plan needed and which options would shrink it. `--watch-dir DIR` rewrites images of the solve's central planes in DIR as the solve progresses, so an image viewer open on DIR shows a long reconstruction or alignment converging:

```bash
tomojax reconstruct scan.nxs -o recon.nxs --method cgls --watch-dir watch
```

## Resume a run after a kill

Pass `checkpoint=DIR` to `tj.reconstruct`, `tj.reconstruct_slabs`, or `tj.align`, `--checkpoint DIR` to `tomojax reconstruct` and `tomojax align`, or `checkpoint` to the worker's and MCP server's `reconstruct`, `reconstruct_slabs` and `align`, to save a run's state in `DIR`. You can also set `checkpoint = "DIR"` in a `--config` TOML file. If the run is killed by a scheduler time limit, preemption, or a node failure, run the same call again with the same directory, input, and settings. TomoJAX resumes from the saved state and returns what an uninterrupted run returns. `DIR` is created if absent and deleted when the call returns (for CLI commands, after all output files are written).

```python
recon = tj.reconstruct(scan, "cgls", iterations=200, checkpoint="recon.ckpt")
result = tj.align(scan, fit="translations", checkpoint="align.ckpt")
```

```bash
tomojax reconstruct scan.nxs -o recon.nxs --method cgls --iterations 200 --checkpoint recon.ckpt
tomojax align scan.nxs -o aligned.nxs --fit translations --checkpoint align.ckpt
```

Canceling a checkpointed run also saves it. Ctrl-C on the command line, `control` with `cancel` in the worker or MCP server, and `Control.cancel()` on a `progress.watch` finish the current iteration, slab, or step, save, and stop with `tomojax.errors.Canceled`, whose `checkpoint` attribute names `DIR`; the CLI's JSON `error` record, the worker's error, and the MCP tool error carry the same `checkpoint` field. A cancel outside a checkpointed loop (while loading, say) stops at once and the error still names `DIR`, which holds whatever was saved.

To find out what a checkpoint holds, inspect its directory: `tomojax inspect DIR`, `tj.inspect(DIR)["checkpoint"]`, or the MCP `inspect` tool. The report names the run (operation, method and settings, input file, TomoJAX version), how it was requested (the command line and its working directory, or the worker operation with its inputs' provenance), when it was created and last saved, each saved entry, its size, and whether a running process holds it. Inspecting a directory with the MCP `inspect` tool lists the checkpoints in its subdirectories.

What each run saves:

| Run | Saved state |
| --- | --- |
| Iterative solvers (`cgls`, `fista`, `sirt`, `spdhg`, `mlem`) | The iterate, every other array the next iteration reads (such as CGLS search direction and residual, or SPDHG primal and dual iterates), counters, and stopping state. A resumed automatic FISTA uses the same TV weight and stops at the same update; host-slab FISTA saves its calibrated noise level and skips the probe backprojection on resume. Warm-start FBP or FDK volumes are not saved because the solver state supersedes them. |
| Host-slab volumes (`fbp`, `fdk`, and `fourier` slabs; independent-plane `sirt` and `mlem` slabs) | Each slab as it finishes, plus the solver state of the slab in progress. |
| `tj.reconstruct_slabs` | Which slabs the caller has finished, plus the solver state of the slab in progress for `sirt` and `mlem`. |
| `tj.align` (joint fits) | Each finished resolution level (volume, poses, and record), the volume, poses, damping, and step history within a level at Gauss-Newton step boundaries, and the poses during refinement. |

For automatic CGLS and SIRT, the saved state includes the stopping statistic alongside the iterate. Each independent axial SIRT slab estimates its own discrepancy and can stop at a different sweep. The saved state retains the slab partition and finished slabs' diagnostics. The result lists each slab's z range, sweep count, and residual in `info["slab_stopping"]`; top-level `effective_iterations` is the largest sweep count across slabs, `termination` is `"iteration_limit"` if any slab reached the cap, and `early_stop` is `True` only when every slab stopped early. A coupled cone reconstruction uses one stopping test over all detector rows.

### Save schedule, time, and disk space

Iterative solvers and alignment save at a loop boundary (an iteration, a ~0.25 s JAX chunk, or an alignment step) once both 10 minutes and 20 times the previous save duration have passed since the previous save. That schedule caps periodic saving overhead at 5% of the run. Finished slabs are saved as each slab completes. Runs without `checkpoint=` take the unmodified code paths with no overhead. See [checkpoint measurements](measurements.md#checkpoint-and-resume) for measured overhead, save times, disk sizes, and host RSS on a synchrotron scan.

Each save writes a new `state-NNNNNN` directory of fsynced `.npy` files beside the previous entry, replaces `checkpoint.json` in one atomic rename (`os.replace`, fsynced), and then deletes the superseded entry. Budget disk space for one copy of the solver state, and up to twice that briefly while a save is written:

- **Iterative solver state:** CGLS saves iterate, search direction, and residual (two volumes and one projection-sized array, plus a gradient volume for CUDA CGLS), or three volumes for host-slab CGLS. Host-slab FISTA saves two volumes (iterate and extrapolate); host SIRT saves one volume (two with acceleration); host MLEM saves one volume and its seen-voxel mask.
- **Finished slabs:** slabs of a volume held in memory or in an unlinked temporary file add one copy of the volume on disk. When `out=` is a writable memmap of a named file (`np.lib.format.open_memmap(...)`), TomoJAX flushes each finished slab to that file and records a CRC fingerprint in the checkpoint instead of copying the planes; if the file's planes later change, those slabs are recomputed.
- **Host memory:** saving copies one device array at a time to host memory. A resumed run keeps restored arrays larger than 1 MiB memory-mapped while the loop runs; those pages are reclaimable file cache, though Linux memory cgroups count them toward RSS.

### Signals, hard kills, and refused checkpoints

On `SIGTERM` inside a checkpointed loop on the main thread, TomoJAX finishes the current iteration, slab, or alignment step, writes a checkpoint, and raises `tomojax.errors.Canceled` with a message naming the directory. The CLI prints that message and exits with status `130`. `SIGTERM` outside a checkpointed loop (for example, while loading a scan) terminates the process as usual. In a background thread of your own program, `SIGTERM` is not intercepted and only periodic saves run. The worker (`tomojax serve`, which the MCP server runs) handles `SIGTERM` itself: it stops admitting requests and cancels every job, so a checkpointed job saves, ends with a `canceled` error naming its checkpoint, and the worker exits once the jobs have ended. When `tomojax mcp` stops on `SIGTERM` or when its client closes it, it forwards `SIGTERM` to its worker and waits up to 120 seconds for a checkpointed job to save before killing the worker. A scheduler may signal only the process it started: when driving `tomojax mcp` under a scheduler, the MCP client must stop its server when signaled so the worker can save ([MCP long jobs](mcp.md#long-jobs-that-must-survive-a-kill)).

On a hard kill (`SIGKILL`, a node failure, or a `SIGTERM` whose current step plus save runs past the scheduler grace period), no final save runs and the next run resumes from the last completed periodic or slab save. Because `checkpoint.json` is replaced atomically after the new entry is synced, a kill during a save leaves the previous entry intact, and any incomplete entry directory is deleted on the next open.

![Two timelines of a checkpointed job. In the first, a SIGTERM shortly before the time limit makes the run save and exit; the requeued job starts from that save. In the second, a hard kill stops the run between two periodic saves; the requeued job starts from the last periodic save and repeats the work done after it, shown in orange.](images/checkpoint-timeline.png)

A schematic of the two cases. An early `SIGTERM` costs no work. A hard kill costs the work done since the last periodic save, at most one save interval of 10 minutes or more.

```bash
uv run python examples/gallery/checkpoint_timeline.py checkpoint-timeline.png
```

TomoJAX refuses a checkpoint with `InvalidParameter` (a `ValueError` subclass) when:

- the directory holds another run's checkpoint (different operation, TomoJAX version, input array digest, geometry, or settings such as method, config, filter, or warm start; arrays above 4 million elements are digested from 8 evenly spaced slices along axis 0, and the error names what differs);
- the directory contains unrelated files without a `checkpoint.json`;
- the checkpoint format version does not match; or
- another process holds the directory's file lock (`checkpoint.lock` via `flock`).

### Caller output with `tj.reconstruct_slabs`

`tj.reconstruct_slabs` marks a slab finished only after your `for` loop body returns for it. On resume, it yields only the slabs after the last finished one, so write each slab to persistent storage that survives the kill and flush it before the body finishes. Leaving the `with` block early (with `break` or an exception) keeps the checkpoint directory so a later run can finish the remaining slabs:

```python
from pathlib import Path
import numpy as np
import tomojax as tj

frames = tj.load_frames("raw.nxs")
grid = frames.geometry.grid
path = Path("volume.npy")
mode = "r+" if path.exists() else "w+"
volume = np.lib.format.open_memmap(
    path, mode=mode, dtype=np.float32, shape=(grid.nz, grid.ny, grid.nx)
)
with tj.reconstruct_slabs(frames, "sirt", checkpoint="slabs.ckpt") as slabs:
    for z_slice, block in slabs:
        volume[z_slice] = block.volume
        volume.flush()
```

### Bit-level agreement and scope

A resumed run continues from the exact arrays and scalars the next iteration reads, so it matches an uninterrupted run bit for bit whenever repeated uninterrupted runs match (on CPU, and in every case checked on one GPU model). A resumed result can differ in the last floating-point bits when:

- you resume on a different GPU model, a different number of devices (`devices=`), or a different JAX or CUDA build;
- you run cone-beam alignment, whose CUDA pose-derivative kernel accumulates per-view column sums with floating-point `atomicAdd` (two uninterrupted runs on the same GPU can differ in the same way);
- free device memory changes between runs so a slabbed method picks a different slab partition, wherever a method's arithmetic depends on slab boundaries (slabs only partly covered by finished slabs are recomputed under the new plan); or
- chunked JAX automatic FISTA recomputes its seeded calibration inside the compiled program on resume, which matches bit for bit only when the device and build repeat that arithmetic identically.

In `tj.align`, step metrics of finished levels are restored from JSON as Python `float` values where an uninterrupted run holds NumPy `float32` scalars (equal values, different types).

`checkpoint=` is available on `tj.reconstruct`, `tj.reconstruct_slabs`, `tj.align`, `tomojax reconstruct`, `tomojax align`, and the worker's and MCP server's `reconstruct`, `reconstruct_slabs` and `align`. A worker slab reconstruction keeps its finished slabs in `DIR/volume.npy` until the run completes, one volume of extra disk, because its staged output does not survive a kill. Direct expert solver calls (`tomojax.recon.cgls`, `cgls_multires`, `fista`, `fbp_host`, `fdk_host`, `fourier`, and the other functions in `tomojax.recon`) and TomoJAX Studio do not expose it. Direct axial center fits (`tj.align(scan, fit="center")` on unposed axial parallel half-turns and `method="opposite-views"`) run on CPU without an iterative loop.

### Checkpoint a Slurm job

Request an early `SIGTERM` before the wall-clock limit, launch TomoJAX with `srun` so the signal reaches the process, enable requeuing, and requeue on exit status `130`:

```bash
#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --signal=TERM@300
#SBATCH --requeue
#SBATCH --open-mode=append

# Add your cluster's partition, GPU, CPU, and memory directives above.
srun tomojax reconstruct scan.nxs -o recon.nxs --method cgls --iterations 500 --checkpoint recon.ckpt
status=$?
if [ "$status" -eq 130 ]; then
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi
exit "$status"
```

Set the `--signal=TERM@N` lead time `N` (in seconds) larger than one iteration (or slab) plus one save. Without `--signal`, Slurm sends `SIGTERM` at the time limit or on `scancel` and follows with `SIGKILL` after `KillWait` (commonly 30 seconds). Small solves finish a step and save within 30 seconds, but large host-slab volumes do not: writing three 57 GB CGLS volumes at ~1.3 GB/s takes roughly two minutes on top of the current iteration. If `SIGKILL` arrives before the save finishes, the requeued job falls back to the previous periodic save. `--requeue` lets Slurm restart the job after exit `130`, preemption, or a node failure, and `--open-mode=append` keeps output from all segments in one log file. Exiting with `$status` for any other code leaves real failures marked as failed in Slurm. When running an MCP client inside `srun`, `--signal` reaches the client rather than the `tomojax mcp` subprocess it spawned; stop the MCP server on `SIGTERM` so the worker saves before the time limit ([MCP long jobs](mcp.md#long-jobs-that-must-survive-a-kill)).

## Reconstruct a volume larger than the GPU

For FBP, FDK, Fourier, cone CGLS/SIRT/MLEM, and CUDA Joseph parallel/laminography CGLS and FISTA-TV, passing a writable C-ordered `float32` memmap as `out=` forces host-slab execution and writes directly to disk without allocating a second full volume:

```python
import numpy as np
import tomojax as tj

scan = tj.load("scan.nxs")
grid = scan.grid
volume = np.lib.format.open_memmap(
    "volume.npy", mode="w+", dtype=np.float32, shape=(grid.nz, grid.ny, grid.nx)
)
recon = tj.reconstruct(scan, "fbp", out=volume)  # FDK for cone, FBP for parallel/lamino
volume.flush()
```

Input projections and `out=` must not overlap (including separate mappings of the same file). Completed slabs are written as they finish, so an interrupted run can leave partial data on disk (pass `checkpoint=` to resume from the last finished slab). `Reconstruction.info["host_slabs"]` is `True` whenever host slabs are used. Host-slab routes are not differentiable.

Geometry-specific slab behavior:

- **Cone-beam FDK:** processes z-slabs, filtering only the detector rows each slab projects onto. Use `fdk_host` with `FDKHostConfig(slices_per_batch=...)` for explicit slab depth (default uses about a third of free VRAM).
- **Axial parallel FBP and Fourier with an unrolled detector:** processes axial z-slabs (at most 16 slices for FBP), reading only matching detector rows. C-ordered `(z, y, x)` keeps xy-planes contiguous on disk.
- **Laminography and posed FBP:** processes x-slabs (sized to about half of free VRAM), streaming and filtering views per slab. Use `fbp_host` with `FBPHostConfig(slices_per_batch=...)` to override slab depth. A 1024³ laminography volume runs in [37 s on an 8 GB laptop GPU](measurements.md#scans-larger-than-the-gpu).

### Stream raw frames directly into slabs

`Frames.corrected()` produces a complete corrected stack in host memory (or `TMPDIR`). `tj.reconstruct_slabs` uses the following storage strategies.

Uncompressed (unfiltered) HDF5 frames, chunked or contiguous and including external links, are read directly at their stored byte offsets with up to 32 parallel positional reads. Chunk offsets are indexed once and cached while the file size and modification time stay the same, and a row band reads only the needed rows of each chunk. Filtered, virtual, compact, and externally stored datasets read through HDF5. The first raw block prefetches while calibration runs, each slab uploads directly from its block with one pitched copy (a host copy runs only when a slab crosses two blocks), and streamed CUDA Fourier slabs check finiteness on the GPU before download.

- **Cone-beam FDK:** `tj.reconstruct_slabs(frames, "fbp")` keeps full-scan angular weights while correcting one row band at a time.
- **Unposed axial parallel with an unrolled detector:** supports `"fbp"`, `"fourier"`, `"sirt"`, and `"mlem"` (with optional `config=` and `warm_start=True`). Pass `config=FourierConfig(filter="ramp")` (or `"hann"` or `"shepp-logan"`) to match FBP's detector response in streamed Fourier reconstruction at the same GPU memory and speed. A full Diamond I12 scan (56.6 GB SIRT volume) ran in 24 minutes using [1.4 GB of VRAM](measurements.md#scans-larger-than-the-gpu).
- **Truncated scans in slabs:** `fbp` (parallel and cone FDK) and `fourier` apply `exterior="cylinder"` by default; pass `FBPConfig(exterior="zero")` or `FourierConfig(exterior="zero")` to disable continuation. Every yielded slab's `block.info["truncation"]` carries the five truncation diagnostics for that slab's detector rows. On full-volume DIAD `k11-52567` (8,000 views of 2,160 × 2,560 into `2,160 × 2,560²`, medians of three runs from raw frames in RAM), continuation adds no measurable time on a Tesla V100S 32 GB GPU (Fourier 31.60 s vs 33.10 s; FBP 324.40 s vs 338.40 s) and up to 5% on an RTX PRO 6000 Blackwell 96 GB GPU (Fourier 12.60 s vs 12.00 s; FBP 55.80 s vs 56.00 s); see [detector exterior continuation](measurements.md#detector-exterior-continuation).
- **Rolled parallel detector.** FBP, SIRT and MLEM require the full corrected projection stack in host memory and a resident volume on the compute device. The raw slab iterator returns the whole volume in one block. Fourier reads bounded bands of rotated detector rows and returns axial slabs.

```python
import numpy as np
import tomojax as tj
from tomojax.corrections import Rings

frames = tj.load_frames("raw.nxs")
grid = frames.geometry.grid
volume = np.lib.format.open_memmap(
    "volume.npy", mode="w+", dtype=np.float32, shape=(grid.nz, grid.ny, grid.nx)
)
with tj.reconstruct_slabs(
    frames, "fbp", corrections=[Rings()]
) as slabs:
    for z_slice, block in slabs:
        volume[z_slice] = block.volume
volume.flush()
```

![The central axial slice and a vertical slice through the axis of a copper oxide specimen at 1.13 µm, each with an enlarged region.](images/zeiss-full-volume.png)

The full Zeiss CuO_2 tomo-B scan (3201 views of 2028 × 2028 pixels, 52 GB corrected) reconstructed this way into a 31 GiB `2028³` volume at 1.13 µm in 1036 s on an 8 GB RTX 4070 Laptop GPU with 30 GB host RAM:

```bash
uv run python examples/gallery/zeiss_full_volume.py CuO_2_tomo-B.txrm VOLUME.npy OUT.png
```

Pass an empty dictionary as `info=info` to `tj.reconstruct_slabs`. After the iterator finishes and the context exits, `info["truncation"]` describes the same sample of the full detector as an in-memory reconstruction. Each sampled detector row counts once, including where slabs overlap. The two fractions use the scan totals, the interior shift is a mean, and the fitted radius is a median. Iterative runs also record stopping diagnostics for each z range in `info["slab_stopping"]` when the solver uses independent planes. The worker and MCP save the record in the volume and return it in the summary.

Only whole-row line-integral corrections (such as `Rings`) are supported in `reconstruct_slabs`, not `Paganin` or `RejectViews`. Release each `block` as you iterate so its volume and projections do not accumulate in memory.

## Bin detectors finer than the grid

When the detector pixel pitch (divided by magnification in cone beam) is finer than the reconstruction voxel pitch, projectors trace more rays than the grid can resolve. `scan.binned(factor)` averages `factor × factor` pixel blocks, reducing projection cost by `factor²` and adjusting the detector center to match (remainder edge pixels are dropped). Iterative `tj.reconstruct` warns with the suggested factor whenever the detector is at least twice as fine as the grid. On the CLI, pass `--binning` to `tomojax correct`. On the FIPS walnut, 2 × 2 binning made iterative reconstruction [3.5× faster with no loss of accuracy](measurements.md#fips-walnuts-real-lab-cone-beam-ct).

## Use several GPUs

Pass `devices=` in Python to split views across multiple GPUs in one process (`jax.devices()` uses all visible GPUs):

```python
import jax
import tomojax as tj

scan = tj.load("scan.nxs")
recon = tj.reconstruct(scan, "fista", nonnegative=True, devices=jax.devices())
```

In the JSON Lines worker (`tomojax serve`) and MCP server (`tomojax mcp`), pass the device IDs reported by `status` (such as `["gpu:0", "gpu:1"]`) in the `devices` field of `reconstruct`, `project`, or `backproject`. Without `devices`, those tools use only the default GPU.

How each solver shares work:

- **FBP and FDK (`"fbp"`):** each GPU filters and backprojects its contiguous share of views against a full volume and sums partial volumes onto the first GPU. In `fdk_host` and automatic FDK slabs (`FDKConfig(devices=...)`), each GPU reconstructs its own z-slabs.
- **CGLS (`"cgls"`):** each GPU projects and backprojects its contiguous share of views against a full copy of the volume inside one `shard_map`, then all-reduces the gradient volume every iteration. On axial parallel scans, the CUDA backprojector assigns neighboring threads along each group's middle ray axis so different angular ranges take matching time.
- **FISTA-TV (`"fista"`):** each GPU projects its view share against a full copy of the volume, and automatic TV-weight calibration (`tv_weight=None`) generates and backprojects each noise-probe view on its owning device. For resident `float32` CUDA volumes of at least 64 MiB (when partitioning the longest axis gives each device an interior of at least `2 * tv_prox_iterations` voxels and a haloed slab smaller than the axis), a reduce-scatter (`psum_scatter`) sums each device's gradient slab with an overlapping `tv_prox_iterations`-voxel halo per side, each device runs the CUDA TV proximal on its slab with the full grid's step size, and an `all_gather` of the disjoint updated interiors restores the full iterate for the next projection and for checkpointing. Both parallel-beam and cone-beam FISTA use these paths. Smaller volumes, non-CUDA devices, or large `tv_prox_iterations` fall back to all-reducing the gradient and running TV on each GPU.

What limits multi-GPU scaling and capacity:

- **Interconnect bandwidth:** every CGLS or FISTA iteration moves volume-sized collectives across the GPUs. On four Tesla V100S 32 GB GPUs paired across two CPU sockets over PCIe (Diamond K11-52567, 2,000 views, `64 × 2560 × 2560` volume, 50 warm iterations), a synchronized 1.56 GiB all-reduce takes 0.240 s on 2 GPUs (7.00 GB/s) and 0.288 s on 4 GPUs (5.83 GB/s). Collectives take 18–20% of 4-GPU wall time, giving 1.83× on 2 GPUs and 3.03× on 4 for CGLS (223.19 / 122.17 / 73.70 s) and 1.72× on 2 GPUs and 2.89× on 4 for nonnegative FISTA-TV (249.14 / 145.20 / 86.06 s). On four H100 80 GB GPUs connected over NVLink, 20 nonnegative FISTA-TV iterations on the 900-view binned 501³ FIPS walnut ran 3.34× faster (7.4 s vs 24.8 s, measured before shared TV slabs). See [multi-GPU measurements](measurements.md#several-gpus) and [axial V100S scaling](measurements.md#single-gpu-host-slab-and-multi-gpu-axial-cgls-and-fista-tv).
- **Short or small solves:** on small volumes or low iteration counts, host preparation and dispatch take a larger fraction of wall time. Ten nonnegative FISTA iterations on a 360-view 257³ cone phantom take 4.71 s on 1 V100S, 3.45 s on 2, and 2.57 s on 4, where host preparation with all GPUs idle takes 0.67 s of a 2.64 s traced 4-GPU run.
- **Volume capacity and streaming:** each GPU holds a full copy of the volume and its view shard in VRAM, and `devices=` cannot be combined with host streaming (`ValueError`). Multiple GPUs speed up compute without increasing maximum volume size, except in `fdk_host` and automatic FDK slabs.
- **Supported operations:** `tj.project`, `tj.backproject`, and `tj.reconstruct` with `"fbp"` (including FDK), `"cgls"`, and `"fista"`, plus the matching worker and MCP operations. `fourier`, `sirt`, `mlem`, `spdhg`, `fbp_host`, `tj.align`, and standalone CLI commands (`tomojax reconstruct`, `tomojax align`) run on one GPU (select which with `CUDA_VISIBLE_DEVICES`).
- **Floating-point rounding and scope:** GPUs must belong to one process on one machine. Cross-device reductions change floating-point addition order, so multi-GPU CGLS and FISTA can differ from single-GPU runs in the last `float32` bits (for example, `5.94 × 10⁻⁷` relative L2 on 4-GPU FISTA for K11-52567).
