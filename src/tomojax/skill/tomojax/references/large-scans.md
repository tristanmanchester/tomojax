# Large scans and several GPUs

TomoJAX streams projections and reconstructs in slabs on its own when a scan or volume exceeds GPU memory, and backs large host arrays with temporary files when they exceed half of the available RAM. The full account, with each method's limits, is at https://tomojax.dev/large-scans/.

## Before a large job

- Run `reconstruct` or `align` with `dry_run=true`. The plan reports the method, grid and memory placement, so you can see whether the volume will stay on the GPU or go to host slabs.
- `status` shows device memory, live handles and their sizes. Close handles you no longer need with `control` and `action="close"`.
- Large host arrays spill to `TMPDIR`. On Linux machines where `/tmp` is RAM-backed, a full synchrotron scan can fill it; ask the user for a fast local disk and start the server with `TMPDIR` pointing there.
- Reduce the work when the question allows: `binning` in `correct`, `edit_scan` to select views or crop rows, or a `preview` slice instead of a full volume.

## Volumes larger than memory

- `reconstruct_slabs` reconstructs raw frames slab by slab straight into a NeXus file, for FBP, Fourier, SIRT, and MLEM, without holding the volume or the corrected stack (on a rolled parallel detector, `fourier` streams bounded bands of rotated rows while `fbp`, `sirt`, and `mlem` read the full detector and return one block). It takes a destination `path` and refuses to replace an existing file unless `force` is set. The saved volume (`volume.reconstruction_info` in `inspect`) and the tool's summary include whole-scan `truncation` and iterative stopping diagnostics (including `slab_stopping` on independent planes).
- In Python, passing a writable `float32` memmap as `out=` to `tj.reconstruct` writes the volume to disk slab by slab for the methods that support host slabs. See [python.md](python.md) under "Memory-mapped reconstruction".

## Several GPUs

Pass `devices=jax.devices()` in Python (`tj.reconstruct`, `tj.project`, `tj.backproject`), or pass `devices` (a list of device IDs from `status`, such as `["gpu:0", "gpu:1"]`) to the MCP `reconstruct`, `project`, or `backproject` tools to split views across GPUs in one process for `fbp`/FDK, `cgls`, and `fista`. Without `devices`, the worker uses only the default GPU. `fourier`, `sirt`, `mlem`, `spdhg`, `align`, and `reconstruct_slabs` run on one GPU.

Each GPU holds a full copy of the volume and its view shard in VRAM (host streaming is refused with `devices`), so several GPUs speed up computation without letting a larger volume fit, except in FDK slabs where each GPU reconstructs its own z-slabs. On multiple CUDA GPUs, `fista` shares automatic TV calibration and, for resident `float32` volumes of at least 64 MiB that split cleanly with a `tv_prox_iterations` halo per side, splits the gradient reduction and TV proximal into overlapping slabs (`psum_scatter` and `all_gather`). Per-iteration volume collectives over the host interconnect limit speedup on large scans (for example, 3.03× for CGLS and 2.89× for FISTA-TV on four PCIe V100S 32 GB GPUs for 50 iterations of 2,000-view `64 × 2560 × 2560` K11-52567; 3.34× on four NVLink H100 80 GB GPUs for 20 iterations of the 900-view 501³ walnut), and host startup limits speedup on short or small solves.

## Clusters and checkpointing

TomoJAX has no scheduler integration of its own. On a cluster managed by Slurm or a similar scheduler, run the work inside an allocation, requesting the GPUs and memory the dry-run plan reports: drive the MCP server there, or run the CLI or Python as a batch job.

To survive time limits, preemption, node failures or a cancel, give `reconstruct`, `reconstruct_slabs` or `align` a `checkpoint` directory (`checkpoint=DIR` in Python, `--checkpoint DIR` on `tomojax reconstruct` and `tomojax align`). The run saves at loop boundaries once 10 minutes and 20 save durations have passed (at most 5% overhead), after each finished slab, and whenever it is stopped at a boundary: a `control` cancel, Ctrl-C, or `SIGTERM` from the scheduler (the MCP server's worker handles `SIGTERM` itself). A stopped run ends with a `canceled` error whose `checkpoint` field names the directory. Calling the same tool or command again with the same arguments, on the same input, resumes from the saved state and deletes the directory on completion; a different input or setting is refused, naming what differs.

Handles do not survive the server, so plan for another session to resume:

1. `save` the corrected or aligned scan to a file and run the long job from that path, with `checkpoint` in the project directory.
2. When a session finds unfinished work, `inspect` the project directory: its `checkpoints` lists checkpoint directories. `inspect` one to read the run (operation, method, settings, input file), the request that started it (its arguments and input provenance), when it was last saved and whether a running process holds it (`in_use`).
3. Repeat the request. The recorded `request` uses the worker's wire form (for example, `source: {"type": "path", "path": "..."}`); pass `source` as the path string when calling the MCP tool. If its source was a handle, rebuild the input the same way first (the provenance lists each step), so its digest matches. A worker or MCP `reconstruct_slabs` run keeps finished slabs in `DIR/volume.npy` until it completes.

On Slurm, run with `srun` so the signal reaches TomoJAX (when you drive the MCP server, the signal reaches your client process, which must stop the server for the worker to save), pass `--signal=TERM@N` with `N` larger than one iteration plus one save (large volumes cannot save inside a 30 s `KillWait`), and for batch jobs pass `--requeue` and call `scontrol requeue "$SLURM_JOB_ID"` on exit `130`. A `SIGKILL` or lost node allows no final save: the run resumes from the last periodic save. In Python, `tj.reconstruct_slabs` resumes after the last slab your loop body finished, so write slabs to storage that survives a kill and flush each one. Studio and direct calls to `tomojax.recon.*` do not take `checkpoint`. See https://tomojax.dev/large-scans/#resume-a-run-after-a-kill for the full Slurm script and caveats.
