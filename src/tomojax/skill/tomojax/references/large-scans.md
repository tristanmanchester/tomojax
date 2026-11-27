# Large scans and several GPUs

TomoJAX streams projections and reconstructs in slabs on its own when a scan or volume exceeds GPU memory, and backs large host arrays with temporary files when they exceed half of the available RAM. The full account, with each method's limits, is at https://tomojax.dev/large-scans/.

## Before a large job

- Run `reconstruct` or `align` with `dry_run=true`. The plan reports the method, grid and memory placement, so you can see whether the volume will stay on the GPU or go to host slabs.
- `status` shows device memory, live handles and their sizes. Close handles you no longer need with `control` and `action="close"`.
- Large host arrays spill to `TMPDIR`. On Linux machines where `/tmp` is RAM-backed, a full synchrotron scan can fill it; ask the user for a fast local disk and start the server with `TMPDIR` pointing there.
- Reduce the work when the question allows: `binning` in `correct`, `edit_scan` to select views or crop rows, or a `preview` slice instead of a full volume.

## Volumes larger than memory

- `reconstruct_slabs` reconstructs raw frames slab by slab straight into a NeXus file, for FBP, Fourier, SIRT and MLEM, without holding the volume or the corrected stack. It takes a destination `path` and refuses to replace an existing file unless `force` is set.
- In Python, passing a writable `float32` memmap as `out=` to `tj.reconstruct` writes the volume to disk slab by slab for the methods that support host slabs. See [python.md](python.md) under "Memory-mapped reconstruction".

## Several GPUs

In Python, `tj.reconstruct(scan, method, devices=jax.devices())` splits views across the GPUs in one process. Each GPU holds a full copy of the volume, so several GPUs speed up the computation but do not let a larger volume fit, except for FDK slabs, where each GPU reconstructs its own slabs. The MCP server's worker uses the devices JAX sees; `status` lists them.

## Clusters and checkpointing

TomoJAX has no scheduler integration of its own. On a cluster managed by Slurm or a similar scheduler, run the reconstruction or alignment as an ordinary batch job inside the allocation with the Python API or the CLI, requesting the GPUs and memory the dry-run plan reports.

To survive time limits, preemption, or node failures, pass `checkpoint=DIR` to `tj.reconstruct`, `tj.reconstruct_slabs`, or `tj.align`, or `--checkpoint DIR` to `tomojax reconstruct` and `tomojax align`. Running the same call again with the same directory, input, and settings resumes from the saved state and deletes `DIR` on completion. Saves run at loop boundaries once 10 minutes and 20 save durations have passed (at most 5% overhead), after each finished slab, and on `SIGTERM` in the main thread (raising `tomojax.errors.Canceled`, CLI exit status `130`). On Slurm, run with `srun`, pass `--signal=TERM@N` with `N` larger than one iteration plus one save (large volumes cannot save inside a 30 s `KillWait`), pass `--requeue`, and call `scontrol requeue "$SLURM_JOB_ID"` on exit `130`. For `tj.reconstruct_slabs`, write slabs to persistent storage that survives a kill (for example, a memmap opened with `mode="r+"` when the file exists) and flush each slab before the loop body returns.

The MCP tools, `tomojax serve`, Studio, and direct calls to `tomojax.recon.*` do not expose `checkpoint=`. The MCP server talks over stdio to a client on the same machine, so drive it inside an allocation or use it locally on a binned or cropped copy to choose settings before submitting the checkpointed batch job. See https://tomojax.dev/large-scans/#resume-a-run-after-a-kill for the full Slurm script and caveats.
