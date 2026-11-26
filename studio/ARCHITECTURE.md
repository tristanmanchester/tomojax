# Studio architecture

## Ownership

The Python worker owns tomography operations, numerical arrays, progress,
validation, cancellation checkpoints and committed scientific exports. The Rust
application owns projects, durable asset identities, parameter drafts, the local
job queue and display preferences. The renderer owns GPU resources derived from
registered scalar data. A display operation never mutates a numerical input.

| Module | Responsibility |
| --- | --- |
| `protocol.rs` | Typed official protocol responses and parameter schema |
| `engine.rs` | Bounded asynchronous JSONL transport, request correlation, process lifecycle |
| `project.rs` | Versioned manifest, captured inputs, job records, single writer and atomic autosave |
| `app/` | Workflow scheduling, dialogs, project actions, presentation model and view coordination |
| `forms.rs`, `app/validation.rs` | Schema-derived drafts, exact field paths, debounced worker validation and plans |
| `app/projection.rs` | Which selected inputs a forward or back projection reads (geometry, volume or projections), chosen once for preflight, the worker command and the result's parent |
| `previews.rs`, `app/previewing.rs` | Temporary per-control requests, warm sources, supersession and explicit Keep |
| `live.rs` | Coherent registered solver checkpoints, pose tables and artifact read leases |
| `progress.rs` | Durable task trees, metric histories, registry units and notes |
| `metadata.rs` | Native stored-array registration and official inspection/product interpretation |
| `imaging/` | Numeric NPY/HDF5 readers, bounded scalar cache, coalescing requests and prefetch |
| `volume/` | Scalar image shaders, transfer textures and volume ray casting within Rune |
| `ui/` | Native layout, controls, scientific overlays and presentation-only drafts |

[STATECHART.md](STATECHART.md) charts the preview channel, job control,
checkpoint lease and save state machines. Under `cargo test`,
`tests/protocol_model.rs` checks every interleaving of each one with the
worker, within small bounds. Property tests run random action sequences
through the real code and check invariants after every step:
`app/state_machine_tests.rs` drives `AppState` through selection, views,
window edits, jobs, Activity, relinking and reopening, and
`previews/supersession_tests.rs` drives one preview channel through edits
and worker admissions, refusals and answers in any order.

`UiModel` is a presentation snapshot derived from the project and live service
state. `UiAction` is the view's route to application effects. It does not contain
an independent copy of the scientific workflow. `UiState` owns editable drafts
and transient view interaction. The progress view borrows retained histories from
the project; routine model refreshes do not copy every metric sample.

## Requests and jobs

The native bridge launches the same interpreter selected by `tomojax studio` and
runs `tomojax serve`. Requests have unique string IDs. Transport threads read and
write bounded messages without blocking Rune's event loop. Stderr is diagnostic
output, never a protocol source.

The scheduler captures a job before admission, resolves its input handles and
submits a public operation. A result that owns a reusable handle is saved through
`run.save`. Only the successful durable terminal makes the derived result complete
in the project. Session handles are not persisted as reusable identifiers.

The bridge tracks observed channel admissions, not speculative revisions. A
rejected replacement cannot invalidate a prior admitted request. Stale terminals
still resolve request bookkeeping. Control acknowledgment is distinct from the
target request's terminal result. Shutdown drains worker output before process
reaping, with a bounded forced-exit fallback and owned temporary directory cleanup.

Pause/resume acknowledgment triggers an authoritative status check. This also
covers a pause reached before a numerical task exists and therefore before the
worker can emit a progress event. The application polls only while that control
transition is unresolved, and a terminal result retires its control watch.

### Interactive requests

`slice`, `center_sweep`, `correction_preview` and `region_preview` have separate
worker channels. The preview controller debounces edits for 35 ms, shares warm
source handles with form preflight, and admits replacements independently of the
durable scheduler and autosave. A local intent generation invalidates an older
answer immediately, including a terminal already buffered before the next
channel admission. The bridge also rejects obsolete admitted revisions.

The worker still executes one device request at a time. Replacing a preview
does not interrupt an unrelated reconstruction. Pausing that reconstruction
retains the device queue; only explicit completion or cancellation frees it.
Studio reports that blocker and routes user controls to the existing job.

Temporary arrays are not project assets. **Keep result** transfers their ownership
to a local copy job with the captured input, parameters, request and revision.
It copies the registered arrays and provenance before publishing the asset.
Superseded results are retired after native reads of their scalar and coordinate
artifacts finish. Unchanged failed settings remain failed until an edit or an
explicit retry; polling does not silently repeat the operation.

### Editing and plans

Form edits trigger worker `validate` after 350 ms. Supported reconstruction and
alignment forms then request `plan`; the exact supported wrapper and parameter
fields come from `describe`. For interactive slice bounds, the plan receives the
intersection with the described reconstruction fields. It does not receive
preview-only parameters. A plan carries its setting signature and full captured
source identity, so relinking an asset cannot reuse the old grid.

Errors retain the worker's complete `path`. A nested object error highlights its
JSON editor; an indexed correction path targets the appropriate guided control
when one exists, otherwise the ordered JSON editor. Correction drafts preserve
explicit JSON null separately from omission.

### Solver checkpoints and progress

Reconstruction and alignment request live checkpoint artifacts when enabled.
The live reader loads each checkpoint's registered z/y/x planes and full-precision
pose table together. The previous coherent frame remains visible while its
replacement loads. Reference-counted read leases prevent supersession, clearing
or prefetch cancellation from releasing a file beneath a native reader. Releases
are deduplicated; disconnect clears the old session's token ledger before reuse.
Successful result handles also own their request's preview artifacts. Studio
defers closing these handles until native reads finish, queues checkpoint releases
before the close, and retains the decoded frame for display. Cleanup admission
uses the same ordered retry queue as temporary previews.

Progress tasks are keyed by request and task ID, with parent IDs interpreted
within their request. Open, run and save phases cannot overwrite one another's
reused task IDs. Each task keeps its own count, ETA, placement and metric samples. Units
come from the worker-generated metric registry in `docs/progress-metrics.md`.
The job retains notes and all metric samples, including null gaps and categories.
Plot rendering selects at most 600 original extrema per history and never joins
across a null sample. Pose rows are virtualized.

## Storage and recovery

The version 2 manifest stores a result's editable name on its asset record.
Opening a version 1 project translates generated result names using the submitted
operation and input. Job inputs and worker provenance retain their original values.
Studio restores the active asset and view on reopen. Without a valid active asset
it selects the newest result, or the newest acquisition in a project without results.

Studio serializes a snapshot. A coalescing background writer atomically replaces the manifest. Acknowledgments
carry generations so an older save cannot clear a newer dirty state. The writer
holds an OS lock for its lifetime. An unsupported manifest version is rejected
without rewriting it. Both reading and writing enforce the existing 32 MiB
manifest limit. Oversized metadata remains in memory with a visible save error;
it does not replace the last valid file. Only the latest generation's
acknowledgment sets or clears the save error: an older successful write cannot
clear a newer error, and an older failed write cannot fail a newer save.

Managed data uses relative project paths. Referenced acquisitions stay external
unless the user selects copying. Captured job inputs retain their original paths
even if the asset is later relinked.

Managed source copies and array-only outputs use owned staging paths. Copies use
bounded chunks, observe cancellation, and publish only after validation. Region
preview handles remain alive until all of their arrays and coordinate vectors
have been copied. Shared artifact tokens are released once.

Restarting does not silently execute stale queued work. Active records become
interrupted, and a retry uses a new job identity. Existing output destinations
require review because an interrupted transport does not prove the numerical
worker failed to commit.

## Registered scientific viewing

Array descriptors carry storage, shape, dtype, named axes, coordinate frames,
units and fixed coordinates. Native I/O reads bounded hyperslabs or memory-mapped
numeric NPY arrays. Stored axis permutations, signed spacing and irregular view
coordinates remain explicit. The application maps physical z/y/x controls onto
the stored axis names.

The scalar cache has a 512 MiB default budget. Active decoded views and a bounded
number of in-flight loads also consume memory; this is a cache budget, not a hard
process RSS limit. New slice requests supersede old requests for the same slot.
Coarse data can arrive before refinement, and adjacent planes are prefetched.
Project replacement, relinking and authoritative metadata refresh call
`Imaging::invalidate()` before requesting new views. Its generation rejects late
reads from the old source. Ordinary navigation does not stat source files.
Replacing an asset's data or calibration clears its saved measurements.

The compositor uses Rune's actual wgpu device and painter order. Plane textures
retain float scalar samples; windowing and color maps are GPU uniforms. The
volume texture is a normalized half-float display representation with a retained
scalar range. The ray marcher accounts for physical anisotropy, integrates
front-to-back opacity and supports transfer functions, lighting and clipping.
Quantitative measurements use the original float plane samples, not rendered
color or the half-float volume texture.

Display-window text is a draft until **Apply window** succeeds. The renderer uses
the committed range; invalid edits remain editable and persist without changing
the image. An untouched field follows the window on screen, even while the
other field holds an edit.
View undo/redo restores display state without rewinding parameter forms
or scientific jobs.

Slice and center-sweep previews retain their fixed spatial axis. Sweep navigation
indexes the explicit center-coordinate vector. Region products retain three fixed
planes and physical bounds without being relabeled as a volume. Correction
previews retain independently registered before/after arrays and their different
units and domains; selecting the other array resets the display window.
During a temporary preview, the source asset remains selected. Completion of an
unrelated durable job appends its result without retargeting the active controls.
Measurements require a durable result identity and are disabled on temporary data.

Volume uploads and ray-cast resolution are bounded. An unchanged camera and
transfer function reuse the rendered volume frame. Interaction uses reduced
rendering effort and a later frame refines it. Rendering remains independent of
the numerical worker's execution queue.

## Build

The native binary depends on HDF5 and the platform graphics libraries and is
built from the source checkout. The Python launcher, `tomojax studio`, is part of
TomoJAX.
