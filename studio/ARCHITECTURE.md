# Studio architecture

## Ownership

The Python executor owns tomography operations, numerical arrays, progress,
validation, cancellation at work boundaries and committed scientific exports. The Rust
application owns projects, durable asset identities, parameter drafts, the local
job queue and display preferences. The renderer owns GPU resources derived from
registered scalar data. A display operation never mutates a numerical input.

The workspace separates the desktop application from the worker on the data host.
`tomojax-studio` handles rendering and Rune views. `studio-core` holds the
project, connection and worker client. `tomojax-worker` owns the session pipes,
request admission, executor and file transfers. It uses `tomojax-io` for NPY
and HDF5 selections, headers and pixel encoding, and `tomojax-jobs` for jobs.
`tests/test_architecture.py` reads `Cargo.lock` and fails if `studio-core` comes
to depend on Rune, wgpu, winit or HDF5, so its tests build and run without them:

```sh
cargo test --manifest-path studio/Cargo.toml -p studio-core
```

| `studio-core` module | Responsibility |
| --- | --- |
| `protocol.rs` | Typed official protocol responses and parameter schema |
| `pixels/` | Binary block framing and credit, plane selection and decoding, per-pane fetch state and bounded decode queue |
| `engine.rs` | Bounded worker transport, request correlation, process lifecycle |
| `project.rs` | Versioned manifest, captured inputs, job records, single writer and atomic autosave |
| `operations.rs` | Operation names, groups and method titles shared by projects and forms |
| `previews.rs` | Temporary per-control requests, warm sources and supersession |
| `project/script.rs`, `project/receipt.rs` | Capture projection roles and requests, then apply progress and the final receipt to the project |
| `progress.rs` | Durable task trees, metric histories, registry units and notes |

| `tomojax-studio` module | Responsibility |
| --- | --- |
| `app/` | Workflow scheduling, dialogs, project actions, presentation model and view coordination |
| `forms.rs`, `app/validation.rs` | Schema-derived drafts, exact field paths, debounced worker validation and plans |
| `app/durable.rs` | Sends captured scripts after a manifest save, polls the job directory and handles controls and receipts |
| `app/previewing.rs` | Edit intents to preview requests, and explicit Keep |
| `live.rs` | Coherent registered solver previews, pose tables and artifact read leases |
| `imaging/` | Registered display geometry, worker-backed plane and volume reads, bounded scalar cache |
| `volume/` | Scalar image shaders, transfer textures and volume ray casting within Rune |
| `preferences.rs` | Recent projects and keyboard shortcuts |
| `ui/` | Native layout, controls, scientific overlays and presentation-only drafts |

The library target holds `imaging`, `live` and `volume`; the binary
holds `app`, `forms`, `preferences` and `ui`. `forms.rs` keeps its drafts in
Rune `TextEdit`s inside `UiState`, shortcuts are Rune chords, and `live.rs` reads
previews through `imaging`, so these stay out of `studio-core`.

[STATECHART.md](STATECHART.md) charts preview channels, job submission and recovery, preview generations and manifest saves. Under `cargo test`,
`crates/studio-core/tests/protocol_model.rs` checks every interleaving of each
modeled workflow within small bounds. `crates/studio-core/tests/job_recovery.rs` also exercises capture, polling, relinking and restart against real project snapshots. Property tests run random action
sequences through the real code and check invariants after every step:
`crates/tomojax-studio/src/app/state_machine_tests.rs` drives `AppState` through
selection, views, window edits, jobs, Activity, relinking and reopening, and
`crates/studio-core/src/previews/supersession_tests.rs` drives one preview
channel through edits and worker admissions, refusals and answers in any order.

`UiModel` is a presentation snapshot derived from the project and live service
state. `UiAction` is the view's route to application effects. It does not contain
an independent copy of the scientific workflow. `UiState` owns editable drafts
and transient view interaction. The progress view borrows retained histories from
the project; routine model refreshes do not copy every metric sample.

## Requests and jobs

The native bridge launches its sibling `tomojax-worker serve`. The worker starts
the interpreter selected by `tomojax studio` and forwards numerical requests to it. Requests have unique string IDs. Transport threads read and
write bounded messages without blocking Rune's event loop. Stderr is diagnostic
output, never a protocol source.

Studio captures the job script, input paths and projection roles and waits for the project writer before submitting. The worker opens the captured inputs, executes the script, publishes the final product and releases its handles. Studio polls each job from its saved byte offset. Receipt adoption updates the asset, job and cursor in one manifest write. Inspection and frame calibration also check the source asset’s intent number.

The bridge tracks observed channel admissions, not speculative revisions. A
rejected replacement cannot invalidate a prior admitted request. Stale terminals
still resolve request bookkeeping. Control acknowledgment is distinct from the
target request's terminal result. Shutdown drains worker output before process
reaping, with a bounded forced-exit fallback and owned temporary directory cleanup.

Controls name a job directory. Studio reads scheduler state from the worker and pause progress from the running inline attempt. A new connection verifies the project, lists its jobs and reads the recorded directories.

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

Explore reads the export format list from `Export.path.x-formats` and each option from the corresponding worker parameter. A change to the destination, input or options invalidates the displayed plan. The validated operation and the queued operation use the same draft. A native save and an array bundle keep their own operations.

### Solver previews and progress

Reconstruction and alignment scripts optionally request solver previews. Each job poll names one published generation. The reader loads its planes and full-precision pose table together. A superseded or expired generation leaves the previous complete frame on screen.

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

Reopening restores each scripted job’s last observation and reads its directory on the project connection. If the worker never received the submission, Activity records that outcome. A detached job can continue through a disconnect. Studio polls again after an interruption because the scheduler may report the runner’s exit before the shared filesystem exposes its receipt. A checkpoint resume uses a new job identity and the line and path from the current worker reply.

## Registered scientific viewing

Array descriptors carry storage, shape, dtype, named axes, coordinate frames,
units and fixed coordinates. The worker reads bounded hyperslabs from HDF5 and numeric NPY arrays. Stored axis permutations, signed spacing and irregular view
coordinates remain explicit. The application maps physical z/y/x controls onto
the stored axis names.

The project manifest's optional `connection` selects the plane source for all its
assets. Local projects connect to a sibling worker over pipes. `Imaging`
requests a windowed JPEG up to 512 pixels across while moving. Each pane keeps
its newest wanted index while a JPEG is in flight. Once the index settles, the
pane requests float32 samples at its displayed size. Moving again supersedes
that lossless request immediately.

The engine's stdout reader consumes and acknowledges pixel blocks before
handing a complete frame to the decoder. The reader sizes the credit window
from the measured block transfer rate, targeting a tenth of a second of data.
One decoder serves the connection and can replace a waiting JPEG with a newer
one for the same slot. The slot checks its generation and source before
installing a frame. An interim JPEG carries its own coordinates. The probe,
profiles, measurements and histogram use only settled samples. A disconnect
clears the old planes. Volume reads, metadata, coordinate vectors and pose tables use the same worker.
Local volume views fetch float32 for the original finite-value statistics and
texture. Remote volume views use uint16 over their reported scalar range.

The scalar cache has a 512 MiB default budget. Active decoded views and a bounded
number of in-flight loads also consume memory; this is a cache budget, not a hard
process RSS limit. New slice requests supersede old requests for the same slot.
A moving frame can arrive before its lossless replacement.
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

The worker links HDF5 and Studio links the platform graphics libraries. The source build and Studio wheel place both executables together. The Python launcher, `tomojax studio`, is part of
TomoJAX.
