# Local worker protocol

Protocol 1 uses UTF-8 JSON Lines. The client writes one request per line. The worker flushes one reply per line on stdout and reserves stderr for diagnostics. The first line is `ready`. Its device IDs are the values to send in operation parameters. Its capabilities name supported top-level command tags. The versioned schemas and fixed conversations in [protocol/](protocol/) are the client contract. Resolve their absolute schema IDs to local files when validating offline.

A request has a unique string `id`, a tagged `command` and an optional `channel`. `type` selects the variant of a command, workflow, product or storage reference. Required fields and defaults are in the schemas. Omitting a defaulted field uses its declared default. An explicit null is valid only for a nullable field.

The `describe` response contains a JSON Schema in `result.parameters`. Its `$defs` describe the inputs to each operation. Form controls can read `description` for help, `x-choices` for canonical domain choices, and `x-unit` for the unit of a value. `geometry_length` uses the scan's length unit. `native_pixel` and `working_pixel` refer to the original and working detectors. The `x-python` table maps public Python arguments to wire fields. A worker-owned argument such as the reconstruction output buffer is allocated by the executor.


Each admitted request receives zero or more `progress` envelopes followed by one `result` or `error`. Match all three by `request`. Each echoes `channel: null` or the channel name and its monotonically increasing revision. When replacing work on a channel, drop progress and artifacts from older revisions, including records already in the pipe. A rejected line receives a separate `protocol_error` of kind `invalid_request`. Its optional `request` is diagnostic. A duplicate ID leaves the original request pending. Unexpected EOF before a terminal means the process or connection was lost.

## Inputs and outputs

`{"type":"path","path":"scan.h5"}` opens a source on the execution queue. A handle member uses `{"type":"member","handle":"h","member":"self"}`. Every root has a `self` member. Named members are direct lookups in the owning result. For example, pass an alignment's `scan` member wherever a scan is accepted. The session pins every referenced member at admission, including refs in configs and correction steps. It also pins referenced session artifacts. Closing a handle prevents further admissions against it. Admitted work retains its inputs through completion.

Columns below name accepted input kinds in addition to the options defined by the indicated schema variant. A path used as a scan runs the normal scan loader. Raw calibration stays with a frames handle. Opening by kind explicitly selects the saved-data loader. Bulk inputs use storage references, including masks, exposure arrays and pose tables.

| Command or `run.operation` | Input and principal parameters | Published result and ownership |
|---|---|---|
| `describe`, `devices`, `status` | No input | Cached `description`, `devices` or `status`. Status includes request state, logical bytes by handle and sampled allocator counters. |
| `inspect` | Path | `inspection.report`, as in Python. |
| `open` | Path and kind. Scan and frames accept loader geometry, calibration and dataset paths. Saved results use their recorded metadata. | Session handle. Scan exposes projections, frames exposes counts. Reconstruction exposes volume, actual method and nullable scan. Alignment exposes scan, poses and nullable image. Simulation exposes scan and phantom. |
| `close`, `release` | Handle ID or session-issued artifact ID, respectively | `ack`. Release accepts an ID issued by this session. |
| `validate` | An open, workflow or interactive operation with its exact parameters | `validated` or typed error with the nested parameter path. |
| `plan` | A reconstruct or align operation | `reconstruction_plan`, `joint_plan` or `opposite_views_plan`. Placement is an estimate against the included snapshot. |
| `run.correct` | Path, scan or frames, ordered corrections and batch size. Frames also take binning and epsilon. | Scan handle with projections. |
| `run.reconstruct` | Path or scan, method, grid/region, config or public solver options | Reconstruction handle with registered volume, actual method and scan. Owner-managed storage supplies the Python output buffer. |
| `run.align` | Path or scan, fit, method, grid/region and fit options | Alignment handle with corrected scan, pose table and optional working image. Opposite-views centre estimation selects method `opposite-views`. |
| `run.simulate` | Phantom, acquisition geometry, artefacts and seed | Simulation handle with scan and phantom. |
| `run.project`, `run.backproject` | Scan or path for geometry, registered volume or projections by storage ref, optional grid/devices | `projections.projections` or `volume.volume`, session npy. The second operation applies the matched transpose. |
| `run.select`, `run.crop` | Scan, frames or path (scan). Ordered view indices or half-open detector row/column intervals | New handle of the input kind. |
| `run.bin`, `run.poses` | Scan or path, integer detector binning or six-column pose reference (null clears) | New scan handle. |
| `run.combine` | Ordered paths or scan members, optional name | Combined cone scan. The manifest retains each input's own path and transformations in order. |
| `run.reconstruct_slabs` | Raw path or frames, slab method, optional grid, corrections, config, warm start and epsilon | `volume.volume`, session npy. Fill the disk-backed volume at each yielded z interval before advancing. Read it with an npy memory map, then release its artifact ID. |
| `run.save` | Scan, reconstruction, alignment or simulation member, path, include_scan, force | `files.paths` includes the dataset and manifest. Caller owns committed destinations. |
| `run.export` | Reconstruction, alignment or simulation member, path, dtype, range, prefix, force | `files.paths` includes the raw file or TIFF stack and sidecars. Caller owns committed destinations. |
| `plane` | Volume member (including phantom or alignment image), axis and index | `plane.plane`, session npy. Works with a saved volume that has no scan. |
| `slice` | Reconstruct parameters, axis and index | `plane.plane` on the full method's grid, session npy. |
| `centre_sweep` | Slice parameters and ordered signed horizontal detector offsets in pixels, in the centre report's sign convention | `sweep.planes`, session npy. Leading coordinate vector preserves duplicates. |
| `correction_preview` | Raw path or frames, correction settings and half-open binned row interval | `correction_preview.before`, `after` and retained original `views`, session npy. |
| `region_preview` | Scan or path | Region handle for subsequent region parameters, physical XYZ bound pairs, selected grid and registered overlay arrays. |
| `control`, `shutdown` | Request ID and pause/resume/cancel, or no input for shutdown | `ack`. A paused request holds the execution queue. Cancel it to run other work. |

Execution results include bounded `info` and an input manifest. Each manifest node identifies the operation, its parameters and its ordered inputs. File roots retain the source path. Include all contributing inputs even when two distinct members come from the same root. Save the manifest alongside durable outputs. Handle registration, durable commit and terminal publication belong to the session owner. Session artifacts live until release, owning-handle close or shutdown. Input files and committed durable destinations belong to their caller.

## Progress and array registration

The standalone v2 progress schema covers task start/end, completed work with nullable ETA, placement, notes, pause/resume, previews and CLI terminals. A note carries the innermost open task ID, or null for a watch-level message between operations. The worker's `progress.record` admits only nonterminal events. `placed` lists actual participating devices. Elapsed time and ETA use seconds. Error kinds are `input`, `invalid_parameter`, `out_of_memory`, `cancelled`, `superseded` and `internal`. Worker admission errors have their own envelope. A failure path such as `corrections[2].alpha` starts at the operation's parameters.

Every descriptor carries shape, NumPy dtype, ordered axes, value unit/domain and storage. `npy` names a file, `hdf5` names a file and dataset, and `member` names a session handle and member. Session-produced npy files carry a release token. Client-owned npy and HDF5 inputs stay caller-owned. Array values travel outside JSON. Each axis gives a name, frame, unit, and either uniform coordinates or a reference to a one-dimensional vector of its coordinates. Coordinate order equals stored index order, including for irregularly selected views.

Volume dimensions are (z, y, x). A uniform origin locates the **centre of the first voxel** in that dimension. A removed axis records its source index and physical coordinate in `fixed`. Projection dimensions are (view, v, u). Poses are in the detector frame with columns (alpha, beta, phi, x, z, y), the first three in degrees and the last three in the scan's length unit. Lengths can be in pixels for uncalibrated data. Values distinguish counts, transmission, line integrals, attenuation and mixed-unit pose columns.

The compact UTF-8 encoding of `info` is limited to 65,536 bytes. Bulk members are externalised before scalar JSON conversion. Their positions in `summary` are null and their descriptors are in `arrays`, keyed by RFC 6901 JSON Pointer. The scalar summary retains method selection details such as `automatic`.

## Python hand-off

Decode a line as `tomojax.cli.envelope.Request`. For a queued command the adapter receives its typed `ExecutionInput` and an `ExecutionContext`. The owner resolves and pins the input references before queuing and supplies the leases, control, event sink, immutable allocator snapshot sink and private staging allocator. Adapters return `Success` or raise. They may only write to paths supplied by staging. An array reservation has distinct private and eventual public paths. Output reservations within one admission form a durable group, including manifests.

A returned `NewHandle` gives logical bytes, a pure `bind(id)` returning the published product and member map, and a disposal callback. The owner assigns the ID and checks the channel revision at commit. Losing the race discards staged files and unpublished resources. Disposal after successful registration waits for the last lease. The owner emits all worker terminal envelopes.

## Driving the process

Start `tomojax serve` with stdin and stdout pipes. Read `ready` before submitting commands and use its capability list for this build. Read stdout concurrently with writing stdin. The worker samples the device inventory at startup and answers `describe`, `devices` and `status` from memory. Allocator samples come from the executor. An empty allocator list means no sample has arrived. The latest sample for each (device, scope) includes its own timestamp. Handle sizes are logical bytes.

Send at most 1 MiB of UTF-8 per request line, excluding its newline. Request and channel IDs are limited to 256 UTF-8 bytes. Up to 64 execution requests may be active or queued. A replacement that displaces a queued admission can enter a full queue. Immediate commands still work at the queue limit. Use a fresh process after 100,000 admitted IDs. Shutdown remains available at the ID limit. An unknown member, mismatched artifact token and path, duplicate ID, unavailable command or exhausted admission limit yields `protocol_error`. A rejected replacement leaves the prior channel revision in effect.

Queued commands run in arrival order. A later admission on the same channel increments its revision, removes queued predecessors and wakes a running predecessor. Cancellation of queued work ends it immediately. Running work stops at a cooperative checkpoint, or at publication if it returns first. Pause moves running work through `pause-requested` to `paused` at a checkpoint. Resume withdraws a pending pause or wakes a paused request. A paused request holds its memory and the execution queue. Send `control` or `status` off-channel when observing existing work.

The writer preserves record order. Consecutive pending `advanced` events for one request and task may collapse to the newest event. Previews and their coordinate files are published before their record and stay available for release, including when a newer revision supersedes the job. The worker discards stale preview callbacks and disposes of their private reservations when the adapter exits. A partially written line always finishes before another begins.

On shutdown or stdin EOF, the worker closes admission, cancels outstanding work, wakes paused requests and joins the executor. Keep draining stdout through EOF to receive every terminal, including those after the shutdown acknowledgement. Broken stdout, more than 16 MiB of queued output in addition to the line being written, or two seconds without any writable pipe capacity during final drain means connection loss. The process exits with status 1 after cleanup. A draining client gets status 0. After an unexpected EOF, use the filesystem to check durable destinations whose terminals were lost.

## Durable files

Save and export adapters reserve a private staging location on the destination filesystem for each file or directory, including sidecars and manifests. The session serialises admission and publication. Cancellation or a higher channel revision admitted before the commit check discards the entire group and preserves existing destinations even with `force`. Once the session enters commit, it finishes the group and queues its terminal before admitting another command. Sending bytes on stdin alone does not establish which command won.

Without `force`, an existing group member causes the commit to fail. With `force`, originals move to private backups while the group is installed. A missing manifest, failed rename or other install error rolls back earlier group members and returns an error. A failed restoration reports the location of the retained original for recovery. Use the successful terminal as the point to open all files. Other filesystem readers can observe individual renames during a group commit. Concurrent external writers and process or machine crashes have no multi-file transaction guarantee.

Release takes only the issued artifact token. A successful release or handle close rejects new references immediately. Earlier admissions retain their leased storage through their terminal. Normal shutdown removes remaining private and published session files. Committed outputs remain caller-owned.

## Ownership after Milestone 0

| Files or registration | Sole owner |
|---|---|
| Array, progress, operations and products schemas under `docs/protocol/`, their Python types, wire option generation, input/result matrix and workflow transcripts | Execution agent |
| Worker schema, `cli/envelope.py`, schema assembly in `tools/build_wire_schemas.py`, session transcript, protocol transport documentation and writer tests | Worker agent |
| `.importlinter`, `tests/guardrails/api_surface.txt`, `tests/guardrails/ratchets.json`, `tools/check_docstrings.py` and all of `tools/guardrails.py` including `CLI_COMMANDS` | Execution agent integrates registration and measured deltas from both branches |
| Product commands in `cli/api.py`, dispatch/help in `cli/main.py` and CLI vocabulary/help entries in `tests/test_architecture.py` | Worker agent integrates command changes from both branches. |

## Standalone commands

Pass `--progress json` to inspect, correct, reconstruct, align, simulate or export. Read one JSON object per stdout line. A successful invocation ends with `result`. Its `outputs` list includes each written file, and its `arrays` reference the stored data. Export reports the source volume in disk axis order and lists the converted files and sidecar. Inspect returns its report in `info.summary`. Auxiliary NumPy files for bulk diagnostics or NPZ members are caller-owned and live alongside the output. A scalar summary that exceeds the byte limit reports the number of omitted fields as `_omitted_fields`, giving method selection priority over solver histories.

Errors end the stream with `error`. An invalid command line exits 2, an operational failure exits 1 and cancellation exits 130. An internal error also produces a traceback on stderr. A process killed before delivery leaves EOF without a terminal. Request the standalone inspect JSON document or a dry-run plan separately from JSON progress.


### Planning a run

`tomojax reconstruct --dry-run` and `tomojax align --dry-run` print the resolved plan without writing an output. Reconstruction reports the selected inverse, its reasons, its voxel grid and the configured iteration limit for iterative methods. A joint fit lists the padded working grid, outer iteration limit and image iterations per solve at each level. The returned working image crops the padding. Opposite-views reports how many angular pairs the estimator will examine. Pairs with unusable measured data can subsequently be rejected.

The placement estimate includes the sample timestamp, device identifiers and available bytes. Host and device byte estimates describe the working arrays. CPU availability comes from host memory and its cgroup limit. Compiled temporaries, allocator fragmentation and other processes can affect execution. Read the run's `info.placement` for the participating devices, actual storage strategy and the number of allocation retries by work unit.
