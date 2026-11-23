# Local worker protocol

Protocol 1 uses UTF-8 JSON Lines. The client writes one request per line. The worker flushes one reply per line on stdout and reserves stderr for diagnostics. The first line is `ready`. Its device IDs are the values to send in operation parameters. Its capabilities name supported top-level command tags. The versioned schemas and fixed conversations in [protocol/](protocol/) are the client contract. Resolve their absolute schema IDs to local files when validating offline.

A request has a unique string `id`, a tagged `command` and an optional `channel`. `type` selects the variant of a command, workflow, product or storage reference. Required fields and defaults are in the schemas. Omitting a defaulted field uses its declared default. An explicit null is valid only for a nullable field.

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

The standalone v2 progress schema covers task start/end, completed work with nullable ETA, placement, notes, pause/resume, previews and CLI terminals. This milestone specifies v2 ahead of the CLI emitter migration. The worker's `progress.record` admits only nonterminal events. `placed` lists actual participating devices. Elapsed time and ETA use seconds. Error kinds are `input`, `invalid_parameter`, `out_of_memory`, `cancelled`, `superseded` and `internal`. Worker admission errors have their own envelope. A failure path such as `corrections[2].alpha` starts at the operation's parameters.

Every descriptor carries shape, NumPy dtype, ordered axes, value unit/domain and storage. `npy` names a file, `hdf5` names a file and dataset, and `member` names a session handle and member. Session-produced npy files carry a release token. Client-owned npy and HDF5 inputs stay caller-owned. Array values travel outside JSON. Each axis gives a name, frame, unit, and either uniform coordinates or a reference to a one-dimensional vector of its coordinates. Coordinate order equals stored index order, including for irregularly selected views.

Volume dimensions are (z, y, x). A uniform origin locates the **centre of the first voxel** in that dimension. A removed axis records its source index and physical coordinate in `fixed`. Projection dimensions are (view, v, u). Poses are in the detector frame with columns (alpha, beta, phi, x, z, y), the first three in degrees and the last three in the scan's length unit. Lengths can be in pixels for uncalibrated data. Values distinguish counts, transmission, line integrals, attenuation and mixed-unit pose columns.

The compact UTF-8 encoding of `info` is limited to 65,536 bytes. Bulk members are externalised before scalar JSON conversion. Their positions in `summary` are null and their descriptors are in `arrays`, keyed by RFC 6901 JSON Pointer. The scalar summary retains method selection details such as `automatic`.

## Python hand-off

Decode a line as `tomojax.cli.envelope.Request`. For a queued command the adapter receives its typed `ExecutionInput` and an `ExecutionContext`. The owner resolves and pins the input references before queuing and supplies the leases, control, event sink, immutable allocator snapshot sink and private staging allocator. Adapters return `Success` or raise. They may only write to paths supplied by staging. An array reservation has distinct private and eventual public paths. Output reservations within one admission form a durable group, including manifests.

A returned `NewHandle` gives logical bytes, a pure `bind(id)` returning the published product and member map, and a disposal callback. The owner assigns the ID and checks the channel revision at commit. Losing the race discards staged files and unpublished resources. Disposal after successful registration waits for the last lease. The owner emits all worker terminal envelopes. Worker transport, admission limits, writer policy and durable commit implementation belong to the worker milestone.

## Ownership after Milestone 0

| Files or registration | Sole owner |
|---|---|
| Array, progress, operations and products schemas under `docs/protocol/`, their Python types, wire option generation, input/result matrix and workflow transcripts | Execution agent |
| Worker schema, `cli/envelope.py`, schema assembly in `tools/build_wire_schemas.py`, session transcript, protocol transport documentation and writer tests | Worker agent |
| `.importlinter`, `tests/guardrails/api_surface.txt`, `tests/guardrails/ratchets.json`, `tools/check_docstrings.py` and all of `tools/guardrails.py` including `CLI_COMMANDS` | Execution agent integrates registration and measured deltas from both branches |
| Product commands in `cli/api.py`, dispatch/help in `cli/main.py` and CLI vocabulary/help entries in `tests/test_architecture.py` | Worker agent integrates command changes from both branches. |
