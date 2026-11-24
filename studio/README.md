# TomoJAX Studio

Studio is a native Rune workspace for TomoJAX projects. It imports acquisitions,
queues public TomoJAX operations, saves derived results and restores the project
when you reopen it. The scientific viewer reads registered scalar arrays on its
own I/O threads and renders them with the same GPU that draws the interface.

This is the third prototype. It uses the checkout's `tomojax serve` protocol;
there is no second reconstruction implementation inside the GUI.

## Build and run

Use this source checkout and its locked Python environment. Studio is an optional
native application; installing the Python wheel alone does not build its binary.

1. Install Rust **1.99.1**, a C compiler, and the HDF5 development library.
   On Debian or Ubuntu, the relevant native packages are `libhdf5-dev`,
   `libfontconfig1-dev`, `libxkbcommon-dev`, `pkg-config`, and a working Vulkan
   driver. On macOS, install HDF5 with your package manager. If HDF5 is outside
   the compiler's search path, set `HDF5_DIR` to its installation prefix.
2. From the TomoJAX repository root, install the Python dependencies:

   ```sh
   uv sync --locked --extra cpu --dev
   ```

   Use the appropriate TomoJAX accelerator extra instead of `cpu` when preparing
   a GPU reconstruction environment. The renderer's device and TomoJAX's compute
   device are separate choices made by their respective runtimes.
3. Build and launch:

   ```sh
   uv run --no-sync tomojax studio --build
   ```

   Later launches reuse the native binary:

   ```sh
   uv run --no-sync tomojax studio /path/to/project
   ```

The launcher forwards the current Python interpreter to Studio. Rune and its
patched dependencies are pinned in `Cargo.toml`; `Cargo.lock` records the full
native dependency graph. The first build needs access to those dependencies.

Linux file dialogs use the desktop's XDG portal. Dragging files or folders into
an open project is also supported. The file chooser selects one input at a time;
drop multiple inputs together, or select a directory containing several NXtomo
files, for a batch import.

## First project

Choose **Create a project** and select its directory. Add your scans in **Prepare**.
Import references the original files by default. Enable **Copy sources into the
project** before importing when you want managed copies. Copies are staged in
the background and the originals are preserved.

Standard NXtomo/HDF5 metadata and the first scientific views load independently
of the numerical worker. **Inspect dataset** requests TomoJAX's full inspection,
including its projection statistics; this can take longer on a large scan.
Other supported acquisition formats use TomoJAX's existing inspection and loading
paths. Raw or vendor acquisitions that need calibration become directly viewable
after preparation produces a registered result.

Use **Open & calibrate** for explicit loader settings such as raw-frame kind,
flats, darks, detector calibration, angle paths and declared geometry. The form
comes from the worker's parameter schema. Missing calibration is an error to
resolve, not an inferred measurement.

For a reproducible small exercise, choose **Simulation**, keep the small default
phantom, and submit it. The result contains the scan and its registered ground
truth. Select that result, choose **Reconstruct**, set a solver and submit a
reconstruction. Both outputs become durable project assets.

## Prepare and reconstruct

The operation selector exposes calibration, corrections, alignment, simulation,
crop, bin, view selection, poses, combine, reconstruction, slab reconstruction,
slice previews, centre sweeps, correction previews, support estimation, planning,
forward projection and backprojection.

Slice, centre, correction and region previews update as you edit. They use
temporary worker results; **Keep result** creates a project result with the
settings, source fingerprints and worker provenance that produced it. Changing a
control replaces its earlier request. While the latest value is pending, Studio
clears the previous preview's pixels. **Stop** cancels that control's preview;
**Update preview** explicitly retries it. Preview requests do not wait for the
project's autosave or become durable queued jobs.

Centre sweeps are browsable at their recorded signed detector offsets. The centre
slider evaluates one offset; the **Centres** field accepts an ordered sweep.
Slice previews retain the requested physical orientation. Correction previews
provide **Before correction** and **After correction** views, with independent
detector registration and count/line-integral domains. **Estimate support** uses
TomoJAX's calibrated laminography estimator. Keeping the preview saves its three
fixed planes and physical bounds. Unsupported geometry produces the worker's
explanation.

Common correction controls cover zingers, stripes, rings and Paganin phase
retrieval. Paganin's physical parameters must be supplied. Ordered correction
objects, repeated steps, solver configuration objects and uncommon structured
settings remain available through the JSON editor or advanced parameter fields.
Their field names and tags are the public worker's schema, not a Studio dialect.

Select several scans to queue the same operation for each. **Combine scans**
captures the selected inputs in project order. Save a named recipe to reuse
settings; applying a recipe fills the form and leaves submission to you.
The form validates after edits settle. A worker parameter error names its exact
path; **Review** reveals and focuses the corresponding control. Nested JSON
errors identify the failing path beside the containing editor. A draft with a
current validation error cannot be queued. Editing clears that old blocker while
a new check is pending. Reconstruction
and alignment forms also request the worker's plan, including method, reasons,
grid, placement and memory estimates. Editing settings invalidates the previous
plan. For a normal batch, the preflight describes the active input; each captured
input is checked when its job runs. **Combine scans** preflights its selected set.

The bundled and described schemas do not yet declare which fields apply to each
method and geometry. Studio therefore keeps those fields available and shows the
worker's validation errors. Conditional visibility needs applicability metadata
in `describe`; the GUI does not infer numerical compatibility rules.

Each submitted job owns a parameter snapshot and its source paths, kinds and
metadata fingerprints. Editing a form, applying another recipe or relinking an
asset does not retarget work already submitted. A changed or missing captured
source causes that job to fail with a reviewable explanation. Fingerprints
detect metadata changes; they are not a cryptographic snapshot of the pixel
content. Secondary array and coordinate files are included, and inputs are checked
again before accepting results. Keep referenced acquisitions stable during processing.

The local queue admits jobs to one official worker. You can reorder waiting
jobs, prepare more work, or browse saved data while processing runs. Pause and
cancel are cooperative requests: the interface waits for the worker's actual
state and terminal response. A paused worker retains its execution slot.
Interactive previews share that single device queue: a running or paused job can
hold them up. The preview footer reports that condition and offers job controls.
Background file copies can be cancelled and publish only completed staged data.

## Follow a running job

**Jobs** shows the worker's task hierarchy, progress counts and active-time ETA.
Iterative reconstruction and joint alignment show their registered axial,
coronal and sagittal checkpoint planes as they arrive. Alignment also shows the
per-view pose table, with the worker's column order, coordinates and units.
The panel identifies a completed run's retained frame as its last checkpoint.
Each live plane shows its physical coordinate and automatic display window.
An unfinished iterate can contain boundary values that dominate its contrast;
the displayed ranges describe those samples without changing them.

Pause, resume and cancel controls stay beside the job heading. Expand **Details**
to inspect its task history, current metric values, captured settings and recorded warnings.
Alignment plots show the objective and accepted pose step against the worker's
step number. Units come from the protocol metric registry. All observed metric
samples remain in the job record; plot geometry is bounded for rendering and
preserves gaps where the worker reports no numeric sample.

## Explore and compare

The main viewport can show the GPU volume or an orthogonal plane. Linked
thumbnails provide axial, coronal and sagittal slices, a projection and a
sinogram when those arrays are present.
The sinogram starts at the central detector row unless you restore an explicit
saved position.

| Interaction | Result |
| --- | --- |
| Drag a volume | Orbit the camera |
| Wheel over a slice | Move through slices |
| Wheel over a volume, or primary-modifier + wheel | Zoom |
| Middle-button drag over a plane | Pan |
| Click or drag a slice in navigation mode | Move linked crosshairs |
| Double-click, or **Fit image** | Reset zoom and pan |
| **Distance**, then drag a slice | Measure between registered sample centres |
| **Region**, then drag a slice | Report finite-sample mean, spread and range |

Measurements retain their asset, plane and source index. They are shown only on
that slice. Distances require axes in the same coordinate frame and unit. ROI
statistics use the displayed scalar samples and explicitly identify downsampled
data. Display windows and colour maps change rendering, not stored values.
**Apply window** commits the entered range; an invalid or unfinished edit leaves
the current image unchanged. Relinking data clears measurements tied to its old
pixels or calibration.
The numeric display-window fields initially show the range used by the current
image. Automatic updates preserve a text edit you have started.

Volume controls include colour/opacity mapping, an opacity threshold, density,
lighting and three clipping positions. Preview quality is labelled. The default
viewer requests at most 1,024 samples per plane edge and 192 samples per volume
edge; it reads a coarse result first when appropriate and refines it. Full
quantitative data remains in the source and in numerical exports.

Select two volumes and choose **Compare**. The two views share the axial index.
**Share display window** applies the same window to both; otherwise the right
image uses its own automatic window. Different grids or units are called out.
The prototype does not silently register, resample or subtract mismatched data.

## Save, reopen and export

A project directory contains:

```text
project.tomojax.json
inputs/
derived/
cache/
```

The manifest holds assets, recipes, job records, provenance and workspace state.
Managed paths are relative to the project directory. External references retain
their source paths. Derived handle results are saved through the official worker
to NXtomo before their jobs are marked complete. **Keep result** copies registered
array-only previews out of the worker's temporary session storage. Unkept
previews are not project assets.

Autosave writes a replacement manifest on a background thread. An OS file lock
prevents two Studio instances from writing the same project. The status bar shows
whether saving is pending or failed. Reopening restores the active asset, slice
positions, window, camera, clipping, comparison selections, measurements and
parameter drafts, including unfinished text edits.

The manifest has a 32 MiB limit. If accumulated metadata exceeds it, Studio
reports the save failure and retains the latest state in memory; it preserves the
previous valid manifest. A large metric history can reach this limit. Continuing
with an unresolved save error does not make those changes durable.

Jobs left active by a lost session become **Interrupted**. Retry is explicit.
If the recorded output destination already exists, inspect it before submitting
new work; the prototype does not assume a missing terminal means a failed commit.

NXtomo export preserves the registered result and included scan. TIFF export
supports float32 and explicitly chosen uint16 conversion through TomoJAX's public
export operation. Outputs use new destinations. Array-only results export as an
NPY bundle with an `arrays.json` registration description.

NPZ import is disabled in Studio because the current core NPZ loader accepts
pickled metadata. The native NPY reader accepts numeric arrays and rejects object
arrays. Use NXtomo/HDF5 or a recognised acquisition format for GUI imports.

## Commands and preferences

Use **Search commands** or `mod+k` to find actions. `mod` means Command on macOS
and Control on other platforms. Keyboard shortcuts are editable, with conflict
and reserved-key checks. View undo/redo restores display context; it does not
undo committed numerical results or rewrite submitted jobs.

Application preferences store recent projects and shortcuts. Set
`TOMOJAX_STUDIO_SETTINGS` to a different preferences file for isolated testing.
`TOMOJAX_STUDIO_PYTHON` selects the worker interpreter when launching the binary
directly; the Python launcher sets it automatically.

## Verification

From the repository root:

```sh
export RUSTUP_TOOLCHAIN=1.99.1
cargo fmt --manifest-path studio/Cargo.toml --check
cargo test --release --locked --manifest-path studio/Cargo.toml
TOMOJAX_TEST_PYTHON="$PWD/.venv/bin/python" cargo test --release --locked --manifest-path studio/Cargo.toml -- --ignored
```

The ignored gate requires the installed worker and a working wgpu adapter. It
exercises the real worker's preview supersession, validation, plans, live
checkpoints and durable workflow. GPU tests check actual pixels and analytic opacity.
Rune's headless renderer uses the same scientific compositor as a native window:

```sh
TOMOJAX_STUDIO_INPUT=/path/to/project studio/target/release/tomojax-studio --snap /tmp/studio.png --size 1440x900 --scale 1 --script /path/to/script.json --at 3
```

This uses the release executable produced by the **Build and run** steps above.
Ordinary headless mode views saved project data without starting a worker. The
optional real-worker test captures freeze observed application state using the
same renderer; they are separate from that saved-project mode.

The accompanying draft review records the hardware, data, timing observations and
test results obtained for this build. Those measurements qualify this prototype;
they do not establish workstation GPU performance, production-scale acquisition
coverage, or validation on every desktop platform.

See [DESIGN.md](DESIGN.md) for the visual system and
[ARCHITECTURE.md](ARCHITECTURE.md) for ownership and persistence boundaries.
