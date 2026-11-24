# TomoJAX Studio

Studio is a native Rune workspace for TomoJAX projects. It imports acquisitions,
queues public TomoJAX operations, saves derived results and restores the project
when you reopen it. The scientific viewer reads registered scalar arrays on its
own I/O threads and renders them with the same GPU that draws the interface.
Every numerical operation runs in `tomojax serve`, the same worker the CLI uses.

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
comes from the worker's parameter schema. Supply any calibration the scan lacks;
the worker reports what is missing.

To try Studio without data, choose **Simulation**, keep the small default
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
control replaces its earlier request. While the new preview computes, the last
one stays on screen with an updating label. **Stop** cancels that control's preview;
**Update preview** explicitly retries it. Preview requests do not wait for the
project's autosave or become durable queued jobs.

Centre sweeps are browsable at their recorded signed detector offsets. The centre
slider evaluates one offset; the **Centres** field accepts an ordered sweep.
Slice previews retain the requested physical orientation. Correction previews
provide **Before correction** and **After correction** views, with independent
detector registration and count/line-integral domains. **Estimate support** uses
TomoJAX's calibrated laminography estimator. Keeping the preview saves its three
fixed planes and physical bounds.

Common correction controls cover zingers, stripes, rings and Paganin phase
retrieval. Paganin's physical parameters must be supplied. Ordered correction
objects, repeated steps, solver configuration objects and uncommon structured
settings use the JSON editor or advanced parameter fields, with the worker
schema's field names and tags.

Select several scans to queue the same operation for each. **Combine scans**
captures the selected inputs in project order. Save a named recipe to reuse
settings; applying a recipe fills the form and leaves submission to you.
The form validates after edits settle. A worker parameter error names its exact
path; **Review** reveals and focuses the corresponding control. Nested JSON
errors identify the failing path beside the containing editor. A draft with a
current validation error cannot be queued. Reconstruction and alignment forms also request the worker's plan, including method, reasons,
grid, placement and memory estimates. Editing settings invalidates the previous
plan. For a normal batch, the preflight describes the active input; each captured
input is checked when its job runs. **Combine scans** preflights its selected set.

Each submitted job keeps its own copy of the parameters and records the source
paths, kinds and metadata fingerprints, including secondary array and coordinate
files. Editing a form, applying another recipe or relinking an asset leaves
submitted work unchanged. Inputs are checked again before a result is accepted; if
a captured source has changed or gone, the job fails and names it. Don't modify
referenced acquisitions while they are being processed.

The local queue admits jobs to one official worker. You can reorder waiting
jobs, prepare more work, or browse saved data while processing runs. Pause and
cancel are sent to the worker, and the job shows its new state once the worker
confirms it. A paused job keeps the worker. Interactive previews share the same
worker, so a running or paused job holds them until it finishes; the preview
footer says so and offers the job controls. Background file copies can be
cancelled, and only completed copies are added to the project.

## Follow a running job

**Jobs** shows the worker's task hierarchy, progress counts and active-time ETA.
Iterative reconstruction and joint alignment show their registered axial,
coronal and sagittal checkpoint planes as they arrive. Alignment also shows the
per-view pose table, with the worker's column order, coordinates and units.
The panel identifies a completed run's retained frame as its last checkpoint.
Each live plane shows its physical coordinate and automatic display window.

Pause, resume and cancel controls stay beside the job heading. Expand **Details**
for current metric values, convergence plots, captured settings and recorded
warnings; **Show worker trace** lists the worker's internal steps. Alignment
plots show the objective and accepted pose step against the worker's step
number, in the units of the protocol metric registry. Every metric sample stays
in the job record, and plots leave a gap where the worker reports no value.
**Activity** in the top bar lists running and queued jobs from any workspace.

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

Open **Compare** and pick a result for each side in the inspector, or select two
results in the navigator and choose **Compare selection**. Choose the plane (**Axial**, **Coronal** or
**Sagittal**) and the mode:

| Mode | Shows |
| --- | --- |
| **Side by side** | Both results at the same slice, zoom and pan |
| **Split** | Left result before the divider, right result after it; drag the divider |
| **Difference** | Left minus right, in a grey window symmetric about zero |

**Split** and **Difference** need both results on the same grid. **Swap**
exchanges the sides. **Share display window** applies one window to both;
otherwise each side uses its own automatic window.

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
their source paths. Rename a result with its edit button in the project tree.
Removing a result takes it out of the project after you confirm; its files stay
on disk. A result used by a queued or running job can't be removed. Derived handle results are saved through the official worker
to NXtomo before their jobs are marked complete. **Keep result** copies registered
array-only previews out of the worker's temporary session storage. Unkept
previews are not project assets.

Autosave writes a replacement manifest on a background thread. An OS file lock
prevents two Studio instances from writing the same project. The status bar shows
whether saving is pending or failed. Reopening restores the active asset, slice
positions, window, camera, clipping, comparison selections, measurements and
parameter drafts, including unfinished text edits.

The manifest is limited to 32 MiB, which a long metric history can reach. When
a save fails, the status bar says so and the previous manifest is kept; changes
since then exist only in memory until a save succeeds.

Jobs left active by a lost session become **Interrupted**, and you retry them
yourself. If an interrupted job's output destination already exists, check it
before retrying.

NXtomo export preserves the registered result and included scan. TIFF export
supports float32 and explicitly chosen uint16 conversion through TomoJAX's public
export operation. Outputs use new destinations. Array-only results export as an
NPY bundle with an `arrays.json` registration description.

Studio does not import NPZ files, because the NPZ loader accepts pickled
metadata. Import NXtomo/HDF5 or a recognised acquisition format instead. The NPY
reader accepts numeric arrays only.

## Commands and preferences

Use **Search commands** or Ctrl+K to find actions. To change a shortcut, open
**Shortcuts**, choose **Record** and press the new key combination; Studio rejects
conflicts and reserved keys. View undo/redo restores display context; it does not
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
Headless mode views saved project data without starting a worker.

See [DESIGN.md](DESIGN.md) for the visual system and
[ARCHITECTURE.md](ARCHITECTURE.md) for ownership and persistence boundaries.
