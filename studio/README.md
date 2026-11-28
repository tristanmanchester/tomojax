# TomoJAX Studio

TomoJAX Studio is a native desktop workspace for tomography. A project on this computer records your scans, processing settings and results. Choose where the data lives and the jobs run when you create the project. You can inspect slices, sinograms and volumes while a job runs.

![TomoJAX Studio showing the aligned FISTA-TV reconstruction of a 29 mm mechanical watch movement as a 3D GPU volume render, with linked axial, coronal, sagittal, projection, and sinogram thumbnails below, the project tree and saved recipes on the left, and volume appearance and clipping controls on the right.](../docs/images/studio-explore-volume.png)

## Install and run

TomoJAX Studio runs on Linux `x86_64` and `aarch64` (for example an NVIDIA DGX Spark) with a working Vulkan driver.

### Install a prebuilt wheel

Install `tomojax-studio` into the same environment as `tomojax` (use `cuda12` on an NVIDIA GPU, or `cpu` for CPU execution; `--prerelease=allow`, or `--pre` with `pip`, is needed while `1.0.0` is in release candidate):

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python --prerelease=allow "tomojax[cuda12]" tomojax-studio
.venv/bin/tomojax studio
```

The `tomojax-studio` wheels (`manylinux_2_28_x86_64` and `manylinux_2_28_aarch64`) bundle HDF5 statically, so they require no Rust toolchain or system `libhdf5` package. Prebuilt wheels and standalone tarballs are also attached to each GitHub Release.

### Build from source

To build Studio from a source checkout, install Rust **1.99.1**, a C compiler, and the native build packages (`libhdf5-dev`, `libfontconfig1-dev`, `libxkbcommon-dev`, `pkg-config` on Debian/Ubuntu; or omit `libhdf5-dev` and build with `--features static-hdf5` when `cmake` is available):

```sh
uv sync --locked --extra cuda12 --dev
uv run --no-sync tomojax studio --build
```

The launcher forwards the active Python interpreter to Studio and checks source builds, the active environment's `bin/`, and `PATH`. Subsequent launches reuse the compiled or installed binary:

```sh
tomojax studio                        # open the start screen
tomojax studio /path/to/project       # reopen a saved project
tomojax studio /path/to/scan.nxs      # launch with an acquisition ready to import
tomojax studio --demo                 # create a demo project and simulate a scan in it
```

## Workspaces at a glance

Five top-bar workspaces cover the full tomography loop, while the left sidebar keeps the project tree and saved recipes in view:

| Workspace | Purpose |
| --- | --- |
| **Prepare** | Import acquisitions, calibrate raw frames, run corrections, crop/bin/select views, align geometry, and preview slices or center sweeps |
| **Reconstruct** | Configure analytical, iterative, or slab reconstructions with live parameter validation and memory plans, then queue jobs |
| **Explore** | Inspect 3D GPU volume renders and linked axial, coronal, sagittal, projection, and sinogram views; take physical distance and ROI measurements; export volumes |
| **Compare** | Compare two reconstructions side by side, with a draggable split curtain, or as a difference map on a shared display window |
| **Jobs** | Follow active progress, live slice previews, per-view pose tables, convergence plots, and the full processing record |

## Start a project and import data

Choose **Create a project**, name its folder, and add acquisitions in **Prepare** using **Add scans**, **Add a scan folder**, or drag-and-drop. Drop multiple files or choose a directory containing several NXtomo files to import a batch.

- **Referenced vs managed copies:** by default, Studio references files in place. Enable **Copy sources into the project** before importing to stage managed copies under `inputs/` while leaving originals untouched. For header-driven formats such as Nikon `.xtekct` or SkyScan `.log`, copying brings along the whole acquisition folder. Reference HDF5 files that use external links, virtual datasets, or external raw storage in place.
- **Supported formats:** NXtomo/HDF5, Data Exchange, ESRF Bliss/EDF/PyHST, Nikon `.xtekct`, Zeiss `.txrm`, SkyScan logs, PSI cSAXS `.mat`, TIFF stacks, and numeric `.npy` arrays. For older TomoJAX datasets, install the migration release with `python -m pip install "tomojax[cpu]==1.0.0rc7"` and re-save them before importing.
- **Native preview and full inspection:** standard NXtomo/HDF5 metadata and initial views load through the Rust worker while the numerical executor starts. Click **Inspect dataset** to run TomoJAX's full inspection report, including projection statistics across the stack.
- **Saved datasets:** Studio reads the registration and default plot of format-2 TomoJAX files directly. Opening an rc6 file gives the Python commands to re-save it before importing.
- **Calibration:** use **Open & calibrate** when an acquisition needs explicit loader settings (raw-frame kind, flats, darks, angles, pixel pitch, or declared geometry). The form is generated from the worker's parameter schema and reports any missing fields before queuing.

To try Studio without scan files on disk, choose **Try a demo project** on the start screen or launch with `tomojax studio --demo`. Studio creates a new project under its data folder (`~/.local/share/tomojax/Demo projects` on Linux) and simulates a phantom scan with its ground truth; reconstruct it in **Reconstruct**. In any project, **Simulation** in **Prepare** does the same, and **Per-view shift (SD)** adds random per-view motion to try **Alignment** on.

## Create a project on another machine

Follow [Reconstruct a scan on another machine](../docs/remote.md) to set up SSH, install the executor, choose a results folder and queue a workstation or Slurm job. Reopen the project folder on the laptop to inspect the finished reconstruction.

## Prepare, preview, and reconstruct

**Prepare** and **Reconstruct** expose the full public TomoJAX operation set: calibration (`Open & calibrate`), flat/dark and artifact corrections, alignment (direct parallel center measurement and joint volume-and-pose alignment), simulation, view selection, detector cropping and binning, pose tables, multi-orbit scan combination, analytical and iterative reconstruction, slab reconstruction, forward projection, and backprojection.

### Live interactive previews

Four interactive previews update automatically as you adjust settings, without adding jobs to the project history:

- **Slice preview:** reconstructs a single plane at its physical orientation.
- **Center sweep:** evaluates one offset with the center slider, or an ordered list of signed detector offsets in pixels in **Centers**, so you can step through candidates and pick the sharpest rotation center.
- **Correction preview:** switches between **Before correction** and **After correction** sinograms on a row band, keeping native count and corrected line-integral registrations separate.
- **Estimate support:** runs TomoJAX's calibrated laminography region estimator and displays its three orthogonal overlays and physical bounds.

Changing a control supersedes any in-flight preview request while keeping the last image on screen until the new one arrives. Use **Stop** to cancel a preview, **Update preview** to rerun it, or **Keep result** to promote a preview into a durable project asset with its parameters and provenance.

### Validation, plans, recipes, and batch queuing

- **Guided controls and advanced settings:** common correction options (zingers, rings, Paganin phase retrieval) have dedicated fields; ordered multi-step corrections, solver `*Config` options, and nested settings are available in the advanced fields and JSON editor using the worker schema's field names.
- **Live validation and plans:** edits validate automatically against the worker and highlight exact parameter paths on error (**Review** focuses the failing field). Reconstruction and alignment forms also request the worker's execution plan—showing the resolved method, reasons and grid, plus device placement and memory estimates for inline jobs—before you submit.
- **Recipes and batches:** save any configured operation as a named recipe in the sidebar to reuse across scans; applying a recipe populates the form for review before queuing. Select multiple scans in the project tree to queue the same operation across all of them, or choose **Combine scans** to merge selected cone-beam orbits in project order.
- **Immutable job capture:** submitting a job snapshots its parameters and input paths. Continuing to edit forms, apply recipes, or relink assets leaves queued and running work unchanged.

On this computer, Studio runs one job at a time in its worker. You can reorder waiting jobs, browse volumes, or prepare the next batch while a job runs. Interactive previews share the same worker, so when a durable job is running or paused the preview footer links directly to that job's controls.

## Explore, measure, and compare

**Explore** pairs a main viewport (**3D** volume render or **Slice** view) with a linked thumbnail strip showing **Axial**, **Coronal**, **Sagittal**, **Projection**, and **Sinogram** views whenever those arrays are present.

![Axial slice through the hairspring and balance wheel of the aligned watch movement in Explore, showing physical distance and region attenuation measurements in millimeters and inverse millimeters alongside the display window histogram.](../docs/images/studio-explore-slice.png)

| Interaction | Action |
| --- | --- |
| Drag in **3D** mode | Orbit the volume camera |
| Scroll wheel over a slice | Step through slices |
| Scroll wheel in **3D**, or primary-modifier + wheel on a slice | Zoom |
| Middle-button drag over a slice | Pan |
| Click or drag in **Navigate** mode | Move linked crosshairs across all orthogonal views |
| Double-click or **Fit image** | Reset zoom and pan |
| **Distance**, then drag on a slice | Measure physical distance between sample centers |
| **Region**, then drag a box on a slice | Compute mean, standard deviation, and range of finite samples |

- **Display controls:** adjust the grayscale window (**Apply window** commits typed bounds; **Auto** resets from the image), choose a color map (**Gray**, **Bone**, **Viridis**, **Magma**), or invert display contrast. Display settings change rendering on the GPU without altering stored float values.
In **Navigate** mode, press a plane to read the nearest displayed sample and its source row and column. Zoom in to inspect finer detail. Studio fetches the visible region at the pane’s resolution, down to individual source pixels. The probe appears when lossless samples arrive.

- **3D volume rendering:** tune **Opacity threshold**, **Density**, **Lighting**, and **X / Y / Z extent** clipping planes. Studio renders interactive 3D previews at up to 192³ voxels.
- **Measurements:** saved distance and ROI measurements stay tied to their asset, slice index, and physical coordinate frame, and reappear whenever you return to that slice.

### Compare results

Open **Compare** and pick a **Left** and **Right** result in the inspector (or select two results in the project tree and choose **Compare selection**). Pick **Axial**, **Coronal**, or **Sagittal**, then switch between three modes:

| Mode | Shows |
| --- | --- |
| **Side by side** | Both results locked to the same slice, zoom, and pan |
| **Split** | Left result on one side of a draggable curtain, right result on the other |
| **Difference** | Left minus right on a grayscale window symmetric around zero |

**Split** and **Difference** compare results on matching grids. Enable **Share display window** to lock both sides to one intensity range, or disable it to scale each side independently.

![Compare workspace in Split mode comparing nominal geometry FISTA-TV on the left against aligned FISTA-TV on the right on the same axial plane and shared display window, showing doubled features before alignment and sharp gear teeth and mainspring coils after.](../docs/images/studio-compare.png)

## Follow jobs and convergence

**Jobs** displays running, queued, and completed work with nested task progress, active-time ETAs, and live slice previews (orthogonal slices for iterative reconstruction and alignment, plus per-view pose tables for alignment). **Activity** in the top bar opens a compact drawer of active jobs from any workspace.

Expand **Details** on any job to inspect its convergence plots (such as alignment objective and accepted pose step across pyramid levels), captured parameters, recorded warnings, and **Show worker trace** for internal worker stages. You can pause an inline solve at its next iteration, slab or alignment step. For a batch job, Activity shows the scheduler state and the devices reported by the job. Cancel acts on the selected job even while the session executor waits in a queue.

![Jobs workspace with Details expanded on the watch-movement alignment job, showing multi-level Alignment objective and Accepted pose step convergence plots alongside the processing record.](../docs/images/studio-jobs.png)

## Project layout, recovery, and export

The manifest lives in the local project folder. Data paths below are relative to the project’s results folder on its connection.

```text
project.tomojax.json   # assets, recipes, job records, provenance, and workspace state
inputs/                # managed source copies (when copying is enabled)
jobs/                  # immutable scripts, progress records, checkpoints and published results
kept/                  # saved interactive previews
cache/                 # local viewer cache
```

- **Autosave and locking:** workspace state—active asset, slice indices, display window, 3D camera, clipping, comparison pair, measurements, and parameter drafts—saves atomically in the background. An OS file lock ensures only one Studio window writes to a project at a time. The manifest holds up to 32 MiB; if a write fails, the previous manifest stays intact and the status bar reports the error.
- **Managing assets:** rename results inline in the project tree, or remove unused entries from the project (files on disk remain untouched). Assets referenced by a queued or running job stay protected until that job completes or is canceled.
- **Session recovery:** process and Slurm jobs keep running after disconnection. Reopening the project retrieves their progress and results. Inline work, including jobs on this computer, ends with the worker session. A cancelled or interrupted solve with a usable checkpoint offers **Resume**. If the scheduler cannot locate a job, **Abandon** leaves a cancellation request against that job’s recorded identity.
- **Export:** export derived results from **Explore** as TomoJAX NeXus (`.nxs`, preserving geometry and included scan), 32-bit float or scaled 16-bit TIFF slice stacks, or NPY bundles with `arrays.json` registration metadata for array-only previews.

## Shortcuts, preferences, and environment variables

Press **Ctrl+K** or click **Search commands** to run any command or open any operation, such as **Alignment**, from the keyboard. Open **Shortcuts** and click **Record** to rebind keys (conflicts and reserved keys are flagged immediately). View undo/redo steps through display and navigation changes without affecting queued or completed jobs.

| Environment variable | Purpose |
| --- | --- |
| `TOMOJAX_STUDIO_PYTHON` | Python interpreter for `tomojax serve` when launching the binary directly (`tomojax studio` sets this automatically) |
| `TOMOJAX_STUDIO_SLOWLINK` | Development override for the whole project. Fetch planes through the worker using a JSON array of relay arguments, or `[]` for a direct pipe. |
| `TOMOJAX_STUDIO_SETTINGS` | Override the preferences file path (recent projects and custom shortcuts) for isolated runs |
| `TOMOJAX_STUDIO_INPUT` | Project directory or acquisition path opened on startup |

## Export from Explore

Select a result and choose its output format in Explore. **TomoJAX result (.nxs)** saves a result for reopening. For interchange, choose a file or a parent directory and enter the options for that format. Leave an option blank to use the worker default. The crop and range fields accept JSON arrays. The worker shows the number of files, uncompressed bytes and coordinates or calibration that the reader can recover. **Queue export** becomes available when the plan matches the current input and settings. Array-only results export as registered NPY bundles.

## Verification

From the repository root, `just studio-check` runs formatting, Clippy and the tests, and `just studio-verify` builds Studio. It captures the Start screen, each workspace of a small simulated project, Prepare and Reconstruct again with TomoJAX ready, the remote file browser, transfers, connection states, and seven Activity states from a worker using a test Slurm scheduler into `.artifacts/studio-snap/`. It fails on a missing, wrongly sized or blank image; look at the images to judge a visual change. The underlying commands:

```sh
export RUSTUP_TOOLCHAIN=1.99.1
cargo fmt --all --manifest-path studio/Cargo.toml --check
cargo clippy --locked --manifest-path studio/Cargo.toml --workspace --all-targets -- -D warnings
cargo test --release --locked --manifest-path studio/Cargo.toml --workspace
TOMOJAX_TEST_PYTHON="$PWD/.venv/bin/python" cargo test --release --locked --manifest-path studio/Cargo.toml --workspace -- --ignored
uv run --no-sync python tools/verify_studio.py --build
```

To run the same capture suite through the local worker at 1.2 MB/s and 54 ms round-trip delay, run this from the repository root.

```sh
TOMOJAX_STUDIO_PYTHON="$PWD/.venv/bin/python" \
TOMOJAX_STUDIO_SLOWLINK='["'"$PWD"'/tools/slowlink.py","--rate","1200000","--delay-ms","27"]' \
just studio-verify
```

Studio and its worker share a Cargo workspace. `studio-core` handles projects, the worker client and previews. `tomojax-studio` renders the desktop. `tomojax-worker` serves files and manages the Python executor, using `tomojax-io` for storage and `tomojax-jobs` for job execution. `studio-core` builds without Rune, wgpu or HDF5, so `cargo test --manifest-path studio/Cargo.toml -p studio-core` checks the protocol, manifest and state-machine logic in seconds.

The `--ignored` suite exercises the live Python worker (preview supersession, validation, plans, pause/cancel boundaries, and durable saves), checks the native NXtomo reader against what `tj.load` and `tj.inspect` read from files TomoJAX writes, and checks GPU rendering pixels and opacity against analytic expectations.

Studio also supports headless offscreen rendering through the same scientific compositor:

```sh
TOMOJAX_STUDIO_INPUT=/path/to/project studio/target/release/tomojax-studio \
  --snap /tmp/studio.png --size 1440x900 --scale 2 --at 1.0
```

See [DESIGN.md](DESIGN.md) for the visual system and [ARCHITECTURE.md](ARCHITECTURE.md) for ownership and persistence boundaries.
