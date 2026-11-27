# tomojax.cli

`tomojax.cli` implements the `tomojax` console script.

## Conventions

- Commands that produce a file (`correct`, `reconstruct`, `align`, `export`) take
  `INPUT -o OUTPUT [options]`.
- `inspect INPUT [--json] [--preview-dir DIR [--slice AXIS=INDEX] [--centers SPEC]]`
  reports on a dataset without `-o`.
- `simulate -o OUTPUT [options]` generates a synthetic scan without an input.
- `serve`, `mcp`, and `studio [PROJECT_OR_INPUT] [--build] [--demo]` launch the
  JSON Lines worker, the MCP stdio server, and the native desktop workspace.
- `skill [-o SKILLS_DIR]` prints where the Agent Skill is, or copies it into
  `SKILLS_DIR/tomojax`.
- An existing output is refused unless `--force` is given, and an output may never
  collide with an input or another output of the same run.
- `tomojax <command> --help` lists the command's options. Expert settings are
  not on the help page: put them in a TOML file and pass it with
  `--config FILE` (`correct`, `reconstruct`, `align` and `simulate`).
  `tomojax <command> --config-keys` lists the keys a file may set, with their
  defaults (`tomojax reconstruct --method NAME --config-keys` filters to one method);
  explicit command-line options override the file.
- Exit status is 0 on success, 1 on failure (for `inspect`, also when the
  dataset is invalid or a preview fails), 2 for a usage error, and 130 when a command is interrupted.
- `tomojax --version` prints the version.
- A usage error prints the message, a hint where the fix is known (`--force` for an
  existing output, the close match for a mistyped command, choice, option or config
  key) and the command's one-line usage. With `--progress json` it is also an
  `error` record on stdout, whose message carries no hint.

The options match the Python API's keywords: `tomojax reconstruct --method cgls
--iterations 50` is `tj.reconstruct(scan, "cgls", iterations=50)`, and
`tomojax align --fit translations roll` is
`tj.align(scan, fit=("translations", "roll"))`.

With no `--fit`, alignment estimates per-view translations jointly with a volume. `--fit` takes one or more of `center`, `translations`, `tilt`, `roll` and `rotations`. On an unposed, unrolled axial parallel scan with a well-sampled half-turn, `--fit center` alone measures the rotation axis directly from the projections with no working image unless `--method joint`, `--grid`, `--region`, `--config` or a joint solver option is given. Reconstruction without `--method` selects its method from the geometry.

## Commands

- `inspect`: `tj.inspect`, describing a dataset and checking that it can be reconstructed;
  `--json` prints the report as JSON. Its Loading section says what loading reads from the
  file alone: pixel units when no pixel size is recorded, a rotation center or geometry
  the file does not record, or the frames a stopped scan was cut to. Given a checkpoint
  directory, it reports the stopped run instead. `--preview-dir DIR` writes PNGs: the central
  projection when the file holds projections, and slices named
  `slice_<axis>_<index>.png`. A saved reconstruction's slices are read from the file
  (its three central planes by default); a scan's are reconstructed (its central z
  plane by default). `--slice AXIS=INDEX` chooses planes and can be repeated.
  `--centers SPEC` sweeps the detector center on a scan: it reconstructs the plane of
  the first `--slice` (or the central z plane) at each center of `START:STOP:STEP`
  or a comma list, in pixels from the detector middle, and writes one image per
  candidate and a contact sheet, `center_sweep.png`. The report lists each candidate
  with the `U` to pass to `tomojax correct --detector-center U V`.
- `correct`: import raw acquisitions or datasets, supply missing metadata (`--flats`,
  `--darks`, `--angles`, `--pixel-size`, `--detector-center`, `--geometry`, `--tilt`
  and cone parameters), and apply optional corrections (`--zingers`, `--paganin`,
  `--rings`, `--reject-views`) and `--views`, `--crop` and `--binning`.
  On raw frames every step runs; on a dataset of line integrals only the steps that
  act on them do. `--flats` and `--darks` accept frame stacks, scalar levels, or
  another scan's HDF5/NeXus file (taking only its flat or dark frames). Bracketing
  multi-frame flat sets are weighted from a sample of the views by default while
  interpolating common row gain by position; when views move the flat by more than
  1% over more than 1% of the detector, `correct` prints a warning naming both
  remedies (`--flats` from another scan, or the expert `--flat-weights position`
  flag, also `flat_weights = "position"` in `--config`; see
  [flat fields](../corrections/README.md#flat-fields)). `--detector-center U V`
  sets the first detector's center, in the
  pixel size's unit, after everything else has loaded, and moves every segment of a
  segmented scan by the same amount. `--preview PNG` draws a band of output rows as
  sinograms before and after the steps and their difference (the expert
  `--preview-rows A:B` chooses the band). `--dry-run` writes no scan: it prints the
  steps in run order, the output shape and size, and writes the preview if asked.
- `reconstruct`: choose automatically, or reconstruct with Fourier, FBP/FDK, CGLS,
  FISTA-TV, SPDHG-TV, SIRT or MLEM with `tj.reconstruct`, with the poses saved in
  the input unless `--no-apply-poses`. Its expert settings are fields of the method's
  configuration class (`tomojax.recon.FBPConfig`, `FourierConfig`, `CGLSConfig`,
  `FISTAConfig`, `SPDHGConfig`, `SIRTConfig` or `MLEMConfig`), each replacing that
  field of the class's defaults, as `tj.reconstruct(config=...)` takes them;
  `--region cyl` zeros the volume outside the cylinder every view sees. By
  default (`exterior = "cylinder"`), `fbp` (including cone-beam FDK) and
  `fourier` continue each detector row past both edges with a cylinder model
  for truncated scans; pass `exterior = "zero"` in `--config` with an explicit
  `--method` to assume zero beyond the detector. `reconstruct` records
  `truncation` (`edge_ratio`, `outside_fraction`, `interior_shift`, `radius`,
  `field_of_view_radius`) in the output file and `--manifest`, and appends
  `, N% of attenuation beyond the detector` to its closing line whenever
  `outside_fraction` rounds to at least 1% (0.5% or above). Omitting
  `--iterations` enables automatic stopping for `cgls` and full-data `sirt`
  (estimated projection discrepancy, capped at 50 updates, with FISTA momentum
  enabled on full-data `sirt` when `relaxation <= 1`) and for `fista` (estimated
  proximal-gradient noise score, capped at 200 updates). Ordered-subset `sirt`
  and `mlem` run 50 sweeps, and `spdhg` runs ceil(600 × √B) updates for B sampled data blocks. Passing
  `--iterations N` fixes the budget (and runs plain sweeps for `sirt` unless
  `acceleration = true` is also set in `--config`). For iterative solvers,
  `reconstruct` prints a closing line stating how many updates ran, why the
  solve stopped, and any automatic TV weight; check `termination`,
  `early_stop`, and `effective_iterations` (plus `resolved_tv_weight` for
  `fista`) in `--manifest` for the same fields in JSON. The output
  holds the volume, its geometry and record, and names the input projections;
  `--include-scan` keeps them too. `--checkpoint DIR` (also accepted as `checkpoint` in
  `--config`) saves solver state and finished slabs in `DIR` so a killed run resumes
  when the same command runs again, exiting `130` on `SIGTERM` or Ctrl-C after saving and
  deleting `DIR` once all outputs are written; `tomojax inspect DIR` shows what it holds
  and the command line that started it
  ([resume a run after a kill](../../../docs/large-scans.md#resume-a-run-after-a-kill)).
  `--seed`, SPDHG's view order, is an expert option not listed by `--help`.
  `--dry-run` prints the plan: the method and why,
  the grid, and the estimated memory against the memory available.
- `align`: estimate the named geometry corrections with `tj.align`, on top of the poses saved in
  the input unless `--no-apply-poses`. On an unposed, unrolled, well-sampled axial parallel scan,
  `--fit center` measures the rotation center directly across the detector width from the half-turn
  sinogram and opposing rays without writing a working image; pass `--method joint` (or `--grid`,
  `--region`, `--config`, or a joint solver option) for a joint image-and-center fit, which starts
  from the measured center. For cone data, `--fit center` fits detector-u and detector-v jointly;
  `--fit translations tilt roll` adds shared setup angles to per-view shifts, and
  `--fit translations rotations` adds per-view sample rotations. `--motion`, `--iterations`,
  `--image-iterations`, `--tolerance`, `--interpolation` and `--resolution` are `tj.align`'s
  keywords; a `--config` file sets the other `tomojax.alignment.AlignConfig` fields
  (including `exclude_views`, `detector_columns`, and `image_tv_weight = "coverage"`).
  Joint fits print the `disturbed_views` screening summary when an object may cross the
  field of view.
  `--method opposite-views` runs the pair-only center check from near-opposite views without an
  image. `--grid` sets image sampling. `--checkpoint DIR` saves finished resolution levels,
  within-level solver state, and refinement poses in `DIR` so a killed fit resumes when the
  same command runs again, exiting `130` on `SIGTERM` or Ctrl-C after saving and deleting `DIR` once
  all outputs are written ([resume a run after a kill](../../../docs/large-scans.md#resume-a-run-after-a-kill)).
  `--dry-run` prints the plan: sampled views and pairs for
  direct axial center measurement, angular pairs for `opposite-views`, or each level's grid and
  budget for a joint fit, along with estimated memory.
- `export`: `tj.export`, writing one registered member as ImageJ TIFF (`.tif`, `.tiff`),
  OME-TIFF (`.ome.tif`, `.ome.tiff`), a TIFF slice directory, raw binary (`.raw`),
  MetaImage (`.mhd`), or AmiraMesh (`.am`) according to OUTPUT, with optional `--crop`,
  `--binning`, `--axis`, `--dtype`, `--range`, `--compress`, and `--member`. Use
  `--dry-run` to inspect the files, axes, value mapping, and space estimates before
  writing. See [export](../../../docs/export.md).
- `simulate`: `tj.simulate`, writing a synthetic scan of a phantom with the phantom as its volume.
  Use `--translation-sigma DIST` to simulate independent object shifts with standard
  deviation `DIST` voxel pitches along each detector axis.
- `serve`: run the local JSON Lines worker over stdin/stdout for GUI and automation clients.
  See the [local worker protocol](../../../docs/protocol.md).
- `mcp`: run the Model Context Protocol server over stdio so AI agents such as Claude Code or
  Codex can inspect, process, view, and script scans directly. Requires `tomojax[mcp]`; see
  [use TomoJAX from an AI agent](../../../docs/mcp.md).
- `skill`: print where the TomoJAX Agent Skill is in this installation and the directories
  common agents read skills from, or copy it there with `-o SKILLS_DIR` (`--force` replaces
  an existing copy). See [use TomoJAX from an AI agent](../../../docs/mcp.md).
- `studio`: launch [TomoJAX Studio](../../../studio/README.md) with the active Python
  environment. Pass an optional `PROJECT_OR_INPUT` path to open a saved project or queue an
  acquisition for import, `--build` to compile the native binary from a source checkout, or
  `--demo` to start a sample simulation workflow.

## A lab CT scan from start to finish

```bash
tomojax correct scan/scan.xtekct --darks dark.tif -o scan.nxs
tomojax inspect scan.nxs
tomojax align scan.nxs -o aligned.nxs --fit center roll
tomojax reconstruct aligned.nxs -o recon.nxs
tomojax inspect recon.nxs --preview-dir previews
tomojax export recon.nxs -o slices/
```

A file that records the pixel size but not the rotation center (external
NXtomo or Data Exchange, ESRF Bliss or EDF) loads with the center at the middle
of the detector, and `reconstruct` reports that status. A file that records no
rotation axis or pixel scale warns on load (`geometry: not recorded` under
`Loading` in `inspect`) and requires `--geometry` on `tomojax correct` (for
example, `--geometry parallel`) before `align` or `reconstruct`. Fit the center
first:

```bash
tomojax align scan.h5 -o aligned.nxs --fit center
tomojax reconstruct aligned.nxs -o recon.nxs
```

Nikon scans record no dark field: `--darks` takes a dark image or one level,
and `--darks 0` suits only counts that are already dark-corrected.
`align --fit center roll` estimates the cone detector's center and roll,
and `reconstruct` uses them. `previews/` then holds the three central slices, such
as `slice_z_256.png`. See the [lab cone-beam CT guide](../../../docs/lab-ct.md).

To judge the center by eye instead, sweep it and look at the contact sheet:

```bash
tomojax inspect scan.h5 --preview-dir sweep --centers -20:20:2
tomojax correct scan.h5 -o centered.nxs --detector-center U V   # U and V as listed
```

## Follow or cancel a command

The six data commands (`inspect`, `correct`, `reconstruct`, `align`, `export`, and `simulate`)
accept `--progress auto|bar|json|none`. The default, `auto`, shows
live progress when stderr is a terminal: a row per running task with its count,
time left and metrics, and a line left behind for each finished level or solver,
such as `✓ binning 2  4/12 steps  objective 4.38 → 0.0833  13.5s`. `bar` off a
terminal prints only those finished lines and warnings; `none` prints warnings only.
Press Ctrl-C to cancel at the next work boundary (batch, iteration, slab, or
alignment step). The
command exits 130 after cleanup. A second Ctrl-C interrupts immediately. When
`--checkpoint DIR` is set on `reconstruct` or `align`, Ctrl-C or `SIGTERM`
finishes the current iteration, slab, or step, saves state to `DIR`, prints a
message naming `DIR`, and exits 130; with `--progress json` the `error` record
carries `checkpoint: DIR`.

`reconstruct --watch-dir DIR` and `align --watch-dir DIR` rewrite PNGs of the
solve's central planes in DIR as the solve progresses (`slice_z.png`, `slice_y.png`,
`slice_x.png`, and `poses.png` for an alignment). Keep an image viewer open on DIR
to watch a long run converge.

```bash
tomojax reconstruct scan.nxs -o recon.nxs --method cgls --progress json > progress.jsonl
```

In JSON mode stdout contains one flushed JSON record per line. TomoJAX's log
messages and warnings are `note` records; other packages' warnings and
tracebacks go to stderr. Each task has a `started` and a `finished` record.
Use `task` and `parent` to follow nested operations. The `name` field uses
a fixed vocabulary. Display the lower-case `label`, such as `binning 2`
for a `level` or `refine poses` for a `refine` task. An `advanced` record counts
completed work in the unit named by `started`. An unknown total is `null`,
and a later advance can revise the total. Times are seconds since the watch began.
The [v2 JSON schema](../../../docs/protocol/progress.schema.json) defines the records.

A successful command exits 0. Operational failures emit an `error` with a
machine-readable kind and exit 1. Cancellation closes active tasks, emits a
`canceled` error and exits 130. An internal exception emits an `internal` error,
prints its traceback on stderr with where to report the bug, and exits 1.
Unexpected EOF before completion can indicate external process termination.
Argument errors exit 2. A dry run with `--progress json` ends with a `result`
record whose `info.summary` is the plan.
