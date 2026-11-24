# tomojax.cli

`tomojax.cli` implements the `tomojax` console script.

## Conventions

- Commands that produce a file (`correct`, `reconstruct`, `align`, `export`) take
  `INPUT -o OUTPUT [options]`.
- `inspect INPUT [--json] [--preview-dir DIR [--slice AXIS=INDEX] [--centres SPEC]]`
  reports on a dataset without `-o`.
- `simulate -o OUTPUT [options]` generates a synthetic scan without an input.
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

With no `--fit`, alignment estimates per-view translations. `--fit` names the parameter blocks
released in the same joint solver; it takes one or more of `center`, `translations`, `tilt`,
`roll` and `rotations`. Reconstruction without `--method` selects its method from the geometry.

## Commands

- `inspect`: `tj.inspect`, describing a dataset and checking that it can be reconstructed;
  `--json` prints the report as JSON. `--preview-dir DIR` writes PNGs: the central
  projection when the file holds projections, and slices named
  `slice_<axis>_<index>.png`. A saved reconstruction's slices are read from the file
  (its three central planes by default); a scan's are reconstructed (its central z
  plane by default). `--slice AXIS=INDEX` chooses planes and can be repeated.
  `--centres SPEC` sweeps the detector centre on a scan: it reconstructs the plane of
  the first `--slice` (or the central z plane) at each centre of `START:STOP:STEP`
  or a comma list, in pixels from the detector middle, and writes one image per
  candidate and a contact sheet, `centre_sweep.png`. The report lists each candidate
  with the `U` to pass to `tomojax correct --detector-center U V`.
- `correct`: import raw acquisitions or datasets, supply missing metadata (`--flats`,
  `--darks`, `--angles`, `--pixel-size`, `--detector-center`, `--geometry`, `--tilt`
  and cone parameters), and apply optional corrections (`--zingers`, `--paganin`,
  `--stripes`, `--rings`, `--reject-views`) and `--views`, `--crop` and `--binning`.
  On raw frames every step runs; on a dataset of line integrals only the steps that
  act on them do. `--detector-center U V` sets the first detector's centre, in the
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
  `--region cyl` zeroes the volume outside the cylinder every view sees. The output
  holds the volume, its geometry and record, and names the input projections;
  `--include-scan` keeps them too. `--dry-run` prints the plan: the method and why,
  the grid, and the estimated memory against the memory available.
- `align`: jointly estimate the image and the named geometry corrections with `tj.align`, on
  top of the poses saved in the input unless `--no-apply-poses`. `--fit center` fits one shared centre
  (two for cone data); `--fit translations tilt roll` adds shared setup angles to per-view shifts.
  `--iterations`, `--image-iterations`, `--tolerance`, `--interpolation` and `--resolution`
  are `tj.align`'s keywords; a `--config` file sets the other `tomojax.alignment.AlignConfig` fields.
  `--method opposite-views` estimates `--fit center` from near-opposite views without an image.
  `--fit translations rotations` adds per-view sample rotations. `--grid` sets image sampling.
  `--dry-run` prints the plan: each level's grid and budget, and the estimated memory.
- `export`: `tj.export`, writing a reconstruction as TIFF slices, or as one raw file when
  OUTPUT ends in `.raw`.
- `simulate`: `tj.simulate`, writing a synthetic scan of a phantom with the phantom as its volume.

## A lab CT scan from start to finish

```bash
tomojax correct scan/scan.xtekct --darks dark.tif -o scan.nxs
tomojax inspect scan.nxs
tomojax align scan.nxs -o aligned.nxs --fit center roll
tomojax reconstruct aligned.nxs -o recon.nxs
tomojax inspect recon.nxs --preview-dir previews
tomojax export recon.nxs -o slices/
```

A file that records the pixel size but not the rotation centre (external
NXtomo or Data Exchange, ESRF Bliss or EDF) loads with the centre at the middle
of the detector, and `reconstruct` reports that status. Fit the centre first:

```bash
tomojax align scan.h5 -o aligned.nxs --fit center
tomojax reconstruct aligned.nxs -o recon.nxs
```

Nikon scans record no dark field: `--darks` takes a dark image or one level,
and `--darks 0` suits only counts that are already dark-corrected.
`align --fit center roll` estimates the cone detector's centre and roll,
and `reconstruct` uses them. `previews/` then holds the three central slices, such
as `slice_z_256.png`. See the [lab cone-beam CT guide](../../../docs/lab-ct.md).

To judge the centre by eye instead, sweep it and look at the contact sheet:

```bash
tomojax inspect scan.h5 --preview-dir sweep --centres -20:20:2
tomojax correct scan.h5 -o centred.nxs --detector-center U V   # U and V as listed
```

## Follow or cancel a command

Every command but `serve` accepts `--progress auto|bar|json|none`. The default, `auto`, shows
live progress when stderr is a terminal: a row per running task with its count,
time left and metrics, and a line left behind for each finished level or solver,
such as `✓ binning 2  4/12 steps  objective 4.38 → 0.0833  13.5s`. `bar` off a
terminal prints only those finished lines and warnings; `none` prints warnings only.
Press Ctrl-C to cancel at the next batch, solver chunk, or alignment step. The
command exits 130 after cleanup. A second Ctrl-C interrupts immediately.

`reconstruct --watch-dir DIR` and `align --watch-dir DIR` rewrite PNGs of the
solve's central planes in DIR at each checkpoint (`slice_z.png`, `slice_y.png`,
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
`cancelled` error and exits 130. An internal exception emits an `internal` error,
prints its traceback on stderr with where to report the bug, and exits 1.
Unexpected EOF before completion can indicate external process termination.
Argument errors exit 2. A dry run with `--progress json` ends with a `result`
record whose `info.summary` is the plan.
