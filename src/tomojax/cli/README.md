# tomojax.cli

`tomojax.cli` implements the `tomojax` console script.

## Conventions

Every command has the shape `tomojax <command> INPUT -o OUTPUT [options]`
(`simulate` takes no input).

- An existing output is refused unless `--force` is given.
- `tomojax <command> --help` lists the command's options. Expert settings are
  not on the help page: put them in a TOML file and pass it with
  `--config FILE` (`reconstruct`, `align` and `simulate`).
  `tomojax <command> --config-keys` lists the keys a file may set, with their
  defaults; explicit command-line options override the file.
- Exit status is 0 on success, 1 on failure (for `inspect`, also when the
  dataset is invalid) and 2 for a usage error.
- `tomojax --version` prints the version.

The options match the Python API's keywords: `tomojax reconstruct --method cgls
--iterations 50` is `tj.reconstruct(scan, "cgls", iterations=50)`, and
`tomojax align --fit translations roll` is
`tj.align(scan, fit=("translations", "roll"))`.

With no `--fit`, alignment estimates per-view translations. `--fit` names the parameter blocks
released in the same joint solver; it takes one or more of `center`, `translations`, `tilt`,
`roll` and `rotations`. Reconstruction without `--method` selects its method from the geometry.

## Commands

- `inspect`: `tj.inspect`, describing a dataset and checking that it can be reconstructed;
  `--json` prints the report as JSON, and `--preview DIR` writes PNGs of the
  central projection, when the file holds projections, and of a reconstruction's
  central slices.
- `load`: `tj.load`, then `tj.save`, for every input `tj.load` reads (raw frames,
  corrected with dark, flat and `-log`, or a dataset). Its flags supply what the
  file lacks under `tj.load`'s names: `--flats`, `--darks`, `--angles`,
  `--pixel-size`, `--detector-center`, `--apply-poses`, and `--geometry` with
  `--tilt` and the cone's distances and detector angles.
- `correct`: the optional corrections (`--zingers`, `--paganin`, `--stripes`,
  `--rings`, `--reject-views`) and `--views`, `--crop` and `--binning`. On raw frames
  every step runs and the same input flags as `load` apply; on a dataset of line
  integrals only the steps that act on them do.
- `reconstruct`: choose automatically, or reconstruct with Fourier, FBP/FDK, CGLS, FISTA-TV
  or SPDHG-TV with `tj.reconstruct`, with the poses saved in the input unless
  `--no-apply-poses`. Its expert settings are fields of the method's configuration
  class (`tomojax.recon.FBPConfig`, `FourierConfig`, `CGLSConfig`, `FistaConfig` or
  `SPDHGConfig`), each replacing that field of the class's defaults, as
  `tj.reconstruct(config=...)` takes them; `--region cyl` zeroes the volume
  outside the cylinder every view sees. The output holds the volume, its geometry and
  record, and names the input projections; `--include-scan` keeps them too.
- `align`: jointly estimate the image and the named geometry corrections with `tj.align`, on
  top of the poses saved in the input unless `--no-apply-poses`. `--fit center` fits one shared centre
  (two for cone data); `--fit translations tilt roll` adds shared setup angles to per-view shifts.
  `--iterations`, `--image-iterations`, `--tolerance`, `--interpolation` and `--resolution`
  are `tj.align`'s keywords; a `--config` file sets the other `tj.AlignConfig` fields.
  `--method opposite-views` estimates `--fit center` from near-opposite views without an image.
  `--fit translations rotations` adds per-view sample rotations. `--grid` sets image sampling.
- `export`: `tj.export`, writing a reconstruction as TIFF slices, or as one raw file when
  OUTPUT ends in `.raw`.
- `simulate`: `tj.simulate`, writing a synthetic scan of a phantom with the phantom as its volume.

## A lab CT scan from start to finish

```bash
tomojax correct scan/scan.xtekct --darks dark.tif -o scan.nxs
tomojax inspect scan.nxs
tomojax align scan.nxs -o aligned.nxs --fit center roll
tomojax reconstruct aligned.nxs -o recon.nxs
tomojax inspect recon.nxs --preview previews
tomojax export recon.nxs -o slices/
```

A file that records the pixel size but not the rotation centre (external
NXtomo or Data Exchange, ESRF Bliss or EDF) loads with the centre at the middle
of the detector, and `recon` says so. Fit the centre first:

```bash
tomojax align scan.h5 -o aligned.nxs --fit center
tomojax reconstruct aligned.nxs -o recon.nxs
```

Nikon scans record no dark field: `--darks` takes a dark image or one level,
and `--darks 0` suits only counts that are already dark-corrected.
`align --fit center roll` estimates the cone detector's centre and roll,
and `recon` uses them. `previews/` then holds `slice_z.png`, `slice_y.png` and
`slice_x.png`. See the [lab cone-beam CT guide](../../../docs/lab-ct.md).
