# tomojax.cli

`tomojax.cli` implements the `tomojax` console script.

## Conventions

Every command has the shape `tomojax <command> INPUT -o OUTPUT [options]`
(`simulate` takes no input).

- An existing output is refused unless `--force` is given.
- `tomojax <command> --help` lists the command's options. Expert settings are
  not on the help page: put them in a TOML file and pass it with
  `--config FILE` (`preprocess`, `recon` and `simulate`).
  `tomojax <command> --config-keys` lists the keys a file may set, with their
  defaults; explicit command-line options override the file.
- Exit status is 0 on success, 1 on failure (for `inspect`, also when the
  dataset is invalid) and 2 for a usage error.
- `tomojax --version` prints the version.

The options match the Python API's keywords: `tomojax recon --method cgls
--iterations 50` is `tj.reconstruct(scan, "cgls", iterations=50)`, and
`tomojax align --fit translations roll` is
`tj.align(scan, fit=("translations", "roll"))`.

With no `--fit`, alignment estimates per-view translations. `--fit` names the parameter blocks
released in the same joint solver; it takes one or more of `centre`, `translations`, `tilt`,
`roll` and `rotations`. Reconstruction without `--method` selects its method from the geometry.

## Commands

- `inspect`: describe a dataset and check that it can be reconstructed;
  `--json` prints the report as JSON, and `--preview DIR` writes PNGs of the
  central projection and, for a reconstruction, the volume's central slices.
- `import`: write a dataset from a Nikon `.xtekct` scan, a TIFF stack (with
  `--angles`, `--pixel-size` and, for cone beams, the distances) or another
  `.nxs`/`.npz` dataset.
- `preprocess`: flat- and dark-correct raw frames into absorption projections.
- `recon`: choose automatically, or reconstruct with Fourier, FBP/FDK, CGLS, FISTA-TV
  or SPDHG-TV with `tj.reconstruct`, with the poses saved in the input unless
  `--no-poses`. Its expert settings are fields of the method's configuration
  class (`tomojax.recon.FBPConfig`, `FourierConfig`, `CGLSConfig`, `FistaConfig` or
  `SPDHGConfig`), each replacing that field of the class's defaults, as
  `tj.reconstruct(config=...)` takes them; `--roi cyl` zeroes the volume
  outside the cylinder every view sees.
- `align`: jointly estimate the image and the named geometry corrections with `tj.align`, on
  top of the poses saved in the input unless `--no-poses`. `--fit centre` fits one shared centre
  (two for cone data); `--fit translations tilt roll` adds shared setup angles to per-view shifts.
  `--fit translations rotations` adds per-view sample rotations. `--grid` sets image sampling.
- `export`: write a reconstruction as TIFF slices, or as one raw file when
  OUTPUT ends in `.raw`.
- `simulate`: write a synthetic scan of a phantom.

## A lab CT scan from start to finish

```bash
tomojax import scan/scan.xtekct -o scan.nxs
tomojax inspect scan.nxs
tomojax align scan.nxs -o aligned.nxs --fit centre roll
tomojax recon aligned.nxs -o recon.nxs
tomojax inspect recon.nxs --preview previews
tomojax export recon.nxs -o slices/
```

`align --fit centre roll` estimates the cone detector's centre and roll,
and `recon` uses them. `previews/` then holds `projection.png`, `slice_z.png`,
`slice_y.png` and `slice_x.png`. See the [lab cone-beam CT guide](../../../docs/lab-ct.md).
