# tomojax.cli

`tomojax.cli` implements the `tomojax` console script.

## Conventions

Every command has the shape `tomojax <command> INPUT -o OUTPUT [options]`
(`simulate` takes no input).

- An existing output is refused unless `--force` is given.
- `tomojax <command> --help` lists the command's options. Expert settings are
  not on the help page: put them in a TOML file and pass it with
  `--config FILE`. `tomojax <command> --config-keys` lists the keys a file may
  set, with their defaults; explicit command-line options override the file.
- Exit status is 0 on success, 1 on failure (for `inspect`, also when the
  dataset is invalid) and 2 for a usage error.
- `tomojax --version` prints the version.

The options match the Python API's keywords: `tomojax recon --method cgls
--iterations 50` is `tj.reconstruct(scan, "cgls", iterations=50)`, and
`tomojax align --mode cor --freeze dy` is
`tj.align(scan, mode="cor", freeze=["dy"])`.

## Commands

- `inspect`: describe a dataset and check that it can be reconstructed;
  `--json` prints the report as JSON, and `--preview DIR` writes PNGs of the
  central projection and, for a reconstruction, the volume's central slices.
- `import`: write a dataset from a Nikon `.xtekct` scan, a TIFF stack (with
  `--angles`, `--pixel-size` and, for cone beams, the distances) or another
  `.nxs`/`.npz` dataset.
- `preprocess`: flat- and dark-correct raw frames into absorption projections.
- `recon`: reconstruct a volume with FBP (FDK for cone beams), CGLS, FISTA-TV
  or SPDHG-TV, applying a saved alignment unless `--ignore-alignment`.
- `align`: estimate the rotation axis and per-view poses (`--mode pose`,
  `cor`, `cor-then-pose` or `full`). Mixed setup and pose stages in `full`
  carry their own gauge policies; an expert direct parameter set mixing them
  needs an explicit `gauge_policy`.
- `export`: write a reconstruction as TIFF slices, or as one raw file when
  OUTPUT ends in `.raw`.
- `simulate`: write a synthetic scan of a phantom.

## A lab CT scan from start to finish

```bash
tomojax import scan/scan.xtekct -o scan.nxs
tomojax inspect scan.nxs
tomojax align scan.nxs -o aligned.nxs --mode cor
tomojax recon aligned.nxs -o recon.nxs
tomojax inspect recon.nxs --preview previews
tomojax export recon.nxs -o slices/
```

`align --mode cor` estimates the cone beam's axis offset and detector roll,
and `recon` uses them. `previews/` then holds `projection.png`, `slice_z.png`,
`slice_y.png` and `slice_x.png`. See the [lab cone-beam CT guide](../../../docs/lab-ct.md).
