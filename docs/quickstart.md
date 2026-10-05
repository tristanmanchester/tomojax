# Reconstruct your first scan

This guide starts from an installed checkout; follow [installation](installation.md)
first. If you do not have data yet, run the [synthetic example](../README.md#first-reconstruction).
Commands use `uv run --no-sync` from the checkout root. Replace example input
paths with your files and use new output paths.

## Inspect the data and geometry

```bash
uv run --no-sync tomojax inspect scan.nxs
uv run --no-sync tomojax validate scan.nxs
```

Check the projection count and shape, angles, detector pitch, voxel pitch, and
geometry type. Validation checks the dataset contract; it cannot prove that
metadata matches the instrument. Arbitrary HDF5 layouts may need explicit paths
during preprocessing. See `tomojax preprocess --help`.

Reconstruction expects absorption/log-attenuation projections. Raw detector
intensities, normalized transmission, and already-corrected attenuation are
different inputs. Establish which you have before preprocessing.

## Correct raw detector frames

For an NXtomo scan containing sample, flat, and dark frames:

```bash
uv run --no-sync tomojax preprocess raw.nxs corrected.nxs
uv run --no-sync tomojax inspect corrected.nxs
uv run --no-sync tomojax validate corrected.nxs
```

By default this applies flat/dark correction and the negative logarithm, then
writes sample-only absorption projections with preprocessing provenance.
`--transmission` writes normalized transmission instead and is not the input
domain expected by the reconstruction commands. Already-corrected absorption
data should bypass this step; applying the logarithm again changes the data.

For TIFF data, use the [TIFF and measured-geometry instructions](real-laminography.md#prepare-tiff-data).
`ingest` packages a stack; it does not perform flat/dark correction.

## Reconstruct and inspect slices

For parallel or laminography data, start with FBP:

```bash
uv run --no-sync tomojax recon --data corrected.nxs --out recon.nxs \
  --algo fbp --roi off --save-manifest recon-manifest.json
uv run --no-sync tomojax validate recon.nxs
uv run --no-sync tomojax slices --data recon.nxs --out quicklooks
```

`recon.nxs` stores the volume and copies the projections. The manifest records
reconstruction settings. `quicklooks/` contains labelled PNGs for the three
central planes and a JSON slice description. PNG contrast is scaled for display;
read the stored floating-point volume for quantitative work.

`--roi off` preserves the recorded grid. The default is `--roi auto`, which may
crop to the detector field of view. `--grid NX NY NZ` changes the dimensions but
keeps the input voxel spacing. Check both field of view and physical units before
comparing reconstructions.

For tilted or irregular-angle data, start with the iterative workflow in the
[real scan guide](real-laminography.md#reconstruct-with-the-recorded-geometry).
If motion remains plausible, use the [alignment guide](alignment-guide.md) after
checking geometry and preprocessing. A visually sharper aligned image does not
by itself establish accurate recovered poses.
