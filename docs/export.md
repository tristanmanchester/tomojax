# Export a reconstruction

For a volume you can reopen in TomoJAX, use `tj.save("recon.nxs", recon)`. To take it into another application, choose the output suffix for that reader. For example, this writes a calibrated TIFF stack with LZW compression. Add `--dry-run` to see the dimensions, value range, scratch space and output files before writing.

```bash
tomojax export recon.nxs -o detail.tif --crop 100:300,:,: --binning 2 --compress
```

<!-- formats -->
| Destination | Format | What the reader receives |
| --- | --- | --- |
| `.tif`, `.tiff` | ImageJ TIFF | ImageJ reads voxel size, origin and value calibration in classic TIFF. Bio-Formats reads voxel size in classic and BigTIFF. |
| `.ome.tif`, `.ome.tiff` | OME-TIFF | Bio-Formats reads voxel sizes. Classic ImageJ and ITK read the in-plane sizes. |
| Slice directory | TIFF slices | ImageJ, Bio-Formats and ITK read the in-plane pixel size. |
| `.raw` | Raw | The adjacent JSON record gives axes, units and value scaling. |
| `.mhd` | MetaImage | ITK and VTK read spacing and origin. The header stores lengths without a unit. |
| `.am` | AmiraMesh | Bio-Formats reads voxel size and assumes micrometers when the unit is unknown. The bounding box records placement. |
| `.h5`, `.hdf5`, `.nxs` | NeXus NXdata | silx reads physical coordinates and stored samples. The signal's offset and scaling factor recover physical values. |
| `.zarr`, `.ome.zarr` | OME-Zarr 0.5 | ome-zarr-py reads the axes, voxel size and origin of every level. The tomojax attribute holds value scaling. |
<!-- /formats -->

TomoJAX writes volumes in z, y, x order. A crop uses half-open intervals in that order. Binning averages complete spatial bins after the crop and places the new voxel center at the mean of their centers. A one-plane crop writes a single section. With `axis="x"`, planes have z rows and y columns. TIFF viewers treat the first dimension as their stack axis, so changing the stack axis can mirror the display. MetaImage keeps the world frame through its direction matrix.

```python
import tomojax as tj

recon = tj.load_reconstruction("recon.nxs")
receipt = tj.export("detail.ome.tif", recon, crop=(slice(100, 300), None, None))
axes = tj.inspect("detail.ome.tif")["contents"]["volume"]["axes"]
```

Float32 retains quantitative values. For uint8 or uint16, `range=(low, high)` maps the bounds to zero and the largest integer. Without bounds TomoJAX samples up to 16 evenly spaced cropped and binned planes for their finite 0.1 and 99.9 percentiles. Supply bounds for constant data. Integer export stops on a nonfinite voxel. Read the clipping counts in the receipt and the inverse mapping in the export record. Multiply the stored integer by `scale` and add `offset` to recover the physical value within half a step inside the range. Integer or boolean volumes supplied as masks keep labels 0–255 and their grid.

Native ImageJ opens classic TIFF and applies its value calibration. Fiji uses Bio-Formats to open BigTIFF. That path returns stored integers and resets the origin. Use float32 when values must remain quantitative in either reader. The writer selects BigTIFF at 4 GiB minus 64 MiB of uncompressed pixels, even when compression is enabled. Physical ImageJ TIFF coordinates are in micrometers. For volumes in pixel units the full spacing is in the record.

The complete record is embedded in a TIFF stack. A slice directory holds its record in `<directory-name>.json`. Raw, MetaImage and AmiraMesh keep it beside the primary file as `<filename>.json`. Keep each file group together. CLI and worker exports require `force` to replace an existing destination. Replacing an owned directory replaces all its contents. A directory without its TomoJAX export record is refused.

Every export checks its primary file, payloads, records and any directory it will replace or recover against live inputs in the same process. Files stay protected while a running operation, open handle, unreleased artifact or reachable Python result depends on them. A hard link or symlink to an input also conflicts. If the error names an input inside an owned directory, choose another destination for the export. `force` preserves this protection.

## Continue an interrupted export

If a slice directory, raw file or MetaImage export is canceled or killed, repeat the same request with the same source and destination. TomoJAX keeps completed slabs, the integer range and clipping counts in a sibling directory named `.<destination>.partial`. Leave that directory in place until publication finishes. A second writer that targets the same partial receives an error.

The export resumes when its source is an identifiable, unchanged file and its options still match. Editing or replacing the source, changing the crop or encoding, or exporting from an in-memory array starts the export again. Single-file TIFF, AmiraMesh, NeXus and OME-Zarr also restart. A dry run reports `resumes` for a source and format that support recovery.

## Projections and sinograms

Pass a scan to write its measured line integrals. An aligned scan carries its fitted poses. To write the projections behind a reconstruction, use `recon.scan` or select the saved `projections` member.

```python
tj.export("projections.h5", scan)
tj.export("sinograms.tif", scan, axis="v", crop=((10, 110), None, None))
```

```bash
tomojax export aligned.nxs -o sinograms.h5 --member projections --axis v
```

The crop is in (view, v, u) order. Binning averages detector rows and columns and retains each view. A sinogram page has one row per selected view. TIFF readers treat these rows as samples. NeXus stores the actual rotation angles in degrees. The record gives the written axis order, detector calibration, angles, poses, their translation frame and acquisition positions. A projection stack can be written as TIFF, raw, MetaImage or NeXus.

## HDF5 and OME-Zarr

Give an HDF5 export a dataset path when another tool expects one.

```bash
tomojax export recon.nxs -o 'results.h5::/entry/final_result/attenuation' --compress
```

The default signal is `/entry/data/data`. A NeXus viewer follows the default group to the signal and its physical coordinates. With integer scaling, multiply the sum of the stored value and `<signal>_offset` by `<signal>_scaling_factor`. silx displays the stored integer. Compression uses gzip with shuffle. With `--force` the export replaces the entire container, including its other datasets.

Install `tomojax[zarr]` to export an OME-Zarr 0.5 directory. Open it with an NGFF reader such as ome-zarr-py. The base level contains the selected voxels. Each coarser level averages neighboring voxels in pairs, drops an unpaired trailing voxel, and records the new voxel centers. Axes of length one retain their coordinate. The pyramid ends when every side fits in 256 voxels. Masks have one level. Zstandard compression is enabled by default. The directory's `tomojax` attribute contains the export record.

![K11 crop in Fiji showing a 50 µm scale bar and a calibrated voxel value](../images/k11-export.png)

Fiji's native ImageJ reader opened this 640 × 640 crop of K11 scan 52567 as a
classic 16-bit TIFF. Its 0.54 µm pixels span 345.6 µm. The marked voxel has an
attenuation of 0.000679 µm⁻¹ after applying the TIFF's value calibration.
