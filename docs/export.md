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
| `.am` | AmiraMesh | Bio-Formats reads voxel size and assumes micrometres when the unit is unknown. The bounding box records placement. |
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

![K11 crop in Fiji showing a 50 µm scale bar and a calibrated voxel value](../images/k11-export.png)

Fiji's native ImageJ reader opened this 640 × 640 crop of K11 scan 52567 as a
classic 16-bit TIFF. Its 0.54 µm pixels span 345.6 µm. The marked voxel has an
attenuation of 0.000679 µm⁻¹ after applying the TIFF's value calibration.
