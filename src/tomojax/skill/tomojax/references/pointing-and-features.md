# Point at a feature and make it a region

Marks let you show the user exactly where you mean; features turn an agreed location into a mask that later steps (scoring, comparison, outlining) can use.

## Coordinates

A position is one of:

- **A pixel of an image you were shown:** `{"pixel": [x, y], "image": "render-12"}`, with x to the right and y down from the image's top-left corner, and the image ID from that view's `images` list. This is usually the most accurate way to point at what you see. The server converts it to array indices and reports them back.
- **Array indices** in the source's storage order: `[z, y, x]` for a volume and `[view, v, u]` for projections. Fractions are allowed.

The same position names the same voxel in every plane, ROI, stride and zoom, so a mark placed on an axial slice appears in the right place on a vertical slice through it.

`view` images stay within 1568 pixels on the long edge and 1.15 megapixels, so you see them at full size and a pixel you read off one is the pixel the server means. To aim at a small or thin target, zoom in first with an `roi` around it; `grid=true` adds a pixel grid, and the index ticks along the edges give array indices. Indices also come from data: the `bounds` of a feature, a `profile`, or a `python` call such as `np.unravel_index(np.argmax(volume), volume.shape)`.

## Marks on `view`

Pass `marks` to `view` in the `planes`, `projection`, `sinogram` or `difference` modes. Each mark has a `type`, an optional short `label` and an optional `color` (`yellow`, the default, `cyan`, `magenta`, `orange`, `green`, `red`).

| `type` | Fields | Shown on a plane when |
| --- | --- | --- |
| `point` | `at` | `at` is within 2 index units of the plane |
| `text` | `at`, `label` | as for `point`; draws only the label |
| `arrow` | `from`, `to` | the head `to` is within 2 units |
| `box` | `corners`: two opposite corners, inclusive | the plane cuts the box (within 2 units) |
| `circle` | `at`, `radius` in index units | the plane cuts the sphere; it draws the cross-section |
| `polyline`, `polygon` | `points` (2 or more, 3 or more) | every point is within 2 units |

```json
{"source": "@recon-1a2b3c4d", "axis": "z", "index": 120,
 "marks": [{"type": "point", "at": [120, 210.5, 388], "label": "shell?"},
           {"type": "circle", "at": [120, 260, 300], "radius": 15, "color": "cyan"}]}
```

The structured result has a `marks` list in the same order. Each entry gives the mark's `positions` in array indices and lists every rendered plane (`panel`, the index into `panels`) with `drawn`, the mark's `pixel` `[x, y]` in the returned image `image`, and the `value` under a point-like mark or the `mean` inside a box, circle or polygon, from the displayed samples. `percentile` is the share of the plane's samples below that value and `display` its place between the display limits (0 the lower, 1 the upper). A mark not drawn has `reason`: `off_plane` (with its `distance`) or `outside_roi`.

A point, or an arrow's head, on flat ground, with no peak, trough or edge within 2 samples, also has a `warning` and `nearest`: the array indices (`at`), `distance` in index units, `value` and `percentile` of the closest structure on that plane, taken at that structure's most extreme sample along the line from the mark. Structure here means a change of at least a quarter of the plane's range and well above its noise, so a plane of air or noise alone gives a warning without `nearest`. Mark `nearest.at` if that is what you meant. A mark on structure has neither; check that its `value` and `percentile` fit the description (bright, dark, an edge) before you show it.

A point can snap to what you mean. Add `"snap": "peak"` (brightest), `"trough"` (darkest) or `"edge"` (steepest change), and optionally `within` (index units, default 5). On each plane the point is drawn on, it moves to that sample within reach, found on a copy smoothed by one sample (a peak or trough then takes the most extreme raw sample beside it, so a shell one sample thin keeps its crest), and the report adds `snapped` (its array indices on that plane) and `moved` (how far, in index units). A point that snaps onto flat ground, because the structure is further than `within`, gets the same `warning` and `nearest`. Use the `snapped` indices as the seed for a feature.

Ticks are on by default when marks or features are given; pass `ticks=false` to hide them, or `ticks=true` to get them without marks.

## Ask, then make a feature

Show the marked image and ask a direct question: "Is the shell you mean the ring under the yellow point?" When you are unsure, mark two candidates in different colors and ask which one. After the user confirms, make a feature with the `feature` tool:

| `recipe.type` | Fields | Makes |
| --- | --- | --- |
| `box` | `corners` | The box between two corners, inclusive |
| `sphere` | `at`, `radius` | The voxels within `radius` of `at` |
| `cylinder` | `at`, `radius`, `axis`, `extent` | A disc across `axis`, over `extent` along it |
| `prism` | `points`, `axis`, `extent` | A polygon across `axis`, over `extent` along it |
| `seeded` | `seeds`, `tolerance`, `tolerance_unit`, `smoothing`, `max_radius` | The voxels connected to the seeds whose value stays near the seeds' level |
| `band` | `feature`, `inner`, `outer` | The voxels at signed distances in (`inner`, `outer`] from another feature |

A confirmed mark converts directly: a box mark's corners make a `box`, a circle mark's `at` and `radius` a `sphere`, a polygon mark's points a `prism`, and a point mark's `at` the seed of a `seeded` region.

- **Seeded.** The level is the median around each seed. `tolerance` (default 3) is in units of the noise measured near the seeds, or of the volume's values with `tolerance_unit="value"`; the result's `recipe` reports the `level`, `noise` and `band` it used. Place seeds well inside the feature. If the region leaks into a neighbor, lower `tolerance` or set `max_radius`; if it is patchy, raise `tolerance` or add `smoothing` (a Gaussian sigma in voxels, which pulls edges in). A structure one or two voxels thick is better made as a `band` inside a larger grown region.
- **Band.** Space beyond the array is background. A feature voxel with a face next to background has distance 0, including at the volume edge. Bands select only voxels inside the grid: `inner=0, outer=3` is a three-voxel shell outside it and `inner=-3, outer=0` the three-voxel rim inside it. `inner=1, outer=4` leaves one voxel between a feature and its surroundings, which keeps partial-volume edge voxels out of both when you compare them.

Name features after what they are (`name="shell"`). The result gives the voxel count, `bounds` (half-open per axis) and the mean and standard deviation of the source inside the mask.

## Use a feature

Keep the feature on its registered grid when measuring another source. Bands and views check the shape, named axes in storage order, frames, axis units and sample centers (absolute tolerance `1e-6` in each axis's unit). A mismatch identifies the differing coordinate or grid property. Rebuild the feature on the target grid or explicitly resample it there. A bare NumPy mask has only indices. When you pass one to `register_feature(mask, like=source)`, `like` declares the grid on which you computed those indices.

- `view` with `features=["@feature-…"]` outlines up to five features, in cyan, magenta, green, orange and red, on any source with the same grid: the volume it was made on, another reconstruction of the same scan, or a difference.
- In `python`, `handles["@feature-…"].mask` is the whole boolean mask. A feature stores only its bounding box: `.box` (a slice per axis) and `.inside` (the mask over that box). On a large volume use those, for example `volume[f.box][f.inside]`, instead of building `.mask`, which takes one byte per voxel of the grid. `register_feature(mask, like="@recon-…", name="…")` registers a mask you computed yourself on that object's grid; `like` also takes a member such as `"@recon-….volume"`.
- `save` writes a feature to an HDF5 file (`.h5`, `.hdf5` or `.nxs`) with its grid, name and recipe; `open` with `kind="feature"` reloads it, and `export` writes the mask as 0/1 TIFF slices or a raw file for other software.
- Features work at any volume size: each recipe works on the feature's own box plus its band or growth margin and reads the source in windows.
- Show the user what you will optimize for before you optimize it: one `view` with the feature outlined.
