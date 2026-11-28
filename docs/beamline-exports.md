# First slices from beamline exports

Each recipe below loads one detector row from a real beamline export so you can inspect a first slice quickly without allocating a full volume. Except for PSI cSAXS (which stores unwrapped phase), every dataset starts from raw counts.

The pixel sizes and rotation centers below belong to these specific acquisitions; substitute your own scan's calibration when adapting a recipe.

![Five slices, one from each recipe on this page, each above an enlarged region: ESRF bamboo, a PSI TOMCAT skin sample, an Elettra polystyrene foam, the PSI cSAXS porous glass and the Diamond I13 pellet.](images/beamline-first-slices.png)

One native detector row from each scan, reconstructed with `tj.reconstruct(scan)`. The example script applies `Rings()` to ESRF and Diamond I13. Elettra extends beyond the detector (local tomography), and cSAXS reconstructs refractive phase rather than attenuation.

```bash
uv run python examples/gallery/beamline_first_slices.py DATA beamline-first-slices.png
```

## ESRF NXtomo export

Download nabu's [bamboo test file](http://www.silx.org/pub/nabu/data/bamboo_reduced_bliss.nx) (HTTP because the server's TLS certificate is invalid), exported from Bliss by `nxtomomill`:

```python
import tomojax as tj

frames = tj.load_frames("bamboo_reduced_bliss.nx")
```

The loader locates the `NXtomo` group regardless of entry name and reads angles, image keys, flats, darks, and unit-labeled pixel pitches. When the file omits the rotation center, loading places it at the detector middle (`detector_center_status="not_recorded"`); supply a calibrated `detector=` or fit it with `tj.align(scan, fit="center")`.

Continue with [Correct, reconstruct, and inspect](#correct-reconstruct-and-inspect) below. For raw Bliss `.h5` or EDF folders instead of NXtomo exports, see [raw ESRF acquisitions](quickstart.md#raw-esrf-acquisitions).

## PSI TOMCAT Data Exchange export

From the [TomoBank catalog](https://tomobank.readthedocs.io/), `tomo_00069.h5` (skin) stores counts, flats, darks, and `/exchange/theta`, but omits the pixel pitch. Supply its accompanying calibration (3.7 µm pixels, rotation axis at column 515.5 on 1024 columns) in millimeters:

```python
import tomojax as tj

pitch = 3.7e-3  # mm (absent from the HDF5 file)
axis_column = 515.5
frames = tj.load_frames(
    "tomo_00069.h5",
    detector=tj.Detector(
        1024, 428, pitch, pitch,
        center=((1023 / 2 - axis_column) * pitch, 0),
    ),
)
```

`Detector.center` takes a physical offset rather than a column index: `((nu - 1) / 2 - axis_column) * pitch` for an axial parallel detector. This recipe reads the Data Exchange conversion rather than TOMCAT's native GigaFRoST format.

## Elettra SYRMEP Data Exchange export

`tomo_00024.h5` (polystyrene foam, from [TomoBank](https://tomobank.readthedocs.io/)) records its 2.04 µm pitch and `/exchange/theta` angles (which TomoJAX uses directly instead of the inconsistent `angular_step` metadata), with rotation axis at column 972 on 2048 columns:

```python
import tomojax as tj

pitch = 2.04e-3  # mm
frames = tj.load_frames(
    "tomo_00024.h5",
    detector=tj.Detector(
        2048, 2048, pitch, pitch,
        center=((2047 / 2 - 972) * pitch, 0),
    ),
)
```

Because the sample extends past the detector edges, the default Fourier inverse continues each detector row past both edges with a cylinder model (`exterior="cylinder"`) to suppress the truncation rim and bowl. Check `result.info["truncation"]` for `outside_fraction` and `interior_shift`, or pass `FourierConfig(exterior="zero")` to disable continuation ([truncated scans](quickstart.md#truncated-and-region-of-interest-scans)).

## PSI cSAXS ptychographic tomography

The nanoporous-glass dataset of Odstrcil et al. (Opt. Express 2019; Zenodo 3539513, `porous_glass_dataset.mat`) is a MATLAB 7.3 file holding unwrapped, aligned phase sinograms rather than raw counts. `tj.load` reads them directly:

```python
import math
import tomojax as tj

scan = tj.load("porous_glass_dataset.mat")  # 720 views of 416 x 1312, parallel
recon = tj.reconstruct(scan)
delta = recon.volume * scan.source.geometry_metadata["wavelength"] / (2 * math.pi)
```

The reconstructed volume is phase per millimeter (`2πδ / λ`); multiplying by `λ / 2π` using the recorded wavelength yields the refractive-index decrement `δ`. The sinograms are pre-centered on the detector middle. Files holding only the unaligned complex `stack_object` are rejected (see [examples/align_ptychography.py](../examples/README.md#prepare-and-align-complex-ptychography-projections) for phase preparation).

## Correct, reconstruct, and inspect

Save one of the `frames = ...` snippets above in `preview.py` and append:

```python
row = frames.counts.shape[1] // 2
scan = frames.cropped(slice(row, row + 1), slice(None)).corrected()
result = tj.reconstruct(scan)
tj.save("preview.nxs", result)
```

Run it and generate slice previews:

```bash
python preview.py
tomojax inspect preview.nxs --preview-dir preview-images
```

Reload the floating-point volume and geometry in Python with:

```python
restored = tj.load_reconstruction("preview.nxs")
print(restored.volume.shape, restored.grid, restored.geometry.detector)
```

Pass `include_scan=True` to `tj.save` if you also want the corrected projections stored in `preview.nxs`. For ring removal, phase retrieval, larger row windows, and explicit solvers, continue with the [quickstart](quickstart.md).

## Half-acquisition with an offset parallel detector

An axial parallel scan can cover a specimen nearly twice as wide as the detector by placing the rotation axis near one detector edge and rotating through 360°. Opposing views then cover opposite sides of the specimen and share an overlapping strip around the axis. For example, a 2,560-column detector with the axis 64 pixels inside its nearer edge measures a 128-pixel strip twice and covers a 4,992-pixel diameter. Pixel distances are measured from the outer detector edge, half a pixel outside the first pixel center.

Keep all 360° views and the original detector coordinates when loading. On a raw acquisition that fits in memory, load the scan, fit the rotation center from the shared strip, remove rings, and reconstruct:

```python
import tomojax as tj
from tomojax.corrections import Rings

scan = tj.load("full-turn.nxs")
fit = tj.align(scan, fit="center")
print(fit.info["diagnostics"])          # the overlap in pixels, opposing pairs and rows used
aligned = fit.scan
print(aligned.grid.shape)               # (z, y, x), expanded to the far edge
recon = tj.reconstruct(aligned.corrected(Rings()))
tj.save("half-acq-recon.nxs", recon)
```

On the command line:

```bash
tomojax align full-turn.nxs -o aligned.nxs --fit center
tomojax inspect aligned.nxs
tomojax reconstruct aligned.nxs -o half-acq-recon.nxs
```

On an unposed, unrolled axial full turn, `tj.align(scan, fit="center")` locates the axis by matching how attenuation changes across opposing view pairs in their shared strip, and records `overlap_px`, `pairs`, and `rows` in `fit.info["diagnostics"]`. A detector-derived grid expands to the far edge and stays centered on the rotation axis, including when you save the corrected scan before aligning it. A grid supplied by the file or the caller keeps its extent.

On a full turn, the opposing views measure the region across the near edge of the panel. With `exterior="cylinder"`, all methods continue only the far edge. On a rolled panel, this edge is determined separately for each horizontal line through the detector. The truncation record reports the far-edge distance as `field_of_view_radius`, using the median over sampled lines when rolled.

Fourier inversion and FBP weight opposing rays smoothly across their overlap before filtering. CGLS, SIRT, FISTA, SPDHG and MLEM fit the exterior to the original line integrals and the far-edge continuation. Set `exterior="zero"` in the method config to fit the original measurements alone.

Inspecting the scan (`tomojax inspect`, `tj.inspect`, or MCP `inspect`) reports `full_turn: true` under `loading`. When the rotation axis is known (recorded in the file, or after saving `aligned.nxs`) and the detector is unrolled or rotated by a multiple of 90°, inspection also reports `axis_to_edge_px`, `field_of_view_px`, and `overlap_px` in detector pixels. If the axis lies outside the detector (`axis_to_edge_px < 0`), `overlap_px` is `0` and `central_gap_px` gives the diameter of the unmeasured central disk. Attenuation confined to that disk produces no signal in any view.

On a rolled detector whose roll is not a multiple of 90°, the overlap and completed field vary with height. Direct reconstruction weights each horizontal detector line over its own measured interval and sizes a detector-derived grid to enclose all of them; inspection omits the single-number overlap and field diameter because no single interval applies to every row. The direct center fit requires an unrolled detector.

Because an offset full turn nearly quadruples the slice area, full-height volumes can exceed host RAM (a 4,992 × 4,992 × 2,160 `float32` volume at 0.54 µm pitch is 215.3 GB). For large scans, fit the center on a narrow row band, then stream the full raw frames straight to disk with the calibrated detector:

```python
from dataclasses import replace
import numpy as np
import tomojax as tj
from tomojax.corrections import Rings

frames = tj.load_frames("full-turn.nxs")
mid = frames.counts.shape[1] // 2
band = frames.cropped(slice(mid - 8, mid + 8), slice(None)).corrected()
center_u = tj.align(band, fit="center").info["calibration"]["center"]["value"]

d = frames.geometry.detector
calibrated = tj.load_frames(
    "full-turn.nxs",
    detector=replace(d, center=(center_u, d.center[1])),
)
g = calibrated.geometry.grid  # or construct a smaller tj.Grid explicitly
volume = np.lib.format.open_memmap("volume.npy", mode="w+", dtype=np.float32, shape=g.shape)
with tj.reconstruct_slabs(calibrated, "fourier", corrections=[Rings()]) as slabs:
    for z_slice, block in slabs:
        volume[z_slice] = block.volume
volume.flush()
```

On the CLI, pass `--detector-center U V` to `tomojax correct`; in the worker or MCP server, pass `geometry={"detector_center": [U, V]}` to `open` (with `kind="frames"`) before calling `reconstruct_slabs`. See [stream raw frames directly into slabs](large-scans.md#stream-raw-frames-directly-into-slabs) for disk and memory placement, and the [gallery](gallery.md#parallel-half-acquisition) for a coverage map and comparison against a full-width detector.

## Diamond I13 NXtomo acquisition

Diamond I13 scan 159815 (`diamond-i13-mg31912-1/raw/159815.nxs`) links externally to `pco1-159815.hdf`; keep both files in the same directory. TomoJAX discovers its 3001 sample views, 200 flats (before and after), 100 darks, and angles automatically. Bracketing multi-frame flat sets are weighted from a sample of the views by default (`flat_weights="views"`), separating common row gain from local texture; pass `flat_weights="position"` to `.corrected(...)` to blend by position alone, or pass a clean neighboring scan's file as `flats="other_scan.nxs"` when a scan's own flats are shadowed ([flat fields](../src/tomojax/corrections/README.md#flat-fields)). Because pixel pitch is unrecorded, the scan loads in pixel units unless you pass `pixel_size=` or `detector=`.

The example below uses pixel units with this scan's Savu crop (`441:1872`) and axis column (`1159.5` on the 2560-column detector, giving offset `120`):

```python
import tomojax as tj

frames = tj.load_frames(
    "diamond-i13-mg31912-1/raw/159815.nxs",
    detector=tj.Detector(2560, 2160, 1, 1, center=(120, 0)),
)
scan = frames.cropped(slice(1080, 1081), slice(441, 1872)).corrected()
result = tj.reconstruct(scan)
tj.save("i13-preview.nxs", result)
```

```bash
python i13_preview.py
tomojax inspect i13-preview.nxs --preview-dir i13-preview-images
```

Recorded angle signs are kept as stored. For TIFF-only acquisitions without NXtomo metadata, see [preparing TIFF data](real-laminography.md#prepare-tiff-data).

### Loading the same scan from a mixed TIFF folder

If the scan is stored as a single folder of TIFFs (`proj_0.tif`–`proj_3300.tif`: 100 darks, 100 flats, 3001 projections, 100 flats), symlink them into separate folders and supply the angles:

```python
from pathlib import Path
import numpy as np
import tomojax as tj

raw, work = Path("diamond-i13-159815/raw"), Path("i13-split")
groups = {
    "darks": range(0, 100),
    "flats": [*range(100, 200), *range(3201, 3301)],
    "proj": range(200, 3201),
}
for name, indices in groups.items():
    (work / name).mkdir(parents=True, exist_ok=True)
    for i in indices:
        link = work / name / f"proj_{i:04d}.tif"
        if not link.exists():
            link.symlink_to((raw / f"proj_{i}.tif").resolve())

frames = tj.load_frames(
    work / "proj",
    flats=work / "flats",
    darks=work / "darks",
    angles=np.load("savu_angles.npy"),
    detector=tj.Detector(2560, 2160, 1, 1, center=(120, 0)),
)
```

A flat folder carries no frame positions, so its flats are averaged instead of interpolated between the pre- and post-scan sets (on this scan, ~6% beam decay causes up to ±0.03 mean difference in the corrected views).
