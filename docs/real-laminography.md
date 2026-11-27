# Reconstruct a real parallel or laminography scan

This guide covers supplying measured geometry for parallel-beam and tilted-axis laminography scans, loading TIFF stacks, and choosing between FBP and FISTA-TV. Replace the example paths, dimensions, and tilt values with your acquisition's measurements. For lab cone-beam CT, see [lab cone-beam CT](lab-ct.md).

## Check an existing NeXus dataset

```bash
tomojax inspect scan.nxs
```

Confirm view count and order, angles in degrees, detector and voxel pitches, and geometry type, and read the `Loading` section (`loading` in `tj.inspect` and MCP `inspect`) to see whether the file omits pixel size, rotation center, or rotation axis. For laminography, verify `tilt` and `tilt_about` (`tilt` is the angle of the rotation axis away from the perpendicular tomography axis, not the angle to the beam). When laminography tilt or the rotation axis is missing from the file, declare geometry when loading: `geometry=` in Python, `--geometry lamino --tilt ...` on `tomojax correct`, or `geometry` on worker or MCP `open`.

For raw NXtomo, NeXus, or Data Exchange frames with flats and darks, start with [correct raw detector frames](quickstart.md#correct-raw-detector-frames); for other HDF5 layouts, pass `--data-path`, `--angles-path`, and `--image-key-path` (or `data_path=`, `angles_path=`, and `image_key_path=` in `tj.load_frames`).

## Prepare TIFF data

TIFF directories are read in natural filename order (`view2` before `view10`), preserving frame order within multi-page files. Supply a matching angle sidecar: either a 1-D `.npy` array or text/CSV with degrees in the first column (leading headers, blank lines, and `#` comments are skipped; angles are not re-sorted).

### Already-corrected absorption projections

Build a `Scan` directly in Python from a folder of line integrals and save it:

```python
import tomojax as tj
from tomojax.io.api import load_angles, read_tiff_frames

projections = read_tiff_frames("projections")
nv, nu = projections.shape[1:]
detector = tj.Detector(nu, nv, 0.65, 0.65, center=(0.0, 0.0))  # measured pitch and center
grid = tj.Grid(nu, nu, nv, 0.65, 0.65, 0.65)
geometry = tj.ParallelGeometry(grid, detector, load_angles("angles.csv"))
# For laminography, use tj.LaminographyGeometry(grid, detector, angles, tilt=30.0, tilt_about="x")
tj.save("corrected.nxs", tj.Scan(projections, geometry))
```

Detector and voxel pitches share one physical length unit, and `Detector.center` is in that same unit (a 2-pixel offset at 0.65 pitch is `1.3`). Use `tomojax align --fit center` to find the rotation center (on an unposed, unrolled axial parallel scan this measures the axis directly from the projections without reconstructing a volume; on laminography it runs the joint fit), and `tomojax reconstruct --grid NX NY NZ` to resample the output grid while preserving physical extent.

### Raw intensities with flat and dark frames

Pass TIFF folders and measured geometry to `tj.load_frames`:

```python
import tomojax as tj
from tomojax.io.api import load_angles

angles = load_angles("angles.csv")
geometry = tj.LaminographyGeometry(
    tj.Grid(nx=256, ny=256, nz=128, vx=0.65, vy=0.65, vz=0.65),
    tj.Detector(nu=256, nv=128, du=0.65, dv=0.65),
    angles,
    tilt=30.0,
    tilt_about="x",
)
frames = tj.load_frames("projections", flats="flats", darks="darks", geometry=geometry)
scan = frames.corrected()
tj.save("corrected-lamino.nxs", scan)
```

Use `tj.ParallelGeometry(grid, detector, angles)` for axial parallel scans. On the CLI, declare geometry for raw frames with `tomojax correct projections/ --flats flats/ --darks darks/ --angles angles.csv --geometry lamino --tilt 30 --pixel-size 0.65 -o corrected-lamino.nxs`.

To add laminography metadata to an already-corrected `.nxs` dataset without re-running preprocessing:

```python
from tomojax.io import load_dataset, save_dataset

scan = load_dataset("corrected.nxs")
scan.geometry_type = "lamino"
scan.geometry_metadata.update(tilt=30.0, tilt_about="x")
save_dataset("corrected-lamino.nxs", scan)
```

Inspect the result with `tomojax inspect corrected-lamino.nxs` before reconstructing.

## Reconstruct with the recorded geometry

The default reconstruction for laminography is FBP (`tj.reconstruct(scan)` or `tomojax reconstruct corrected-lamino.nxs -o recon.nxs`), which runs in seconds (5.5 s on a 1441-view Diamond K11 scan on a 704 × 704 × 234 grid).

Laminography leaves a cone of frequencies around the rotation axis unmeasured. FBP reconstructs measured frequencies cleanly and leaves the missing cone empty, elongating features along the axis. For sharper images and partial missing-cone recovery from volume bounds and total variation, use FISTA-TV:

```bash
tomojax reconstruct corrected-lamino.nxs -o recon.nxs \
  --method fista --nonnegative \
  --manifest recon-manifest.json
tomojax inspect recon.nxs --preview-dir previews
```

With default settings, FISTA-TV calibrates its TV weight from detector noise and interpolation error and stops when the proximal gradient reaches the estimated noise level or at 200 iterations (override with `--iterations` or `--tv-weight`, and inspect `termination`, `resolved_tv_weight`, `effective_iterations`, and `early_stop` in the manifest, `recon.info`, or the closing line of `tomojax reconstruct`). On simulated laminography benchmarks, automatic nonnegative FISTA-TV cuts FBP's measured-band error roughly in half and suppresses missing-cone fog (see [laminography measurements](measurements.md#laminography-the-default-fbp-against-fista)). Enable `--nonnegative` only when attenuation is nonnegative.

![Diamond K11 scan 54286 reconstructed by FBP and by FISTA-TV: in-plane slices through a layer of glass spheres, enlarged regions and vertical sections. The FISTA-TV slice has sharper sphere edges and a clean background, and its vertical section is dark around the layer where FBP's is gray.](images/k11-fbp-fista.png)

Diamond K11 scan 54286, aligned with `tj.align(scan)` and reconstructed on a 704 × 704 × 234 grid (6.48 µm voxels): FBP (top, 7 s) versus 100-iteration nonnegative FISTA-TV (bottom, 651 s). FISTA-TV sharpens the glass spheres and clears the missing-cone haze above and below the layer. Iterative solvers require the whole sample inside the grid (at native 2.16 µm pitch that grid exceeds an 8 GB GPU).

```bash
uv run python examples/gallery/k11_fbp_fista.py k11-54286.nxs k11-fbp-fista.png
```

## Reconstruct scans larger than GPU memory

FBP streams projections from host memory and switches automatically to host slabs when the volume exceeds GPU memory; see [large scans and several GPUs](large-scans.md) for slab solvers and multi-GPU execution.

## Automatic NeXus laminography

NeXus scan files (including Diamond/GDA layouts) need no dataset path overrides when they declare a default plot, detector `image_key`, and angular coordinate, though raw frames omit laminography tilt and often pixel pitch (parallel scans without a recorded pitch load in pixel units unless you pass `pixel_size=` or `detector=`). Supply the nominal geometry before aligning, as in [examples/align_laminography.py](../examples/align_laminography.py):

```python
geometry = tj.LaminographyGeometry(grid, tj.Detector(nu, nv, pitch, pitch), angles, tilt=35)
scan = tj.load("scan.nxs", geometry=geometry)
result = tj.align(scan, fit=("translations", "tilt", "roll"))
tj.save("aligned.nxs", result)
```

Or run the example script directly:

```bash
python examples/align_laminography.py scan.nxs --tilt 35 --pixel 0.00216 -o aligned.nxs
```

By default, `tj.align` fits independent per-view lab-frame translations jointly with the volume (pass `motion="smooth"` for a steady stage). Add `"tilt"` and `"roll"` for shared setup angles, or `"rotations"` for per-view wobble. `result.scan` retains the scan's original grid, native detector samples, and fitted poses so `tj.reconstruct(result.scan)` reconstructs at full resolution; `result.image` is the alignment's working-scale volume on `result.grid`. Pass an explicit `grid=` to fix sampling (explicit grids are not coarsened automatically).

Save an unaligned reconstruction first to compare slices, and see [align a scan](align-a-scan.md) for parameter selection and result checks.
