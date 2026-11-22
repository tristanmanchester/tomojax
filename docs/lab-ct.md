# Lab cone-beam CT

TomoJAX reconstructs and aligns lab cone-beam CT scans (point source, flat detector, turntable) from Nikon `.xtekct`, Zeiss `.txrm`, SkyScan logs, ASTRA geometries, or generic TIFF folders, including multi-orbit acquisitions.

## Import

### Nikon (X-Tek)

A Nikon scan consists of an `.xtekct` parameter file alongside projection TIFFs. `tomojax correct` reads source and detector distances, pixel pitch and offsets, calibrated rotation axis (`ObjectOffsetX`, `ObjectRoll`, `ObjectTilt`, or legacy `CentreOfRotationTop/Bottom`), white level, and nominal angles (`InitialAngle` and `AngularStep`; `.ang` and `_ctdata.txt` sidecars are ignored). Only numbered projection TIFFs matching `<Name>` are loaded, skipping flat fields, thumbnails, and reconstructions. If a folder holds multiple `.xtekct` files, pass the calibrated one (typically `<name>_recon.xtekct`; an acquisition file with `AutomaticCentreOfRotation=1` has no centre and loads at the detector middle).

Nikon files do not store dark frames; supply measured darks with `--darks`, or `--darks 0` if counts are already dark-subtracted:

```bash
tomojax correct scan/part.xtekct --darks 0 -o scan.nxs
tomojax inspect scan.nxs
```

TomoJAX reverses Nikon's top-right panel origin and converts angles and offsets into its own frame while keeping lengths in millimetres. Any extra closing frame on a full turn is excluded with a warning. To override angles with a measured schedule, pass `angles=` (or `--angles`) in TomoJAX's object-angle frame.

### Generic TIFF cone scans

For scanners without a dedicated reader, pass measured `ConeGeometry` to `tj.load_frames(..., geometry=..., flats=..., darks=...)`, or declare it on the CLI using one consistent length unit:

```bash
tomojax correct projections/ --flats flats/ --darks darks/ \
  --angles angles.csv --geometry cone \
  --source-to-axis 120.5 --source-to-detector 980.0 \
  --pixel-size 0.2 --detector-center 0.0 0.0 -o scan.nxs
```

`--flats` and `--darks` accept either frame directories or scalar levels (`--flats 60000 --darks 0`). `--pixel-size` accepts one value (square pixels) or `du dv`. For already-corrected line-integral TIFFs, read them with `tomojax.io.api.read_tiff_frames` and construct `tj.Scan(projections, geometry)` in Python.

Optional detector orientation flags (`--axis-offset`, `--detector-roll`, `--detector-pitch`, `--detector-yaw`) apply right-handed rotations about the detector normal (pointing towards the source), u, and v in that order: positive roll turns u towards v, positive pitch tilts the +v edge towards the source, and positive yaw swings the +u edge away from it.

Without a recorded grid, cone scans default to one voxel per detector pixel at the rotation axis (`pixel_size / magnification`). Use `--grid NX NY NZ` on the CLI or `scan.grid.resampled((64, 64, 64))` in Python to resample the grid while preserving physical extent.

### Artefact corrections

Pass `--stripes WIDTH`, `--rings`, or `--zingers` to `tomojax correct` (or `Stripes`, `Rings`, `Zingers`, `BeamHardening` to `frames.corrected(...)` in Python). For beam-hardening linearisation with an explicit polynomial or X-ray spectrum, see [beam hardening](../src/tomojax/corrections/README.md#beam-hardening); vendor beam-hardening parameters are kept in metadata unapplied.

### Zeiss Xradia preview

Load the **projection acquisition** (`.txrm`), not a reconstructed `.txm` volume or scout image. TomoJAX reads embedded reference images, timestamps, exposure times, angles, distances, and per-view shifts (`alignment/x-shifts`, `y-shifts`), storing those shifts in per-view detector centres so measured pixels are never resampled.

- **Centre:** shifts are relative to `AutoRecon/CenterShift`. Without it, loading puts the centre at the detector middle (`detector_center_status="not_recorded"`); fit `"center"` with `tj.align` or supply `detector=tj.Detector(nu, nv, pitch, pitch, (u, 0))` (where `pitch` is recorded object-plane pixel size × magnification). Passing a full `geometry=` uses it verbatim and drops embedded shifts with a warning.
- **References and darks:** pass a separate flat with `--flats reference_Front.xrm` (or `flats=` + `flat_exposure_times=` in seconds for raw arrays). Files with `AcquisitionSettings/CameraStaticBackground = 1` need no darks; otherwise supply `--darks` / `darks=`.
- **Detector saturation:** Zeiss records the camera's linear ceiling in `MaxLinearRange` (45 000 counts on typical scans). Long exposures can push air pixels past that ceiling; `tj.load` warns when counts exceed it and records the affected fraction and `linear_limit_line_integral` in the `flat_dark` record. Saturated air reads as artificial absorption (e.g. ~0.33 instead of 0 on CuO_2) and appears as a bright background around the sample.

For a fast 64³ CPU preview (every 8th view, 16 × 16 binning):

```bash
tomojax correct scan.txrm --darks 0 -o corrected.nxs --views ::8 --binning 16
tomojax reconstruct corrected.nxs -o preview.nxs --method cgls --iterations 10 --grid 64 64 64 --region off
tomojax inspect preview.nxs --preview-dir previews
```

In Python:

```python
import tomojax as tj

scan = tj.load_frames("scan.txrm").selected(slice(None, None, 8)).corrected(binning=16)
grid = scan.grid.resampled((64, 64, 64))
recon = tj.reconstruct(scan, "cgls", iterations=10, grid=grid)
tj.save("preview.nxs", recon)
```

Per-view detector shifts produce a `ConeSegments` geometry, for which automatic reconstruction selects matched CGLS rather than single-arrangement FDK.

## Calibrate the detector centre and setup angles

A wrong detector centre doubles features, while detector roll introduces height-dependent blur. Fit both shared parameters together:

```bash
tomojax align scan.nxs -o calibrated.nxs --fit center roll
```

For cone data, `"center"` fits detector-u and detector-v offsets in the detector frame (distinct from `ConeBeam.axis_offset`), `"roll"` fits detector roll, and adding `"tilt"` fits the rotation-axis tilt. Source distances, pitch, and yaw stay fixed. Compare slices across several heights against an unaligned reconstruction.

## Reconstruct

For a single cone-beam arrangement, `--method auto` selects FDK (`--method fbp` selects it explicitly):

```bash
tomojax reconstruct calibrated.nxs -o fdk.nxs
```

FDK handles full turns, offset-detector (half-fan) full turns, and Parker-weighted short scans (≥ 180° + fan angle). Like any circular-orbit FDK, it is exact in the orbit plane and develops cone artefacts away from it.

![FDK central slices of three lab scans, each above an enlarged region: a copper oxide specimen from a Zeiss Xradia at 1.13 µm, a box of seeds from a Nikon X-Tek at 94 µm and grains in a SkyScan 1172 at 14.1 µm.](images/lab-cone-slices.png)

Orbit-plane FDK slices of three lab scans loaded directly from vendor files at native axis-plane resolution without alignment: Zeiss Xradia CuO_2 tomo-B (3201 views, recorded centre and shifts), Nikon X-Tek seeds (1571 views, `_recon.xtekct`), and SkyScan 1172 grains (996 views over 199°, nominal detector middle):

```bash
uv run python examples/gallery/lab_cone_slices.py CuO_2_tomo-B.txrm \
  "47209 testscan korn01_recon.xtekct" jyvat.log lab-cone-slices.png
```

Iterative solvers model the exact cone-beam geometry and per-view poses, using the exact transpose of the forward projector so CGLS stays stable across iterations (ASTRA's approximate backprojector diverges after ~20 iterations on the same scan; see [cone-beam measurements](measurements.md#cone-beam-projection-and-fdk)):

```bash
tomojax reconstruct calibrated.nxs -o cgls.nxs --method cgls --iterations 20
tomojax reconstruct calibrated.nxs -o tv.nxs --method fista --iterations 50 --tv-weight 0.002
```

Pass method-specific settings via `--config` TOML files or `tomojax.recon.*Config` classes ([quickstart](quickstart.md#reconstruct-and-inspect-slices)). When the volume exceeds GPU memory, FDK and iterative cone solvers switch automatically to host z-slabs ([large scans and several GPUs](large-scans.md)).

## Export

```bash
tomojax export fdk.nxs -o fdk_slices/               # 32-bit float TIFF z-slices
tomojax export fdk.nxs -o fdk.raw --dtype uint16    # little-endian z-major raw file
```

Outputs ending in `.raw` write a single binary file; any other path writes a directory of z-slices (y rows × x columns, bottom to top). A JSON sidecar (`fdk.json` or `slice.json`) records shape, voxel pitch, and any `uint16` scaling range (`--range`, defaulting to the 0.1 and 99.9 percentiles). Export streams one slice at a time.

## Correct per-view motion

Sample drift, stage wobble, and thermal motion shift the object between views. For cone data, `tomojax align` defaults to three lab-frame translations per view (including along-beam magnification shifts); add `rotations` when wobble remains:

```bash
tomojax align scan.nxs -o aligned.nxs --fit translations rotations
tomojax reconstruct aligned.nxs -o recon.nxs --method cgls
```

Fitted poses compose on top of any input poses; pass `--no-apply-poses` to `reconstruct` to ignore stored poses. See [align a scan](align-a-scan.md) for settings and gauges.

## Speed

On a 256³, 360-view cone scan on an RTX 4070 Laptop GPU (host-to-host), FDK takes 0.10 s (ASTRA 0.29 s, TIGRE 0.54 s) and forward projection 0.14 s (ASTRA 0.17 s); the exact transpose takes 0.12 s versus ASTRA's approximate backprojector at 0.073 s. See [cone-beam measurements](measurements.md#cone-beam-projection-and-fdk) for full timings, device-resident numbers, and 1024³ FDK.

## Python workflow

```python
import tomojax as tj

scan = tj.load_frames("scan/part.xtekct", darks=0).corrected()
calibrated = tj.align(scan, fit=("center", "roll")).scan
recon = tj.reconstruct(calibrated, "cgls")
tj.save("part-recon.nxs", recon)
```

For volumes larger than host RAM, pass a writable memmap via `out=` or stream raw frames with `tj.reconstruct_slabs` ([large scans and several GPUs](large-scans.md)).

## Coming from ASTRA

`tj.Scan.from_astra` converts ASTRA `(rows, views, columns)` projections with `cone` or `cone_vec` projection geometry and a 3-D volume geometry directly into a TomoJAX `Scan`:

```python
import astra
import tomojax as tj

proj_geom = astra.create_proj_geom("cone_vec", rows, cols, vectors)
vol_geom = astra.create_vol_geom(501, 501, 501)
scan = tj.Scan.from_astra(data, proj_geom, vol_geom)
volume = tj.reconstruct(scan).volume            # (z, y, x), matching ASTRA
data, proj_geom, vol_geom = scan.to_astra()     # round-trip back to ASTRA
```

ASTRA rotates the source and detector around a fixed object, whereas TomoJAX fixes the beam and rotates the object; `from_astra` fits a circular orbit (reversing angles automatically) and places axis tilts or per-view perturbations into `scan.poses`. Runs of views with different source/detector arrangements become segments of a `ConeSegments` scan (vectors whose source/detector arrangement changes arbitrarily every view are rejected). Converted projections match ASTRA to 0.03% across tilted, offset, and jittered geometries.

## Try one FIPS walnut orbit on a CPU

Download and unpack a `WalnutN` directory from the [FIPS walnut collection](../examples/README.md#bring-a-real-multi-orbit-scan-into-register). `examples/import_walnut.py` handles portrait orientation, reversed view order, flat/dark correction, and ASTRA vector geometry without needing ASTRA installed, defaulting to every 20th view of orbit 2, 12 × 12 binning, and a 64³ grid over the 50.1 mm cube:

```bash
python examples/import_walnut.py ~/data/walnuts/Walnut1 -o orbit.nxs
tomojax reconstruct orbit.nxs -o fdk.nxs --method fbp --region off
tomojax reconstruct orbit.nxs -o cgls.nxs --method cgls --iterations 20 --region off
tomojax inspect cgls.nxs --preview-dir walnut-previews
```

In Python after the import:

```python
import tomojax as tj

scan = tj.load("orbit.nxs")
fdk = tj.reconstruct(scan, "fbp")
cgls = tj.reconstruct(scan, "cgls", iterations=20)
tj.save("walnut-cgls.nxs", cgls)
```

Pass `--orbit 1` or `--orbit 3` for the other orbits, or reduce `--every` and `--binning` and raise `--size` for higher resolution. Open all three slice planes to inspect cone artefacts away from the midplane. For all three orbits together, use the multi-orbit workflow below.

## Multi-orbit scans

Scans whose source/detector arrangement changes between orbits or stacked sections use `ConeSegments` (created by `tj.Scan.from_astra` or `tj.Scan.combine([a, b, c])`). Iterative solvers reconstruct all segments jointly; `warm_start=True` averages each segment's FDK volume when every segment covers an arc FDK supports. FDK alone refuses multi-arrangement scans.

Because recorded orbit heights often drift slightly, run `tj.align(scan)` to register all segments jointly (only relative orbit heights are observable, so the combined object can shift slightly in absolute z). [examples/align_walnut_orbits.py](../examples/align_walnut_orbits.py) runs the full three-orbit FIPS walnut workflow at 2 × 2 binning in ~6 minutes on an 8 GB GPU, recovering orbit heights within 0.04 mm of the authors' calibration ([walnut measurements](measurements.md#fips-walnuts-real-lab-cone-beam-ct)).

## Limitations

TomoJAX models circular orbits (not helical scans) and flat detectors, uses single-material beam-hardening curves, and does not model X-ray scatter. See [known limitations](known-limitations.md) for full details.
