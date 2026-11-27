# Support matrix

This page lists supported workflows, acquisition formats, and numerical backends. See [installation](installation.md) for dependencies and [measurements](measurements.md) for benchmark results.

## Workflows

| Workflow | Entry point |
| --- | --- |
| Inspect, validate, and preview datasets (including slice previews and center sweeps) | `tomojax inspect scan.nxs [--json] [--preview-dir DIR] [--slice AXIS=INDEX] [--centers SPEC]` |
| Convert dataset containers (`.npz`, `.nxs`, `.h5`) | `tomojax correct scan.npz -o scan.nxs` |
| Preview corrections on a row band before writing | `tomojax correct raw.nxs --dry-run --preview band.png` or `tomojax.interactive.correction_preview` |
| Load and correct HDF5 raw frames (NXtomo, NeXus scan files, Data Exchange, Diamond/GDA) | `tomojax correct raw.nxs -o corrected.nxs` or `tj.load("raw.nxs")` |
| Load and correct TIFF stacks (parallel, laminography, cone) | `tomojax correct ./projections --flats ./flats --darks ./darks --angles angles.csv --pixel-size ... -o corrected.nxs` |
| Load already-corrected TIFF line integrals | `tj.Scan(read_tiff_frames(path), geometry)` |
| Lazy cropping, view selection, and custom correction steps | `tj.load_frames(path).cropped(...).selected(...).corrected(*steps)`; `scan.corrected(*steps)` |
| Artifact removal and phase retrieval | `--rings`, `--zingers`, `--reject-views`, `--paganin` (expert); `BeamHardening`, `RadialDistortion` in Python |
| Analytical and iterative reconstruction (Fourier, FBP/FDK, CGLS, FISTA-TV, SPDHG-TV, SIRT, MLEM) | `tomojax reconstruct corrected.nxs -o recon.nxs [--method ...]` or `tj.reconstruct(scan, method, config=...)` |
| Nikon `.xtekct` cone-beam import | `tomojax correct scan/part.xtekct --darks 0 -o scan.nxs` or `tj.load_frames("scan/part.xtekct", darks=0).corrected()` |
| Zeiss `.txrm` projection acquisitions | `tj.load("scan.txrm")` or `tj.load_frames("scan.txrm", flats="reference_Front.xrm").corrected(...)` ([CPU preview](lab-ct.md#zeiss-xradia-preview)) |
| Multi-orbit cone-beam scans (`ConeSegments`) | `tj.Scan.from_astra(...)` or `tj.Scan.combine([...])`, then iterative `tj.reconstruct` and `tj.align` |
| Alignment (direct parallel center fit; joint fit for center, translations, tilt, roll, per-view rotations) | `tomojax align corrected.nxs -o aligned.nxs --fit ...` or `tj.align(scan, fit=..., grid=...)` |
| Export volume to 32-bit TIFF z-slices or raw binary (`float32`/`uint16`) | `tomojax export recon.nxs -o slices/` or `-o recon.raw --dtype uint16` |
| Synthetic scan generation | `tomojax simulate -o synthetic.nxs --size 64 --geometry parallel\|lamino\|cone` |
| Volumes larger than GPU or host memory | Automatic host slabs for FBP, FDK, Fourier, CGLS, FISTA-TV, SIRT, MLEM; `out=` memmap and `tj.reconstruct_slabs` ([large scans](large-scans.md)) |
| Plan, watch, or stream JSON progress for CLI and Python runs | `--dry-run`, `--watch-dir DIR`, `--progress auto\|bar\|json\|none`; `tomojax.progress.watch` |
| Native desktop workspace (projects, live jobs, linked views, 3D volume rendering, comparison) | `tomojax studio [PROJECT_OR_INPUT] [--build] [--demo]` ([Studio](../studio/README.md)) |
| AI agent integration over MCP stdio | `tomojax mcp` with `tomojax[mcp]` ([MCP guide](mcp.md)) |
| Local JSON Lines worker for GUI and custom clients | `tomojax serve` ([worker protocol](protocol.md)) |
| Multi-GPU reconstruction (single machine, Python only) | `devices=` on `tj.project`, `tj.backproject`, and `tj.reconstruct` (`fbp`, `cgls`, `fista`) ([several GPUs](large-scans.md#use-several-gpus)) |

## Metadata and calibration rules

- **Missing pixel pitch:** parallel-beam scans fall back to pixel units (`length_unit: "pixel"`); override with `pixel_size=`, `detector=`, or `--pixel-size`. Cone-beam scans require a physical pixel size.
- **Missing rotation center:** defaults to the detector middle `(0.0, 0.0)` with `detector_center_status="not_recorded"`. Supply `detector=` or fit it with `tj.align(scan, fit="center")`.
- **Missing angles or laminography tilt:** raise an error naming the argument (`angles=` or `geometry=`) rather than assuming defaults.
- **Missing flats or darks:** supply `flats=` (frames or scalar white level) and `darks=`. Nikon and SkyScan require explicit `darks=` (SkyScan also requires `flats=`); Zeiss skips darks when `AcquisitionSettings/CameraStaticBackground = 1`.
- **Cropping and binning:** compressed TIFF crops decode only intersecting strips or tiles. Detector binning reduces output resolution automatically when no explicit grid is recorded; `Scan.binned()` and file-owned grids keep their sampling unless overridden with `grid=`.

## Acquisition formats

`tj.load_frames` recognizes supported raw exports by file content, even if renamed. Physical pitches are normalized to millimeters.

| Format | Frames and calibration | Angles and geometry metadata |
| --- | --- | --- |
| External NXtomo (ESRF, Diamond, generic) | Group declaring `NXtomo` (`instrument/detector/data`, `image_key`); ESRF return frames (`image_key_control = -1`) excluded; single-row `(view, u)` sinograms stay lazy as `(view, 1, u)` | `sample/rotation_angle` (or `sample/transformations/rotation_angle`) with units (default degrees); `x_pixel_size`/`y_pixel_size` with length units |
| NeXus scan files (Diamond GDA, generic) | Default plot frames (`@default`, NXdata `@signal`), local or externally linked; `image_key` of their `NXdetector`; scans stopped early read the common prefix covered by frames, key, and angles with a warning | NXdata coordinate along frames with angular units, or units recorded on the same-named positioner field in the entry; loads in pixel units (`length_unit: "pixel"`) when no pixel size is recorded |
| Data Exchange (APS, PSI TOMCAT, Elettra SYRMEP) | `/exchange/data`, `data_white`, `data_dark`, or interleaved `image_key` | `/exchange/theta`; `/measurement/instrument/detector/actual_pixel_size_x`/`_y`; facility labels kept |
| ESRF raw Bliss HDF5 | Sequence-owned detector blocks, image keys, and flat positions (synthetic-tested; all linked files required) | Unit-labeled angular commands; `technique/optic/sample_pixel_size`; distances from `technique/scan` |
| ESRF SPEC EDF | Ordered single-image EDFs, references, accumulated darks divided by `DARK_N` | `.info`/XML `PixelSize` and `Distance`; supply `angles=` explicitly |
| ESRF PyHST `.par` | Pass `.par` path explicitly; raw single-image axial parallel EDFs with dark/flat schedules (multi-frame, cone, helical, binning, distortion unsupported) | Recorded angles, pixel pitch, rotation axis, and reconstruction region |
| Nikon `.xtekct` | Numbered `<Name>` projection TIFFs and `WhiteLevel`; supply `darks=` explicitly; both TIFF axes and angle signs normalized | Source/detector distances, pixel pitch, `ObjectOffsetX`/`Roll`/`Tilt` (or legacy `CentreOfRotationTop/Bottom`), `InitialAngle`/`AngularStep`; vendor ROI offsets kept as metadata |
| Zeiss `.txrm` | Lazy OLE image streams, embedded or external `.xrm` references (`flats=`), exposure-time count-rate normalization, timestamp interpolation | `ImageInfo/Angles`, object-plane `pixelsize`, `XrayGeometry`, distances, `AutoRecon/CenterShift`, per-view `alignment/*-shifts` |
| SkyScan log + TIFFs | Lazy numbered TIFFs (excluding scouts and drift records); requires geometric and flat-field correction enabled in export; supply `flats=` and `darks=` | Source/detector distances, pixel pitch/binning, vertical optical-axis line; supply horizontal center via `detector=` |
| PSI cSAXS `.mat` (MATLAB 7.3) | `tj.load` reads unwrapped `aligned_phase_sinogram` directly as line integrals (unaligned complex `stack_object` rejected) | `rotation_angle` (degrees), `pixel_size` (mm), `wavelength` (saved in `geometry_metadata`) |

Tested against real acquisitions from ESRF NXtomo (`nxtomomill`), APS TomoBank (13-BM-D, 32-ID), PSI TOMCAT, Elettra SYRMEP, Diamond I12/K11/I13, Nikon X-Tek, Zeiss Xradia `.txrm`, SkyScan 1172, and PSI cSAXS. Native TOMCAT GigaFRoST is not supported.

## Numerical and accelerator scope

| Operation | Coverage and backends |
| --- | --- |
| Geometries | `ParallelGeometry`, `LaminographyGeometry`, `RotationAxisGeometry` (arbitrary axis), `ConeGeometry` (`ConeBeam` with offsets, roll, pitch, yaw, `axis_offset`, arbitrary axis), `ConeSegments`; arbitrary per-view rigid poses; fan beam as a 1-row cone detector |
| Forward projection and matched adjoint | Linear and Keys cubic Joseph plane-sampling models; JAX reference and CUDA C kernels (exact transpose via gather, no scattered atomics); first-order volume and pose AD (`jax.grad`, `jvp`, `vjp`, `vmap`) on CUDA, higher-order AD on JAX |
| Fused Joseph pose helpers (`tomojax.forward`) | `joseph_l2_value_and_grad` (unweighted half squared error + volume/pose gradients) and `joseph_pose_normal_equations` (per-view loss, gradient, Gauss-Newton `J.T @ J`, residual for 1–16 directions on CUDA without storing full Jacobians) |
| Parallel FBP and Cone FDK | JAX (CPU/GPU) and CUDA C kernels (auto-selected on CUDA); automatic host-slab fallback (`fbp_host`, `fdk_host`) and raw-frame streaming (`tj.reconstruct_slabs`); full turns, offset-detector full turns, Parker-weighted short scans |
| Fourier-slice inverse | `fourier` for axial parallel scans (uniform/nonuniform angles, offsets, anisotropic grids, per-view translations); optional FBP detector-response `filter` (`"ramp"`, `"hann"`, `"shepp-logan"`); NumPy reference or CuPy CUDA (`cuda12` / `fourier-cuda12`); host slabs; non-differentiable |
| Iterative solvers (`cgls`, `cgls_multires`, `fista`, `spdhg`, `sirt`, `mlem`) | Exact matched JAX/CUDA projectors; automatic projection streaming and host-slab volume fallback (except SPDHG, which needs 5 volumes in VRAM); TV/Huber-TV (with detector-uncertainty weight calibration and noise-score stopping in FISTA), box bounds, support masks, `data_term="per_view_contrast"`, ordered subsets |
| Alignment (`tj.align`) | Direct half-turn sinogram and opposing-ray center measurement on CPU for `fit="center"` on unposed, unrolled axial parallel scans (no working volume); single Gauss-Newton/Schur joint solver for `center`, `translations`, `tilt`, `roll`, and `rotations` across all geometries (initializing axial parallel center fits from the measured center); `method="opposite-views"` pair-only check for axial parallel scans |
| PyTorch bridge (`tomojax.torch.Projector`) | Zero-copy DLPack wrapper around CUDA forward and exact-transpose projectors with autograd |
| Hardware tested | Linux x86_64 CPU; CUDA 12 on Tesla V100S (Volta, 32 GB), RTX 4070 Laptop GPU (Ada, 8 GB) and 4× H100 (80 GB); CUDA kernels target Volta (compute capability 7.0) and newer GPUs |

Volumes use `(z, y, x)` and projections use `(view, v, u)`, with voxel and detector spacings sharing one physical length unit. Forward projection integrates over physical ray length. FBP returns attenuation in inverse length units (normalized differently from the Euclidean transpose used by iterative solvers).
