# Support matrix

This page lists supported workflows, acquisition formats, and numerical backends. See [installation](installation.md) for dependencies and [measurements](measurements.md) for benchmark results.

## Workflows

| Workflow | Entry point |
| --- | --- |
| Inspect, validate, and preview datasets (including loader findings, slice previews, center sweeps, and checkpoint directories) | `tomojax inspect INPUT [--json] [--preview-dir DIR] [--slice AXIS=INDEX] [--centers SPEC]`, `tj.inspect(path)`, or worker/MCP `inspect` |
| Convert dataset containers (`.npz`, `.nxs`, `.h5`) | `tomojax correct scan.npz -o scan.nxs` |
| Preview corrections on a row band before writing | `tomojax correct raw.nxs --dry-run --preview band.png` or `tomojax.interactive.correction_preview` |
| Load and correct HDF5 raw frames (NXtomo, NeXus scan files, Data Exchange, Diamond/GDA) | `tomojax correct raw.nxs -o corrected.nxs` or `tj.load("raw.nxs")` |
| Load and correct TIFF stacks (parallel, laminography, cone) | `tomojax correct ./projections --flats ./flats --darks ./darks --angles angles.csv --pixel-size ... -o corrected.nxs` |
| Load already-corrected TIFF line integrals | `tj.Scan(read_tiff_frames(path), geometry)` |
| Lazy cropping, view selection, and custom correction steps | `tj.load_frames(path).cropped(...).selected(...).corrected(*steps)`; `scan.corrected(*steps)` |
| Parallel scan with an offset axis and a full turn | `align --fit center` followed by `reconstruct`, or the corresponding Python and MCP calls. Automatic field sizing, overlap weighting, and streamed Fourier/FBP apply to stationary axial scans ([half-acquisition](beamline-exports.md#half-acquisition-with-an-offset-parallel-detector)) |
| Artifact removal and phase retrieval | `--rings`, `--zingers`, `--reject-views`, `--paganin` (expert); `BeamHardening`, `RadialDistortion` in Python |
| Analytical and iterative reconstruction (Fourier, FBP/FDK, CGLS, FISTA-TV, SPDHG-TV, SIRT, MLEM) | `tomojax reconstruct corrected.nxs -o recon.nxs [--method ...]` or `tj.reconstruct(scan, method, config=...)` |
| Nikon `.xtekct` cone-beam import | `tomojax correct scan/part.xtekct --darks 0 -o scan.nxs` or `tj.load_frames("scan/part.xtekct", darks=0).corrected()` |
| Zeiss `.txrm` projection acquisitions | `tj.load("scan.txrm")` or `tj.load_frames("scan.txrm", flats="reference_Front.xrm").corrected(...)` ([CPU preview](lab-ct.md#zeiss-xradia-preview)) |
| Multi-orbit cone-beam scans (`ConeSegments`) | `tj.Scan.from_astra(...)` or `tj.Scan.combine([...])`, then iterative `tj.reconstruct` and `tj.align` |
| Alignment (direct parallel center fit; joint fit for center, translations, tilt, roll, per-view rotations; crossing screening and manual view exclusion) | `tomojax align corrected.nxs -o aligned.nxs --fit ...`, `tj.align(scan, fit=..., grid=..., config=AlignConfig(...))`, or `tomojax.alignment.api.disturbed_views` |
| [Export a registered volume](export.md) as ImageJ TIFF, OME-TIFF, TIFF slices, raw binary, MetaImage, or AmiraMesh | `tomojax export recon.nxs -o slices/`, `tj.export("recon.tif", recon)`, or worker/MCP `export` |
| Synthetic scan generation | `tomojax simulate -o synthetic.nxs --size 64 --geometry parallel\|lamino\|cone` |
| Volumes larger than GPU or host memory | Automatic host slabs for FBP, FDK, Fourier, CGLS, FISTA-TV, SIRT, MLEM; `out=` memmap, `tj.reconstruct_slabs`, and worker/MCP `reconstruct_slabs` ([large scans](large-scans.md)) |
| Checkpoint and resume long runs (Python, CLI, worker, and MCP) | `checkpoint=DIR` on `tj.reconstruct`, `tj.reconstruct_slabs`, and `tj.align`; `--checkpoint DIR` on `tomojax reconstruct` and `tomojax align`; `checkpoint` on worker and MCP `reconstruct`, `reconstruct_slabs`, and `align` ([resume a run after a kill](large-scans.md#resume-a-run-after-a-kill)) |
| Plan, watch, or stream JSON progress for CLI and Python runs | `--dry-run`, `--watch-dir DIR`, `--progress auto\|bar\|json\|none`; `tomojax.progress.watch` |
| Native desktop workspace (projects, live jobs, linked views, 3D volume rendering, comparison) | `tomojax studio [PROJECT_OR_INPUT] [--build] [--demo]` ([Studio](../studio/README.md)) |
| AI agent integration over MCP stdio | `tomojax mcp` with `tomojax[mcp]` ([MCP guide](mcp.md)) |
| Local JSON Lines worker for GUI and custom clients | `tomojax serve` ([worker protocol](protocol.md)) |
| Multi-GPU reconstruction (single machine; Python, worker, and MCP) | `devices=` on `tj.project`, `tj.backproject`, and `tj.reconstruct` (`fbp`, `cgls`, `fista`), or `devices` on worker and MCP `project`, `backproject`, and `reconstruct` ([several GPUs](large-scans.md#use-several-gpus)) |

## Metadata and calibration rules

- **Missing pixel pitch:** parallel-beam scans fall back to pixel units (`length_unit: "pixel"`); override with `pixel_size=`, `detector=`, or `--pixel-size`. Cone-beam scans require a physical pixel size.
- **Missing rotation center:** defaults to the detector middle `(0.0, 0.0)` with `detector_center_status="not_recorded"`. Supply `detector=` / `--detector-center U V`, or fit it with `tj.align(scan, fit="center")`, `tomojax align --fit center`, or MCP `align`.
- **Missing rotation axis:** a file that records no rotation axis or pixel scale warns on load (`geometry_status="not_recorded"`, shown under `loading` in `inspect`). Reconstruction and alignment refuse it until geometry is declared: pass `geometry=...` to `tj.load`, `--geometry parallel` (or `lamino` / `cone`) to `tomojax correct`, or `geometry: {"kind": "parallel"}` to worker/MCP `open`.
- **Missing angles or laminography tilt:** raise an error naming the argument (`angles=` or `geometry=`) rather than assuming defaults.
- **Missing flats or darks:** supply `flats=` (frames, another scan's HDF5/NeXus file whose flat frames are extracted, or a scalar white level) and `darks=` (frames, another scan's file, or a scalar dark level). Passing `flats=` or `darks=` on an unlabeled stack identifies it as detector counts regardless of storage dtype (including `float32`/`float64`); files with a recorded log correction or PSI phase measurement reject flat/dark calibration. Exposure normalization requires flat frames (not a white level) and both durations. Nikon and SkyScan require explicit `darks=` (SkyScan also requires `flats=`); Zeiss skips darks when `AcquisitionSettings/CameraStaticBackground = 1`.
- **Dataset path selectors and geometry overrides:** `data_path`, `image_key_path`, and `angles_path` select HDF5 arrays in both raw acquisitions and saved scans (eager and lazy loads, CLI, worker, and MCP); Bliss accepts `data_path` for its sequence, and fixed-storage formats (TIFF, NPZ, MATLAB phase, Nikon, SkyScan, Zeiss, EDF) reject them. Geometry overrides on load preserve recorded `view_positions` (acquisition gaps and restarts).
- **Bracketing flat sets:** multi-frame flat sets before and after a scan default to `flat_weights="views"` ([flat fields](../src/tomojax/corrections/README.md#flat-fields)), weighting local texture from a sample of the views while interpolating common row gain by position (using local flat noise so narrow stable strips remain effective, raising the common level over 16-row bands where views transmit more than the inferred beam, and masking pixels at or below the dark count before taking logarithms); single-frame sets interpolate by position. When views move the flat by more than 1% over more than 1% of the detector, TomoJAX warns that the views do not follow the flat sets' positions and names both remedies: borrowing clean flats from another scan with `flats=` (`--flats`), or passing `flat_weights="position"` (`--flat-weights position` on `tomojax correct`, or `flat_weights: "position"` on worker/MCP `correct`, `reconstruct_slabs`, and `correction_preview`).
- **Cropping and binning:** compressed TIFF crops decode only intersecting strips or tiles. Detector binning reduces output resolution automatically when no explicit grid is recorded; `Scan.binned()` and file-owned grids keep their sampling unless overridden with `grid=`.

## Saved objects

`tj.save` writes scans, reconstructions, alignments and simulations as
format-2 NeXus in `.nxs`, `.nx`, `.h5`, `.hdf` or `.hdf5` files. NeXus viewers
can open the default array at its physical coordinates. `tj.inspect` reports
the file format, arrays and processing record. This release also reads files
written by 1.0.0rc6 for migration.

## Acquisition formats

`tj.load_frames` recognizes supported raw exports by file content, even if renamed. Physical pitches are normalized to millimeters.

NXtomo files written by GDA can store the commanded rotation in the imaging group. If the NXtomo entry omits its angles, TomoJAX reads the SmarAct ψ or z-rotation command for the same frame dataset. It records the angle source and warns when a recognized angular command carries the positioner's millimeter label. A missing physical pitch leaves lengths in detector pixels.

| Format | Frames and calibration | Angles and geometry metadata |
| --- | --- | --- |
| External NXtomo (ESRF, Diamond, generic) | Group declaring `NXtomo` (`instrument/detector/data`, `image_key`); ESRF return frames (`image_key_control = -1`) excluded; single-row `(view, u)` sinograms stay lazy as `(view, 1, u)`; scans stopped early read the common prefix covered by frames, `image_key`, and angles with a warning and record original counts in `frame_records` | `sample/rotation_angle` (or `sample/transformations/rotation_angle`) with units (default degrees); `x_pixel_size`/`y_pixel_size` with length units |
| NeXus scan files (Diamond GDA, generic) | Default plot frames (`@default`, NXdata `@signal`), local or externally linked; `image_key` of their `NXdetector`; scans stopped early use the same common-prefix path and `frame_records` record as standard NXtomo | NXdata coordinate along frames with angular units, or units recorded on the same-named positioner field in the entry; loads in pixel units (`length_unit: "pixel"`) when no pixel size is recorded |
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
| Parallel FBP and Cone FDK | JAX (CPU/GPU) and CUDA C kernels (auto-selected on CUDA); automatic host-slab fallback (`fbp_host`, `fdk_host`) and raw-frame streaming (`tj.reconstruct_slabs`); full turns, offset-detector full turns (parallel and cone), Parker-weighted short scans; rolled detectors filtered along physical horizontal detector lines (exact axis exchange at 90° multiples, cubic resampling otherwise); FDK angular weights and short-scan coverage derived from the effective source orbit (including pose-carried angles); default cylinder exterior continuation for truncated scans (`exterior="cylinder"\|"zero"`) and `info["truncation"]` diagnostics |
| Fourier-slice inverse | `fourier` for axial parallel half turns and full turns (uniform/nonuniform angles, offset-detector overlap weighting, rolled detectors on horizontal lines with exact 90° permutation, anisotropic grids, per-view axial rotations about `z`, per-view translations); optional FBP detector-response `filter` (`"ramp"`, `"hann"`, `"shepp-logan"`); default cylinder exterior continuation (`exterior="cylinder"\|"zero"`); NumPy reference or CuPy CUDA (`cuda12` / `fourier-cuda12`); host slabs; non-differentiable |
| Iterative solvers (`cgls`, `cgls_multires`, `fista`, `spdhg`, `sirt`, `mlem`) | Exact matched JAX/CUDA projectors; automatic projection streaming and host-slab volume fallback (except SPDHG, which needs 5 volumes in VRAM); discrepancy stopping up to 50 updates for unpenalized CGLS and full-data SIRT (with default momentum for full-data SIRT when `relaxation <= 1`), TV/Huber-TV with isolated-voxel noise calibration in FISTA (with noise-score stopping up to 200 updates) and SPDHG (scaling its default budget as `ceil(600 × √B)` for `B` view blocks), fixed budgets for OS-SART, MLEM/OSEM, and `cgls_multires`, box bounds, support masks, `data_term="per_view_contrast"`, ordered subsets. CGLS, SIRT, FISTA, SPDHG and MLEM estimate the exterior of the grid by default. Their configurations accept `exterior="zero"` to disable the estimate, and `info["truncation"]` records its contribution |
| Alignment (`tj.align`) | Direct axial parallel center measurement on CPU for `fit="center"` on unposed, unrolled scans (half-turn sinogram focus and full-turn opposing-ray overlap, no working volume); single Gauss-Newton/Schur joint solver for `center`, `translations`, `tilt`, `roll`, and `rotations` across all geometries (initializing axial parallel center fits from the measured center), reporting `info["disturbed_views"]` screening and supporting `AlignConfig(exclude_views=..., detector_columns=..., image_tv_weight="coverage")`; `tomojax.alignment.api.disturbed_views` preflight; `method="opposite-views"` pair-only check for axial parallel scans |
| PyTorch bridge (`tomojax.torch.Projector`) | Zero-copy DLPack wrapper around CUDA forward and exact-transpose projectors with autograd |
| Hardware tested | Linux x86_64 CPU; CUDA 12 on 1× and 4× Tesla V100S (Volta, 32 GB), RTX 4070 Laptop GPU (Ada, 8 GB) and 4× H100 (80 GB); CUDA kernels target Volta (compute capability 7.0) and newer GPUs |

Volumes use `(z, y, x)` and projections use `(view, v, u)`, with voxel and detector spacings sharing one physical length unit. Forward projection integrates over physical ray length. FBP returns attenuation in inverse length units (normalized differently from the Euclidean transpose used by iterative solvers).
