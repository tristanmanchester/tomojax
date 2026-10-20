# TomoJAX support matrix

Available workflows and their entrypoints. “Supported” identifies an implemented
and tested path, not a guarantee of accuracy for every scan. Alignment remains
experimental. See [installation](installation.md) and [measurement scope](measurements.md).

| Workflow | Status | Supported entrypoint |
|---|---|---|
| Dataset inspection and validation | Supported | `tomojax inspect scan.nxs` (`--json` for a JSON report; exit status 1 when invalid) |
| Preview PNGs (central projection and central volume slices) | Supported | `tomojax inspect recon.nxs --preview previews` |
| TIFF stack import | Supported | `tomojax import ./tiffs --angles angles.csv --pixel-size ... --geometry parallel --config calibration.toml -o scan.nxs` |
| Dataset container conversion (`.npz`, `.nxs`, `.h5`) | Supported | `tomojax import scan.npz -o scan.nxs` |
| HDF5 flat/dark correction (NXtomo `image_key`, or APS Data Exchange `data_white`/`data_dark`; flat sets before and after interpolated) | Supported | `tomojax preprocess raw.nxs -o corrected.nxs`, or `tj.load("raw.nxs")` |
| Frame corrections in Python (crop, select views, correction steps) | Supported | `tj.load_frames(path).cropped(...).selected(...).corrected(*steps)`; `Scan.corrected(*steps)` for line-integral steps |
| TIFF flat/dark preprocessing | Supported | `tomojax preprocess ./projections -o corrected.nxs --flats ./flats --darks ./darks --angles angles.csv` |
| FBP, CGLS, FISTA-TV, SPDHG-TV from corrected projections | Supported | `tomojax recon corrected.nxs -o recon.nxs --method fbp\|cgls\|fista\|spdhg` |
| Fourier-slice inverse of supported parallel scans | Supported | `tomojax recon corrected.nxs -o recon.nxs --method fourier`; matching available CUDA or NumPy, with explicit backend configuration |
| Cone-beam (lab CT) scans: FDK and iterative reconstruction | Supported | `tomojax import ./tiffs --geometry cone --source-to-axis ... --source-to-detector ... --pixel-size ... --angles angles.csv --config calibration.toml -o scan.nxs`, then `tomojax recon` (automatic FDK for one arrangement; `--method fbp` selects it explicitly) |
| Per-projection 5-DOF pose alignment | Experimental | `tomojax align corrected.nxs -o aligned.nxs --mode pose` |
| Cone-beam 6-DOF pose alignment (adds `dy` along the beam) | Experimental | `tomojax align cone_scan.nxs -o aligned.nxs --mode pose` |
| Ring (stripe), zinger, outlier-view and beam-hardening corrections; Paganin phase retrieval | Supported | `tomojax preprocess raw.nxs -o scan.nxs --stripes 9 --zingers --reject-views --beam-hardening 1,0.05` (and the expert `--paganin`), or `tomojax.corrections` steps |
| Volume export as TIFF slices or raw (`uint16` or `float32`) | Supported | `tomojax export recon.nxs -o slices/`; `tomojax export recon.nxs -o recon.raw --dtype uint16` |
| Nikon (X-Tek) scan import, including its calibrated rotation axis | Supported | `tj.load_frames("scan/part.xtekct", darks=0).corrected()`, or `tomojax preprocess scan/part.xtekct --darks 0 -o scan.nxs` |
| Zeiss Xradia projection acquisitions | Supported | `tj.load_frames("scan.txrm", darks=0).corrected(...)`; separate reference images use `flats="reference_Front.xrm"`; [CPU preview](lab-ct.md#zeiss-xradia-preview) |
| Multi-orbit cone-beam scans (`ConeSegments`): iterative reconstruction and pose alignment | Supported; alignment experimental | `tj.Scan.from_astra(...)` or `tj.Scan.combine([...])`, then `tj.reconstruct(scan, "cgls")` and `tj.align(scan)`; `orbit_heights` in `tomojax.alignment.api` |
| Cone-beam axis-offset (centre of rotation) and detector-roll calibration | Experimental | `tomojax align cone_scan.nxs -o calibrated.nxs --mode cor`; `calibrate_cone_axis` in Python |
| Detector-centre/COR alignment | Experimental | `tomojax align corrected.nxs -o aligned.nxs --mode cor` |
| Detector-centre offset with per-view motion | Experimental | `tomojax align corrected.nxs -o aligned.nxs --mode cor-then-pose` |
| Mixed setup and pose alignment | Experimental; each stage fixes its gauge policy | `tomojax align corrected.nxs -o aligned.nxs --mode full` |
| Deterministic synthetic dataset generation | Supported | `tomojax simulate -o synthetic_scan.nxs --size 64 --geometry parallel\|lamino\|cone` |
| Python API reconstruction | Supported | `tj.reconstruct(scan, method, config=...)`, `tj.project`, `tj.backproject`; building blocks in `tomojax.geometry`, `tomojax.forward`, `tomojax.recon` |
| Python API alignment, with resumable checkpoints | Experimental | `tj.align(scan, mode=..., grid=..., checkpoint=..., config=...)` |
| Volumes larger than the GPU | Supported | FBP and FDK switch to host slabs automatically; `fbp_host` and `fdk_host` in Python ([large scans](large-scans.md)) |
| Several GPUs in one process | Supported (Python only) | `devices=` on `tj.project`, `tj.backproject`, `tj.reconstruct` (`fbp`, `cgls`, `fista`) and `tj.align` ([large scans](large-scans.md#use-several-gpus)) |

No-flag `tomojax align` and `tomojax recon` use the Python automatic policies.
Their recorded outcome is conditional: reported limitations are not accuracy
certificates. Choose an explicit mode/method when supplying expert solver settings.

## Scope

NXtomo and generic TIFF/HDF5 frame loading requires recorded or explicitly
supplied detector pixel sizes and view angles. External NXtomo/Data Exchange
use their format-defined axial parallel model. If the pitch is recorded but the
rotation centre is absent, loading estimates detector-u from a few corrected
rows with opposite-view gradient correlation and records `inferred` provenance;
vertical zero defines the free axial origin. Insufficient coverage/contrast
requires explicit `detector=`. Laminography requires a recorded tilt/axis or
`geometry=`. Nikon and Zeiss files lacking dark calibration require `darks=`; zero is
only an explicit choice for already corrected counts. SkyScan additionally
requires `flats=`/`white_level=` rather than deriving it from bit depth.
Compressed TIFF crops decode only their intersecting strips/tiles. In folders,
TIFF-suffixed entries are image storage, not acquisition text sidecars.
TIFF line-integral import requires explicit `--geometry`, `--pixel-size` and
`detector_center = [u, v]` in `--config`; laminography also needs `tilt_deg`.
For raw frames without an explicit grid, detector binning also reduces output
resolution. Axial-parallel crops follow the measured row window; other models
retain their output box. File/caller-supplied grids and `Scan.binned()` retain
their chosen sampling.

Missing values produce an error
naming the override, not centred unit pixels or a 30° tilt.

Workflows outside the table above are research or expert diagnostics.

## Acquisition metadata

`load_frames` recognises these raw exports by content, even with a renamed file.
A partial detector record is retained for inspection, not completed with a
zero rotation centre. Missing axial centres may be estimated from data; an
unavailable pitch or failed estimate requires `detector=`/`geometry=`. Physical
pitches below are normalised to millimetres.

| Export | Frames and calibration | Angles and detector metadata |
|---|---|---|
| ESRF raw Bliss step sequence | Sequence-owned detector blocks, image keys and flat positions; external/virtual pixel links checked | Unit-labelled angular commands before raw motors; `technique/optic/sample_pixel_size`; distances from `technique/scan`; missing axial centre is estimated or supplied |
| ESRF SPEC EDF | Ordered EDF files, reference files, accumulated darks divided by recorded `DARK_N` | `.info`/XML `PixelSize` and propagation `Distance`; caller supplies `angles=` (range alone has no origin), and `detector=` if needed |
| External NXtomo (including ESRF exports) | A group declaring `NXtomo`, with `instrument/detector/data` and `image_key`; the entry name is unrestricted | `sample/rotation_angle` (or `sample/transformations/rotation_angle`), recorded angular units; detector `x_pixel_size`/`y_pixel_size` with length units |
| Data Exchange (including PSI TOMCAT and Elettra SYRMEP TomoBank exports) | `/exchange/data`, `data_white`, `data_dark`, or interleaved `image_key` | `/exchange/theta`; `/measurement/instrument/detector/actual_pixel_size_x` and `_y` with `unit`/`units` when present; sample/source/beamline labels are retained |

These readers do not turn an estimated motor position into a calibrated
rotation centre, or infer pixel pitch from image dimensions. Data Exchange
exports lacking facility labels remain Data Exchange, not a guessed beamline.
This support does not establish support for every native TOMCAT/SYRMEP layout.
Real-file coverage includes ESRF NXtomo converted by nxtomomill, TOMCAT/SYRMEP
Data Exchange, Diamond I12 HDF5 and K11 GDA, Nikon TIFF acquisitions, Zeiss TXRM,
SkyScan logs/TIFFs, Diamond I13 compressed TIFF folders with explicit calibration,
and TomoJAX NPZ/NXtomo round trips. Available raw Bliss/EDF
samples are incomplete and are rejected; a complete raw Bliss/EDF workflow and
native TOMCAT GigaFRoST are not established by those checks.
Missing calibration does not disable content recognition: supply `flats=`/
`white_level=` and `darks=`. NXtomo/Data Exchange angular fields use their
format-defined degree convention when units are absent; generic motor datasets
without units require explicit `angles=`.

SkyScan logs supply distances, camera pitch/binning, optional resize and the
vertical optical-axis line, but not a calibrated horizontal centre. Supply
`detector=`/`geometry=` plus explicit flat/white and dark calibration.

Zeiss reads `ImageInfo/Angles`, object-plane `pixelsize`, `XrayGeometry`,
cone distances, both `alignment/*-shifts` and `AutoRecon/CenterShift`;
missing calibration requires `geometry=` rather than zero image shifts.
The embedded reference uses `ReferenceData/DataType`; separate XRM flat/dark
files use their own `ImageInfo/DataType`. Both are read lazily.
`MultiReferenceData/ImageN` references use their recorded exposure durations and
timestamps; the same count-rate correction handles separate XRM references.

Nikon detector pitch/offsets and source distances come from the `.xtekct`
fields; its axis requires either the recorded modern offset/roll/tilt or both
legacy COR endpoints. Vendor reconstruction dimensions, pitches and ROI offsets
remain source metadata, not an applied reconstruction region: TomoJAX uses its
field-of-view grid unless the caller supplies `grid=`.

## Numerical and accelerator scope

| Operation | Geometry and backend coverage |
|---|---|
| Forward model and matched adjoint | Parallel rays with rigid poses, anisotropic voxels, and shifted grids/detectors; JAX reference and optional CUDA kernels |
| Parallel FBP | JAX on CPU/CUDA; CUDA selected automatically on CUDA for built-in `ParallelGeometry` without an explicit detector grid |
| Host-output FBP | `fbp_host`: NumPy/memmap or device input and optional FP32 output for every parallel-beam geometry `fbp` accepts (cone beams use `fdk_host`); parallel axial slabs or laminography x-slabs via JAX or native CUDA kernels; not differentiable. `fbp` itself streams NumPy/memmap projections batch by batch |
| Fourier-slice inverse (Python API) | `fourier_reconstruct`: uniform/nonuniform half-turn parallel scans, including redundant/opposing exposures, offsets/anisotropic spacing/cropped grids; NumPy reference or optional CuPy CUDA; host arrays/memmaps and axial slabs; not differentiable |
| General posed/laminography filtered adjoint | JAX; approximate initialisation for incomplete-angle data |
| Public FISTA-TV and SPDHG-TV | JAX with matched discrete adjoints; convergence regressions for parallel and tilted scans; NumPy/memmap projections stream from host memory when large (SPDHG-TV also keeps its dual and weights there) |
| Public CGLS (Python API and `tomojax recon --method cgls`) | Matched FP32 JAX/CUDA operators, scalar damping, optional squared physical voxel differences, nonzero starts; CUDA selected automatically on CUDA with canonical detector grids; large NumPy/memmap projections stream from host memory through the equivalent normal equations |
| Joseph plane model (Python API) | `project_joseph` and the default discretisation of CGLS, FISTA-TV and SPDHG-TV; explicit bilinear or Keys cubic interpolation; JAX reference and matched CUDA gather transpose; CUDA first-order volume/pose AD, JVP/VJP and batching; rigid poses and canonical detector grids required |
| Fused Joseph least squares (Python API) | `joseph_l2_value_and_grad`: raw half squared error plus volume/matrix-pose gradients; CUDA retains residuals and tile reductions without a ray-by-plane tape; not called by the alignment pipeline |
| Joseph pose normal equations (Python API) | `joseph_pose_normal_equations`: per-view raw loss, directional gradient, Gauss-Newton matrix and residual; 1–16 caller-supplied pose directions; CUDA avoids a full projection Jacobian; caller chooses damping, priors and gauges |
| Coarse-to-fine CGLS (Python API) | Optional `cgls_multires` with explicit budgets and a final full-data solve; tested on odd/shifted grids and sharp/noisy phantoms, with known performance regressions |
| Internal differentiable FISTA core | JAX and explicit CUDA forward/adjoint variants, used by alignment workflows |
| Cone-beam geometry | `ConeGeometry`: point source and flat detector with offsets, roll, pitch and yaw, a laterally offset (`axis_offset`), tilted or arbitrary axis, any per-view poses; Joseph sampling of diverging rays with a matched transpose; JAX reference (differentiable in volume and poses) and CUDA kernels (CuPy); CGLS, FISTA-TV, SPDHG-TV and FDK (full turns, offset-detector full turns, Parker-weighted short scans); `tomojax simulate`, `tomojax import` and saved datasets. Fan beam is a one-row cone detector |
| Real GPU validation | CUDA on RTX 4070 Laptop (Ada); CPU reference mode is tested separately |

Volumes use `(x, y, z)` array order and projections use `(view, v, u)`. Voxel and
detector spacings must use consistent physical length units. Forward projection
integrates over physical ray length. FBP returns attenuation in the volume's
units; its backprojection normalisation differs from the Euclidean transpose
used by iterative solvers. See [accuracy and performance evidence](measurements.md) for
the measured cases, backend choices, and limitations.

For choosing an alignment mode and checking its result, see the
[alignment guide](alignment-guide.md).
