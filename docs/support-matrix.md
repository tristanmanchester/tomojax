# TomoJAX support matrix

Available workflows and their entrypoints. “Supported” identifies an implemented
and tested path, not a guarantee of accuracy for every scan. Alignment remains
experimental. See [installation](installation.md) and [measurement scope](measurements.md).

| Workflow | Status | Supported entrypoint |
|---|---|---|
| Dataset inspection and validation | Supported | `tomojax inspect scan.nxs` (`--json` for a JSON report; exit status 1 when invalid) |
| Preview PNGs (central projection and central volume slices) | Supported | `tomojax inspect recon.nxs --preview previews` |
| TIFF stack import | Supported | `tomojax import ./tiffs --angles angles.csv --pixel-size ... -o scan.nxs` |
| Dataset container conversion (`.npz`, `.nxs`, `.h5`) | Supported | `tomojax import scan.npz -o scan.nxs` |
| HDF5 flat/dark correction (NXtomo `image_key`, or APS Data Exchange `data_white`/`data_dark`; flat sets before and after interpolated) | Supported | `tomojax preprocess raw.nxs -o corrected.nxs`, or `tj.load("raw.nxs")` |
| Frame corrections in Python (crop, select views, correction steps) | Supported | `tj.load_frames(path).cropped(...).selected(...).corrected(*steps)`; `Scan.corrected(*steps)` for line-integral steps |
| TIFF flat/dark preprocessing | Supported | `tomojax preprocess ./projections -o corrected.nxs --flats ./flats --darks ./darks --angles angles.csv` |
| FBP, CGLS, FISTA-TV, SPDHG-TV from corrected projections | Supported | `tomojax recon corrected.nxs -o recon.nxs --method fbp\|cgls\|fista\|spdhg` |
| Fourier-slice inverse of supported parallel scans | Supported | `tomojax recon corrected.nxs -o recon.nxs --method fourier`; NumPy default, optional CuPy via config |
| Cone-beam (lab CT) scans: FDK and iterative reconstruction | Supported | `tomojax import ./tiffs --geometry cone --source-to-axis ... --source-to-detector ... --pixel-size ... --angles angles.csv -o scan.nxs`, then `tomojax recon` (`--method fbp`, the default, runs FDK) |
| Per-projection 5-DOF pose alignment | Experimental | `tomojax align corrected.nxs -o aligned.nxs --mode pose` |
| Cone-beam 6-DOF pose alignment (adds `dy` along the beam) | Experimental | `tomojax align cone_scan.nxs -o aligned.nxs --mode pose` |
| Ring (stripe), zinger, outlier-view and beam-hardening corrections; Paganin phase retrieval | Supported | `tomojax preprocess raw.nxs -o scan.nxs --remove-stripes 9 --zingers --reject-outliers --beam-hardening 1,0.05` (and the expert `--paganin`), or `tomojax.corrections` steps |
| Volume export as TIFF slices or raw (`uint16` or `float32`) | Supported | `tomojax export recon.nxs -o slices/`; `tomojax export recon.nxs -o recon.raw --dtype uint16` |
| Nikon (X-Tek) `.xtekct` scan import | Experimental (tested on synthetic files only) | `tomojax import scan/part.xtekct -o scan.nxs` |
| Multi-orbit cone-beam scans (`ConeSegments`): iterative reconstruction and pose alignment | Supported; alignment experimental | `tj.Scan.from_astra(...)` or `tj.Scan.combine([...])`, then `tj.reconstruct(scan, "cgls")` and `tj.align(scan)`; `orbit_heights` in `tomojax.alignment.api` |
| Cone-beam axis-offset (centre of rotation) and detector-roll calibration | Experimental | `tomojax align cone_scan.nxs -o calibrated.nxs --mode cor`; `calibrate_cone_axis` in Python |
| Detector-centre/COR alignment | Experimental | `tomojax align corrected.nxs -o aligned.nxs --mode cor` |
| Detector-centre offset with per-view motion | Experimental | `tomojax align corrected.nxs -o aligned.nxs --mode cor-then-pose` |
| Mixed setup and pose alignment | Experimental; each stage fixes its gauge policy | `tomojax align corrected.nxs -o aligned.nxs --mode full` |
| Deterministic synthetic dataset generation | Supported | `tomojax simulate -o synthetic_scan.nxs --size 64 --geometry parallel\|lamino\|cone` |
| Python API reconstruction | Supported | `tj.reconstruct(scan, method, config=...)`, `tj.project`, `tj.backproject`; building blocks in `tomojax.geometry`, `tomojax.forward`, `tomojax.recon` |
| Python API alignment, with resumable checkpoints | Experimental | `tj.align(scan, mode=..., grid=..., checkpoint=..., config=...)` |
| Volumes larger than the GPU | Supported | FDK switches to host slabs automatically; `fbp_host` and `fdk_host` in Python ([large scans](large-scans.md)) |
| Several GPUs in one process | Supported (Python only) | `devices=` on `tj.project`, `tj.backproject`, `tj.reconstruct` (`fbp`, `cgls`, `fista`) and `tj.align` ([large scans](large-scans.md#use-several-gpus)) |

## Scope

Workflows outside the table above are research or expert diagnostics.

## Numerical and accelerator scope

| Operation | Geometry and backend coverage |
|---|---|
| Forward model and matched adjoint | Parallel rays with rigid poses, anisotropic voxels, and shifted grids/detectors; JAX reference and optional Pallas kernels |
| Parallel FBP | JAX on CPU/CUDA; Pallas selected automatically on CUDA for built-in `ParallelGeometry` without an explicit detector grid |
| Host-output FBP | `fbp_host`: NumPy/memmap input and optional FP32 output for every parallel-beam geometry `fbp` accepts (cone beams use `fdk_host`); parallel axial slabs or laminography x-slabs on JAX CPU/CUDA or Pallas CUDA; not differentiable. `fbp` itself streams NumPy/memmap projections batch by batch |
| Fourier-slice inverse (opt-in Python API) | `fourier_reconstruct`: uniform unique half-turn parallel scans, offsets/anisotropic spacing/cropped grids; NumPy reference or optional CuPy CUDA; host arrays/memmaps and axial slabs; not differentiable |
| General posed/laminography filtered adjoint | JAX; approximate initialisation for incomplete-angle data |
| Public FISTA-TV and SPDHG-TV | JAX with matched discrete adjoints; convergence regressions for parallel and tilted scans; NumPy/memmap projections stream from host memory when large (SPDHG-TV also keeps its dual and weights there) |
| Public CGLS (Python API and `tomojax recon --method cgls`) | Matched FP32 JAX/Pallas operators, scalar damping, optional squared physical voxel differences, nonzero starts; Pallas selected automatically on CUDA with canonical detector grids; large NumPy/memmap projections stream from host memory through the equivalent normal equations |
| Joseph plane model (Python API) | `project_joseph` and the default discretisation of CGLS, FISTA-TV and SPDHG-TV; explicit bilinear or Keys cubic interpolation; JAX reference and matched CUDA gather transpose; CUDA first-order volume/pose AD, JVP/VJP and batching; rigid poses and canonical detector grids required |
| Fused Joseph least squares (Python API) | `joseph_l2_value_and_grad`: raw half squared error plus volume/matrix-pose gradients; CUDA retains residuals and tile reductions without a ray-by-plane tape; not called by the alignment pipeline |
| Joseph pose normal equations (Python API) | `joseph_pose_normal_equations`: per-view raw loss, directional gradient, Gauss-Newton matrix and residual; 1–16 caller-supplied pose directions; CUDA avoids a full projection Jacobian; caller chooses damping, priors and gauges |
| Coarse-to-fine CGLS (Python API) | Optional `cgls_multires` with explicit budgets and a final full-data solve; tested on odd/shifted grids and sharp/noisy phantoms, with known performance regressions |
| Internal differentiable FISTA core | JAX and explicit Pallas forward/adjoint variants, used by alignment workflows |
| Cone-beam geometry | `ConeGeometry`: point source and flat detector with offsets, roll, pitch and yaw, a laterally offset (`axis_offset`), tilted or arbitrary axis, any per-view poses; Joseph sampling of diverging rays with a matched transpose; JAX reference (differentiable in volume and poses) and CUDA kernels (CuPy); CGLS, FISTA-TV, SPDHG-TV and FDK (full turns, offset-detector full turns, Parker-weighted short scans); `tomojax simulate`, `tomojax import` and saved datasets. Fan beam is a one-row cone detector |
| Real GPU validation | CUDA on RTX 4070 Laptop (Ada); the multi-device tests and several-GPU timings also on four H100 GPUs; CPU interpretation is tested separately |

Volumes use `(x, y, z)` array order and projections use `(view, v, u)`. Voxel and
detector spacings must use consistent physical length units. Forward projection
integrates over physical ray length. FBP returns attenuation in the volume's
units; its backprojection normalisation differs from the Euclidean transpose
used by iterative solvers. See [accuracy and performance evidence](measurements.md) for
the measured cases, backend choices, and limitations.

For choosing an alignment mode and checking its result, see the
[alignment guide](alignment-guide.md).
