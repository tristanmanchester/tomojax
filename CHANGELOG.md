# Changelog

## Unreleased

The CUDA cone backprojector is again the exact transpose of the forward
projector where a ray is nearly tied between two axes: both now round the ray's
coordinates identically (it differed by up to 1.6% there). Adjoints are about
1.5% slower; forward projections are unchanged.

The CUDA FDK no longer prints `cudaErrorNotPermitted` texture errors: its
textures are freed once their kernel's event completes, never in a stream
callback.

Mistakes now fail early with a clear message rather than giving wrong results:
a reversed `scan.selected(...)` of a multi-orbit scan (which paired views with
the wrong orbit's geometry), invalid `batch_views`, `epsilon`, `white_level` or
flat and view positions in corrections, invalid FISTA and SPDHG settings, and
invalid `fdk_host` slab depths or output arrays. SPDHG now honours `tau`,
`sigma_data` or `sigma_tv` given alone, both solvers accept zero iterations,
and `fdk_host` returns zeros for slabs the detector does not see.

This release adds lab cone-beam CT (geometry, FDK, Nikon import, axis
calibration, ASTRA conversion and multi-orbit `ConeSegments` scans) and a
workflow API at the package root: `tj.load`, `tj.reconstruct`, `tj.align` and
`tj.save`. Alignment now corrects scans that already carry poses and brings
the orbits of multi-orbit scans into register, with the coupled pose solver and
Joseph projection as its defaults. Reconstruction (FBP, FDK, CGLS, FISTA)
and pose alignment share a scan's views among several GPUs (`devices=`), the
solvers stream large scans from host memory,
and most of them need much less device memory. The command line is now a thin
layer over the Python API and every option has one name, so most 0.3 scripts,
config files and command lines need changes; see Migrating from 0.3.

### Migrating from 0.3

A renamed `--config` key fails with its new name ("config key 'lambda_tv' ...
was renamed 'tv_weight'"); a removed key fails with the list of valid keys. A
renamed Python keyword raises `TypeError`.

**Python names**

| 0.3 | Now |
|---|---|
| `tomojax.align` (the package) | `tomojax.alignment`; `tomojax.align` is now the function `tj.align` |
| `tomojax.align.AlignConfig`, `align_multires` | `tomojax.alignment.AlignConfig`, `align_multires` |
| `tomojax.align.align` | `tomojax.alignment.api.align`, or `tj.align(scan)` |
| `align(init_params5=...)`, `se3_from_5d` | `init_pose_params=`, `se3_from_pose_params` |
| `AlignResumeState.params5` and `.L`; `AlignMultiresResumeState`'s `*_outer_iters_*` fields | `.pose_params`, `.lipschitz`; `*_outer_iterations_*` |
| `ReconstructionAlgorithmOptions`, `ReconstructionAlgorithmRequest`, `ReconstructionResult`, `run_reconstruction_algorithm` (`tomojax.recon.api`) | `tj.reconstruct`; on arrays, `tomojax.recon.api.method_config` and `reconstruct_arrays` |
| `thetas_deg=` on `ParallelGeometry`, `LaminographyGeometry`, `RotationAxisGeometry` | `angles=` (degrees) |
| `ProjectionDataset.angles_deg`, `load_tiff_stack(angles_deg=)` | `angles` |
| `NXTomoMetadata.thetas_deg`, `RealLaminographyInput.thetas_deg`, `simulate(cfg)["thetas_deg"]` | `angles` |
| `Detector(det_center=...)`, `Detector.det_center` | `center` (`tomojax inspect` reports `center`; `Detector.from_dict` reads `to_dict`'s output) |
| `build_geometry_from_dataset_metadata(apply_saved_alignment=)` | `poses=` |
| `cgls_multires(iters_per_level=)` | `iterations_per_level=` |
| Configuration fields given by position; `sphere`, `cube` and `blobs` options (`size`, `value`, `seed`, `n_blobs`); `preprocess_nxtomo`'s `config` | keywords only |

**Solver configuration fields** (`FistaConfig`, `SPDHGConfig`, `CGLSConfig`,
`FBPConfig`, `FBPHostConfig`; also saved reconstruction `info`)

| 0.3 | Now |
|---|---|
| `iters` | `iterations` |
| `lambda_tv` | `tv_weight` |
| `positivity` | `nonnegative` |
| `L` | `lipschitz` |
| `tv_prox_iters`, `power_iters` | `tv_prox_iterations`, `power_iterations` |
| `filter_name` | `filter` |

**`AlignConfig` fields** (also the `tomojax align` `--config` keys)

| 0.3 | Now |
|---|---|
| `align_profile="lightning"` / `"tortoise"` | `quality="fast"` / `"reference"` |
| `outer_iters` | `outer_iterations` |
| `recon_iters` | `iterations` |
| `recon_algo` (`fista_tv`, `spdhg_tv`, ...) | `reconstruction` (`"fista"` or `"spdhg"`) |
| `lambda_tv`, `tv_prox_iters` | `tv_weight`, `tv_prox_iterations` |
| `recon_positivity` | `nonnegative` |
| `spdhg_seed` | `seed` |
| `recon_L` | `lipschitz` |
| `gn_joint_iters` | `gn_joint_iterations` |
| `freeze_dofs` | `freeze` |
| `gauge_fix` | removed; alignment returns the least-motion estimate (see New) |
| `quality_tier`, `fallback_policy`, `fold_rigid_detector_grid` | removed (they echoed the profile or were always on) |

Each setting takes one spelling. Removed: `opt_method` values `lbfgsb`,
`l_bfgs` and `l_bfgs_b`; hyphenated `gauge_policy` values (`anchor-mean`,
`prior-required`, `diagnose-only`); `pose_model="per-view"`; hyphenated or
upper-case schedule names. Alignment `info` uses the new names
(`reconstruction`, `lipschitz`, `quality`, `completed_outer_iterations`,
`total_outer_iterations`, and per outer iteration `lipschitz_measured` and
`lipschitz_next`).

`AlignConfig()` keeps its own defaults (object-frame translations, the
alternating solver), and `tj.align(config=...)` uses a config as given. To
change one setting of what a mode runs, start from the mode's configuration:
`replace(alignment_plan("pose", scan.grid).config, ray_integrator="exact")`.

**Exports.** Package roots export what users call. Names no longer at a root
are in that package's `.api` module: from `tomojax.recon`, `Regulariser`,
`clear_filter_caches`, `default_fbp_scale` and `run_parallel_fbp_direct_pallas`;
from `tomojax.geometry`, the axis constants, calibration and gauge helpers,
`axes_to_perm`, `transpose_volume`, `read_geometry_json` and
`read_pose_params_csv`; from `tomojax.io`, `NXTomoMetadata`, `LoadedNXTomo`,
`convert_dataset`, `load_nxtomo`, `save_nxtomo`, `validate_nxtomo` and the
payload and JSON helpers. Removed, as nothing used them: `VolumeSupportKind`,
`centered_volume_support`, `sum_backproject_views_chunked` and
`supports_parallel_fbp_z_integer` (recon); `build_calibration_manifest`,
`canonicalize_geometry_gauges`, `write_geometry_json`, `write_pose_params_csv`,
`write_pose_decomposition_csv` and the detector-grid transform helpers
(geometry); `write_json_object`, `spatial_bin`, `pad_to_multiples`,
`volume_chunks`, `flat_dark_to_transmission`, `transmission_to_absorption`,
`flat_dark_to_absorption` and `absorption_to_transmission` (io; `tj.load_frames`
and `Frames.corrected` correct frames, `np.exp(-p)` is transmission);
`preprocess_nxtomo`, `preprocess_tiff_stack`, `PreprocessConfig` and
`PreprocessResult` (io; `tj.load_frames(path, ...).corrected(*steps)` and
`tj.save`, with `PreprocessConfig`'s settings as `load_frames` keywords,
`Frames.selected`, `Frames.cropped` and steps); the profile, fallback and gauge-fix types, `schedule_preset`,
`level_detector_grid`, `build_loss_adapter`, `ScheduleResumeState`,
`normalize_schedule_resume_state`, and
`build_alignment_checkpoint_metadata_from_input` with its input classes
(alignment, replaced by `tomojax.alignment.api.AlignmentRun`), among others;
and the synthetic-sidecar helpers in `tomojax.datasets.api`.

**Commands.** Every command is `tomojax <command> INPUT -o OUTPUT` and refuses
an existing output unless `--force`. Exit status is 0 for success, 1 for
failure and 2 for a usage error. `--help` lists the options most runs need;
expert settings are `--config` keys, listed by `--config-keys`.

| 0.3 | Now |
|---|---|
| `--data IN --out OUT`; `preprocess IN OUT`; `convert --in IN --out OUT` | `IN -o OUT`; `import IN -o OUT` converts `.npz` and `.nxs` |
| `tomojax ingest` | `tomojax import` |
| `tomojax validate` | `tomojax inspect` (exits 1 for an invalid dataset; `--json`) |
| `tomojax slices` | `tomojax inspect --preview DIR` |
| `--quicklook`, `--save-preview` | `--preview` |
| `--save-manifest` | `--manifest` |
| `--volume-axes`, `--transfer-guard` (`recon`, `align`) | removed; volumes are saved as `tj.save` saves them (`tomojax export` writes other layouts), and JAX's own `jax.transfer_guard` |
| `ingest --du --dv` | `import --pixel-size SIZE [SIZE_V]` |
| `ingest --sample-name` | `import --name` |
| `ingest --det-center-u/-v`, `--grid`, `--voxel-size` | removed; `tomojax align --mode cor` estimates the centre, and `recon --grid` or `tj.reconstruct(grid=...)` sets the grid |
| `preprocess --format`, `--domain`, `--log`, `--transmission`, `--dtype`, `--clip-min` | removed: `preprocess` writes a float32 scan of line integrals, bounded below by `--epsilon` |
| `preprocess --assume-flat-field V`, `--assume-dark-field V` | `--flats V`, `--darks V` (a level for every pixel) |
| `preprocess --auto-reject`, `--outlier-z-threshold Z` | `--reject-outliers [Z]` (views that jump from their neighbours; non-finite values are set to zero and counted) |
| `preprocess --select-views-file`, `--reject-views-file` | `--select-views`, `--reject-views` with the ranges |
| `simulate --nx --ny --nz`, `--nu --nv`, `--n-views` | `--size N` (or the `grid` and `detector` keys), `--views` |
| `simulate --rotation-deg`, `--tilt-deg` | `--rotation`, `--tilt` |

**`tomojax recon`** takes `--method`, `--filter`, `--iterations`,
`--tv-weight`, `--nonnegative`, `--warm-start`, `--seed`, `--grid`, `--roi`,
`--poses`/`--no-poses`, `--preview`, `--manifest` and `--progress`. Expert
settings are fields of the method's configuration class (`FBPConfig`, also for
FDK, `CGLSConfig`, `FistaConfig`, `SPDHGConfig`); a setting the method does not
take fails, naming the ones it does.

| 0.3 | Now |
|---|---|
| `--algo` | `--method` (`fbp`, `cgls`, `fista`, `spdhg`) |
| `--iters`, `--lambda-tv`, `--spdhg-seed` | `--iterations`, `--tv-weight`, `--seed` |
| `--positivity` / `--no-positivity` | `--nonnegative` (off by default) |
| `--warm-start fbp` | `--warm-start` |
| `--apply-saved-alignment` / `--ignore-saved-alignment` | `--poses` (now the default) / `--no-poses` |
| `--mask-vol cyl` | `--roi cyl`: the `auto` crop, the volume zeroed outside the cylinder every view sees, and that cylinder as FISTA's and SPDHG's `support` |
| `--regulariser`, `--huber-delta`, `--lower-bound`, `--upper-bound`, `--theta`, `--views-per-batch`, `--gather-dtype` | `--config` keys of the same names (`views_per_batch` has no `auto`) |
| `--tv-prox-iters`, `--L` | keys `tv_prox_iterations`, `lipschitz` |
| `--spdhg-tau`, `--spdhg-sigma-data`, `--spdhg-sigma-tv` | keys `tau`, `sigma_data`, `sigma_tv` |
| `--no-checkpoint-projector` | `checkpoint_projector = false` |
| `--det-u-px`, `--det-v-px` | removed; `tomojax align --mode cor`, or shift `scan.detector.center` in Python |
| `--frame` | removed (it only labelled the file) |

The manifest records `method` (was `algorithm`), the resolved configuration
and the solver's record under `reconstruction` (was `algorithm_config`), and
an `roi` block.

**`tomojax align`** takes `--mode`, `--quality`, `--levels`, `--freeze`,
`--roi`, `--grid`, `--checkpoint`, `--poses`/`--no-poses`, `--manifest`,
`--progress` and `--dry-run`. Expert settings are `AlignConfig` fields, each
replacing that field of the configuration the mode and quality give (which
`--dry-run` prints).

| 0.3 | Now |
|---|---|
| `--mode auto` | `--mode full` |
| `--mode max` | `--mode full --quality reference` |
| `--mode cor_then_pose`, `tj.align(mode="cor_then_pose")` and other spellings | `cor-then-pose`: each mode has one spelling, the same in Python and the CLI |
| `--quality` aliases `normal`, `full`; `--align-profile` | `--quality fast` or `reference` |
| `--freeze-dofs` | `--freeze` |
| `--print-plan-json` | `--dry-run` |
| `--resume PATH`, `--checkpoint-every N` | `--checkpoint PATH`: written after every outer iteration, resumed by itself |
| `--save-params-json`, `--save-params-csv` | `save_alignment_params_json` and `save_alignment_params_csv` in `tomojax.alignment.api`, on `tj.load("aligned.nxs").poses` |
| `--loss`, `--loss-param`, `--loss-schedule` | the `loss` key: a name (`"huber"`), a table (`{ name = "huber", delta = 1.0 }`) or a level schedule (`"4:phasecorr,2:ssim,1:l2_otsu"`) |
| `--outer-iters`, `--recon-iters`, `--recon-algo`, `--lambda-tv`, `--tv-prox-iters`, `--recon-positivity`, `--spdhg-seed`, `--recon-L`, `--early-stop-rel` | keys `outer_iterations`, `iterations`, `reconstruction`, `tv_weight`, `tv_prox_iterations`, `nonnegative`, `seed`, `lipschitz`, `early_stop_rel_impr` |
| The other expert flags (`--regulariser`, `--views-per-batch`, `--gather-dtype`, `--opt-method`, `--gn-damping`, `--lbfgs-*`, `--lr-rot`, `--w-rot`, `--optimise-dofs`, `--schedule`, `--bounds`, `--gauge-policy`, `--pose-model`, `--seed-translations`, `--early-stop`, `--mask-vol`, `--log-summary`, ...) | `--config` keys of the same names |
| `--gauge-fix` | removed |

The aligned file holds six-column, detector-frame poses. The manifest gives
the run's inputs and settings with the alignment's record (losses, gauge,
calibrated setup geometry) under `alignment`.

**Changed behaviour and defaults**

- `tj.load` corrects a file of raw detector frames (an NXtomo `image_key`
  marking flats or darks) to line integrals, as `load_frames(path).corrected()`;
  it used to return the flats and darks as projections, in counts. A file of
  integer counts without flats, or a TIFF stack, raises and points to
  `tj.load_frames`. A Nikon scan's line integrals are `-log(I / WhiteLevel)`
  as before, with `I` no longer clipped to 1 first.
- `tomojax recon` applies the poses saved in its input unless `--no-poses`
  (0.3 ignored them unless `--apply-saved-alignment`).
- `tomojax align` corrects on top of the poses saved in its input, as
  `tj.align(tj.load(path))` does; 0.3 started from the nominal geometry and
  replaced them. `--no-poses` starts from the nominal geometry. A posed input
  takes only `--mode pose`.
- `tomojax align --mode pose` (and `cor-then-pose`) runs the coupled
  volume-and-pose solver: Joseph projection, least squares without TV, fp32
  gathers, up to 30 early-stopped outer iterations, detector-frame
  translations, a global shift search first, and coarse-to-fine levels that
  keep at least 32 voxels on the grid's shortest axis (4, 2, 1 from 128).
  `gn_coupling = "fixed_volume"` gives the alternating scheme. Every mode uses
  Joseph projection.
- Pose translations are in the detector frame by default
  (`pose_translation_frame = "object"` restores the 0.3 tables): object-frame
  translations cannot shift the image sideways where the sample's x axis lies
  along the beam.
- Alignment returns the estimate with the least per-view motion, so its poses
  and volume can differ from 0.3's by a rigid motion.
- `spdhg` runs 400 iterations in `tomojax recon` too (was 50), each one block
  of views, and does not clip at zero unless `nonnegative`
  (`SPDHGConfig.nonnegative` defaults to False, was True).
- `gather_dtype` is `fp32` on every device (0.3's `tomojax recon` used `bf16`
  on GPUs).
- CGLS, FISTA and SPDHG project with `projector_model="auto"`: Joseph plane
  sampling, with Pallas kernels on CUDA (CGLS used the ray model).
  `projector_model="ray"` restores it; explicit detector grids and exact
  integration keep the ray model.
- `FistaConfig.views_per_batch` defaults to None: 64 views with batched
  operators, one on the ray-model path (was 1). `power_iterations` is 3 (was
  5), started from the backprojected data. FISTA's loss history and early
  stopping follow the extrapolated point.
- `AlignConfig.views_per_batch` defaults to 0, which sizes batches from free
  GPU memory (was 1).
- `warm_start` is a bool and starts `cgls`, `fista` and `spdhg`.
- `FBPHostConfig.slices_per_batch` defaults to None, sized to free device
  memory (was 16).
- FBP weights every circular parallel-beam scan exactly, and backprojects
  voxel by voxel in every geometry (see Fixed). Tilted, partial-turn,
  irregular, laminography and anisotropic FBP volumes change; an explicit
  `FBPConfig.scale` keeps the old uniform weighting.
- Compiled JAX programs are cached on disk (`TOMOJAX_JAX_CACHE=off` disables
  it, `TOMOJAX_JAX_CACHE_DIR` moves it; an existing JAX cache setting wins).
  The command line sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` unless it is set.

**Checkpoints.** Alignment checkpoints are schema 4. Checkpoints from 0.3 do
not resume ("schema version 1 predates this version of TomoJAX"); restart the
alignment. A run is identified by its mode, `AlignConfig`, levels, grid,
detector and a fingerprint of its projections and geometry; a checkpoint of
other settings is refused, naming the difference. `CheckpointError` is a
`ValueError`.

**Saved files.** Files from 0.3 still load. `.nxs` files keep
`rotation_angle`, `.npz` files their `thetas_deg` key, and detector metadata
its `det_center` key. Five-column pose tables load with `dy = 0`. Poses saved
without a translation frame (0.3's) are read as object-frame poses, and
`Scan.poses` reports them in the detector frame.

### New

- **Workflow API.** `tj.Scan` holds projections and the geometry that made
  them. `tj.load` reads TomoJAX datasets and Nikon `.xtekct` scans, with their
  saved poses unless `poses=False`. `tj.reconstruct(scan, method)` runs `fbp`
  (FDK for cone beams, in host slabs when the volume is too large for the
  device), `cgls`, `fista` or `spdhg`; `config=` takes the method's
  configuration class (another method's class raises `ValueError`), keywords
  replace its fields, and `result.info` holds the resolved config and the
  solver's record. `tj.align(scan, mode=...)` returns an `Alignment` of
  `(volume, scan, poses, info)` whose `scan` carries the corrections.
  `tj.save` writes scans, reconstructions and alignments, and
  `tj.load_reconstruction` reads a reconstruction with its `info`.
  `tj.project` and `tj.backproject` work on every geometry. `import tomojax`
  does not import JAX.
- **Geometry from data consistency.** `tomojax.alignment.api.orbit_heights`
  places each orbit of a multi-orbit cone-beam scan relative to the first
  without reconstructing: any plane through two views' sources must have the
  same plane-integral derivative in both views' data (Grangeat), so a wrong
  height makes them disagree. On the FIPS walnut, from its uncorrected
  geometry, it finds +0.37 and +0.74 mm in 23 s (pose alignment: +0.38 and
  +0.76 in about 250 s; held-out views favour about +0.39 and +0.76). It is
  good to about half a pixel at the axis. `tomojax.geometry.api.view_frames`
  gives each view's source and detector frame.
- **Corrections.** `tj.load_frames(path)` reads detector frames as `tj.Frames`:
  the sample frames of an NXtomo file (read from the file only as they are
  corrected), its flats and darks by `image_key`, a Nikon scan with its white
  level, or a TIFF stack with `angles=` (and `flats=`/`darks=` as arrays or
  TIFFs). `Frames.corrected(*steps)` makes the scan of line integrals
  `-log((I - D) / (F - D))` on the device, a batch of views at a time, each
  view's flat interpolated between the flat sets taken around it; steps from
  `tomojax.corrections` run on counts, transmission or line integrals:
  `Stripes(width)` removes rings (sorting-based, on the GPU a detector row at
  a time), `RejectViews(z)` drops views whose median is an outlier (and their
  geometry), `Zingers` replaces bright specks with their neighbours'
  median, `Paganin` retrieves single-material phase from propagation-based
  phase contrast, and `BeamHardening` linearises with a polynomial.
  `Scan.corrected` runs line-integral steps on a scan; `selected` and
  `cropped` (on `Scan` and `Frames`) keep some views or a detector block with
  their geometry (each view's flat still interpolated at its place in the
  scan; a lazily read file reads only the block). `load_frames` finds frames,
  `image_key` and angles (radians converted) in other HDF5 layouts or at
  `data_path=`..., takes NXtomo pixel sizes, and a whole `geometry=`. APS
  Data Exchange files (`exchange/data` with `data_white`, `data_dark` and
  `theta`; TomoPy's and tomocupy's layout) load and correct as they are. `Scan.corrections` records
  what was done, and is saved with the scan.
- **Cone-beam CT.** `tomojax.geometry.ConeGeometry` with a `ConeBeam` source
  and a flat detector (offsets, roll, pitch and yaw; turntable, tilted or any
  rotation axis; per-view poses). Rays are sampled on voxel planes (Joseph)
  with an exact matched transpose, in JAX (differentiable in the volume and
  the poses) and as CUDA kernels. CGLS, FISTA and SPDHG reconstruct cone
  scans. `tomojax.recon.fdk` (and `fbp` on cone data) reconstructs full turns,
  Parker-weighted short scans and offset detectors (Wang's weights: a
  detector covering 5.5 columns on one side of the axis reconstructs a 32³
  phantom with error 0.071, against 0.075 for a centred detector twice as
  wide). `fdk_host` reconstructs scans larger than device memory in z-slabs,
  from and into memmaps. Cone datasets without a grid get one voxel per
  detector pixel at the rotation axis. `tomojax simulate --geometry cone` and
  `tomojax import --geometry cone --source-to-axis ... --source-to-detector ...`
  make cone datasets.
- **Lab CT import.** `tomojax import scan.xtekct -o scan.nxs`, `tj.load` and
  `tomojax.io.load_nikon_xtekct` read Nikon (X-Tek) scans: distances, detector
  pixels and offsets, volume, white level and angles (from `_ctdata.txt` when
  present), converted to absorption. A [lab cone-beam CT guide](docs/lab-ct.md)
  covers import, calibration, FDK, iterative reconstruction and motion
  correction.
- **Cone axis calibration.** `ConeBeam.axis_offset` places the rotation axis
  laterally, and `tomojax.recon.calibrate_cone_axis` estimates it and the
  detector roll from the sharpness of thin FDK slabs at three heights.
  `tomojax align --mode cor` runs it on cone data, and `cor-then-pose` and
  `full` run it before their pose stages. On 128³ and 256³ blob scans it
  recovers offsets of up to 11 voxels to 0.08 voxels and rolls of up to 1.2°
  to 0.04°, in about 4 s at 256³.
- **Multi-orbit scans.** `tomojax.geometry.ConeSegments` holds scans whose
  source and detector arrangement changes partway (several orbits, stacked
  sections of a tall sample): one arrangement per run of views, reconstructed
  together by CGLS, FISTA and SPDHG and saved like any scan. `tj.Scan.combine`
  joins scans. Segments with detectors of different pixel pitch are refused.
  `tomojax.geometry.ScanGeometry` names what every scan geometry provides.
- **ASTRA Toolbox conversion.** `tj.Scan.from_astra(data, proj_geom,
  vol_geom)` reads `cone` and `cone_vec` geometries as a fitted circular orbit
  plus per-view pose corrections, splitting into segments where the
  arrangement changes; `scan.to_astra()` goes back. Converted scans project
  like ASTRA to 0.03% on tilted, offset and jittered vector geometries.
- **Aligning posed and multi-orbit scans.** `tj.align` corrects scans that
  already carry poses (ASTRA imports, earlier alignments) on top of them, and
  aligns `ConeSegments` scans as one, bringing their orbits into register.
  From the FIPS walnut's uncorrected record it recovers the authors' orbit
  heights (orbit 2 -0.381 mm against -0.397, orbit 3 -0.755 against -0.794),
  and the reconstruction then matches the corrected one (error 0.155 against
  0.154; 0.280 uncorrected); see `bench/walnut_alignment.py`,
  `examples/align_walnut_orbits.py` and docs/lab-ct.md. `cor`,
  `cor-then-pose` and `full` refuse posed and segmented scans.
- **Six-parameter poses.** Pose tables (`pose_params`) have a sixth column,
  `dy` along the beam, which changes cone-beam magnification and stays zero
  for parallel beams. Alignment estimates it with the other five and anchors
  its mean; `--freeze dy` keeps it fixed.
- **Alignment.** Modes are `pose`, `cor`, `cor-then-pose` and `full`, planned
  by `tomojax.alignment.alignment_plan` for Python and the command line alike.
  - The coupled solver recovers five of the six free-voxel pilot cells
    (rotations to 0.0006–0.008°), where the alternating scheme left 0.1–4°.
  - Joseph plane sampling (`ray_integrator="joseph"` or `"joseph_cubic"`)
    with a matched gather transpose: on analytic 128³ scans pose alignment
    recovers parallel and laminography rotations to 0.0061° and 0.0024° (in
    41 and 45 s; see Performance), where exact integration took 588 and
    1044 s for 0.0059° and 0.0016°. Joseph coefficients accept calibrated
    (offset and rolled) detectors.
  - A global per-view shift search seeds pose alignment (`seed_translations`,
    on in `pose` mode): on 64³ scans with ±0.5° tilts and ±15 px shifts,
    rotation errors fall from 7.7° and 12.6° to 0.034° and 0.012° (parallel,
    laminography). Coarse to fine, ±3° motion is recovered in 64³
    laminography, where one level stopped at 0.037°.
  - The least-motion estimate. Rotating or shifting the object, and every
    pose the opposite way, predicts the same data; alignment now returns the
    estimate with the least per-view motion and moves the volume to match
    (volume errors 0.26 to 0.0037 on a 64³ cone scan, 0.55 to 0.019 on a
    half-turn parallel scan with a detector offset). `info["gauge"]` records
    the motion removed; `tomojax.alignment.api.least_motion_estimate` moves
    any volume and pose table, a truth say, to the same estimate.
  - `cor-then-pose` on parallel data runs the pose solver and saves the
    constant part of the detector-u shifts as the detector centre. With a
    +3.7 px offset and ±0.5°/±8 px motion on analytic 128³ scans, rotation
    errors fall from 0.23–0.26° to 0.005–0.006°; on the gVXR chip phantom a
    3.2 px offset is recovered as 3.18 px. The mode used to search for the
    offset before correcting any motion, which biased it.
  - `cor` mode on parallel and laminography data starts from the detector-u
    offset whose FBP reprojects most consistently, which needs no opposite
    views. With a +3.7 px offset on analytic 128³ scans it recovers 3.693 and
    3.677 px in 41 and 84 s, where it previously reached 3.50 and 3.59 px in
    about 380 s.
  - `tj.align(scan, grid=...)` aligns on another grid than the scan's.
    `tj.align(scan, checkpoint=path)` saves progress after each outer
    iteration and resumes a checkpoint of the same alignment (a finished one
    returns at once); a checkpoint of another alignment raises `ValueError`
    naming what differs, and is left untouched.
  - `result.info` holds the resolved `config` and, for pose alignment of
    parallel and laminography scans, `implied_detector_u_px`, the
    detector-centre offset the translations hold, which `tomojax align` logs
    (`tomojax.alignment.api.implied_detector_offset` computes it).
  - `tomojax.alignment.api.coupled_pose_config(**overrides)` is the
    configuration `pose` mode runs. The README example, a 96³ laminography
    scan with ±1° and ±2 px motion, goes from 0.57 to 0.087 relative error in
    14 s, with rotations to 0.0026°.
  - An alignment whose next level cannot fit in device memory stops at the
    level before, with a warning; `info["factors"]` lists the levels that ran
    and `info["factors_skipped"]` the others. A first level that cannot fit
    raises `AlignmentMemoryError` before any work.
- **Several GPUs.** `tj.project`, `tj.backproject` and `tj.reconstruct` with
  `cgls` or `fista` take `devices=` (one device or several; `jax.devices()`
  for all), as do `CGLSConfig` and `FistaConfig`. Each device projects its
  share of the views and holds the whole volume, and their backprojections
  are summed, so the transpose stays exact; each device reads only its own
  views, and the CUDA kernels launch on the GPU holding their buffers.
  Twenty FISTA iterations on the binned walnut take 24.8 s on one H100,
  13.6 s on two and 7.4 s on four (docs/performance.md).
  - FBP and FDK (`FBPConfig.devices`, `FDKConfig.devices`) weight each view
    for the whole scan and filter and backproject each device's share in a
    thread of its own; `fdk_host` gives each device its own z slabs, so a
    volume larger than any one GPU reconstructs on several.
  - `tj.align(devices=)` shares the views in `pose` alignment's joint pose and
    volume update (one `shard_map`: each device holds its views' projections,
    pose columns and pose increments) and in cone-beam reconstruction steps.
    The devices are not a setting of the alignment, so a checkpoint made on
    some resumes on others. SPDHG stays on one device: each of its steps
    would sum a whole volume across the devices for one block of views.
- **Host streaming.** FISTA, CGLS and SPDHG read NumPy or memmap projections
  larger than 40% of free device memory one view batch at a time inside the
  compiled solve (`stream_projections` forces either way). On a 512³,
  3072-view laminography scan (3.2 GB) on an 8 GB GPU, FISTA peaks at 4.3 GB,
  CGLS at 3.2 GB and SPDHG at 4.3 GB; where both fit, streaming costs FISTA
  4%. `fbp` streams NumPy or memmap input, and `fbp_host` takes every
  geometry `fbp` does: a 1024³, 1024-view laminography FBP from and to
  memmaps (4.3 GB each) runs in 37 s on an 8 GB GPU, where 768³ previously
  ran out of memory.
- `Scan.binned(n)` averages n x n detector pixels. Iterative
  `tj.reconstruct` and `tj.align` warn when the detector samples the rotation
  axis at least twice as finely as the voxels, naming the `binned(n)` that
  makes them up to n² times cheaper. On the FIPS walnut (pixels half a voxel
  at the axis) it makes iterative reconstruction 3.7 times faster with no
  loss of accuracy.
- `tomojax preprocess` is a thin layer over `tj.load_frames` and the
  corrections: `--flats`/`--darks` (TIFFs or levels, for any input),
  `--select-views`, `--reject-views`, `--crop`, `--zingers`,
  `--remove-stripes WIDTH` (rings), `--reject-outliers`, `--beam-hardening
  C1,C2,...` and, expert, `--paganin`, `--epsilon` and the HDF5 paths. The
  scan written records each correction, and `tomojax inspect` lists them.
- `tomojax export` writes a reconstruction as 32-bit or scaled 16-bit TIFF
  z-slices, or one raw file, with a JSON sidecar of shape, voxel size and
  scaling, reading one slice at a time.
- `tomojax --version`; `tomojax simulate` needs only `-o`, and sizes a cone
  detector to see the whole volume. `Scan.source` is the dataset record a
  scan was loaded from; `LaminographyGeometry.axis_unit_lab` gives its
  rotation axis; geometries can supply a vectorised `stack_poses`.

### Fixed

- `tj.align(mode="cor")` (and `tomojax align --mode cor`) found a wrong axis
  for a parallel scan of a sample larger than the field of view: on an APS
  2-BM scan (720 views of 22 x 1536) it seeded −88.5 pixels for a true −15.1
  and spent about 30 minutes refining from there. A parallel scan over a half
  turn is now seeded by Vo, Atwood and Drakopoulos's (2014) sinogram method
  (−15.25 here, in 1.6 s), refined on a slab of its central rows with the
  batched operators, scored without linearising, and stopped when a round
  gains little: 61 s to −15.11. Other geometries keep the reprojection
  search. The validation residuals batch as many views as fit the device
  when `views_per_batch` is 0, instead of one.
- `tomojax align --mode pose` (and `align_multires` with pose-only schedules)
  optimised the poses against an all-zero volume and returned the nominal
  geometry. Pose stages alternate with reconstruction again, and a given
  `lipschitz` is honoured at full resolution and on resume.
- The coupled solver shifted poses alone to fix the translation gauge, which
  capped its accuracy at about 0.08°. `gn_coupling="joint"` now applies to
  every pose-only stage of a named schedule.
- Each view's 5 x 5 pose block is solved in scaled variables, with Marquardt
  damping only where its FP32 Cholesky factor fails. Unscaled factorisations
  of weakly determined blocks returned NaN and rejected whole Gauss-Newton
  steps.
- The translation seed search never ran at the first coarse-to-fine level.
- `tomojax recon --algo spdhg` always clipped the volume at zero, whatever
  `--no-positivity` said.
- `--warm-start fbp` did not start FISTA.
- FBP weighted tilted (laminography) axes, partial or full turns and
  irregular angles as a uniform untilted half turn. It now fits the axis, arc
  and angular spacing from the poses and filters each view to match: tilted
  reconstructions match the measured-frequency truth to 5% (14% and 42%
  before, for full and half turns). Uniform untilted half turns are
  unchanged.
- Laminography and anisotropic FBP backprojected with the ray-model adjoint,
  which blurred the result (5% error on a smooth parallel phantom, 0.1% now).
  FBP now backprojects voxel by voxel with bilinear interpolation in every
  geometry, with the Pallas kernel on CUDA; explicit `det_grid` inputs keep
  the ray model.

### Performance

- **Cone beam on CUDA.** One forward kernel serves every view and is 1.3 to
  3.7 times faster than this release's first cone kernels; the transpose
  keeps full occupancy at 40 registers (20% faster). On the FIPS walnut's
  three orbits (every 4th view, 20 iterations), non-negative least squares
  takes 86 s binned 2 x 2 (ASTRA 91 s) and 304 s unbinned (ASTRA 217 s).
- **FDK** filters rows up to 2048 pixels with one matrix product and samples
  the filtered images through the texture unit from half floats, with
  interpolation weights rounded to 1/256 as in ASTRA's FDK. On the FIPS walnut
  (1200 views, 501³) it takes 1.7–1.9 s against ASTRA's 2.13 s and agrees
  with ASTRA's volume to 0.05%; a synthetic 256³ case takes 0.048 s (ASTRA
  0.29 s), and a 1024³ host FDK 8.0 s. FDK compiles a third as many programs
  on its first call and reads the next batch of views while the current one
  runs.
- **FISTA and SPDHG** run on the batched operators CGLS uses: 50 FISTA
  iterations on a 64³ scan fall from 20 s to 0.15–0.19 s, and 40 SPDHG
  iterations from 1.3 s to 0.05–0.07 s, with equal or lower error. FISTA also
  runs one projection fewer per iteration (a quarter faster), reuses its
  first backprojection as its first gradient, and skips the TV step when the
  TV weight is zero. On the 128³ TV comparison (`bench/compare_tv.py`) FISTA
  reaches 0.076 relative error in 1.13 s; TIGRE's FISTA reaches 0.101 in
  9.7 s and ASD-POCS 0.146 in 5.7 s.
- **CUDA C Joseph transpose**, compiled at run time with CuPy (now in the
  `cuda12` extra), for volumes of 2^24 voxels and more (it costs about 0.3 s
  per process to start): laminography backprojection is 1.6x and parallel
  1.3x faster at 256³, structured 256³
  laminography CGLS takes 1023 ms instead of 1372 ms warm, and a 512³
  FISTA-TV solve 48.7 s instead of 63.1 s. `TOMOJAX_CUDA_KERNELS` forces the
  CUDA kernels on (1) or off (0).
- The Joseph forward kernel skips the planes a ray does not cross (8–13%
  faster at 256³, bitwise unchanged), and its 8 x 16 ray tiles make
  laminography projection 18% faster. Joseph kernels are traced once per
  configuration, cutting a CGLS solve's tracing time by about a third.
- **Alignment.** The coupled solver caches pose Jacobian columns whenever
  five sinograms fit in a quarter of free device memory: full-resolution
  `tomojax align --mode pose` on a 256³, 361-view laminography scan takes
  188 s, with rotations to 0.0030°. Reconstruction batches sized from free
  memory make warm alignment of the free-voxel cells 1.9–2.3x faster and cold
  1.4–2x. Alignment's reconstruction step uses the matched Joseph transpose
  and the CUDA kernels with dynamic poses: pose alignment of analytic 128³
  scans takes 41 s (parallel) and 45 s (laminography).
- A saved detector roll of zero no longer sends `tomojax recon` down the
  ray-model path one view at a time: 100 FISTA iterations on the 720-view
  chip phantom take 29 s, where one took more than 400 s.
- **Start-up.** Importing `tomojax.geometry` and `tomojax.recon` loads JAX
  and SciPy only when needed (a Fourier reconstruction imports in about 50 ms
  instead of 450 ms). CGLS checks its inputs inside the solve: a cold 64³
  laminography call takes 675 ms instead of about 930 ms, or 385 ms with a
  warm cache.
- **Device memory.** On a 512³, 768-view laminography scan on an 8 GB GPU,
  CGLS (previously out of memory) peaks at 5.6 GB, FISTA-TV falls from 7.2 to
  4.6 GB and SPDHG-TV (previously out of memory) peaks at 4.9 GB. The Joseph
  and cone kernels read and write the `(view, row, column)` layout, so no
  transposed copies of the projections are made; the transposes accumulate view batches
  in place; FISTA's TV step keeps three volumes instead of five; and CGLS,
  `project_joseph` and parallel `fbp_host` no longer copy NumPy inputs to the
  device twice. `tj.project` and `tj.backproject` run compiled (peak 3.1 GB,
  was 5.6, on the unbinned walnut). Alignment's joint update works through
  view batches (7.2 to 5.8 GiB at the walnut's finest level), and the shift
  search correlates 32 views at a time.

### Development

- Design rules for contributors, held by `tests/test_architecture.py`: the
  public API is recorded in `tests/guardrails/api_surface.txt` so every
  change shows in review, CLI options must map to Python keywords, and
  ratchets in `tests/guardrails/ratchets.json` (configuration fields,
  exported names, CLI flags, long files and functions, lint suppressions,
  complex functions, type errors, positional options, first-device probes,
  jaxlib and test private imports) may fall but not rise.
- Ruff rejects positional boolean parameters, private member access across
  objects, `print` in the library, shadowed builtins and commented-out code.
- The CPU tests run on four CPU devices, so CI exercises the multi-GPU split.
  `just test-cuda` runs every test with the GPU visible. The
  checkpoint-resume test runs on the CPU, whose arithmetic is reproducible.
- About 1,350 lines of unused code are deleted.
- Benchmarks: `bench/walnut.py` (starting JAX's and ASTRA's GPU runtimes
  before timing either), `bench/walnut_alignment.py`, `bench/compare_tv.py`,
  and a gVXR chip-package laminography phantom in `bench/phantoms` with an
  exact mesh voxeliser for the truth.

## 0.3.0 — 2026-10-04

This release adds matched Joseph/CGLS reconstruction, exact trilinear ray
integration, opt-in coupled alignment, and revised installation and data guides.
It remains an early research release: the frozen reconstruction comparison has
accepted pairs in 26/27 cells, and the opt-in free-voxel alignment pilot passes
5/6 cells. The stretch performance and recovery targets are not met. See
[measurement scope](docs/measurements.md) and [known limitations](docs/known-limitations.md).

- Fix NaN derivatives through Huber-TV updates in flat image regions and through
  the Huber conjugate proximal map at zero duals. Preserve the ordinary update
  values; verify regularized unrolled and implicit reconstruction derivatives
  against dense systems across parallel, anisotropic, and tilted geometry.

- Store bulk benchmark evidence as lossless gzip archives with verified original
  and archive hashes. Keep summaries readable and exclude regenerated raw logs
  from the source change list; preserve every recorded measurement and failure.

- Rewrite the README and task guides with explicit installation, physical units,
  input domains, supported scope, and failure-inclusive measurement links. Add
  a reproducible public-API reconstruction figure and image provenance catalog.
- Fix the Python example's removed projector import and use a matched public
  Joseph/CGLS workflow. Run it in CI. Extend the fresh installed-wheel check
  through reconstruction and labelled slice export using shared smoke steps.
- Share angle-sidecar parsing between TIFF ingestion and preprocessing. Reject
  empty/nonfinite vectors, non-vector NPY arrays, and malformed rows after
  numeric data begins; preserve acquisition order and leading CSV headers.
- Correct TIFF ingestion's `--det-center-u/v` conversion from detector pixels
  to physical length. Previously non-unit detector pitch produced wrong stored
  offsets. Re-ingest affected stacks from the original data and requested pixel
  offsets; existing files are not silently rewritten.
- Make developer recipes preserve the selected environment with `--no-sync`.
  Replace the blanket surface-test marker with explicit API/CLI/IO/checkpoint
  selections; keep numerical tests and aggregate coverage in the full gate. Add an optional plotting dependency group.

- Add opt-in exact line integration of the zero-extended trilinear voxel basis
  to alignment and iterative reconstruction. Carry `ray_integrator="exact"`
  through reconstruction, pose scoring, setup validation and translation seeds;
  reject resuming under a different operator. CUDA provides a matched volume
  adjoint, while alignment pose scoring retains the differentiable JAX reference.

- Add opt-in coupled volume/pose Gauss–Newton updates, with a stacked solve
  or exact elimination of the small pose blocks. Preserve the fixed-volume
  default. The frozen free-voxel pilot passes five of six cells; noisy
  anisotropic recovery still misses the rotation gate.
- Reuse compiled joint objectives across scans with matching shapes and solver
  options, while passing each scan's measurements, geometry arrays and weights
  as inputs. The six-cell comparison shows faster warm calls, with unchanged
  cold startup, process GPU memory and recovery coverage.

- Stop inflating the reused FISTA step bound by 20% after every alignment
  outer iteration. Preserve the solver's effective bound instead of shrinking
  voxel updates exponentially, including across checkpoint resume. Convert the
  carried total bound to the public solver's data bound on fallback, so Huber-TV
  curvature is included once.
- Replace alignment's dimension-only initial step estimate with the maximum
  row sum of the nonnegative normal operator. Account for physical voxel and
  detector spacing, offsets, sampling density and tilted geometry; a change of
  length units no longer changes unregularized voxel updates.

- Keep rigid pose composition, inverse-pose translation and Gauss–Newton
  normal equations at full FP32 multiplication precision on CUDA. Reduced
  matrix-multiply defaults could bias recovered small rotations.

- Add an explicit central finite-difference Gauss–Newton Jacobian for sharp
  object recovery at trilinear interpolation boundaries, with physically scaled
  translation and rotation stencils. Preserve default autodiff and exact
  forward/volume-adjoint semantics; numerical columns cost extra projections.

- Add explicit detector-frame translations to the public alignment API,
  preserving object-frame defaults and checkpoint compatibility. Carry the
  convention through pose updates, reconstruction, multiresolution stages,
  checkpoints and parameter exports. Reject incompatible mean-shift gauge
  constraints and resuming under a different frame.
- Apply existing per-view poses when reconstructing training subsets for setup
  objectives; previously those subsets discarded the pose offsets.
- Correct the direction of phase-correlation translation seeds and convert
  detector displacements into the chosen pose frame. Preserve frozen offsets
  and discard unobservable seed directions in the legacy object-frame model.
- Correct comparison-worker startup isolation: external ASTRA/TIGRE fixture
  loading no longer imports JAX/Pallas. Retain measured warm attempts for failed
  reconstruction budgets without counting them as accepted-result timings.

- Overlap bounded host row preparation and output writes with large CUDA
  Fourier reconstructions, including asynchronous transfers through pinned
  buffers and two pending GPU slabs. Cache bounded immutable small-geometry
  arrays and reduce fractional-row interpolation allocations while preserving
  arithmetic and independent per-call working buffers. Reuse device queues
  across calls to prevent growth of cached stream-specific allocations.

- Extend the exact unit detector-row CUDA adjoint specialization to cubic
  interpolation, reusing vertical weights while preserving negative lobes and
  absolute-weight error bounds. Tilted and non-unit mappings retain the general
  gather path.

- Add opt-in `CGLSConfig.gradient_damping` for squared physical voxel differences,
  with free boundaries and a matched regularized normal-residual stopping check.
  The zero default preserves unregularized execution.

- Add `joseph_pose_normal_equations` for per-view least-squares pose updates.
  CUDA accumulates the gradient and small Gauss–Newton matrices directly and
  returns residuals for line search, without storing a full projection Jacobian.

- Add explicit Keys cubic interpolation to Joseph projection, fused loss and
  CGLS, with matched CUDA volume/pose derivatives and a gather transpose that
  retains negative weights. Linear interpolation remains the default. CGLS
  accounts for those negative weights in its componentwise roundoff estimate.

- Add `project_joseph` with bounded-memory first-order CUDA volume/pose
  derivatives and `joseph_l2_value_and_grad` with fused projection/pose-gradient
  accumulation. Preserve a JAX reference, physical coordinates, batching and
  first-order forward/reverse transformations; existing alignment defaults stay
  on the trilinear ray model.
- Remove repeated dynamic integer division from the Joseph gather transpose
  while preserving its footprint and accumulation order.

- Add opt-in `fourier_reconstruct`/`FourierConfig` for uniform parallel scans, with an independently checked NumPy reference, CuPy FFTs and a CUDA Kaiser–Bessel interpolation kernel. Support detector offsets, anisotropic and cropped grids, opposing/reordered views and bounded host-output slabs; keep the default reconstruction selection unchanged.

- Fix CGLS stopping when a bright region masked meaningful updates in a weak region. Verify convergence with recomputed residuals and report componentwise FP32 stagnation separately.
- Check CGLS residuals periodically near the FP32 noise floor, preventing occasional JAX CUDA atomic-reduction noise from evading stagnation checks until the iteration limit.

- Add opt-in Joseph plane sampling to CGLS, with a deterministic CUDA gather transpose and independent physical-ray matrix checks. The default ray model is unchanged.
- Add optional isolated gVXR material fixtures with verified units and geometry, synthetic spectral attenuation and reproducible Poisson noise.
- Add optional `cgls_multires` with explicit per-level iteration budgets, physical-coordinate-preserving initialization, and a final full-data solve.
- Preserve all physical volume faces when coarsening odd or explicitly positioned grids; keep coarse detector rays uniformly spaced without duplicated edge samples. Old multiresolution alignment checkpoints are rejected because their sampling convention can differ; restart those runs. Single-resolution checkpoint compatibility is unchanged.
- Add frozen sharp-ellipsoid and noisy-ellipsoid comparison suites with independent chord-length data and retained failed quality gates.
- Add public `cgls`/`CGLSConfig` for matched-adjoint least squares with scalar damping, JAX/CUDA backends, dynamic iteration budgets, and explicit termination diagnostics.
- Accumulate the Pallas batch adjoint into one volume instead of allocating a volume per view.
- Factor Pallas interpolation and share ray traversal across kernels; preserve masked boundary loads and the forward sample recurrence.
- Build built-in laminography pose stacks with one device transfer and preserve custom parallel-geometry pose overrides.
- Add isolated, fixed-quality reconstruction comparisons against ASTRA/TIGRE, including unsuccessful budgets, cold search costs, and sampled process GPU memory.
- Reuse compiled FISTA/SPDHG loops and shared operator-norm estimation with dynamic data and geometry inputs; transfer loss histories to the host in one operation.
- Accumulate FP32 batches of adjoint rays into one volume, reducing temporary memory while preserving mixed-precision adjoint semantics.
- Cache alignment gauge projection and compile L-BFGS evaluations/updates once per optimization call.
- Fix false rejection of finite anisotropic volumes caused by float32 finiteness-average rounding.
- Remove duplicate implicit-alignment reconstruction and propagate its measurement-data gradient through the implicit adjoint.
- Remove exact detector-size tuning overrides and preserve explicitly requested JAX reconstruction backends.
- Add full public workflow benchmarks and regressions for fresh inputs, weighted solver steps, operator norms, implicit gradients, and alignment validity.
- Correct FBP attenuation scaling and replace circular FFT filtering with a zero-padded discrete ramp. Quantitative FBP output changes; regenerate reconstructions that depend on the old normalization.
- Preserve filtered detector tails through the full volume in built-in parallel FBP, removing corner bias from premature cropping. Raw data remain zero outside the measured detector; genuinely truncated acquisitions still need an appropriate correction.
- Stream direct FBP padding, filtering and backprojection in bounded view batches, avoiding full-stack FFT temporaries. Preserve partial batches, shifted detector coordinates and the explicit Pallas helper's translated poses.
- Add `fbp_host` and `FBPHostConfig` for axial-slab parallel reconstruction from NumPy/memmap inputs into optional host output storage, bounding projection and volume device storage by slab dimensions while preserving physical coordinates and partial slabs.
- Add automatic voxel-driven CUDA/Pallas parallel FBP and `FBPConfig.backprojector` (`auto`, `jax`, `pallas`); remove the superseded tiled FBP kernel.
- Fix long-ray adjoint coordinate drift and scatter FP32 gradients directly into the accumulator.
- Synchronize FBP fallback chunks before committing progress, so asynchronous out-of-memory retries preserve every view exactly once.
- Preserve useful Pallas tiles on odd detector dimensions with masked loads/stores; reject unsafe integer-slice specializations for tiny tilts and accumulated row-spacing drift.
- Fix Pallas loop unrolling and CPU interpretation of masked/repeated-index atomic updates.
- Upgrade JAX to 0.11.2 with a `<0.12` compatibility bound and use ImageIO's maintained tifffile backend.
- Add analytic accuracy, adjoint, weighted-gradient, real-CUDA, and iterative convergence regressions, plus reproducible ASTRA/TIGRE and FBP benchmarks with retained results.
- Consolidate the public CLI/API, minimal examples, and product workflow tests.
- Remove obsolete benchmark harnesses, historical/v1-parity gates, one-off runners, oversized development logs, and diagnostic artifact builders from the shipped tree; retain them in a separate development archive. Current reproducible development benchmarks live in `bench/` outside the installed package.
- Remove `tomojax.bench` and `tomojax.verify` entirely from the shipped tree.
- Laminography geometry now aligns the rotation axis with the volume z-axis, so lamino reconstructions produce z-stacks with square x–y slices; regenerate datasets/recons if you relied on the old orientation.
- Recon CLI now crops to detector FOV by default (`--roi auto`); pass `--roi off` to keep legacy behavior.
- Normalize NX volume IO: write volumes on disk in `zyx` order with `@volume_axes_order` metadata, transpose on load, and warn (silence via `TOMOJAX_AXES_SILENCE`).
- Add CLI `--volume-axes` override for `recon`/`align` and update NX data wrangler to tag volumes.
- Fix CUDA “invalid image” faults on Turing GPUs by replacing the projector’s small GEMM with element-wise transforms, ensuring SPDHG/FWD projections compile cleanly on RTX 6000/8000 while keeping mixed-precision gather heuristics.

## 0.2.0 — v2 at repo root
- Promote the v2 implementation to the primary package at `tomojax`.
- Move v2 code under `src/tomojax`, tests under `tests`, docs under `docs`.
- Add root `pyproject.toml` (src/ layout) and update `pixi.toml` tasks.
- Remove legacy root examples and experimental folders.
