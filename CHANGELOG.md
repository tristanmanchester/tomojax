# Changelog

## Unreleased

- Breaking: reconstruction options have one name each, the one
  `tj.reconstruct` already used, in Python, on the command line, as
  `--config` keys and in saved `info`. In the solver configurations
  (`FistaConfig`, `SPDHGConfig`, `CGLSConfig`, `FBPConfig`, `FBPHostConfig`,
  `FDKConfig`, `FistaCoreConfig`), `ReconstructionAlgorithmOptions` and
  `fista_multires`/`cgls_multires`: `iters` is `iterations` (and
  `tv_prox_iters`, `power_iters`, `iters_per_level` are `tv_prox_iterations`,
  `power_iterations`, `iterations_per_level`), `lambda_tv` is `tv_weight`,
  `positivity` is `nonnegative`, `filter_name` is `filter`, `L` is
  `lipschitz`, and the options' `algorithm` is `method` and `spdhg_seed` is
  `seed`; their `warm_start` is a bool. `tomojax recon` takes
  `--tv-prox-iterations` (was `--tv-prox-iters`) and `--lipschitz` (was
  `--L`). A `--config` file using a retired key (`algo`, `iters`, `lambda_tv`,
  `positivity`, `spdhg_seed`, `tv_prox_iters`, `L`) fails with its new name,
  for example "config key 'lambda_tv' ... was renamed 'tv_weight'". Saved
  reconstruction `info` and solver `info` use the new keys
  (`effective_iterations`, `lipschitz`, `tv_weight`, `nonnegative`, ...), and
  the `tomojax recon` manifest records `method` (was `algorithm`).
- Breaking: alignment settings use the same vocabulary. An `AlignConfig` field
  that sets the inner reconstruction has its `FistaConfig` or `SPDHGConfig`
  name: `recon_iters` is `iterations`, `lambda_tv` is `tv_weight`,
  `tv_prox_iters` is `tv_prox_iterations`, `recon_positivity` is
  `nonnegative`, `spdhg_seed` is `seed`, `recon_L` is `lipschitz` and
  `recon_algo` is `reconstruction`. Also `outer_iters` is `outer_iterations`,
  `gn_joint_iters` is `gn_joint_iterations`, `freeze_dofs` is `freeze` (as in
  `tj.align`), and `align_profile` (`"lightning"`/`"tortoise"`) is `quality`
  (`"fast"`/`"reference"`, as in `tj.align`). `quality_tier` and
  `fallback_policy` are gone: the first only echoed the profile, and the
  second was always reset to `"fallback"`. `ReconLayerConfig` and
  `FoldReconstructionConfig` follow (`iterations`, `tv_weight`, `lipschitz`,
  `nonnegative`, `implicit_cg_iterations`), as do `AlignResumeState`,
  `AlignMultiresResumeState` and `AlignmentCheckpointProgress` (`lipschitz`,
  `*_outer_iterations_*`). Alignment `info` uses the new keys:
  `reconstruction`, `lipschitz`, `quality`, `completed_outer_iterations`,
  `total_outer_iterations`, and per-outer `lipschitz_measured` and
  `lipschitz_next`. `tomojax align` options are the field names with hyphens:
  `--outer-iterations`, `--iterations`, `--reconstruction`, `--tv-weight`,
  `--tv-prox-iterations`, `--seed`, `--nonnegative`/`--no-nonnegative`,
  `--lipschitz`, `--pose-translation-frame` and `--early-stop-rel-impr`, and
  every option's `--config` key is its own name (`quality`, `freeze`,
  `manifest`, `dry_run`). A `--config` file using a retired key
  (`align_profile`, `outer_iters`, `recon_iters`, `recon_algo`,
  `recon_positivity`, `recon_L`, `freeze_dofs`, `translation_frame`,
  `early_stop_rel`, `save_manifest`, `print_plan_json`, and those above)
  fails with its new name.
- Breaking: alignment options take one spelling each. Removed: the
  `--align-profile` option and the `quality` spellings `lightning` and
  `tortoise`; `recon_algo` values `fista_tv`, `spdhg_tv`, `fista-tv` and
  `spdhg-tv`; `opt_method` and stage optimizer values `lbfgsb`, `l_bfgs` and
  `l_bfgs_b`; hyphenated `gauge_policy` values (`anchor-mean`,
  `prior-required`, `diagnose-only`); `pose_model="per-view"`; hyphenated or
  upper-case schedule names; and the gauge-fix spellings `off`, `false`,
  `disabled` and `disable`.
- Breaking: alignment checkpoints written before this version (schema 2 or
  earlier) do not resume; resuming one fails with "schema version 2 predates
  this version of TomoJAX". Restart the alignment.
- Fixed: `tj.reconstruct(scan, "spdhg")` ignored `nonnegative` and always
  clipped the volume at zero. It now honours it and, like `fista`, does not
  clip unless asked.
- Fixed: `tj.align` in `cor`, `cor-then-pose` or `full` mode on a scan that
  already carries poses ran and found the wrong centre; those modes now
  refuse posed and segmented scans (align them with `mode="pose"`).
- Fixed: `Scan.poses` is in the detector frame for every scan, as documented;
  files saved with object-frame poses reported those as they were, and a
  segmented scan saved its poses as detector-frame whatever their frame.
- Fixed: the binning suggestion takes a segmented scan's least magnified
  segment (it took the first) and, for laminography, the smallest voxel side;
  it no longer warns before `tj.align` rejects a bad option.
- Fixed: an alignment ended early by a level that does not fit in device
  memory is now complete, so resuming it does not retry that level;
  `info["factors"]` lists the levels that ran, and the notice is a Python
  warning. The memory check counts the cached pose columns and per-pixel
  weights with a margin of two, asks the device holding the data, and the
  update it compiles is the one that runs.
- Breaking: configuration classes (`FistaConfig`, `CGLSConfig`, `FBPConfig`,
  `AlignConfig`, ...) take keywords only, as the design rules ask of options;
  so does `ConeGeometry.poses(thetas_deg=...)`. `Reconstruction.grid` is now
  the scan's grid, a property rather than a field, and `tj.load_reconstruction`
  returns the `info` saved with it. `least_motion_estimate` takes
  `cone_beam=` (was `beam=`).
- `ConeSegments` refuses detectors of different pixel pitch when made, not
  when first projected. `tomojax align` warns when its input carries pose
  corrections, which it replaces (`tomojax.align` corrects on top of them).
- New ratchets: the length of every file over 800 lines and function over 100
  (which may only shrink), options passed positionally to public methods and
  configurations, `jax.devices()[0]` probes, jaxlib private imports and
  private imports in tests.
- `tj.project`, `tj.backproject` and `tj.reconstruct` with `cgls` or `fista`
  take `devices=` (one device or several; `jax.devices()` for every GPU): each
  device projects its share of the views and holds the whole volume, and their
  backprojections are summed, so the transpose stays exact and the result, on
  the first device, is the one-device result up to that sum's order. Each
  device reads only its own views of the projections. `FistaConfig` and
  `CGLSConfig` take `devices` too. The CUDA kernels now launch on the GPU
  holding their buffers, not the current one. The CPU tests run on four CPU
  devices, so CI exercises the split.
  Twenty FISTA iterations on the binned walnut take 24.8 s on one H100, 13.6 s
  on two and 7.4 s on four (docs/performance.md).
- Fixed: aligning a scan loaded from a file returned `result.scan` without
  the corrections (`result.poses` had them). A loaded dataset rebuilt its
  geometry from the metadata it was read with, so later changes to its poses,
  angle offsets or geometry were ignored; it now uses its current fields.
- Iterative `tj.reconstruct` and `tj.align` warn when the detector samples the
  rotation axis at least twice as finely as the grid's voxels (a cone beam's
  pixel pitch divided by its magnification), naming the `scan.binned(n)`
  that makes them up to n² times cheaper, as on the FIPS walnut.
- `examples/align_walnut_orbits.py` brings the FIPS walnut's three orbits into
  register from the scanner's uncorrected geometry with the public API (it
  needs the data download and a GPU; see examples/README.md).
- Large scans need less device memory. `tj.project` and `tj.backproject` run
  compiled, so their view loop fills one projection stack in place (peak
  3.1 GB, was 5.6, on the unbinned walnut). The cone kernels read and write
  JAX's `(view, row, column)` layout, so no transposed copy of the
  projections is made (and the forward is a little faster). Alignment's joint
  pose and volume update works through view batches and stores at most one
  projection-sized array (7.2 to 5.8 GiB at the walnut's finest level), with
  a scalar weight for plain least squares. A level whose update cannot fit
  in device memory now ends alignment at the level before, with a warning and
  `info["factors_skipped"]`, instead of failing after the coarser levels; a
  first level that cannot fit raises `AlignmentMemoryError` before any work.
  The translation pre-search streams segmented scans' CGLS from the host.
- `just test-cuda` runs every test with the GPU visible, not only the
  `gpu`-marked ones: unmarked tests take CUDA paths there that CI's CPU runner
  never does. The checkpoint-resume test runs on the CPU device, whose
  arithmetic is reproducible (GPU atomics are not).
- `tj.align` takes scans that already carry poses (ASTRA imports, earlier
  alignments) and corrects them on top, and aligns multi-orbit `ConeSegments`
  scans as one, bringing their orbits into register. From the FIPS walnut's
  uncorrected record it recovers the authors' orbit heights (orbit 2 -0.381 mm
  against -0.397, orbit 3 -0.755 against -0.794), and the reconstruction then
  matches the corrected one (error 0.155 against 0.154; 0.280 uncorrected);
  see `bench/walnut_alignment.py` and docs/lab-ct.md. Underneath, cone kernels
  take per-view lab frames (`tomojax.core.cone.ConeModel`, built for the
  solver's binned detector), so every solver and alignment level projects each
  segment in its own arrangement; the least-motion gauge ignores the faint
  background a reconstruction leaves at the grid edge; FDK lets views
  repeating an angle share it; and the translation pre-search reconstructs
  segmented scans with CGLS.
- The CUDA cone-beam forward projector is 1.3 to 3.7 times faster, and one
  kernel now serves every view: a warp's rays step through the planes
  together (rays entering through the volume's top or bottom had left lanes
  on different planes), blocks run views fastest and detector rows slowest
  (so concurrent blocks share a slab of the volume in L2), and the inner loop
  is half the instructions. The separable forward kernel was slower and is
  gone. The non-separable transpose loads each run of pixels at once and
  holds to 40 registers for full occupancy (20% faster). FISTA estimates its step from three power iterations started from
  the backprojected data (`power_iters` now defaults to 3; five from a
  constant volume were less accurate on the FIPS walnut) and reuses that
  backprojection as its first gradient, saving two and a half projections.
  On the walnut's three orbits (every 4th view, 20 iterations, binned 2 x 2)
  non-negative least squares now takes 86 s, ASTRA 91 s (was 162 s);
  unbinned 304 s, ASTRA 217 s (was 593 s).
- FDK on CUDA is about twice as fast: rows up to 2048 pixels are ramp-filtered
  by one matrix product (cuBLAS, three times faster than the FFTs at 768
  pixels), and the backprojection samples the filtered images through the
  texture unit from half floats, each batch scaled to its peak. Its
  interpolation weights are rounded to 1/256, as in ASTRA's FDK. On the FIPS
  walnut (1200 views, 501³) FDK takes 1.7–1.9 s against ASTRA's 2.13 s and
  agrees with ASTRA's volume to 0.05%; the synthetic 256³ case takes 0.048 s
  (ASTRA 0.29 s), and a 1024³ host FDK 8.0 s (was 12.3 s). FDK also compiles
  a third as many programs on its first call. `bench/walnut.py` starts JAX's
  and ASTRA's GPU runtimes before timing either.
- FISTA runs one projection fewer per iteration, a quarter faster on every
  geometry: it tracks the objective at the extrapolated point, where the
  gradient's residual already gives it, instead of projecting the new iterate
  again; its loss history and early stopping follow that point. With no TV
  weight it skips the TV step. `Scan.binned(n)` averages n x n detector pixels,
  for detectors that sample finer than the grid: on the FIPS walnut scan
  (pixels half a voxel at the axis) it makes iterative reconstruction 3.7 times
  faster with no loss of accuracy. The cone-beam CUDA kernels are in
  `tomojax/core/cone_kernels.cu`.
- Scans whose source and detector arrangement changes partway (multi-orbit
  scans, stacked sections of a tall sample) are `tomojax.geometry.ConeSegments`:
  one arrangement per run of views, reconstructed together by CGLS, FISTA and
  SPDHG, saved and loaded like any scan. `Scan.from_astra` splits vectors into
  segments where the arrangement changes, and `tj.Scan.combine` joins scans.
  `tomojax.geometry.ScanGeometry` names what every scan geometry provides (grid,
  detector, angles). FDK streams the next batch of views while the current one
  is filtered and backprojected.
- Move ASTRA Toolbox scans to TomoJAX and back: `tj.Scan.from_astra(data,
  proj_geom, vol_geom)` reads `cone` and `cone_vec` geometries (per-view source,
  detector and pixel vectors) as a fitted circular orbit plus per-view pose
  corrections, and `scan.to_astra()` returns ASTRA data and geometries.
  Converted scans project like ASTRA to 0.03% on tilted, offset and jittered
  vector geometries.
- **Breaking:** a workflow API at the package root. `tomojax.Scan` holds
  projections and the geometry that produced them; `tomojax.load` reads
  TomoJAX datasets and Nikon `.xtekct` scans (applying a saved alignment),
  `tomojax.reconstruct(scan, method)` runs `fbp` (FDK for cone beams, host
  slabs when large), `cgls`, `fista` or `spdhg` and rejects options the method
  does not take, `tomojax.align(scan, mode=...)` returns the scan with its
  corrections applied, and `tomojax.save` writes scans, reconstructions and
  alignments. `tomojax.project` and `tomojax.backproject` project any geometry
  (cone beams included, which had no public projector). `import tomojax` still
  does not import JAX.
- **Breaking:** `tomojax.align` (the subpackage) is now `tomojax.alignment`, so
  the name `tomojax.align` is the function. Alignment modes are `pose`, `cor`,
  `cor-then-pose` and `full` (formerly `auto`; `max` is `full` at
  `quality="reference"`), planned by `tomojax.alignment.alignment_plan` for
  Python and the CLI alike; the cone-beam axis calibration moved from the CLI
  into the library.
- **Breaking:** package roots export what users call; implementation helpers
  moved to each package's `.api` module (`tomojax.recon` 28 names to 22,
  `tomojax.geometry` 38 to 16, `tomojax.io` 25 to 12). The geometry-level
  single-resolution `align` and `coupled_pose_config` are in
  `tomojax.alignment.api`.
- **Breaking:** one command-line shape, `tomojax <command> INPUT -o OUTPUT`,
  with an existing output refused unless `--force`, exit status 0 for success,
  1 for failure and 2 for a usage error, and `tomojax --version`. Seven
  commands: `inspect` (now also validating, exiting 1 for an invalid dataset,
  with `--json` to stdout and `--preview DIR` PNGs of the central projection
  and volume slices; replaces `validate` and `slices`), `import` (formerly
  `ingest`; also converts `.npz` and `.nxs`, replacing `convert`),
  `preprocess`, `recon`, `align`, `export` and `simulate`. Options share the
  Python names: `recon --method --iterations --tv-weight --nonnegative`
  (formerly `--algo --iters --lambda-tv --positivity`), `--preview`,
  `--manifest`, and `align --freeze` and `--dry-run`. `recon` applies a saved
  alignment unless `--ignore-alignment`. `--help` lists the options most runs
  need; expert settings are keys of the TOML file given with `--config`,
  listed with their defaults by `--config-keys` (they still parse as flags).
  `preprocess` and `export` infer their formats from the paths; `import`
  takes `--pixel-size`; `simulate` needs only `-o` (`--size`, `--views`, and
  a cone detector sized to see the whole volume). Alignment quality is `fast`
  or `reference`; the aliases `normal` and `full` are gone.
- **Breaking:** per-view pose tables are `pose_params` everywhere (formerly
  `params5`, though they have six columns): `align(init_pose_params=...)`,
  `AlignResumeState.pose_params`, `se3_from_pose_params`. Alignment checkpoints
  store them under that name, as schema version 2; older checkpoints are
  refused.
- **Breaking:** the expert `.api` modules drop 67 names nothing used
  (`tomojax.geometry.api` 77 to 43, `tomojax.alignment.api` 93 to 79,
  `tomojax.io.api` 43 to 34, `tomojax.recon.api` 26 to 22,
  `tomojax.datasets.api` 21 to 15), and 1,350 lines of code that nothing
  called are deleted, among them the geometry CSV and JSON writers
  (`write_pose_params_csv`, `write_geometry_json`), `build_calibration_manifest`,
  `canonicalize_geometry_gauges`, `spatial_bin`, `pad_to_multiples` and
  `run_active_lbfgs`. Option names are normalised in one place
  (`tomojax.core.validation.option_name`).
- Alignment reports where the object is unambiguously. Rotating or shifting
  the object, and every pose the opposite way, predicts the same data, and the
  solver could end anywhere along that motion: a 64-cubed cone scan came back
  shifted a voxel along its axis (volume error 0.26 against the truth) and a
  half-turn parallel scan with a detector offset three voxels sideways (0.55).
  `tj.align`, `align_multires` and `tomojax align` now return the estimate
  with the least per-view motion, moving the volume to match (errors 0.0037
  and 0.019), and record the motion removed in `info["gauge"]`. The detector
  centre's share of a constant u shift is found in the same fit, replacing
  `fold_detector_offset`; `tomojax.alignment.api.least_motion_estimate` moves
  any volume and pose table, for example a truth, to the same estimate.
  **Breaking:** the `gauge_fix` setting is gone; setup stages still anchor
  object-frame translations internally.
- **Breaking:** options are keyword-only throughout the public API:
  `tomojax.datasets.sphere`, `cube` and `blobs` (`size`, `value`, `seed`,
  `n_blobs`) and `tomojax.io.preprocess_nxtomo(config=...)`. `Scan.source`
  (formerly private) is the dataset record a scan was loaded from, and
  `LaminographyGeometry.axis_unit_lab` gives its rotation axis like
  `RotationAxisGeometry`'s.
- Design rules for contributors, and `tests/test_architecture.py` to hold the
  code to them: the public API is recorded so every change shows in review,
  CLI options must map to Python keywords, and measures of debt
  (configuration fields, exported names, CLI flags, long files, lint
  suppressions, complex functions, type errors outside the type-checked
  modules) may fall but not rise. Ruff now rejects
  positional boolean parameters, private member access across objects,
  `print` in the library, shadowed builtins and commented-out code.
- Fix the CUDA cone-beam transpose for volumes smaller than its 32- or
  64-voxel tiles: a tile's far edge, close to the source, projected through
  infinity and dropped detector columns (errors up to 30% at 8-cubed).

- `tomojax preprocess` corrects two lab-CT artefacts in absorption data:
  `--beam-hardening C1,C2,...` maps each value p to `C1 p + C2 p^2 + ...`, and
  `--remove-stripes WIDTH` removes rings by subtracting each detector pixel's
  constant offset, judged from its values sorted over views against those of
  its neighbouring columns. Data without such offsets pass unchanged.
- `tomojax export` writes a reconstruction as 32-bit or scaled 16-bit TIFF
  z-slices, or one raw file, with a JSON sidecar of shape, voxel size and
  scaling, reading one slice at a time. `tomojax recon --algo fbp` on cone
  data now reconstructs volumes too large for the device in z-slabs on the
  host (`fdk_host`) instead of failing.
- Cone-beam pose alignment runs about four times faster: the reconstruction
  step stacked the pose-adjusted views one at a time, which dominated each
  outer iteration. A 96-cubed-phantom, 240-view `tomojax align --mode
  cor_then_pose` now takes 44 s instead of 163 s. Geometries can supply a
  vectorised `stack_poses` for `stack_view_poses`.
- Import Nikon (X-Tek) lab CT scans: `tomojax ingest scan.xtekct --out
  scan.nxs` (and `tomojax.io.load_nikon_xtekct`) reads the source and
  detector distances, detector pixels and offsets, reconstruction volume,
  white level and angles (from `_ctdata.txt` when present) and converts the
  projection TIFFs to absorption. The axis offset and detector roll are left
  to `tomojax align --mode cor`. A [lab cone-beam CT guide](docs/lab-ct.md)
  covers import, calibration, FDK, iterative reconstruction and motion
  correction.
- Cone datasets without a grid now reconstruct one voxel per detector pixel
  at the rotation axis (the pixel size divided by the magnification) instead
  of one per pixel at the detector.
- FDK reconstructs full turns on an offset detector (the rotation axis
  projecting off the detector centre, as lab scanners use to widen the field
  of view): Wang's weights blend each ray's two measurements and the filtered
  rows keep their tail past the detector's short side, so a detector covering
  5.5 columns on one side of the axis reconstructs a 32-cubed phantom as well
  as a centred detector twice as wide (error 0.071 against 0.075). Angular
  weights now apply before the ramp filter, as FDK requires; this lowers
  Parker-weighted short-scan errors (0.114 against 0.075 for a full turn).
- Calibrate a cone-beam scan's rotation axis: `ConeBeam.axis_offset` places
  the axis laterally (the lab-CT centre of rotation; `tomojax ingest
  --axis-offset`), and `tomojax.recon.calibrate_cone_axis` estimates it with
  the detector roll from the sharpness of thin FDK slabs at three heights,
  coarse to fine on binned data. On cone data `tomojax align --mode cor`
  calibrates both, and `cor_then_pose`, `auto` and `max` calibrate them before
  their pose stages, saving the calibrated beam. On 128- and 256-cubed blob
  scans it recovers offsets of up to 11 voxels to 0.08 voxels and rolls of up
  to 1.2 degrees to 0.04 degrees, in about 4 s at 256-cubed. `fdk_host` now cuts
  rolled detectors to the rows each slab needs, and the FDK backprojector no
  longer wastes threads on thin slabs.
- Add cone-beam (lab CT) geometry: `tomojax.geometry.ConeGeometry` with a
  `ConeBeam` source and flat detector (detector offsets, roll, pitch and yaw;
  turntable, tilted or arbitrary rotation axis; any per-view poses). Rays from
  the source are sampled on voxel planes (Joseph) with a matched transpose, in
  JAX (differentiable in the volume and the poses) and as CUDA kernels; views
  of an unperturbed turntable use two-pass separable kernels. CGLS, FISTA-TV
  and SPDHG-TV reconstruct cone scans, and `fdk` (also `fbp` and `tomojax
  recon --algo fbp` for cone data) adds Feldkamp reconstruction for full turns
  and Parker-weighted short scans. `tomojax simulate --geometry cone` and
  `tomojax ingest --geometry cone --source-to-axis ... --source-to-detector ...`
  create cone datasets, which save and load with their beam. On a 256-cubed,
  360-view scan with a 384-squared detector, forward projection takes 0.12 s
  (ASTRA 0.17 s, TIGRE 0.48 s), its exact transpose 0.18 s (ASTRA's
  approximate one 0.075 s) and FDK 0.10 s (ASTRA 0.29 s, TIGRE 0.52 s) at the
  same accuracy. `fdk_host` reconstructs scans larger than device memory in
  z-slabs from and into memmaps, filtering only the rows each slab needs: a
  1024-cubed scan takes 11.7 s in RAM and 14.3 s memmap to memmap (ASTRA
  20.3 s, TIGRE 30.7 s).
- Align cone-beam scans in six degrees of freedom: pose tables gain a sixth
  column, `dy` along the beam, which changes cone-beam magnification and
  stays zero for parallel beams. `tomojax align` on cone data estimates it
  with the other five parameters, anchoring its mean (a common `dy` is the
  volume's scale), and `tomojax recon --apply-saved-alignment` replays all six.
  Five-column tables, sidecars and checkpoints from earlier versions load with
  `dy = 0`. Cone rays are sampled along their own dominant axis, so
  projections change continuously as a pose turns a ray through 45 degrees.
- Estimate a detector-centre offset together with per-view motion:
  `tomojax align --mode cor_then_pose` now runs the pose solver and saves the
  constant part of the recovered detector-u shifts as the detector centre,
  leaving the per-view motion in the pose table (`auto` and `max` add it to
  their setup estimate). The fold runs in `align_multires`, so the Python
  API's `cor_then_pose` schedule, which requires detector-frame translations,
  behaves the same; the API also adds `fold_detector_offset`. With a +3.7 px offset and ±0.5°/±8 px motion on
  analytic 128³ scans, rotation errors fall from 0.23–0.26° to 0.005–0.006°,
  and the offset matches its identifiable value to 0.004 px. On the gVXR chip
  phantom it recovers a 3.2 px offset to 3.18 px. The previous mode searched
  for the offset before correcting any motion, which biased it.
- `tomojax align` now uses detector-frame translations by default
  (`--translation-frame detector`, with `--gauge-fix none`). Object-frame
  translations move the sample along its own x and z axes, so at views where
  its x axis lies along the beam they cannot shift its image horizontally; a
  constant shift needed 70 px translations near 90° and 270° on the chip
  phantom, with 0.37 px errors.
  Detector-frame recovery is as accurate or better on every benchmark tried.
  `--translation-frame object` restores the previous pose tables.
- `tomojax recon --apply-saved-alignment` applies saved poses in the
  translation frame they were estimated in, which `tomojax align` now records
  with the gauge metadata; files without it are read as object-frame poses.
- A saved detector roll of zero no longer gives `tomojax recon` an explicit
  detector grid, which forced FISTA-TV onto the ray-model reference path one
  view at a time. On the 720-view chip phantom, one iteration took more than
  400 s; 100 iterations now take 29 s.
- `tomojax recon` streams host projections for CGLS and SPDHG-TV as well as FBP
  and FISTA-TV.
- Stream projections from host memory in CGLS and SPDHG-TV, as FISTA-TV
  already did. Streamed CGLS solves the equivalent normal equations: only
  volume-sized arrays stay on the device, and each residual recomputation
  reads the views once. Streamed SPDHG-TV keeps the data, any weights and its
  sinogram-sized dual variable in host memory and moves one block per
  iteration (bitwise identical to device-resident data). On a 512-cubed,
  3072-view laminography scan (3.2 GB) on an 8 GB GPU, CGLS peaks at 3.2 GB
  and SPDHG-TV at 4.3 GB. `CGLSConfig` and `SPDHGConfig` gain
  `stream_projections` (``None`` streams stacks above 40% of free device memory).
- Report the detector-u (centre-of-rotation) offset implied by pose
  alignment: `tomojax align` logs it and writes `implied_detector_u_px` to the
  manifest, and `tomojax.alignment.api.implied_detector_offset` computes it. Pose
  mode absorbs such an offset exactly into the per-view translations; the
  report separates the constant part from the view-dependent shift of a rigid
  object translation.
- Add a gVXR chip-package phantom for laminography (`bench/phantoms`): a
  Blender script models eight closed, disjoint material meshes (glass-epoxy
  substrate, copper ground plane, vias and traces, silicon die, gold bond
  wires, silica-filled mold, voided SAC solder balls), and a simulator turns
  gVXR path lengths into 25 keV measurements with xraylib attenuation and
  refraction, Fresnel phase contrast, detector blur, pixel integration,
  Poisson noise, pixel gain and noisy flats, with per-view motion and a
  detector offset. An exact mesh voxeliser provides the truth.
- Seed `tomojax align --mode cor` with a search for the detector-u offset
  whose FBP reprojects most consistently (a coarse scan over a quarter of the
  detector, then golden section), replacing the opposite-view pairing that
  needs a parallel half or full turn. All CLI alignment modes now default to
  Joseph integration. With a +3.7 px offset on analytic 128-cubed scans, COR
  mode recovers 3.693 and 3.677 px (parallel, laminography) in 41 and 84 s,
  where it previously reached 3.50 and 3.59 px in about 380 s.
- Cache the pose Jacobian columns of the coupled solver whenever five
  sinograms fit in a quarter of free device memory (previously a fixed 64 MB)
  and apply cached columns to all views in single batched contractions. On a
  256-cubed, 361-view laminography scan the columns were recomputed in every
  conjugate-gradient iteration, 60% of GPU time; full-resolution
  `tomojax align --mode pose` now takes 188 s instead of 13.6 minutes
  (22 minutes this morning) with rotations to 0.0030 deg.
- Solve each view's 5-by-5 pose block in symmetrically scaled variables and,
  only where its FP32 Cholesky factor still fails, with Marquardt damping of
  1e-5 relative to its diagonal. Rotation and translation columns differ in
  size by orders of magnitude, and unscaled factorisations of weakly
  determined blocks returned NaN, which rejected whole Gauss-Newton steps.
- Seed translations at the first coarse-to-fine level, where the shift search
  previously never ran because that level starts from explicit zero poses.
- Fix the reconstruction step inside alignment with Joseph integration. Its
  explicit gradient backprojected with the ray model's transpose (2% from the
  Joseph adjoint), and traced poses sent it to the JAX reference operators
  instead of the CUDA kernels, which accept dynamic poses. It now uses the
  matched plane transpose and the CUDA kernels, with 64 views per batch.
  `tomojax align --mode pose` on analytic 128-cubed scans takes 45 and 41 s
  instead of 86 and 72 s (laminography, parallel) at unchanged accuracy, and
  the README example 14 instead of 35 s.
- Add a CUDA C gather transpose for linear Joseph plane sampling, compiled at
  run time with CuPy (now part of the `cuda12` extra) and launched on XLA's
  stream inside compiled solvers. Each thread loops once over the joint
  footprint of four consecutive z voxels, and weights round exactly like the
  forward projection. Laminography backprojection is 1.6x and parallel 1.3x
  faster at 256-cubed scale; structured 256-cubed laminography CGLS runs in
  1023 instead of 1372 ms warm and a 512-cubed FISTA-TV solve in 48.7 instead
  of 63.1 s, and a full-resolution 256-cubed laminography alignment 13.6 instead
  of 22 minutes (rotations to 0.0029 deg). It is used for volumes of 2^24 voxels and more, since its start-up
  costs about 0.3 s per process; `TOMOJAX_CUDA_KERNELS` forces it on (1) or
  off (0).
- The alignment shift search reprojects, shifts and correlates 32 views at a
  time, so its padded correlation spectra stay bounded for long scans, and
  host-streamed FBP reads the next view batch while the current one runs.
- Stream projections from host memory in FISTA-TV. NumPy or memmap stacks
  larger than 40% of free device memory are read one view batch at a time
  inside the compiled solve (`FistaConfig(stream_projections=...)` forces
  either way), with identical results. A 512-cubed laminography scan with 3072
  views (3.2 GB) runs on an 8 GB GPU at a 4.3 GB peak; where both fit,
  streaming costs 4%. `tomojax recon --algo fista` passes host projections.
- Add `tomojax recon --algo cgls`, optionally FBP-initialised with
  `--warm-start fbp`, so the command line has the fastest-converging
  unregularised solver the Python API already offered. Document aligning a
  large scan at reduced resolution and reconstructing the full data with the
  recovered poses: a 256-cubed laminography alignment stopped at half
  resolution takes 185 s instead of 22 minutes (rotations to 0.0051 instead
  of 0.0030 deg).
- Seed pose alignment with a global per-view shift search. Each pass
  reconstructs with the current shifts removed (FBP), reprojects, and moves
  every view to its cross-correlation peak, searching up to a quarter of the
  detector and discarding the shift pattern of a rigid object translation.
  `--seed-translations` is now on by default for `tomojax align --mode pose`
  (`--no-seed-translations` disables it), and `coupled_pose_config` enables
  it; it now also runs in single-resolution alignment, and replaces the
  previous single phase correlation against a ray-model FISTA reconstruction.
  On 64-cubed scans with +/-0.5 deg tilts and +/-15 px shifts (23% of the
  detector), rotation errors fall from 7.7 and 12.6 deg to 0.034 and 0.012 deg
  in parallel and laminography; 128-cubed results are unchanged.
- Reconstruct laminography and posed scans larger than device memory with
  FBP. `fbp` given a NumPy array or memmap streams view batches from host
  memory (bitwise-identical to device input), and `fbp_host` now accepts every
  geometry `fbp` does, writing x-slabs sized to the free device memory. A
  1024-cubed, 1024-view laminography FBP from and to memmaps (4.3 GB each) runs
  in 37 s on an 8 GB GPU, where 768 cubed previously ran out of memory.
  `tomojax recon --algo fbp` keeps projections on the host. The FBP kernel
  accumulates view batches in place. `cgls`, `project_joseph` and parallel
  `fbp_host` no longer make a transient second device copy of NumPy inputs.
- Cut iterative-solver GPU memory. On a 512-cubed, 768-view laminography scan
  on an 8 GB GPU, CGLS (previously out of memory) peaks at 5.6 GB, FISTA-TV
  falls from 7.2 to 4.6 GB and SPDHG-TV (previously out of memory) peaks at
  4.9 GB. The Joseph CUDA kernels keep sinograms in their (view, v, u) layout,
  so XLA no longer stores transposed copies of every sinogram around solver
  loops, and the gather transpose accumulates view batches in place.
  FISTA-TV computes its data gradient batch by batch without a sinogram-sized
  temporary, its TV prox runs projected gradient on the dual (Chambolle) with
  three persistent volumes instead of five, and zero starting states are
  created inside the compiled solves. CGLS always adopts the recomputed
  residual and no longer retains a second state for breakdown. SPDHG-TV
  recomputes the TV divergence instead of differencing dual fields and no
  longer allocates a sinogram of unit weights. The forward kernel's 8-by-16
  ray tiles also make laminography projection 18% faster. The CLI sets
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` unless already set.
  FISTA-TV on the 128-cubed TV comparison now reaches 0.076 in 1.13 s.
- Add `bench/compare_tv.py`. On a 128-cubed structured parallel scan with 3%
  noise and 50 iterations, TomoJAX FISTA-TV reaches 0.077 relative error in
  1.28 s; TIGRE's FISTA reaches 0.101 in 9.7 s and ASD-POCS 0.146 in 5.7 s.
- Align coarse to fine by default in `tomojax align --mode pose` when the
  reconstruction grid's shortest axis has at least 64 voxels (factors 4, 2, 1
  from 128). A 256-cubed, 361-view laminography alignment takes 22 instead of
  44 minutes with the same 0.003 deg rotation accuracy and a better volume, and
  +/-3 deg motion is recovered in 64-cubed laminography where a single level
  stopped at 0.037 deg.
- Add `tomojax.alignment.coupled_pose_config(**overrides)`, the configuration
  `tomojax align --mode pose` runs, for Python callers; `AlignConfig()` keeps
  its older alternating defaults. Add an alignment example and README figure:
  a 96-cubed laminography scan with +/-1 deg and +/-2 px per-view motion goes
  from 0.57 to 0.087 relative error in 35 s, with rotations recovered to
  0.0026 deg, on analytic data from continuous objects.
- Add Joseph plane sampling to alignment (`ray_integrator="joseph"` or
  `"joseph_cubic"`, CLI `--ray-integrator`) and make it the coupled pose
  solver's default. Its matched gather transpose replaces the exact
  integrator's atomic scatter: on analytic 128-cubed scans of continuous
  objects (181 views), `tomojax align --mode pose` recovers parallel and
  laminography poses to 0.0061 and 0.0024 deg in 46 and 93 s, where exact
  integration took 588 and 1044 s for 0.0059 and 0.0016 deg. Joseph plane
  coefficients now accept affine detector grids (calibrated offsets and roll),
  so the model also serves calibrated detectors.
- Fix pose alignment through `tomojax align` and `align_multires`. Pose
  stages skipped reconstruction, so pose-only schedules (the default `pose`
  mode) optimised against an all-zero volume and returned the nominal poses.
  Pose stages now alternate with reconstruction again; a supplied `recon_L` is
  honoured at full resolution.
- Make the coupled volume-and-pose solver the default for `tomojax align
  --mode pose` (`--pose-solver coupled`; `alternating` keeps the previous scheme,
  and remains the default for modes with setup stages). It
  implies exact ray integration, least squares without TV, fp32 gathers and up
  to 30 early-stopped outer iterations, and rejects explicitly conflicting
  options. On the six free-voxel pilot cells the default CLI now recovers five
  (rotation 0.0006-0.008 deg), where the alternating scheme left 0.1-4 deg.
  `gn_coupling="joint"` now applies to every pose-only stage of a named
  schedule, and the coupled solver no longer shifts poses alone to fix the
  translation gauge, which had capped its accuracy at about 0.08 deg.
  The pilot's data share the exact integrator's voxel-basis model; on analytic
  data from continuous objects all solvers settle at a 0.09-0.5 deg
  discretisation floor at 32 cubed, with reconstructions as good as with the
  true poses.
- Run FISTA-TV and SPDHG-TV on the same batched operators as CGLS, and make
  Joseph plane sampling with Pallas kernels on CUDA the default projector for
  all three (`projector_model="auto"`). FISTA-TV previously used the JAX ray
  model one view at a time: 50 iterations on a 64-cubed scan fall from 20 s to
  0.15-0.19 s, and 40 SPDHG-TV iterations from 1.3 s to 0.05-0.07 s, with equal
  or lower reconstruction error. Joseph matches analytic line integrals as well
  as the ray model. Explicit detector grids and the exact ray integrator keep
  the ray-model path; `projector_model="ray"` restores the previous operator.
  FISTA's `views_per_batch` now defaults to 64 batched views (CLI default too).
  Alignment's internal reconstructions keep the ray model of its pose objective.
- Size alignment's reconstruction batches from free GPU memory by default
  (`views_per_batch=0`, also the CLI default). Both alignment profiles used one
  view per batch, launching a projector call for every view in every
  reconstruction pass. On the six free-voxel alignment cells, warm alignment is
  1.9-2.3x faster and cold 1.4-2x, with the same accepted cells and unchanged
  peak GPU memory. An explicit positive value is still honoured.
- Clip each Joseph forward ray to the planes it crosses. Skipped planes
  contributed exact zeros, so projections are bitwise unchanged; forward
  projection is 8-13% faster at 256 cubed in parallel, anisotropic and tilted
  scans. Trace each Joseph kernel once per configuration instead of at every
  call site, cutting a CGLS solve's tracing time by about a third.
- Weight filtered backprojection exactly for every circular parallel-beam
  scan. `fbp` now fits the rotation axis, arc and angular spacing from the view
  poses and applies the matching per-view filter: a ramp along u scaled by the
  sine of the ray-axis angle, divided by how many acquired views measure each
  frequency. Tilted (laminography) axes, partial or full turns and irregular
  angles were previously weighted as a uniform untilted half turn. In tests,
  tilted reconstructions now match the measured-frequency truth to 5% (14% and
  42% before for full and half turns); uniform untilted half turns are unchanged.
  An explicit `FBPConfig.scale` keeps the previous uniform weighting.
- Backproject FBP voxel by voxel with bilinear detector interpolation in every
  geometry, using the Pallas kernel on CUDA. Laminography and anisotropic FBP
  previously used the ray-model adjoint, which blurred the result (5% error on
  a smooth parallel phantom, against 0.1% now). Explicit `det_grid` inputs keep
  the ray-model path.
- Start faster. Importing `tomojax.geometry` and `tomojax.recon` no longer loads
  JAX or SciPy until a JAX-based function is used, so a Fourier reconstruction
  imports in about 50 ms instead of 450 ms. Compiled JAX programs are cached on
  disk by default (`TOMOJAX_JAX_CACHE=off` disables, `TOMOJAX_JAX_CACHE_DIR`
  relocates; an existing JAX cache setting wins). CGLS checks its inputs inside
  the solve instead of compiling separate programs, cutting a cold 64-cubed
  laminography call from about 930 ms to 675 ms, or 385 ms with a warm cache.

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
