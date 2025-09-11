% TomoJAX v2 Implementation Checklist (next/)

This checklist tracks the cohesive rewrite while keeping the current repo intact. Each item includes acceptance criteria and quick validation steps. Default on‑disk format: HDF5 with NeXus (NXtomo) + TomoJAX extras.

## Phase 0 — Foundation
- [x] Create `next/` workspace (no changes to existing files)
  - DoD: `next/README.md` present; this checklist committed
  - Verify: repo still runs existing examples
- [x] Decide package/name: `tomojax_next` and CLI namespace
  - DoD: Recorded in `README_v2.md`

## Phase 1 — HDF5/NXtomo IO
- [x] Define schema (datasets, groups, attrs, units)
  - Datasets: `/entry/data/projections (n_views,nv,nu,float32)`, chunks `(1,256,256)`; `/entry/processing/tomojax_next/volume (nz,ny,nx)` optional; `/entry/processing/tomojax_next/align/thetas (n_views,5)`; angles (deg) under NXtransformations
  - NX attrs: `NX_class`, `units`, `depends_on`; TomoJAX JSON blobs for `grid`, `det`
  - DoD: Markdown schema doc in `next/docs/schema_nxtomo.md`
- [x] Implement reader/writer `data/io_hdf5.py`
  - Reader: build `Geometry` + `Grid` + `Detector` from NXtransformations and attrs
  - Writer: compression `lzf` default; optional `gzip` level flag
  - DoD: Load → Save → Load round‑trip equals within tolerance
- [x] CLI convert
  - `tomojax-next convert --in sim.npz --out sim.nxs` and reverse
  - DoD: Ends with valid NXtomo per validator
- [x] Validation tools
  - `datasets.validate_nxtomo(path)` checks groups/attrs/units
  - DoD: Test fails on deliberately broken files

## Phase 2 — Geometry Abstraction
- [x] Base interfaces `core/geometry/base.py`
  - `Geometry.rays_for_view(i) -> (origin_fn(u,v), dir_fn(u,v))`
  - `Geometry.pose_for_view(i) -> SE3` (homogeneous 4×4)
  - DoD: Type‑checked stubs + docs
- [x] SE(3) transforms `core/geometry/transforms.py`
  - Twist `Exp/Log`, compose, invert; jit‑friendly
  - DoD: Numeric tests vs reference on small random cases
- [x] Parallel CT `core/geometry/parallel.py`
  - Axis‑aligned detector, parallel rays; matches current defaults
  - DoD: Unit tests for orthogonality and pixel center mapping
- [x] Laminography `core/geometry/lamino.py`
  - Rotation axis tilted by `tilt_deg` about x/z; per‑view angle
  - DoD: Unit tests: correct tilt orientation and angle progression

## Phase 3 — Projector & Operators
- [x] Geometry‑agnostic forward projector `core/projector.py`
  - Streaming integrate: `o(u,v) + s·d(u,v)`; step size heuristic `Δs ≈ min(vx,vy,vz)`
  - JIT with checkpoint
  - DoD: Produces `(nv,nu)` per view
- [x] Adjoint test `operators.adjoint_test_once`
  - DoD: Relative error < 1e-2 on tiny grid
- [x] Basic tests for projector and adjoint
  - DoD: Uniform-volume integral equals path length; adjoint test passes

## Phase 4 — Reconstruction
- [x] FBP `recon/fbp.py`
  - Ramp/Shepp‑Logan/Hann; frequency‑domain filtering; via VJP backprojection
  - DoD: Basic PSNR test passes on tiny case
- [x] FISTA‑TV `recon/fista_tv.py`
  - TV prox (Chambolle); Lipschitz via power method; loss logging
  - DoD: Loss decreases on tiny case
- [x] Multi‑resolution hooks `recon/multires.py`
  - Binning for projections/volume; parameter transfer; coarse→fine FISTA
  - DoD: Tiny test asserts multi-res loss <= single-level

## Phase 5 — Data Simulation
- [x] Phantoms `data/phantoms.py`
  - Shepp‑Logan-like, cubes, blobs; 3D; deterministic by seed
- [x] Simulate projections `data/simulate.py`
  - Noise: Poisson, Gaussian; parallel and laminography support
  - DoD: Writes valid `.nxs`; seed stored in meta
- [x] Datasets API `data/datasets.py`
  - Loaders, schema validation, convenience wrappers
  - DoD: Round‑trip and validation tests

## Phase 6 — Alignment Pipeline
- [x] Parametrizations `align/parametrizations.py`
  - 5DOF mapping to SE3 (alpha,beta,phi,dx,dz)
- [x] Alternating pipeline `align/pipeline.py`
  - Recon step (FISTA-TV) + per-view gradient alignment
  - DoD: Quick tiny test with loose RMSE thresholds
- [x] Metrics `align/metrics.py`
  - RMSE helpers

## Phase 7 — CLIs & Config
- [x] Config utils `utils/config.py`
  - JSON/YAML loader, dataclass dump helper
- [x] CLI: `simulate`, `recon`, `align`, `convert` (argparse)
  - Help strings and examples added to README
- [x] Logging utils `utils/logging.py`
  - Device/backend logging; integrated into CLIs

## Phase 8 — Tests (pytest)
- [x] Unit tests: geometry, transforms, projector, IO, phantoms
  - Added tests for random shapes phantom determinism/bounds
- [x] Schema & round‑trip: HDF5 validators (negative test added)
  - Validator flags missing rotation_angle
- [x] Integration: simulate→FBP; simulate→FISTA; simulate→align (parallel, lamino)
  - Added integration tests under `next/tests/test_integration.py`
- [ ] Performance sanity: microbench snapshot
  - Stored in text under `next/tests/_perf/`

## Phase 9 — Parity & Validation vs Current Repo
- [ ] Tiny shared case (e.g., 64³, 64 views)
  - DoD: v1 vs v2 forward projection L2 diff < 1e-4; align converges to comparable metrics
- [ ] Script `examples_v2/compare_v1_v2.py`
  - DoD: Prints deltas and pass/fail

## Phase 10 — Docs & Examples
- [ ] `README_v2.md` with quick start and CLIs
- [ ] `examples_v2/` tiny end‑to‑end scripts
- [ ] Schema doc and geometry overview diagrams (optional)
  - DoD: New users can simulate→align in 5 commands

## Phase 11 — Release Quality
- [ ] Type hints, docstrings, API reference consistency
- [ ] Naming and style (4‑space, snake_case, <=100 cols)
- [ ] Changelog `CHANGELOG_v2.md`
  - DoD: All checked; ready for wider use

---

## Acceptance Criteria & Quick Commands

- IO
  - [ ] `pixi run python -m tomojax_next.cli.simulate --geometry parallel --nx 128 --n-proj 128 --out data/sim_parallel.nxs`
  - [ ] `pixi run python -m tomojax_next.cli.convert --in data/sim_parallel.nxs --out data/sim_parallel.npz`
- Reconstruction
  - [ ] `pixi run python -m tomojax_next.cli.recon --algo fbp --data data/sim_parallel.nxs --out runs/fbp.npy`
  - [ ] `pixi run python -m tomojax_next.cli.recon --algo fista --data data/sim_parallel.nxs --iters 50 --lambda-tv 0.005 --out runs/fista.npy`
- Alignment
  - [ ] `pixi run python -m tomojax_next.cli.align --data data/sim_parallel.nxs --levels 4 2 1 --outer-iters 15 --lambda-tv 0.005 --out runs/align_parallel`
  - [ ] `pixi run python -m tomojax_next.cli.simulate --geometry lamino --tilt-deg 30 --nx 128 --n-proj 128 --out data/sim_lamino.nxs`
  - [ ] `pixi run python -m tomojax_next.cli.align --data data/sim_lamino.nxs --levels 4 2 1 --outer-iters 15 --lambda-tv 0.005 --out runs/align_lamino`
- Tests
  - [ ] `pytest -q next/tests` passes locally (CPU) in < 5 min

## Notes & Gotchas
- Use `float32` by default; document any `float64` hotspots
- HDF5: Prefer local SSD during writes (NFS locking); `lzf` default compression
- JIT stability: keep `static_argnames` stable; avoid Python control flow on tensor shapes
- Seeds: store under `/entry/processing/tomojax_next/meta/seeds`

## Definition of Done (Project)
- All phases complete; tests pass; CLIs documented; example end‑to‑end runs reproducible; parity script passes; alignment thresholds met for parallel CT and laminography tiny cases; schema validated.

---

## Remaining Tasks (Prioritized)

1. Multi‑Resolution Hooks (Phase 4)
   - Implement `recon/multires.py` (binning for projections/volume, parameter transfer)
   - Acceptance: Coarse→fine improves final metric in test

2. Expand Tests (Phase 8)
   - Unit: geometry, transforms, projector, IO — fast CPU suite
   - Integration: simulate→FBP, simulate→FISTA, simulate→align (parallel, lamino)
   - Schema & round‑trip: stricter HDF5 validators; negative tests
   - Performance sanity: microbench snapshot under `next/tests/_perf/`

3. Parity & Validation vs Current Repo (Phase 9)
   - Tiny shared case; v1 vs v2 forward diff and convergence parity
   - Comparison script printing deltas and pass/fail

4. Docs & Examples (Phase 10)
   - README_v2 quick start, examples_v2 scripts, optional diagrams

5. Release Quality (Phase 11)
   - Type hints/docstrings, consistent naming/style, CHANGELOG_v2
