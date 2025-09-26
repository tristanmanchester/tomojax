# TomoJAX Axes Unification & I/O Normalization — Implementation Checklist

This plan fixes the reported slice/width/height swap by normalizing volume axis
order at the I/O boundary while keeping the current internal memory layout.

Context today:
- Internal volume order in memory: `x, y, z` → array shape `(nx, ny, nz)`.
- External ecosystem (NX/HDF5 viewers): expects `z, y, x` → `(nz, ny, nx)`.
- Root cause of your 320×270×320 result: volumes were written to disk in
  internal order without an axes attribute, so tools interpreted axis 0 as `z`.

Track A below (recommended immediate fix) is a boundary‑normalization patch that
adds explicit axes metadata and transposes on save/load. Track B outlines a
future full internal reorder if we ever want internal `zyx` as well.

---

## Track A — Boundary Normalization (Recommended)

Normalize at I/O only; leave math and memory layout unchanged.

### A0. Decision & constants
- [x] Decide canonical on‑disk order: `zyx` (most common; matches viewers).
- [x] Introduce constant: `DISK_VOLUME_AXES = "zyx"` (internal default remains `xyz`).
- [x] Choose attribute name: `/entry/processing/tomojax/@volume_axes_order`.

### A1. Utilities (helpers)
- [x] Add `src/tomojax/utils/axes.py` with:
  - [x] `axes_to_perm(src: str, dst: str) -> tuple[int,int,int]` (e.g., `"xyz"->"zyx" → (2,1,0)`).
  - [x] `transpose_volume(vol, src: str, dst: str) -> jnp.ndarray`.
  - [x] `infer_disk_axes(vol_shape, grid) -> str|None` using `(nx,ny,nz)` vs `(nz,ny,nx)` heuristics.
  - [x] `is_shape_xyz(vol_shape, grid)`, `is_shape_zyx(vol_shape, grid)`.
  - [x] Minimal numpy fallback for non‑JAX contexts (used by IO).

### A2. Write path — `save_nxtomo`
- [x] Update signature in `src/tomojax/data/io_hdf5.py`:
  - [x] Add kwarg `volume_axes_order: str = "zyx"` (disk order).
  - [x] Document that input `volume` is internal `xyz`.
- [x] When `volume is not None`:
  - [x] If `volume_axes_order == "zyx"`, transpose `(nx,ny,nz) → (nz,ny,nx)` before write.
  - [x] Else if `"xyz"`, write as‑is (reserved for testing/interops).
  - [x] Write attribute `/entry/processing/tomojax/@volume_axes_order = volume_axes_order`.
  - [x] Keep existing `@frame` semantics unchanged (`"sample"|"lab"`).
- [x] Unit‑test new behavior (see A6).

### A3. Read path — `load_nxtomo`
- [x] After reading `grid` and optional `volume`:
  - [x] Read `/entry/processing/tomojax/@volume_axes_order` if present.
  - [x] If attr is `"zyx"`, transpose data to internal `xyz`.
  - [x] If attr is `"xyz"`, keep as‑is.
  - [x] If attr missing:
    - [x] Heuristic: if `vol.shape == (grid["nx"], grid["ny"], grid["nz"])` → treat as legacy disk=`xyz`; keep and set `out["volume_axes_order"] = "xyz_legacy"`.
    - [x] Else if `vol.shape == (grid["nz"], grid["ny"], grid["nx"])` → disk likely `zyx`; transpose and set `out["volume_axes_order"] = "zyx"`.
    - [x] Else: leave untouched, set `out["volume_axes_order"] = "unknown"` and log a warning.
- [x] Always return `out["volume"]` in internal `xyz` order.

### A4. CLI integration
- [x] `src/tomojax/cli/recon.py`:
  - [x] Pass `volume_axes_order="zyx"` to `save_nxtomo`.
- [x] `src/tomojax/cli/align.py`:
  - [x] Pass `volume_axes_order="zyx"` to `save_nxtomo`.
- [x] Optional: add `--volume-axes {zyx,xyz}` for expert overrides; default `zyx`.

### A5. Data wrangling script
- [x] `scripts/nexus_data_wrangler.py`:
  - [x] When writing placeholder `volume` shaped `(nz,ny,nx)`, add `@volume_axes_order = "zyx"`.
  - [x] In help text, document axes order and consistency with viewers.

### A6. Tests
- [x] Add `tests/test_axes_io.py` with:
  - [x] `test_save_volume_writes_zyx_and_attr()`: save `xyz` tensor, assert file dataset is `(nz,ny,nx)` and attr=`"zyx"`.
  - [x] `test_load_legacy_xyz_without_attr()`: create HDF5 in‑memory `(nx,ny,nz)` without attr, loader returns `(nx,ny,nz)` and marks `volume_axes_order="xyz_legacy"`.
  - [x] `test_roundtrip_xyz_to_zyx_to_xyz()`: save then load; equality within tolerance.
  - [x] `test_no_change_for_projections_shape()`: ensure `(n, nv, nu)` untouched.

### A7. Documentation
- [x] Update `docs/schema_nxtomo.md`:
  - [x] Add `@volume_axes_order` under `/entry/processing/tomojax` with default `"zyx"`.
  - [x] Clarify internal memory order is `xyz` and always returned by loaders.
- [x] Add a short “viewer tips” note in `README.md` about `zyx` on disk.

### A8. Migration tooling (optional but recommended)
- [ ] Add `scripts/migrate_volume_axes.py`:
  - [ ] Detect legacy files (missing attr) and transpose `(nx,ny,nz) → (nz,ny,nx)` in place, then set attr to `"zyx"`.
  - [ ] `--dry-run`, `--backup .bak`, bulk glob support, progress output.
  - [ ] Log decisions per file (legacy vs already normalized).

### A9. Logging & safety
- [x] Add concise warnings when heuristics are used in `load_nxtomo`.
- [x] Env var `TOMOJAX_AXES_SILENCE=1` to suppress noisy axes logs in batch runs.

### A10. Release notes & CI
- [x] `CHANGELOG.md`: document new behavior and backward compatibility.
- [ ] Add CI job to run new tests.
- [ ] Verify GPU/CPU code paths unaffected (no compute‑path changes).

### A11. Manual validation
- [ ] Simulate small dataset: `pixi run simulate --nx 64 --ny 64 --nz 32 --nu 64 --nv 32 --n-views 180 --out runs/sim.nxs`.
- [ ] Reconstruct: `pixi run recon --data runs/sim.nxs --algo fbp --out runs/fbp.nxs`.
- [ ] Inspect saved volume via HDF5: assert dataset shape `(nz,ny,nx)` and attr `"zyx"`.
- [ ] Load with TomoJAX API and confirm returned shape `(nx,ny,nz)` and numerics match roundtrip.
- [ ] Repeat with `align` CLI; verify expected slice count equals detector rows.

---

## Track B — Full Internal Reorder (Longer‑Term, Optional)

Adopt `zyx` inside the library as well (heavier refactor; not required for the fix).

### B0. Decision & scoping
- [ ] Confirm migration window, branch name, and deprecation policy.

### B1. Core data model
- [ ] Change internal canonical order to `zyx`.
- [ ] Update `Grid` docs to emphasise `nx,ny,nz` fields but clarify array order is `zyx`.
- [ ] Add `VolumeView` helpers to create named‑axis accessors.

### B2. Projector & geometry
- [ ] Update `_flat_index`, `_trilinear_gather`, and stepping logic to index `z,y,x` in memory.
- [ ] Adjust any `reshape`/`ravel(order="C")` sites accordingly.
- [ ] Re‑run adjoint tests and uniform‑path tests.

### B3. Recon, align, filters
- [ ] Touch FBP, FISTA‑TV, multires binning/upsampling to use new order.
- [ ] Audit all `axis=` parameters in FFTs, gradients, and pads.

### B4. I/O and scripts
- [ ] Keep Track A’s on‑disk `zyx` (unchanged) and drop heuristics after grace period.
- [ ] Provide a `TOMOJAX_LEGACY_XYZ=1` feature gate for one release.

### B5. Tests & docs
- [ ] Update shapes in tests, regenerate test data, and expand property tests on axis semantics.
- [ ] Publish a migration guide with code snippets.

---

## Cross‑Cutting Quality Gates
- [ ] Preserve numerical equivalence on projector sanity tests (`tests/test_projector.py`).
- [ ] Adjoint test remains within tolerance.
- [ ] No regression in wall time or memory on representative configs (document device and sizes).
- [ ] Confirm saved NX files load correctly in common viewers (slice count equals detector rows).

## Rollout
- [ ] Land Track A behind a minor version bump.
- [ ] Announce in `CHANGELOG.md` and `README.md` with examples and FAQs.
- [ ] Monitor for user reports; consider Track B only if needed by downstreams.

