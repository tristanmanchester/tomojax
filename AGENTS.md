# TomoJAX

TomoJAX is a general tomography library: parallel-beam, cone-beam and
laminography projection, reconstruction and alignment, in JAX with CUDA
kernels. The aim is to be faster and more accurate than ASTRA and TIGRE on
real scans while `tj.load`, `tj.reconstruct` and `tj.align` just work.

## Constraints

- Keep the backprojector the exact transpose of the forward projector.
- Improvements must be general, not tuned to one benchmark, phantom or dataset.
- Breaking changes are fine before 1.0. Omit compatibility shims.
- A faster or lower-residual result only counts once the reconstruction or pose
  error has been checked too.
- Real data is within a few orders of magnitude of 1 in sensible units; do not
  engineer for values like 1e-30 or 1e30.
- Fix failures in the method itself. Prefer deletion or simplification over
  defensive guards, especially when bad inputs already fail clearly.
- Start review with deterministic checks. Limit prose review to one or two rounds.

## Checks

- Start with `just surface-check` for Ruff, public imports and surface tests.
  Use `just typecheck` (basedpyright), `just imports` (import-linter) and
  `tests/test_architecture.py` for targeted static and architecture checks.
- Run `just guardrails` after intentional API changes or ratchet decreases.
  Regenerate wire files in order with `uv run python tools/build_wire_options.py`
  then `uv run python tools/build_wire_schemas.py`.
- Run the tests your change touches, on GPU for GPU changes. The maintainer runs
  the CPU gate (`just ci`) and full GPU suite (`just test-cuda`, including slow
  alignment fits) when merging into main. Hand off a ready commit.

## Studio

Use Rust 1.99.1. From [`studio/`](studio/), run `cargo fmt --check`,
`cargo clippy --locked --all-targets -- -D warnings` and `cargo test --locked`.
Verify visual changes with a headless `--snap` capture.
[Studio verification](studio/README.md#verification) · [Studio architecture](studio/ARCHITECTURE.md)

See the [architecture map](docs/architecture-map.md), [conventions](CONTRIBUTING.md),
[measurements](docs/measurements.md) and [benchmark drivers](bench/).

Say what changed and how it was checked in a few plain sentences. Evidence
(logs, scripts, archived results) stays out of the repository.
