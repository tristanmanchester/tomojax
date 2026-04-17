# TomoJAX test strategy

The test suite is organized around owned surfaces rather than raw coverage percentage. Add
tests to the surface that owns the user-visible contract you are changing.

## Test surfaces

- Core math and geometry contracts: small CPU-friendly unit tests that lock numerical behavior,
  shapes, validation, and schema conversions in `tomojax.core.*`, `tomojax.data.*`, and
  `tomojax.recon.*`.
- Alignment and reconstruction workflows: focused regression tests for multires orchestration,
  loss contracts, convergence, and failure behavior. These should protect the public pipeline
  and config contracts rather than re-testing projector internals.
- CLI workflows: file-backed smoke tests for `simulate`, `misalign`, `recon`, `align`, and
  `loss_bench`. Prefer real temporary `.nxs` or JSON artifacts over deep monkeypatch stacks so
  the test covers the boundary users actually invoke.
- Benchmark harness and support modules: contract tests for `bench/`, `scripts/`, and
  `tomojax.bench.*` that lock profile parsing, shared helpers, and report or artifact schemas.
  Controller-specific policy belongs here instead of leaking into product CLI tests.
- Small-module coverage: `tests/test_small_module_coverage.py` is only for leaf modules with no
  stronger owning surface. When a module gains a real caller or user-facing behavior, move the
  coverage to that surface instead of growing the catch-all file indefinitely.

## Placement rules

- Prefer a direct test for each changed public module even if another workflow already exercises
  it transitively.
- Put new workflow tests next to the boundary they protect, not into broad kitchen-sink
  integration files.
- Use monkeypatching to cut off external cost or nondeterminism, not to fake the contract under
  test. If the filesystem or NXtomo boundary matters, exercise the real boundary with temporary
  files.
- Keep default test runs CPU-friendly. Expensive benchmark or GPU validation belongs in the
  harness, optional profiles, or documented manual commands.
- When promoting code between `scripts/`, `bench/`, and `src/tomojax/bench/`, move or add tests
  with the owning surface in the same change.

## Practical defaults

- Full default suite: `uv run pytest -q tests`
- Focused module checks: `uv run pytest -q tests/test_recon.py tests/test_align_quick.py`
- Boundary smoke checks after CLI changes:
  `uv run pytest -q tests/test_cli_entrypoints.py tests/test_cli_geometry_build.py tests/test_loss_bench.py`
- Benchmark contract checks after harness changes:
  `uv run pytest -q tests/test_bench_convergence.py tests/test_exp_spdhg_bench.py tests/test_exp_spdhg_report.py tests/test_support_modules.py`
