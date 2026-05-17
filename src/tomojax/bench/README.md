# tomojax.bench

## Status

`tomojax.bench` is a developer and verification package, not a production public
API. Its commands are exposed through `tomojax dev ...` rather than as installed
top-level console scripts.

## Responsibilities

- Benchmark fixtures and benchmark result comparison helpers.
- Diagnostic performance probes.
- Synthetic alignment diagnostic runners.
- Synthetic result report helpers used by implementation and regression work.
- Real-laminography developer workflow contracts, report semantics, and bounded
  planning helpers used by scripts under `scripts/real_laminography`.
- Reference-regression adapters that intentionally bridge developer diagnostics
  to private product internals without exposing those internals as production
  alignment API.

## Boundary Rule

Production modules should not depend on `tomojax.bench`. Benchmark code may
depend on public production facades to exercise them, but benchmark-only helpers
must not leak back into `tomojax.io`, `tomojax.recon`, `tomojax.align`,
`tomojax.forward`, or `tomojax.geometry`.

## Public API

`tomojax.bench.api` and the package root expose reusable developer benchmark
helpers. User workflows should reach benchmark commands only through
`tomojax dev ...`.
