# Contributing to TomoJAX

Make scientific behavior reviewable: describe the input and geometry, the
expected result, the change, and how you checked it. A lower residual or faster
kernel is not sufficient evidence for a more accurate or faster reconstruction.

## Set up development

From a checkout with Python 3.12 and uv:

```bash
uv sync --locked --extra cpu --dev
```

On Linux CUDA, select `--extra cuda12` instead. Add `--group examples` only when
generating figures, or `--group benchmark` for optional comparison dependencies.
See [bench/README.md](bench/README.md) for external solver setup.

The [just](https://github.com/casey/just) recipes below use `uv run --no-sync`:
setup is explicit, so checks preserve the selected CPU/CUDA extras. `uv sync`
can remove manually installed packages, including locally built comparison
libraries; use a separate environment for those or reinstall them deliberately.
You can also run each recipe's commands directly from the [justfile](justfile).

## Run checks

| Command | Purpose |
| --- | --- |
| `just format` | Apply Ruff formatting and automatic lint fixes |
| `just surface-check` | Read-only formatting/lint/import checks, CLI/accelerator smoke checks, and explicitly marked surface tests (excluding numerical/GPU markers) |
| `just test` | CPU test suite, including numerical tests |
| `just typecheck` | Configured public API, IO, CLI, and tool type checks |
| `just imports` | Dependency layers and public-import guardrails |
| `just examples` | Public projection/reconstruction example |
| `just package` | Build sdist/wheel, check metadata, and run a fresh installed-wheel workflow |
| `just ci` | Full local gate: static checks, package/workflow checks, example, CPU coverage, and small analytic benchmarks |
| `just test-cuda` | Require a real CUDA accelerator, then run GPU-marked tests |

`just check` also formats files before running checks. Use `just ci` when you
want validation without automatic source edits. Package recipes replace `dist/`;
keep release artifacts elsewhere. GPU tests are separate from CPU CI and must
be run on suitable hardware for kernel or accelerator behavior changes.

The installed-wheel check imports outside the checkout and exercises simulation,
inspection, validation, FBP reconstruction, and labelled slice export. A source
import passing is not enough to establish that the wheel works.

## Engineering conventions

- Import application code through public module roots or `.api` facades. Keep
  CLI orchestration separate from numerical implementation and honor the
  [import contracts](.importlinter).
- Keep geometry explicit: Python volumes are `(x, y, z)`, projections are
  `(view, v, u)`, and voxel/detector lengths use consistent physical units.
  Document parameter frames and units at API boundaries.
- Preserve matched forward/adjoint semantics. Test numerical changes against
  independent physical or dense references, including shifted, anisotropic,
  tilted, odd-sized, and boundary cases when applicable.
- Validate user inputs before expensive work. Do not silently discard malformed
  projection metadata or convert a failed solve into a success status.
- Keep solver defaults and checkpoint semantics stable unless a change is
  justified, tested, and recorded in the [changelog](CHANGELOG.md).
- Add regression tests for behavior and correctness bugs. Avoid tests that only
  repeat implementation details or timing thresholds that depend on the host.

## Documentation and figures

Start from the reader's task. Provide prerequisites, complete commands, expected
outputs, and material limitations alongside each feature. Check commands against
the actual CLI or public API. Keep user guides in `docs/`; module-level READMEs
explain the corresponding Python interfaces.

Use [examples/plot_reconstruction.py](examples/plot_reconstruction.py) for the
README's synthetic figure. Keep scientific image scales explicit and retain its
JSON provenance. Record missing provenance for historical assets in
[images/README.md](images/README.md); visual quality is not an accuracy metric.

## Repository layout and measurement records

| Location | Contents |
| --- | --- |
| `src/tomojax/` | Installable library and CLI |
| `tests/` | Public behavior, numerical, and accelerator regressions |
| `examples/` | Small runnable public API examples |
| `tools/` | Development and package checks |
| `docs/` | User guides and dated measurement reports |
| `bench/` | Comparison drivers; not part of the installed API |
| `bench/reference/` | Retained raw results and measured source snapshots |
| `images/` | Documented presentation assets |
| `reports/`, `research_notes/` | Research material and literature notes |
| `.artifacts/` | Ignored local runs, logs, and generated working files |

Keep failed experiments and their source identity when they support a report.
Store bulk raw JSON as lossless gzip archives with hashes in the
[evidence catalog](bench/reference/README.md); keep summaries readable. Expanded
experiment logs should not dominate the source change list.
Do not combine measurements from different timing protocols or silently replace
historical data. New performance claims should account for accepted-result
quality, fresh-process and warm time, and process GPU memory across the declared
geometry/object set. See [measurement scope](docs/measurements.md).
