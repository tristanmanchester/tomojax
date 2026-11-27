# Contributing to TomoJAX

In every change, state the input and geometry, the expected result, what changed, and how you checked it. Check reconstruction or pose error alongside residuals and timings.

## Set up development

With Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/), run this from the worktree root.

```bash
./tools/worktree_setup.sh
```

The script installs the locked development dependencies with `cuda12` when `nvidia-smi` succeeds on Linux and with `cpu` otherwise. Use `--cpu` or `--cuda` to select an extra explicitly, and `--dry-run` to preview the setup.

In T3 Code, add a project script with `./tools/worktree_setup.sh` as its command and enable `runOnWorktreeCreate` to run it when T3 creates a worktree.

The script uses `$HOME/.cache/tomojax-tmp` as `TMPDIR` for the sync unless you supply a non-empty `TMPDIR`. On Linux it also creates `$HOME/.cache/tomojax-studio-target`, the Cargo target directory that `just studio-check` and `just studio-verify` share across worktrees; set `CARGO_TARGET_DIR` to use another. The justfile sets the same `TMPDIR` default for its recipes.

To install optional groups, run `uv sync --locked --extra cpu --dev --group examples` for figures or use `--group benchmark` for comparison libraries. Use `--extra cuda12` for a CUDA environment. See [bench/README.md](bench/README.md).

The [just](https://github.com/casey/just) recipes below run with `uv run --no-sync` so they keep your chosen CPU or CUDA extra. Note that `uv sync` removes manually installed packages such as a locally built TIGRE; keep comparison builds in a separate environment or reinstall them afterwards. You can also run the underlying commands directly from [justfile](justfile).

## Run checks

| Command | Purpose |
| --- | --- |
| `just format` | Format and auto-fix lint issues with Ruff |
| `just fast-check` | Read-only format, lint, type-error, and import checks and the architecture tests, in parallel |
| `just surface-check` | Read-only format, lint, and import checks, CLI and accelerator smoke checks, and surface tests (excluding numerical and GPU markers) |
| `just test` | Every CPU test, in parallel, including the alignment fits marked `slow` |
| `just test-cov` | The same tests with branch coverage |
| `just docstrings` | Validate NumPy-style public API docstrings |
| `just docs` | Build the Python reference HTML into `.artifacts/docs/` |
| `just site` | Sync Markdown guides, generate API pages with `griffe`, and build `tomojax.dev` in `site/`. The Site workflow runs the same steps and deploys on a push to `main` that changes a guide, an image, the package source or `site/` |
| `just typecheck` | Run `basedpyright` on the configured public API, IO, CLI, and tools |
| `just imports` | Check dependency layers and public-import guardrails |
| `just examples` | Run the public projection and reconstruction example |
| `just package` | Build sdist and wheel, run `twine check --strict`, and smoke-test the installed wheel |
| `just ci` | CPU gate: static checks, package build and smoke test, example, the CPU tests except those marked `slow`, and small analytic benchmarks |
| `just test-cuda` | Require CUDA, then run every test with the GPU visible, the `slow` ones included: first the tests that need the GPU to themselves, then the rest on workers that share it, up to one per available CPU and one per 1,300 MiB of GPU memory. Failed parallel tests run again in one process. A per-user lock makes a second run wait for the first |
| `just guardrails` | Update the public API and code-health ratchet records after an intentional change |
| `just studio-check` | Studio's `cargo fmt --check`, Clippy with `-D warnings`, and `cargo test` on Rust 1.99.1 |
| `just studio-verify` | Build Studio and capture each workspace headless into `.artifacts/studio-snap/`, failing on a missing or blank image |

`just check` formats files before running checks; use `just ci` for read-only validation. Mark a test that takes more than about 10 s on a CPU `slow`: it starts first in a parallel run, and `just ci` leaves it to `just test-cuda`, where the same fit takes seconds. Package recipes overwrite `dist/`, so store release artifacts elsewhere. Run `just test-cuda` on NVIDIA hardware whenever a change touches GPU kernels or accelerator paths.

`tools/smoke_installed_wheel.py` creates a disposable environment outside the repository, installs the built wheel without a package cache (requiring a real GPU under `--extra cuda12`), and runs simulation, automatic reconstruction, and slice export.

## Make a release

1. Update the version in `pyproject.toml` and add a `## <version> (<date>)` heading to `CHANGELOG.md`.
2. Run `just ci` and `just test-cuda` on that commit, then push to `main`.
3. Optionally run `gh workflow run Release` to publish `main` to TestPyPI after the same checks.
4. Tag and push: `git tag v<version> && git push origin v<version>`. The release workflow runs the CPU gate and installed-package matrix, builds the `tomojax` wheel/sdist and the `manylinux_2_28_x86_64` `tomojax-studio` wheel/tarball (with HDF5 statically linked), publishes to PyPI through the `pypi` environment (which only `v*` tags may deploy to), and creates the GitHub release with both sets of artifacts from the changelog section. Versions containing `a`, `b`, `rc`, or `dev` are marked as pre-releases.

PyPI accepts each version once; fix a failed release with a new version number.

## Engineering conventions

- Import application code through public module roots or `.api` facades. Keep CLI orchestration separate from numerical code and follow the [import contracts](.importlinter). Keep alignment orchestration separate from the private joint solver. White-box unit tests may import private modules with an explicit `check-public-imports: allow-private` comment; do not export internals just for tests.
- Keep geometry explicit: Python volumes use `(z, y, x)`, projections use `(view, v, u)`, and voxel and detector spacings share one physical length unit. Document parameter frames and units at API boundaries.
- Keep backprojectors as exact transposes of their forward projectors. Test numerical changes against independent analytic or dense references, including shifted, anisotropic, tilted, odd-sized, and boundary cases.
- Validate inputs before expensive work. Reject malformed metadata instead of silently discarding it, and never report a failed solve as success.
- Keep solver defaults and result semantics stable unless a change is tested and recorded in the [changelog](CHANGELOG.md).
- Add regression tests for correctness bugs. Avoid tests that mirror implementation details or rely on host-dependent timing thresholds.

## Design rules

TomoJAX should feel like one small, consistent tool. Check API changes against these rules:

- **One vocabulary.** Each concept has one name across Python, the CLI, and the docs: `method`, `iterations`, `tv_weight`, `nonnegative`, `fit`, `poses`. CLI flags are the Python keywords with hyphens. Internal names (profiles, schedules, solver stages) stay internal.
- **One way to do a thing.** No aliases, no duplicate option spellings, and no modes that merely change defaults.
- **Small interfaces, deep modules.** Hide implementation detail behind small entry points. Compute settings from the data rather than adding options; add a config field only with a clear reason. Export user-facing calls from package roots and expert helpers from `.api`.
- **Keyword-only options.** Accept at most one or two positional arguments for the main subject (`tj.reconstruct(scan, "cgls", iterations=50)`); make all other options keyword-only.
- **Results carry their context.** Return objects that bundle their geometry and settings rather than bare arrays plus separate metadata.
- **Fix the method, don't wrap it.** When a method gets a case wrong, fix its model or statistics instead of detecting the case and bypassing it. For example, an earlier aligner reported ~0.1 px of motion on already aligned scans. A "consistency guard" was added that split the data in two, compared resolutions, and discarded alignment below a threshold. That hid the bias, flipped discontinuously between aligned and unaligned, and still let bad results through on drifted data. The right fix lives in the estimator: noise-weighted residuals, regularization scaled by data observability, and reported uncertainties.
- **A helper earns its name** by having two or more callers, being tested on its own, or naming a concept its caller should not need to know. Keep single-use helpers inline.
- **Fixed CLI shape.** File-producing commands take `tomojax <command> INPUT -o OUTPUT`, refuse to overwrite existing outputs without `--force`, and use exit codes 0 (success), 1 (failure), 2 (usage error), and 130 (interrupted). Keep `--help` short and put expert settings in `--config` TOML files. Interactive and server entry points (`inspect`, `simulate`, `serve`, `mcp`, `studio`) omit `-o` or `INPUT` where not applicable.

`tests/test_architecture.py` enforces the mechanical checks:

- The public API matches `tests/guardrails/api_surface.txt`, so API changes appear in diff review.
- The ratchets in `tests/guardrails/ratchets.json` may decrease but never rise: `*Config` fields, exported names, CLI flags, files over 800 lines, functions over 100 lines (excluding docstrings), `basedpyright` warnings by rule, explicit `Any`, `cast()` calls, broad `except` handlers, CUDA kernels and compile-time switches, environment-variable reads, lint suppressions, complexity counts, single-use private helpers, untyped module errors, positional options on public APIs, first-device assumptions, private imports in tests, and in Studio's Rust sources, lint `allow` attributes, files over 800 lines and functions over 100 lines. `python tools/guardrails.py` also prints lines per package, for information.
- CLI flags map cleanly to Python keywords, and `--help` stays concise.

After an intentional change (or when a ratchet drops), run `just guardrails` and commit the updated records. Do not raise a ratchet to make a change pass; fix the code instead, or explain why the debt is justified in the commit message.

## Documentation and figures

Write public API docstrings in NumPy style with `Parameters`, `Returns` (or `Yields`), and `Raises` sections, stating array shapes, coordinate frames, and units at boundaries. Include short `Examples` on main workflow entry points. Run `just docstrings` to validate coverage and `just docs` to build the HTML reference.

Keep user guides in `docs/` (using US English consistently across code, CLI, schemas, and prose) and module-level interfaces in each package's `README.md`. Check commands against the actual CLI and public API.

The README watch figures come from [examples/reconstruct_watch.py](examples/reconstruct_watch.py), [examples/align_watch.py](examples/align_watch.py), and [examples/render_watch.py](examples/render_watch.py). Keep image scales explicit with their JSON records, and note provenance for historical assets in [images/README.md](images/README.md).

## Repository layout and benchmark records

| Location | Contents |
| --- | --- |
| `src/tomojax/` | Installable library, CLI, worker (`serve`), and MCP server (`mcp`) |
| `studio/` | Native Rust/Rune desktop workspace (`tomojax studio`); see [studio/README.md](studio/README.md#verification) for `cargo` checks |
| `plugins/` | Claude Code plugin wrapping `tomojax mcp` and workflow skills |
| `site/` | Documentation website (`tomojax.dev`) generated from repository Markdown and docstrings |
| `tests/` | Public behavior, numerical, and GPU regression tests |
| `examples/` | Small runnable public API examples |
| `tools/` | Development and packaging checks |
| `docs/` | User guides and measured benchmarks |
| `bench/` | Comparison drivers (not part of the installed package) |
| `images/` | Presentation figures |
| `reports/`, `research_notes/` | Research notes and literature reviews |
| `.artifacts/` | Ignored local runs, logs, and build outputs |

Keep raw benchmark records out of the repository. Benchmark scripts write under `bench/results/` (gitignored); attach any raw JSON files a reviewer needs to the PR or issue. [Measurements](docs/measurements.md) summarizes published speed, accuracy, scaling, and memory results with their conditions (full runner commands are in [bench/README.md](bench/README.md)). When adding new performance numbers, report reconstruction quality alongside cold time, warm time, and peak GPU memory.

## Add an export format

1. Write a block writer and add one `Format` entry to `FORMATS` in
   `src/tomojax/io/_export_formats.py`. Specify the suffixes, file layout,
   supported options and what independent readers recover. A file with an
   embedded record also supplies its record reader. The planner validates those
   options for Python, the CLI, the worker and MCP.
2. Extend the independent readback in `tests/test_export.py` for the format.
   The parameterized test exercises every table entry, each stack axis and
   widths 1, 8 and 11. Add the consumer to `tools/check_export_readers.py` to
   check which calibration and sample values survive import.
3. Run `uv run python tools/build_wire_options.py`, then
   `uv run python tools/build_wire_schemas.py`. Their `--check` modes fail on
   stale client metadata, schemas or the format table in `docs/export.md`.
4. Run `uv run pytest tests/test_export.py`, `just interop` and
   `just fast-check`. The temporary-format test exercises dispatch, inspection,
   CLI and generated wire and MCP field descriptions with a format defined only
   inside the test.
