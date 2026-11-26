# TomoJAX architecture map

Use the same names across interfaces, including `method`, `iterations`, `tv_weight`, `nonnegative`, `fit` and `poses`. CLI flags hyphenate Python keywords, as in `--tv-weight`. [Contributing guide](../CONTRIBUTING.md#design-rules)

## Entry points

| Surface | Where to work |
| --- | --- |
| Python library | `import tomojax as tj` exposes `tj.load`, `tj.reconstruct`, `tj.align` and related operations. Import through package roots or expert `.api` facades. [`src/tomojax/`](../src/tomojax/) · [IO](../src/tomojax/io/) · [Projection](../src/tomojax/forward/) · [Reconstruction](../src/tomojax/recon/) · [Alignment](../src/tomojax/alignment/) |
| CLI | Command dispatch, argument parsing, TOML configuration and file-oriented workflows sit above the Python API. [`src/tomojax/cli/`](../src/tomojax/cli/) · [`main.py`](../src/tomojax/cli/main.py) |
| JSONL worker | `tomojax serve` owns session handles, validation, execution, progress and scientific exports. Studio and MCP use this worker. [`serve.py`](../src/tomojax/cli/serve.py) · [`transport.py`](../src/tomojax/cli/transport.py) · [Worker protocol](protocol.md) |
| MCP and bundled Skill | `tomojax mcp` exposes worker operations to agents and serves guidance from the bundled Skill. Install the Skill with `tomojax skill -o SKILLS_DIR`. [`src/tomojax/cli/mcp/`](../src/tomojax/cli/mcp/) · [`src/tomojax/skill/tomojax/`](../src/tomojax/skill/tomojax/) |
| Studio | `tomojax studio` launches the Rust app. Studio owns projects, durable asset identities, the local job queue and display preferences. [`studio-core`](../studio/crates/studio-core/) (projects, protocol, jobs) · [`tomojax-studio`](../studio/crates/tomojax-studio/) (I/O, rendering, views) · [Studio architecture](../studio/ARCHITECTURE.md) |

## Library packages

`src/tomojax/__init__.py` resolves each `tj.*` name lazily from one of three private packages. Each package may import those below it in this table, never those above. [Import contracts](../.importlinter)

| Package | Contents |
| --- | --- |
| [`_workflow/`](../src/tomojax/_workflow/) | `tj.reconstruct`, `tj.align`, `tj.project`, `tj.save` and the `tj.load_*` readers in `operations.py`, with the steps they call: settings planning, field of view and region selection, memory estimates, the joint alignment workflow and held-out evaluation. |
| [`_loading/`](../src/tomojax/_loading/) | `tj.load` and `Frames` in `loader.py`, record geometry, frame sources, and `tj.inspect` and `tj.export` in `files.py`. |
| [`_scan/`](../src/tomojax/_scan/) | `Scan`, `Alignment` and `Reconstruction` in `model.py`, `tj.simulate` in `simulation.py`, and ASTRA geometry conversion. |
| Domain packages | `alignment`, `datasets`, `io`, `corrections`, `recon`, `forward`, `backends`, `nuisance`, `motion`, `geometry` and `core`, which the private packages build on. |

Three private modules stay at the root: `_parameters.py` holds the parameter vocabulary every layer shares, `_typed_arrays.py` the array types that public signatures name, and `_version.py` the version.

## Studio workspaces

| Workspace | Responsibilities and starting points |
| --- | --- |
| **Prepare** | Import, calibrate, correct and align acquisitions. [Inspector](../studio/crates/tomojax-studio/src/ui/inspector.rs) · [Preview control](../studio/crates/tomojax-studio/src/app/previewing.rs) |
| **Reconstruct** | Configure and queue reconstructions with worker validation and memory plans. [Inspector](../studio/crates/tomojax-studio/src/ui/inspector.rs) · [Validation and plans](../studio/crates/tomojax-studio/src/app/validation.rs) |
| **Explore** | View linked slices and volumes, measure in physical coordinates and export results. [Viewer](../studio/crates/tomojax-studio/src/ui/viewer.rs) · [Viewing state](../studio/crates/tomojax-studio/src/app/viewing.rs) |
| **Compare** | Inspect paired results side by side, with a split curtain or as a difference. [Viewer](../studio/crates/tomojax-studio/src/ui/viewer.rs) |
| **Jobs** | Track the queue, solver checkpoints, pose tables and convergence. [Job scheduling](../studio/crates/tomojax-studio/src/app/jobs.rs) · [Progress views](../studio/crates/tomojax-studio/src/ui/progress.rs) |

## Sources of truth

| Contract | Owner and derived files |
| --- | --- |
| Geometry and frames | The geometry guide defines physical `(x, y, z)`, volume storage `(z, y, x)`, projection storage `(view, v, u)` and positive object rotations in degrees. Grid and detector types and transforms live in the geometry core. [Frame conventions](../src/tomojax/geometry/README.md#frames-and-positive-angles) · [`core/geometry/`](../src/tomojax/core/geometry/) |
| Parameters | Public `*Config` dataclasses own fields and defaults. The options generator produces typed ingress in `wire_options.py` and the operation descriptions in `parameters.json` that Studio uses for forms. [Reconstruction configs](../src/tomojax/recon/) · [`AlignConfig`](../src/tomojax/alignment/_config.py) · [Generator](../tools/build_wire_options.py) · [Wire options](../src/tomojax/cli/wire_options.py) · [Parameter descriptions](../src/tomojax/cli/parameters.json) |
| Protocol schemas | Python wire types define envelopes, operations, products, arrays and progress. The schema generator exports their JSON Schema contract. [`envelope.py`](../src/tomojax/cli/envelope.py) · [`operations.py`](../src/tomojax/cli/operations.py) · [`products.py`](../src/tomojax/cli/products.py) · [`wire/`](../src/tomojax/wire/) · [Generator](../tools/build_wire_schemas.py) · [Generated schemas](protocol/) |
| Public API and ratchets | Package exports define the API. Architecture tests compare the live surface and code-health counts with the recorded API and ratchets. [Root exports](../src/tomojax/__init__.py) · [API record](../tests/guardrails/api_surface.txt) · [Ratchets](../tests/guardrails/ratchets.json) · [Architecture tests](../tests/test_architecture.py) |
| Studio UI | The app derives `UiModel` from project and live service state. `UiState` holds editable drafts and transient interactions. Views send `UiAction` to the app for project, worker and filesystem effects. [View contract](../studio/crates/tomojax-studio/src/ui/model.rs) · [Action dispatch](../studio/crates/tomojax-studio/src/app/actions.rs) |
