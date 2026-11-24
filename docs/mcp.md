# Use TomoJAX from an AI agent

`tomojax mcp` runs a [Model Context Protocol](https://modelcontextprotocol.io) server over stdio. An agent connected to it can inspect, load, correct, align, reconstruct, simulate and export scans, look at the results as images, and run Python against the data it has loaded. No tool waits for a person, so an agent can work through a scan unattended.

![Three central planes of FIPS walnut 1 as an agent receives them from the reconstruct tool, each labelled with its handle, plane index, physical extent and display limits.](images/mcp-walnut-panels.png)

What the agent sees after `reconstruct` on the FIPS walnut (three orbits, FISTA): the central planes, labelled with the result's handle, the plane, its extent and the display limits, beside a short text summary.

## Install and connect

The server needs the `mcp` extra. Install it with the CUDA 12 extra on a GPU machine, and the `spectra` extra if you want `tube_spectrum`:

```bash
uv pip install "tomojax[mcp,cuda12,spectra]"
```

### Claude Code

Add the server, then allow its tools so that calls run without a permission prompt:

```bash
claude mcp add tomojax -- uvx --from "tomojax[mcp,cuda12,spectra]" tomojax mcp
```

```json
{
  "permissions": {
    "allow": ["mcp__tomojax"]
  }
}
```

Put the `permissions` block in `.claude/settings.json` (shared with the project) or `.claude/settings.local.json` (yours only). To share the server with a project instead, commit a `.mcp.json`:

```json
{
  "mcpServers": {
    "tomojax": {
      "command": "uvx",
      "args": ["--from", "tomojax[mcp,cuda12,spectra]", "tomojax", "mcp"]
    }
  }
}
```

A project server needs approval once per checkout. Pre-approve it with `"enabledMcpjsonServers": ["tomojax"]` in `.claude/settings.local.json`.

From a development checkout, run the server from source with `uv run --extra mcp --extra cuda12 tomojax mcp`.

The repository also carries a Claude Code plugin in `plugins/tomojax/`, with this server and a skill that points the agent at the workflow guides.

### Codex and other clients

Any MCP client that starts stdio servers can run the same command. For Codex, add to `~/.codex/config.toml`:

```toml
[mcp_servers.tomojax]
command = "uvx"
args = ["--from", "tomojax[mcp,cuda12,spectra]", "tomojax", "mcp"]
```

## Tools

| Tool | What it does |
| --- | --- |
| `status` | Versions, devices, GPU memory, live handles and running jobs |
| `inspect` | Report on a file, list the datasets in a directory, or show an HDF5 tree |
| `open` | Load a scan, raw frames or a saved result as a handle, with metadata overrides |
| `describe` | Expert settings for an operation or solver, with defaults and units |
| `correct` | Flat/dark correction, zingers, phase retrieval, stripes, rings, beam hardening and binning |
| `reconstruct` | Reconstruct with an automatic or named method; returns central planes |
| `reconstruct_slabs` | Reconstruct raw frames slab by slab straight into a file, for volumes too large to hold |
| `align` | Fit the centre, setup angles or per-view motion; returns a pose plot and slices |
| `simulate` | Simulate a scan of a phantom, with optional detector artefacts |
| `project`, `backproject` | Apply the forward projector or its exact transpose |
| `edit_scan` | Select views, crop, bin, replace poses or combine scans |
| `preview` | Single slices, centre sweeps, before/after corrections and region estimates |
| `view` | Planes, projections, sinograms, differences, profiles, statistics and pose tables |
| `evaluate` | Score candidate settings on held-out views |
| `save`, `export` | Write a dataset with its manifest, or TIFF slices and raw volumes |
| `control` | Cancel, pause or resume a job; close a handle; restart the worker |
| `python` | Run Python in the worker, with the live handles and `tj` in scope |

Tools that accept `dry_run` report what a call would do without running it: the method and grid a reconstruction would use and its memory estimate, the files an export would create, or the shapes a simulation would produce. Use `describe` before setting expert keys in `config`.

## Handles and references

Each result is kept in the server as a handle with an ID such as `@scan-12-k7m2`. Pass that ID to any tool that takes data. Add a member to choose part of a result, for example `@align-3-k7m2.scan` for the aligned scan or `@sim-1-k7m2.phantom` for a simulation's phantom. A bare handle gives the whole object to `save` and `export`, and its main array (volume, projections, phantom or working image) to an array field.

Anything that is not a handle is a file path, resolved against the project directory. Name a dataset inside an HDF5 file as `scan.h5::/entry/data/data`, and write `./@name` for a file whose name starts with `@`.

Handles last until you close them, the worker restarts, or the server exits. Save results you want to keep. An ID from an earlier session returns an error saying the handle is gone.

## Results, images and long jobs

Every result has a short text summary, the same fields as structured content, and labelled PNG panels where an image helps: three central planes after `reconstruct`, a pose plot and slices after `align`, and the requested planes from `view` and `preview`. Panels that are meant to be compared, such as a centre sweep, share one display range. Large results are cut to fit the client's limits, and the rest is served as `tomojax://result/...` resources that the result names.

Long jobs send progress with the current stage, its metrics and an estimate of the time left. Clients that support the MCP tasks extension receive long calls as tasks they can poll and cancel; other clients wait for the result. Cancel a job from the client or with `control`; it stops at the next checkpoint.

The server runs one worker process with one queue, so GPU jobs run one after another. Running out of GPU memory returns an ordinary error with suggestions, and the existing handles stay. If the worker process itself dies, the server reports which handles were lost and starts a new worker.

## Files

`save`, `export` and `reconstruct_slabs` refuse to replace an existing file unless the call sets `force`, and never overwrite their own inputs. Each write completes in full or not at all: if the server or its worker dies during a write, the next start restores the files that were there before.

## Python

`python` runs code in the worker with `tj`, `np` and `handles` in scope. `handles` maps live IDs to their TomoJAX objects. `register(obj)` publishes a new handle from a `Scan`, `Reconstruction`, array or other result, and `show(array)` adds an image to the reply. Variables persist between calls until `reset` or a worker restart.

Use it for anything the typed tools do not cover: a custom geometry, a direct solver such as `tomojax.recon.cgls`, `tomojax.forward`, phase unwrapping or your own analysis. The `tomojax://guide/python` resource has worked examples.

## Guides and prompts

The server publishes workflow guides as resources and as prompts, which Claude Code lists as slash commands: `reconstruct-scan`, `lab-ct-start-to-finish`, `find-the-centre` and `check-a-reconstruction`. Read them as `tomojax://guide/{topic}`, alongside `python`, `solvers`, `geometry` and `calibration`. `tomojax://handle/{id}` gives a handle's summary and provenance, and `tomojax://describe/{operation}` its settings.

## Limits

The server speaks stdio only, so the agent and the GPU run on the same Linux machine. Handles and tasks live only as long as the server process.
