# validate

The `tomojax-validate` command checks whether an HDF5 file conforms
to the NXtomo schema expected by TomoJAX. It runs lightweight
structural checks and reports any issues found.

```
tomojax-validate <input.nxs>
```

## Usage

The command takes one positional argument: the path to an `.nxs`,
`.h5`, or `.hdf5` file. There are no other flags or options.

```bash
uv run tomojax-validate data/scan.nxs
```

## Exit codes

The process exit code tells you the validation outcome at a glance.

| Code | Meaning |
|------|---------|
| `0`  | File is valid NXtomo |
| `1`  | Validation issues found |
| `2`  | File not found or path isn't a file |

You can use the exit code in scripts to gate downstream steps:

```bash
uv run tomojax-validate data/scan.nxs && uv run tomojax-recon \
  --data data/scan.nxs --out out/recon.nxs
```

## Example

A valid file prints a single confirmation line:

```
$ uv run tomojax-validate data/sim_aligned.nxs
OK: data/sim_aligned.nxs
```

When validation finds problems, it lists every issue:

```
$ uv run tomojax-validate data/broken.nxs
INVALID: data/broken.nxs (2 issues)
- /entry/data/data: dataset missing
- /entry/data/rotation_angle: dataset missing
```

If the path doesn't exist or points to a directory, you get an
error on stderr and exit code 2:

```
$ uv run tomojax-validate /tmp/no_such_file.nxs
ERROR: file not found: /tmp/no_such_file.nxs
```

> [!TIP]
> Run `tomojax-validate` before feeding new data to `tomojax-recon`
> or `tomojax-align`. It catches schema problems early without
> loading projections into memory.

## See also

- [Data format reference](../reference/data-format.md) for the
  full NXtomo schema TomoJAX expects
- [inspect](inspect.md) for metadata and projection statistics
  beyond structural validation
