# convert

The `tomojax-convert` command converts datasets between `.npz`
(NumPy archive) and `.nxs` (HDF5 NXtomo) formats. NXtomo files
follow the TomoJAX schema with all metadata preserved.

```
tomojax-convert --in <input> --out <output>
```

## Supported conversions

The command infers the conversion direction from file extensions.
Two conversions are available:

- **`.nxs` to `.npz`** -- extracts projections, angles, and
  metadata arrays into a flat NumPy archive.
- **`.npz` to `.nxs`** -- packs arrays back into the NXtomo HDF5
  layout with all required groups and datasets.

Both `--in` and `--out` are required. The input extension must be
one of `.nxs`, `.h5`, or `.hdf5` for NXtomo files, or `.npz` for
NumPy archives.

## Examples

Convert an NXtomo file to a NumPy archive:

```bash
uv run tomojax-convert \
  --in data/sim_aligned.nxs \
  --out data/sim_aligned.npz
```

Convert the archive back to NXtomo:

```bash
uv run tomojax-convert \
  --in data/sim_aligned.npz \
  --out data/sim_aligned_roundtrip.nxs
```

> [!NOTE]
> Round-tripping (`.nxs` to `.npz` and back) produces a
> structurally identical NXtomo file. You can verify with
> `tomojax-validate`.

## Notes

NXtomo extras -- geometry parameters, reconstruction grid, and
detector metadata -- are preserved in both directions. The `.npz`
archive stores these as named arrays alongside the projection data,
so no information is lost during conversion.

See the [data format reference](../reference/data-format.md) for
the full NXtomo schema and a list of all stored fields.

## See also

- [Data format reference](../reference/data-format.md)
- [CLI overview](index.md)
