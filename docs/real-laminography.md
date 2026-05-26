# Real Laminography Workflow

The package-facing real-data path starts with inspection and validation, then
uses the public alignment command for pose correction, detector-centre/COR
correction, or expert mixed setup and pose correction.

```bash
uv run tomojax inspect /path/to/scan.nxs
uv run tomojax validate /path/to/scan.nxs
uv run tomojax align --data /path/to/scan.nxs \
  --mode pose \
  --out aligned.nxs
```

## Choose the correction path

Use the correction path that matches the scan problem you want to solve. The
same command writes a reconstructed `.nxs` file with alignment metadata.

- Use `--mode pose` when the sample appears to move from projection to
  projection.
- Use `--mode cor` when the detector centre or centre of rotation appears
  wrong.
- Use `--mode auto --gauge-policy anchor_mean` only when you deliberately want
  expert mixed setup and pose correction.

Pose-only correction can produce a good reconstruction while absorbing setup
errors into the pose parameters. Use setup-specific modes when you need
calibrated geometry, not only a better volume.

## Ingest TIFF projection stacks

When data arrive as TIFF projection stacks, ingest them into the standard
contract before reconstruction or alignment:

```bash
uv run tomojax ingest ./projections \
  --angles angles.csv \
  --du 0.65 \
  --dv 0.65 \
  --out scan.nxs
```

## Next steps

For more detail about choosing `pose`, `cor`, or `auto`, see
[`alignment-guide.md`](alignment-guide.md). For support boundaries, see
[`support-matrix.md`](support-matrix.md) and
[`known-limitations.md`](known-limitations.md).
