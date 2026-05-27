# Real Laminography Workflow

Start with inspection and validation, then use alignment for pose correction,
detector-centre/COR correction, or mixed setup and pose correction.

```bash
uv run tomojax inspect /path/to/scan.nxs
uv run tomojax validate /path/to/scan.nxs
uv run tomojax align --data /path/to/scan.nxs \
  --mode pose \
  --out aligned.nxs
```

## Choose the correction path

Each mode writes a reconstructed `.nxs` file with alignment metadata.

- Use `--mode pose` when the sample appears to move from projection to
  projection.
- Use `--mode cor` when the detector centre or centre of rotation appears
  wrong.
- Use `--mode auto --gauge-policy anchor_mean` for mixed setup and pose
  correction.

Pose-only correction can produce a good reconstruction while absorbing setup
errors into the pose parameters. Use setup-specific modes when you need
calibrated geometry, not only a better volume.

## Ingest TIFF projection stacks

For TIFF projection stacks, ingest into NXtomo format before reconstruction
or alignment:

```bash
uv run tomojax ingest ./projections \
  --angles angles.csv \
  --du 0.65 \
  --dv 0.65 \
  --out scan.nxs
```

## Next steps

For more detail about choosing `pose`, `cor`, or `auto`, see
[`alignment-guide.md`](alignment-guide.md). For limitations, see
[`support-matrix.md`](support-matrix.md) and
[`known-limitations.md`](known-limitations.md).
