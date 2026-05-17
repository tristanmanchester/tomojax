# Synthetic Tomography Workflow

The product-facing synthetic workflow is small and reproducible: generate a
standard projection dataset with `tomojax simulate`, then reconstruct it with
`tomojax recon`.

```bash
uv run tomojax simulate \
  --out synthetic_scan.nxs \
  --nx 64 --ny 64 --nz 64 \
  --nu 64 --nv 64 \
  --n-views 64 \
  --phantom random_shapes

uv run tomojax recon synthetic_scan.nxs --out synthetic_recon.nxs
```

The equivalent public-Python path is kept in
[`examples/simulate_and_reconstruct.py`](../examples/simulate_and_reconstruct.py).
It imports only from package facades such as `tomojax.geometry`,
`tomojax.forward`, and `tomojax.recon`.

Historical synthetic gates that fixed the volume to truth, swept hard geometry
cases, or generated article diagnostics were removed from the product spine and
kept in the development archive. They are useful research evidence, but they are
not part of the supported user workflow.
