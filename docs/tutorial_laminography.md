# Laminography End-to-End Tutorial (Draft)

This guide mirrors the tomography walkthrough but focuses on the laminography
workflow: generating a thin specimen, simulating a tilted-axis scan, and running
FBP/FISTA reconstructions with the new geometry support.

> **Prerequisites**
> - You have a working TomoJAX checkout and the `pixi` environment installed.
> - JAX can see either a GPU or the CPU backend (set `JAX_PLATFORM_NAME=cpu`
>   if your system lacks CUDA libraries).

## 1. Environment

```bash
pixi install          # one-time dependency install
pixi shell            # enter the managed environment
pixi run test-cpu     # optional sanity check (use test-gpu if CUDA is ready)
```

## 2. Simulate a Laminography Dataset

We use the `lamino_disk` phantom, which clamps voxel intensities to a thin slab
orthogonal to the laminography rotation axis.

```bash
pixi run simulate \
  --out runs/lamino_demo.nxs \
  --nx 256 --ny 256 --nz 256 \
  --nu 256 --nv 256 --n-views 360 \
  --geometry lamino \
  --tilt-deg 35 --tilt-about x \
  --phantom lamino_disk \
  --lamino-thickness-ratio 0.15 \
  --n-cubes 80 --n-spheres 80 \
  --min-size 4 --max-size 14 \
  --seed 3
```

Key metadata written to the NXtomo file:

- `/entry/geometry/type = "lamino"`
- `/entry/geometry/geometry_meta_json` with `tilt_deg` and `tilt_about`
- Ground-truth volume constrained to a thin slab.

## 3. Filtered Backprojection (FBP)

The reconstruction CLI automatically rebuilds either a parallel or laminography
geometry from the metadata:

```bash
pixi run python -m tomojax.cli.recon \
  --data runs/lamino_demo.nxs \
  --algo fbp \
  --filter ramp \
  --views-per-batch auto \
  --out runs/lamino_demo_fbp.nxs
```

The result is stored under `/entry/processing/tomojax/volume`. Use tools such
as `scripts/volume_slice.py` or your favourite viewer to inspect the slices.

## 4. TV-Regularised FISTA Reconstruction

```bash
pixi run python -m tomojax.cli.recon \
  --data runs/lamino_demo.nxs \
  --algo fista \
  --iters 75 \
  --lambda-tv 1e-3 \
  --tv-prox-iters 20 \
  --views-per-batch auto \
  --projector-unroll 2 \
  --out runs/lamino_demo_fista.nxs
```

Monitor the log output for the FISTA loss curve. If you are memory-limited,
reduce `views-per-batch`, disable checkpointing, or switch to `gather-dtype
bf16/fp16`.

## 5. Optional: Misalignment Stress Test

```bash
pixi run misalign \
  --data runs/lamino_demo.nxs \
  --out runs/lamino_demo_misaligned.nxs \
  --rot-deg 0.5 \
  --trans-px 4 \
  --seed 11

pixi run align \
  --data runs/lamino_demo_misaligned.nxs \
  --outer-iters 4 --recon-iters 20 \
  --lambda-tv 5e-3 --tv-prox-iters 15 \
  --opt-method gn --gn-damping 1e-3 \
  --views-per-batch auto \
  --out runs/lamino_demo_aligned.nxs
```

The alignment CLI automatically honours the stored laminography tilt. Compare
`/entry/processing/tomojax/align/thetas` before/after to quantify alignment
improvements.

## 6. Tips & Next Steps

- For analytic benchmarking, keep the voxel spacing isotropic and the detector
  sampling identical along the thin axis to avoid anisotropic discretisation
  artefacts.
- If you plan to extend this tutorial (e.g. multiresolution alignment or
  custom phantoms), copy the commands above and adjust only the relevant
  parameters.
- Contributions welcome: open an issue with feedback or ideas for additional
  laminography examples.

---

*Last updated:* WIP — expect changes as laminography support evolves.
