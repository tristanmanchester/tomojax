# Runnable examples

Run these from the checkout root after [installation](../docs/installation.md).
They use public APIs and write no data unless an output is requested.

## Project and reconstruct a volume

```bash
uv run --no-sync python examples/simulate_and_reconstruct.py
```

[simulate_and_reconstruct.py](simulate_and_reconstruct.py) creates a 32³
Shepp–Logan object, generates 60 parallel projections with the Joseph model,
and runs 24 CGLS iterations with that same model. It prints the reference and
reconstruction shapes and the full-volume relative L2 error.
The iteration budget may end before convergence. The JAX backend runs on CPU or
an available GPU; prefix the command with `JAX_PLATFORMS=cpu` to require CPU.

Simulation and reconstruction share a discretization, so this is a usage example,
not independent accuracy evidence. See the [measurements](../docs/measurements.md)
for independent data and accepted-result comparisons.

## Reproduce the README figure

Install the optional plotting group in your chosen environment. For CPU:

```bash
uv sync --locked --extra cpu --group examples --no-dev
JAX_PLATFORMS=cpu uv run --no-sync python examples/plot_reconstruction.py \
  --size 64 --views 90 --iterations 40 \
  --out images/reconstruction-example.png
```

The checked-in figure was generated on CUDA using:

```bash
uv sync --locked --extra cuda12 --group examples --no-dev
uv run --no-sync python examples/plot_reconstruction.py \
  --size 64 --views 90 --iterations 40 \
  --out images/reconstruction-example.png
```

CPU and CUDA use matched models but floating-point reductions can differ; byte
identical images across backends or plotting-library versions are not expected.
Use another output path to preserve the checked-in illustration.

[plot_reconstruction.py](plot_reconstruction.py) writes the PNG and an adjacent
JSON file containing solver settings, full-volume error, termination, display
ranges, versions, and script/image hashes. Both planes share attenuation and
absolute-error scales, with no per-panel contrast fitting. Error is measured
before display and uses the entire volume. The figure includes negative
reconstruction values rather than clipping them away.

## Align a scan with per-view motion

```bash
uv sync --locked --extra cuda12 --group examples --no-dev
uv run --no-sync python examples/align_misaligned_scan.py
```

[align_misaligned_scan.py](align_misaligned_scan.py) simulates a 96³, 240-view,
full-turn 30° laminography scan of continuous Gaussian blobs, with analytic
line integrals at randomly perturbed poses (±1° rotations, ±2 px shifts per
view) and 0.2% noise. The data match no voxel discretisation. It reconstructs
once with the nominal poses (CGLS) and once jointly with the poses using
`tomojax.alignment.align` and `coupled_pose_config()`, then writes
`images/alignment-example.png` and its JSON metrics. On an RTX 4070 Laptop GPU
it recovers rotations to 0.0027° RMS in about 14 s, lowering the volume error
from 0.57 to 0.063. Pose errors are reported after removing their common
offset, which only moves the object.
