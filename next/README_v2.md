% TomoJAX v2 Overview

Package name: `tomojax_next` (scoped under `next/src/`).

Goals
- Clean, stable APIs for projection, reconstruction (FBP, FISTA-TV), alignment.
- Beamline-friendly IO: HDF5 with NeXus NXtomo + TomoJAX extras.
- Geometry abstraction for parallel CT and laminography.
- Reproducible CLIs and tests.

Pixi usage
- Enter env: `pixi shell`
- Run tests: `pixi run test-next` (sets `PYTHONPATH=next/src`)
- Convert example: `pixi run convert-example`

GPU/CUDA
- On DLS systems, load CUDA before using GPU: `module load cuda`
- Verify backend/devices: `pixi run test-gpu`
- Force CPU if needed: `pixi run test-cpu` (sets `JAX_PLATFORM_NAME=cpu`)

CLIs (module executes)
- Convert: `python -m tomojax_next.cli.convert --in data/sim.npz --out data/sim.nxs`
- Simulate: `python -m tomojax_next.cli.simulate --nx 64 --ny 64 --nz 64 --nu 64 --nv 64 --n-views 64 --geometry parallel --phantom shepp --out data/sim_parallel.nxs`
- Simulate random shapes: `python -m tomojax_next.cli.simulate --nx 64 --ny 64 --nz 64 --nu 64 --nv 64 --n-views 64 --phantom random_shapes --n-cubes 20 --n-spheres 20 --min-size 4 --max-size 16 --out data/sim_random.nxs`
- Reconstruct (FBP): `python -m tomojax_next.cli.recon --data data/sim_parallel.nxs --algo fbp --filter ramp --out runs/fbp.nxs`
- Reconstruct (FISTA): `python -m tomojax_next.cli.recon --data data/sim_parallel.nxs --algo fista --iters 50 --lambda-tv 0.005 --out runs/fista.nxs`
- Align: `python -m tomojax_next.cli.align --data data/sim_parallel.nxs --outer-iters 5 --recon-iters 10 --lambda-tv 0.005 --out runs/align.nxs`

Notes
- v2 lives in `next/` to avoid breaking existing code during development.
- Default dtypes are float32; angles stored in degrees on disk.
