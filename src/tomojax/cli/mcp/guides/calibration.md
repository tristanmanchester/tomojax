# Detector and sample models

Build a phantom on a known grid to check image error alongside held-out
projection error. The phantom builders use (nx, ny, nz) arguments and return
volumes in (z, y, x) order.

```python
from tomojax import datasets
dims = (32, 32, 32)
objects = [builder(*dims) for builder in
           (datasets.blobs, datasets.cube, datasets.sphere, datasets.rotated_centered_cube,
            datasets.shepp_logan_3d, datasets.lamino_disk, datasets.random_cubes_spheres)]
cfg = datasets.SimConfig(nx=32, ny=32, nz=32, nu=32, nv=32, n_views=48)
phantom = datasets.make_phantom(cfg)
synthetic = datasets.simulate(cfg)
```

With repeated measured flat and dark frames, estimate signal-dependent noise
then generate a reproducible detector exposure.

```python
model = datasets.DetectorModel.from_flats(flats, darks)
exposure: datasets.DetectorSimulation = model.simulate(scan.projections, flat_frames=8, dark_frames=8, seed=3)
float(np.std(exposure.projections - exposure.expected))
```

## Beam spectrum and phase

Given a measured phase stack and a boolean detector air mask, remove the air
ramp after unwrapping. For absorption data, inspect a beam-hardening response
before applying it. Spectrum modelling requires the optional physics libraries.

```python
from tomojax import corrections
unwrapped = corrections.unwrap_phase(phase)
phase_without_ramp = corrections.remove_phase_ramp(unwrapped, air_mask=air_mask)
energies, signal = corrections.tube_spectrum(80)
physical = corrections.BeamHardening.from_spectrum(energies, signal, "Al")
empirical = corrections.BeamHardening.from_polynomial([0, 1, 0.02], upto=2)
p = np.linspace(0, 2, 50)
residual = empirical.inverse(physical.curve(p)) - p
```

Inspect the settings and findings recorded by a corrected scan with
`saved = [step.to_dict() for step in scan.corrections]` and restore a record
with `corrections.Correction.from_dict(saved[0])`.
