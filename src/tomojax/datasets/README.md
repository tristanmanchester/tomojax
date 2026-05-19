# tomojax.datasets

`tomojax.datasets` owns deterministic synthetic generation for the supported
`tomojax simulate` workflow. It exposes the simulation config/results,
validated simulation artefact settings, simple procedural phantoms, and sidecar
loaders used by examples and tests.

## Public API

- `SimConfig`
- `SimMetadata`
- `SimulatedData`
- `SimulationArtefacts`
- `apply_simulation_artefacts`
- `make_phantom`
- `simulate`
- `simulate_to_file`
- `validate_simulation_artefacts`
- simple phantom helpers such as `shepp_logan_3d`, `cube`, `sphere`, `blobs`,
  `random_cubes_spheres`, and `lamino_disk`
- sidecar metadata loaders for generated datasets

Historical synthetic benchmark manifests, synthetic128 spec loaders, recovery
criteria tables, and publication artifact writers are not part of the product
facade. They were moved to the development archive.

## Dependency policy

This package may depend on low-level retained data-generation helpers and public
geometry/forward/IO surfaces. Product code should import from
`tomojax.datasets`, not from historical benchmark modules.
