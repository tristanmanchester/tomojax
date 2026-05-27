# Tests

Coverage:

- Package imports and private-module import guardrails
- CLI routing for `inspect`, `validate`, `ingest`, `preprocess`, `convert`, `recon`, `simulate`, `align --mode cor`, and `align --optimise-dofs`
- TIFF stack, NPZ, and NX/HDF5 projection payload roundtrips
- NXtomo `image_key` splitting and TIFF stack preprocessing
- FBP and FISTA reconstruction smoke tests
- Deterministic synthetic phantoms, simulation metadata, and artefact contracts
- Numerical correctness of Barron loss kernel and geometry rotation helpers
