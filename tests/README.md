# TomoJAX product tests

The retained suite is intentionally focused. It proves supported user workflows and public boundaries without restoring the historical benchmark, diagnostic, v1-parity, or article-artifact corpus.

Current coverage:

- public package facades import cleanly;
- the grouped `tomojax` CLI exposes only product commands;
- benchmark and verification compatibility namespaces are absent from the shipped tree;
- lower-level data implementation code is private under `tomojax._data` and is not used from product-facing tests, examples, or CLI modules;
- private-module import guardrails still run;
- TIFF stack, NPZ, and NX/HDF5 projection payloads roundtrip through public IO;
- NXtomo `image_key` sample/flat/dark splitting and TIFF stack preprocessing produce the expected absorption or transmission output;
- FBP and FISTA reconstruction smoke tests produce finite tiny volumes;
- deterministic synthetic phantoms, simulation metadata, artefact metadata, and loadable synthetic outputs keep their public contracts;
- product CLI smoke tests cover `inspect`, `validate`, `ingest`, `preprocess`, `convert`, `recon`, `simulate`, and cheap `align --mode cor` routing.

Historical benchmark, v1-parity, article artifact, and diagnostic runner tests were archived with the removed code. Restore tests only when they prove a supported product workflow or a public compatibility boundary.
