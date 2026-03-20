# Benchmark fixtures

Only the tiny smoke fixture is tracked in Git.

The representative screening and canary profiles are intentionally too large to vendor as checked-in
`.npz` files. They are generated on first use into `bench/data/` and should then be left in place on
persistent worker storage.

Each fixture stores:

- geometry metadata;
- detector metadata;
- thetas;
- the ground-truth volume;
- measured projections;
- optional ground-truth alignment parameters.

Normal benchmark runs should load these fixtures read-only.
