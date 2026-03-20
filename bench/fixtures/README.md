# Benchmark fixtures

These small `.npz` bundles are tracked on purpose.

They keep the benchmark harness deterministic and avoid having benchmark runs regenerate
projection data in-process before the first timed call. That helps keep the first-run
measurement closer to a true compile+execute path.

Each fixture stores:

- geometry metadata;
- detector metadata;
- thetas;
- the ground-truth volume;
- measured projections;
- optional ground-truth alignment parameters.

Normal benchmark runs should load these fixtures read-only.
