# TomoJAX

TomoJAX is a general tomography library: parallel-beam, cone-beam and
laminography projection, reconstruction and alignment, in JAX with CUDA
kernels. The aim is to be faster and more accurate than ASTRA and TIGRE on
real scans while `tj.load`, `tj.reconstruct` and `tj.align` just work.

## Constraints

- The backprojector stays the exact transpose of the forward projector; an
  approximate one is out of scope.
- Improvements must be general, not tuned to one benchmark, phantom or dataset.
- Breaking changes are fine before 1.0; no compatibility shims.
- A faster or lower-residual result only counts once the reconstruction or pose
  error has been checked too.

## Checks

`just ci` is the CPU gate (about 9 minutes); `just test-cuda` runs everything on
a CUDA GPU (about 10 minutes). A change is ready when the tests it touches
pass, on the GPU if it touches GPU code. The maintainer runs `just ci` and the
full GPU suite when merging into main, so a ready commit on a branch is the
whole hand-off.
Conventions are in CONTRIBUTING.md, measured numbers in docs/measurements.md,
and the comparisons with ASTRA and TIGRE in bench/.

Say what changed and how it was checked in a few plain sentences. Evidence
(logs, scripts, archived results) stays out of the repository.
