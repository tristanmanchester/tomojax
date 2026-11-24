# Python in the local worker

The last expression is the result. Variables persist until reset or restart.
Use the handle IDs from status as dictionary keys. For example,
`scan = handles[scan_id]` reads a scan and `scan = handles[simulation_id].scan`
reads a simulation's acquisition. Handle arrays are read-only. Register a new
object to use it in typed tools. Registration takes a snapshot immediately.
Use `show(plane)` to return a PNG. The examples below assume a parallel scan.

## Direct solver

```python
from tomojax import recon
args = (scan.geometry, scan.grid, scan.detector, scan.projections)
volume, info = recon.cgls(*args, config=recon.CGLSConfig(iterations=3))
volume_id = register(volume)
show(volume[volume.shape[0] // 2])
volume_id
```

Pass the resulting array ID as the volume of a typed project call.
Read that call's output in Python with `handles[projection_id]`.

## Memory-mapped reconstruction

```python
out = np.lib.format.open_memmap("reconstruction.npy", mode="w+", dtype="float32",
                                shape=scan.grid.shape)
reconstruction = tj.reconstruct(scan, "fbp", out=out)
out.flush()
float(out.max())
```

## Acquisition metadata

For a saved scan at `scan.nxs`, read the calibrated geometry and the acquisition record.

```python
from tomojax import io
record: io.ProjectionDataset = io.load_dataset("scan.nxs")
grid, detector, geometry = io.build_geometry_from_dataset_metadata(record.geometry_inputs())
metadata = record.to_nxtomo_metadata()
{"kind": record.geometry_type, "views": len(record.angles), "shape": grid.shape}
```

For a TIFF acquisition with recorded degree angles, use
`record = io.load_tiff_stack("radiographs", angles=scan.angles)`.
`record.image_key` identifies sample, flat and dark frames when supplied.
A parallel dataset missing detector metadata can use
`io.save_dataset("completed.nxs", record.with_parallel_detector())`.
When reading an NXtomo payload directly, `io.ProjectionDataset.from_nxtomo(payload)`
constructs the same record.

## Custom geometry

This example adds a measured lab-frame translation to each parallel projection. A custom scan
carries the acquisition, so typed tools can use its ID immediately.

```python
from dataclasses import dataclass

@dataclass
class DriftGeometry(tj.ParallelGeometry):
    def pose_matrix(self, i):
        pose = np.array(super().pose_matrix(i))
        pose[0, 3] += 0.1 * i * self.detector.du
        return tuple(map(tuple, pose))

geometry = DriftGeometry(scan.grid, scan.detector, scan.angles)
custom = register(tj.Scan(scan.projections, geometry))
custom
```

## Torch

This example needs PyTorch with CUDA. The adjoint maps the predicted detector
images back to the reconstruction grid. Autograd uses the same transpose.

```python
import torch
from tomojax.torch import Projector
operator = Projector(scan.geometry)
x = torch.ones(scan.grid.shape, device="cuda", requires_grad=True)
prediction = operator.forward(x)
prediction.sum().backward()
bool(torch.allclose(x.grad, operator.adjoint(torch.ones_like(prediction))))
```
