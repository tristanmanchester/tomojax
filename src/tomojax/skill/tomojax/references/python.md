# Python in the local worker

The `python` tool runs your code in the worker process that holds the session's data, so a sweep or a custom measure runs beside the scans and volumes without copying them. Use it for compound work: several reconstructions in one call, loops over slices, measures the typed tools do not have.

## Namespace

| Name | What it is |
| --- | --- |
| `tj`, `np` | `tomojax` and NumPy |
| `handles` | Live IDs to their objects: `handles["@scan-…"]` is a `tj.Scan`, `handles["@recon-…"]` a `tj.Reconstruction`, `handles["@feature-…"]` a feature with a boolean `.mask` (and `.box`, `.inside` for its bounding box), `handles["@array-…"]` a NumPy array. `handles["@sim-….scan"]` selects a member, as the typed tools do. |
| `register(obj)` | Publish a TomoJAX object or array as a new handle and return its ID |
| `register_feature(mask, like=..., name=...)` | Publish a boolean mask as a feature on the grid of `like` (an ID or object) and return its ID |
| `view(source, mode=, axis=, index=, roi=, stride=, other=, marks=, features=, ticks=, limits=, grid=)` | Render planes like the `view` tool, add the image to the reply and return where each mark landed; marks here take array indices, not pixels |
| `show(plane, marks=)` | Add an image of a 2D array to the reply; marks use the array's own `[row, col]` indices |

These are the worker's own objects: derive new arrays from them and leave theirs unchanged.

The last expression is the result. Arrays of up to 16 values come back as lists; larger ones as their shape, dtype, minimum, maximum and mean, so `register` an array you want to keep or view. Printed output is clipped to its first and last 10,000 characters; write long output to a file instead. An error reports the exception and the lines of your code it passed through; set `traceback=true` for the full trace into TomoJAX.

Variables persist between calls. `reset=true` clears them; a worker restart (`control` with `action="restart"`) clears them and closes every handle. A long call runs as an MCP task when the client supports tasks: cancelling it stops at the next TomoJAX checkpoint, so a loop of `tj.reconstruct` calls stops within one iteration, but pure Python or NumPy work runs on until it returns.

The examples below assume a parallel scan, `scan = handles[scan_id]`, or `scan = handles[simulation_id].scan` for a simulation's acquisition.

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
