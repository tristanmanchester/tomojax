"""CGLS with its volumes in host memory, for cone scans larger than the device.

The solve is conjugate gradients on the normal equations, as for streamed
projections: one sweep applies ``A^T A`` to the direction and also returns
``||A d||^2``, so no projection-sized vector is ever stored. The iterate,
normal residual, direction and ``A^T A`` direction are host volumes read and
written one slab at a time; measurements are read only to form the initial
and recomputed normal residuals. Stopping and verification follow the
resident solvers; a verification takes two sweeps, the recomputed residual
and its FP32 cancellation estimate, so each holds one channel of rows.
Sources and sinks work in place on uploaded slabs, so besides the ring of
rows only a few slab-sized arrays occupy the device at a time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax.core.cone import ConeModel
from tomojax.core.host_volume import fill, host_volume, scratch_directory

from ._cgls_stopping import precision_stalled, termination
from ._cone_slabs import Batch, RowReader, cone_problem, host_projections, host_volumes
from ._host_arrays import validate_host_arrays
from ._quadratic import gradient_normal

if TYPE_CHECKING:
    from collections.abc import Callable

    import jax

    from tomojax.geometry import Detector, Geometry, Grid

    from .cgls import CGLSConfig

_EPS = float(np.finfo(np.float32).eps)
_TINY = np.finfo(np.float32).tiny


def solve_on_host(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: Any,
    initial: Any,
    cfg: CGLSConfig,
    *,
    model: Any,
    backend: str,
    poses: jax.Array,
    shared: bool,
    out: np.ndarray | None,
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Solve with host volumes when the device can't hold the solver or ``out`` is given.

    Returns None for the device solvers: other geometries, or views ``shared``
    among several devices.
    """
    cone, rows = not shared and isinstance(model, ConeModel), isinstance(projections, RowReader)
    if (out is not None or rows) and not cone:
        raise ValueError("cgls: out and row readers need a cone-beam geometry on one device")
    if not cone or (out is None and not rows and not host_volumes(grid, 6, backend)):
        return None
    start = None if initial is None else np.asarray(initial)
    return solve(
        geometry, grid, detector, projections, start, cfg,
        model=model, backend=backend, poses=poses, out=out,
    )  # fmt: skip


def solve(
    geometry: Geometry,
    grid: Grid,
    detector: Detector,
    projections: np.ndarray,
    initial: np.ndarray | None,
    cfg: CGLSConfig,
    *,
    model: ConeModel,
    backend: str,
    poses: jax.Array,
    out: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Run validated CGLS settings with host volumes, in ``out`` or a C-ordered volume."""
    projections = host_projections(projections, "cgls")
    problem = cone_problem(
        geometry, grid, detector, projections, model=model, backend=backend, poses=poses,
        batch=cfg.views_per_batch, channels=1,
    )  # fmt: skip
    solver = _HostCGLS(problem, cfg, initial, out)
    info = solver.run()
    volume = problem.result(solver.x)
    return volume, {**info, "projector_backend": backend, "projector_model": model}


class _HostCGLS:
    """One solve's host volumes and sweeps; :meth:`run` mirrors the resident loop."""

    def __init__(
        self, problem: Any, cfg: CGLSConfig, initial: np.ndarray | None, out: np.ndarray | None
    ) -> None:
        self.problem, self.cfg = problem, cfg
        self.plan, self.sweep, self.xp = problem.plan, problem.sweep, problem.engine.xp
        grid = self.plan.grid
        shape = (grid.nz, grid.ny, grid.nx)
        data = problem.data
        self.zero = initial is None
        self.x = fill(validate_host_arrays(data, out, shape, "cgls", order="C"), initial)
        work = [host_volume(shape, scratch_directory(out)) for _ in range(3)]
        self.X, self.G, self.D, self.W = (problem.volume(a) for a in (self.x, *work))
        self.views = np.arange(data.shape[0])
        self.damp2 = np.float32(cfg.damping) ** 2
        self.smooth2 = np.float32(cfg.gradient_damping) ** 2
        self.total = problem.engine.norm2
        self.keep = problem.keep

    def _block(self, k: int, centre: Any, halo: Callable[[int], Any], *, below: bool) -> Any:
        """Slab ``k`` with the next plane above (and, with ``below``, the one below)."""
        (z0, z1), nz = self.plan.planes(k), self.plan.grid.nz
        parts = [halo(z0 - 1)] if below and z0 > 0 else []
        parts += [centre[: z1 - z0, :, :]] + ([halo(z1)] if z1 < nz else [])
        return self.xp.concatenate(parts, axis=0)

    def smoothing(
        self, k: int, centre: Any, halo: Callable[[int], Any], *, absolute: bool = False
    ) -> Any:
        """``gradient_damping^2 D^T D`` (or its absolute weights) on slab ``k``'s planes."""
        block = self._block(k, centre, halo, below=True)
        if absolute:
            block = self.xp.abs(block)
        spacing = (self.plan.grid.vx, self.plan.grid.vy, self.plan.grid.vz)
        applied = gradient_normal(block, spacing, absolute_weights=absolute, xp=self.xp)
        z0, z1 = self.plan.planes(k)
        low = int(z0 > 0)
        return self.smooth2 * applied[low : low + z1 - z0, :, :]

    def edges(self, k: int, centre: Any, halo: Callable[[int], Any]) -> float:
        """``||D d||²`` over the edges from slab ``k``'s planes, the z edge above included."""
        grid, (z0, z1) = self.plan.grid, self.plan.planes(k)
        planes = centre[: z1 - z0, :, :]
        total = sum(
            self.total(self.xp.diff(planes, axis=axis) / step)
            for axis, step in ((2, grid.vx), (1, grid.vy))
            if planes.shape[axis] > 1
        )
        block = self._block(k, centre, halo, below=False)
        if block.shape[0] > 1:
            total += self.total(self.xp.diff(block, axis=0) / grid.vz)
        return total

    def gradient(self) -> tuple[float, float]:
        """Replace G by ``A^T (y - A x) - damping^2 x``; return its norm² and the change²."""
        plan, stats = self.plan, {"gamma": 0.0, "gap": 0.0}

        def source(k: int) -> tuple[Any, ...]:
            return () if self.zero else (self.X.get(*plan.planes(k), retain=not self.keep),)

        def finish(batch: Batch, first: int, stop: int, rows: tuple[Any, ...]) -> None:
            rows[0][...] = self.problem.measured(batch, first, stop) - rows[0]

        def sink(k: int, accumulated: tuple[Any, ...], sources: tuple[Any, ...]) -> None:
            z0, z1 = plan.planes(k)
            value = accumulated[0][: z1 - z0, :, :]
            if not self.zero:
                sources = sources or (self.X.get(z0, z1),)
            if self.smooth2 and sources:
                value -= self.smoothing(k, sources[0], self.iterate_plane)
            if self.damp2 and sources:
                current = sources[0][: z1 - z0, :, :]
                current *= self.damp2
                value -= current
            stats["gamma"] += self.total(value)
            previous = self.G.view(z0, z1)
            previous -= value
            stats["gap"] += self.total(previous)
            self.G.put(z0, z1, value)

        self.sweep.run(
            self.views, source=source, finish=finish, sink=sink,
            prefetch=self._prefetch(self.X), keep=self.keep,
        )  # fmt: skip
        self.zero = False
        return stats["gamma"], stats["gap"]

    def bounded(self) -> bool:
        """Whether every voxel of G lies within the FP32 cancellation estimate of x."""
        plan, xp, within = self.plan, self.xp, [True]

        def finish(batch: Batch, first: int, stop: int, rows: tuple[Any, ...]) -> None:
            rows[0][...] = xp.abs(self.problem.measured(batch, first, stop)) + xp.abs(rows[0])

        def sink(k: int, accumulated: tuple[Any, ...], sources: tuple[Any, ...]) -> None:
            z0, z1 = plan.planes(k)
            cancellation = accumulated[0][: z1 - z0, :, :]
            sources = sources or (self.X.get(z0, z1),)
            if self.smooth2:
                cancellation += self.smoothing(k, sources[0], self.iterate_plane, absolute=True)
            if self.damp2:
                current = xp.abs(sources[0][: z1 - z0, :, :])
                current *= self.damp2
                cancellation += current
            cancellation *= np.float32(8 * _EPS)
            within[0] &= not bool(xp.any(xp.abs(self.G.view(z0, z1)) > cancellation))

        def source(k: int) -> tuple[Any, ...]:
            return (self.X.get(*plan.planes(k), retain=not self.keep),)

        self.sweep.run(
            self.views, source=source, finish=finish, sink=sink,
            prefetch=self._prefetch(self.X), keep=self.keep,
        )  # fmt: skip
        return within[0]

    def normal(self, beta: float | None) -> tuple[float, float]:
        """Set D to ``G + beta D`` (``G`` for None), W to the normal operator on D.

        Returns ``||A D||²`` and ``||D||²``; :attr:`edge_energy` becomes ``||∇D||²``.
        """
        plan, stats = self.plan, {"energy": 0.0, "direction": 0.0}
        updated: set[int] = set()
        self.edge_energy = 0.0

        def halo(z: int) -> Any:
            # A plane of the new direction, whether or not its slab is updated yet.
            plane = self.D.view(z, z + 1)
            if z // plan.depth in updated:
                return plane
            if beta is None:
                return self.G.view(z, z + 1)
            plane *= np.float32(beta)
            plane += self.G.view(z, z + 1)
            return plane

        def source(k: int) -> tuple[Any, ...]:
            updated.add(k)
            z0, z1 = plan.planes(k)
            direction = self.G.view(z0, z1)
            if beta is not None:
                previous = self.D.view(z0, z1)
                previous *= np.float32(beta)
                previous += direction
                direction = previous
            self.D.put(z0, z1, direction)
            stats["direction"] += self.total(direction)
            return (self.D.kernel(direction),)

        def finish(_batch: Batch, _first: int, _stop: int, rows: tuple[Any, ...]) -> None:
            stats["energy"] += self.total(rows[0])

        def sink(k: int, accumulated: tuple[Any, ...], sources: tuple[Any, ...]) -> None:
            value, direction = accumulated[0], (sources or (self.D.get(*plan.planes(k)),))[0]
            if self.smooth2:
                z0, z1 = plan.planes(k)
                value[: z1 - z0, :, :] += self.smoothing(k, direction, halo)
                self.edge_energy += self.edges(k, direction, halo)
            if self.damp2:
                direction *= self.damp2
                value += direction
            self.W.put(*plan.planes(k), value)

        self.sweep.run(
            self.views, source=source, finish=finish, sink=sink,
            prefetch=self._prefetch(self.G, self.D), keep=self.keep,
        )  # fmt: skip
        return stats["energy"], stats["direction"]

    def iterate_plane(self, z: int) -> Any:
        """Plane ``z`` of the iterate on the device."""
        return self.X.view(z, z + 1)

    def _prefetch(self, *volumes: Any) -> Callable[[int], None]:
        """A callback starting to read slab ``k`` of each volume."""

        def start(k: int) -> None:
            for volume in volumes:
                volume.prefetch(*self.plan.planes(k))

        return start

    def update(self, alpha: np.float32) -> tuple[float, bool, float]:
        """``x += alpha d``, ``g -= alpha w``; return ``||g||²``, voxel roundoff and ``||x||²``."""
        xp, gamma, roundoff, size = self.xp, 0.0, True, 0.0
        ahead = self._prefetch(self.D, self.X, self.W, self.G)
        for k in range(self.plan.count):
            z0, z1 = self.plan.planes(k)
            if k + 1 < self.plan.count:
                ahead(k + 1)
            step = self.D.view(z0, z1)
            step *= alpha
            current = self.X.view(z0, z1)
            current += step
            bound = xp.maximum(xp.abs(current), _TINY)
            bound *= np.float32(_EPS)
            roundoff &= not bool(xp.any(xp.abs(step) > bound))
            applied = self.W.view(z0, z1)
            applied *= alpha
            residual = self.G.view(z0, z1)
            residual -= applied
            gamma, size = gamma + self.total(residual), size + self.total(current)
            self.X.put(z0, z1, current)
            self.G.put(z0, z1, residual)
        return gamma, roundoff, size

    def run(self) -> dict[str, object]:
        """Iterate as the resident streamed solver does; return its record."""
        cfg = self.cfg
        gamma, _ = self.gradient()
        first, initial_norm = gamma, float(np.sqrt(gamma))
        threshold = cfg.atol + cfg.rtol * initial_norm
        state = {"failed": not np.isfinite(gamma), "roundoff": False, "verified": True}
        count, recomputations, beta = 0, 0, None
        while (
            count < cfg.iterations
            and gamma > threshold**2
            and not (state["failed"] or state["roundoff"])
        ):
            energy, direction_norm2 = self.normal(beta)
            denominator = (
                energy
                + float(self.damp2) * direction_norm2
                + float(self.smooth2) * self.edge_energy
            )
            if not np.isfinite(denominator) or denominator <= 0:
                state["failed"] = True
                break
            alpha = np.float32(gamma / denominator)
            gamma_new, state["roundoff"], size = self.update(alpha)
            verify = (
                state["roundoff"]
                or float(alpha) ** 2 * direction_norm2 <= _EPS**2 * size
                or gamma_new <= threshold**2
                or (((count + 1) % 16 == 0 or state["verified"]) and gamma_new <= _EPS * first)
            )
            restart = False
            if verify:
                restart, gamma_new = self.verify(gamma, gamma_new, threshold, state)
                recomputations += 1
            count += 1
            if not np.isfinite(gamma_new):
                state["failed"] = True
                break
            beta = None if restart else gamma_new / gamma
            gamma, state["verified"] = gamma_new, verify
        for volume in (self.X, self.G, self.D, self.W):
            volume.close()
        return self.record(gamma, threshold, initial_norm, count, recomputations, state)

    def verify(
        self, gamma: float, gamma_new: float, threshold: float, state: dict[str, bool]
    ) -> tuple[bool, float]:
        """Recompute the normal residual; return whether to restart and its norm²."""
        exact, gap = self.gradient()
        limited = self.bounded()
        candidate = gamma_new <= threshold**2
        restart = candidate or state["roundoff"] or limited or gap > 0.01 * gamma_new
        stalled = precision_stalled(exact, gamma, state["verified"], limited)
        state["roundoff"] = state["roundoff"] or bool(stalled)
        return restart, exact

    def record(
        self,
        gamma: float,
        threshold: float,
        initial_norm: float,
        count: int,
        recomputations: int,
        state: dict[str, bool],
    ) -> dict[str, object]:
        """The resident solvers' record, plus the slab depth and device traffic."""
        converged = bool(not state["failed"] and gamma <= threshold**2)
        return {
            "effective_iterations": count,
            "converged": converged,
            "termination": termination(
                failed=state["failed"], converged=converged, roundoff=state["roundoff"]
            ),
            "normal_residual_norm": float(np.sqrt(gamma)),
            "normal_residual_is_recomputed": bool(state["verified"]),
            "residual_recomputations": recomputations,
            "initial_normal_residual_norm": initial_norm,
            "normal_residual_tolerance": float(threshold),
            "joseph_interpolation": "linear",
            "views_per_batch": self.sweep.batch,
            "formulation": "host_normal_equations",
            "damping": float(self.cfg.damping),
            "gradient_damping": float(self.cfg.gradient_damping),
            "host_slabs": True,
            "slices_per_batch": self.plan.depth,
            "bytes_to_device": self.problem.engine.traffic["to_device"],
            "bytes_to_host": self.problem.engine.traffic["to_host"],
        }
