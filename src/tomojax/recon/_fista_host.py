"""The resident FISTA recurrence with volume state accessed a host slab at a time."""

from __future__ import annotations

from contextlib import ExitStack
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from tomojax import progress
from tomojax.core.host_volume import close_work_file, host_volume, scratch_directory
from tomojax.core.operator_norm import bounds_settled
from tomojax.errors import InvalidParameter
from tomojax.wire.metrics import Metric

from ._cuda_tv import proximal_host
from ._noise_probe import gaussian_probe
from ._tv_metric import raise_tv_weight
from ._tv_ops import gradient_norm_squared

if TYPE_CHECKING:
    import jax

    from tomojax.geometry import Grid, ScanGeometry

    from .fista_tv import FISTAConfig


class HostFista:
    """Three host volumes suffice for gradient, proximal and momentum updates."""

    def __init__(
        self, problem: Any, config: Any, *, metric, noise_scale=None, initial=None, out=None
    ) -> None:
        from ._host_arrays import validate_host_arrays

        self.problem, self.cfg, self.metric = problem, config, metric
        self.noise_scale = noise_scale
        self.plan, self.xp = problem.plan, problem.engine.xp
        grid = self.plan.grid
        shape = (grid.nz, grid.ny, grid.nx)
        x = validate_host_arrays(problem.data, out, shape, "fista_tv", copies=3)
        directory = scratch_directory(x)
        arrays = [x, host_volume(shape, directory), host_volume(shape, directory)]
        self.X, self.Z, self.G = [problem.volume(array) for array in arrays]
        self.views = np.arange(problem.data.shape[0])
        from ._view_statistics import ViewStatistics

        covered = np.zeros(self.plan.detector.nv, bool)
        for first, stop in self.plan.windows:
            covered[first:stop] = True
        self.statistics = (
            ViewStatistics(problem, self.views)
            if config.data_term == "per_view_contrast" or not covered.all()
            else None
        )
        self.absolute_outside = (
            float(self.statistics.outside_energy(0))
            if config.data_term == "absolute" and self.statistics is not None
            else 0.0
        )
        self.lower = config.lower_bound if config.lower_bound is not None else -math.inf
        self.upper = config.upper_bound if config.upper_bound is not None else math.inf
        self.initial = initial
        if initial is not None and np.may_share_memory(initial, x):
            # Norm estimation uses X as scratch; preserve a caller's in-place start.
            from tomojax.core.host_volume import fill

            self.initial = fill(host_volume(shape, directory), initial)
        self.shape = shape

    def apply(self, source, target, *, residual=False, step=None):
        """Write a normal application or data gradient, returning the residual's energy."""
        value = self.absolute_outside if residual else 0.0

        def read(k):
            return () if source is None else (source.get(*self.plan.planes(k)),)

        mean = None
        prepare = None
        if residual and self.cfg.data_term == "per_view_contrast":
            statistics = self.statistics
            assert statistics is not None
            mean = -statistics.mean
            if source is not None:
                if self.plan.ring == self.plan.detector.nv:

                    def prepare_rows(rows):
                        nonlocal mean, value
                        mean = statistics.projection_mean(rows[0]) - statistics.mean
                        value = float(statistics.outside_energy(-mean))

                    prepare = prepare_rows
                else:
                    mean = mean + statistics.project(read)
            value = float(statistics.outside_energy(-mean))

        def finish(batch, first, stop, rows):
            nonlocal value
            if residual:
                rows[0][...] -= self.problem.measured(batch, first, stop)
                if mean is not None:
                    rows[0][...] -= mean[batch.index, None, None]
                value += self.problem.engine.norm2(rows[0])

        def sink(k, accumulated, _sources):
            z0, z1 = self.plan.planes(k)
            value = accumulated[0]
            if step is not None:
                assert source is not None
                value = source.view(z0, z1) - np.float32(step) * value
            target.put(z0, z1, value)

        self.problem.sweep.run(
            self.views, source=read, finish=finish, sink=sink, keep=False, prepare=prepare
        )
        return 0.5 * value

    def noise_gradient_rms(self):
        """Use the existing row-window transpose without another host volume."""
        import jax
        import jax.numpy as jnp

        detector = self.plan.detector
        draw = jax.jit(gaussian_probe, static_argnames=("detector_shape",))
        energy = 0.0

        def finish(batch, first, stop, rows):
            # Draw and center the FULL detector before selecting a row window.
            # Each view's draw is seeded by its index, so drawing views in chunks
            # no larger than the planned row window bounds the full-view probe.
            views = np.asarray(batch.views)
            chunk = max(1, rows[0].nbytes // (4 * detector.nv * detector.nu))
            for start in range(0, len(views), chunk):
                probe = draw(jnp.asarray(views[start : start + chunk]), (detector.nv, detector.nu))
                # DLPack shares the draw with the native transpose's buffer.
                native = self.xp.from_dlpack(probe)
                rows[0][start : start + chunk] = native[:, first:stop]
                # Finish the borrowed-buffer copy before JAX can reclaim the draw.
                self.xp.cuda.get_current_stream().synchronize()
                del probe, native

        def sink(_k, accumulated, _sources):
            nonlocal energy
            energy += self.problem.engine.norm2(accumulated[0])

        self.problem.sweep.run(
            self.views, source=lambda _: (), finish=finish, sink=sink, keep=False
        )
        return math.sqrt(energy / math.prod(self.shape))

    def initialize(self):
        """The unchanged backprojected-data power start, then the feasible initial iterate."""
        xp = self.xp
        if self.noise_scale is not None:
            self.metric = raise_tv_weight(self.metric, self.noise_scale * self.noise_gradient_rms())
        value = self.apply(None, self.G, residual=True)
        limit = self.cfg.lipschitz
        if limit is None:
            peak = 0.0
            for k in range(self.plan.count):
                peak = max(peak, float(xp.max(xp.abs(self.G.view(*self.plan.planes(k))))))
            norm = 0.0
            for k in range(self.plan.count):
                z0, z1 = self.plan.planes(k)
                start = xp.abs(self.G.view(z0, z1)) + np.float32(1e-3 * peak if peak else 1)
                norm += self.problem.engine.norm2(start)
                self.X.put(z0, z1, start)
            del start
            power, normal = self.X, self.Z
            engine, iterations = self.problem.engine, max(1, self.cfg.power_iterations)
            steps, upper = 0, math.inf
            # The step-safe Collatz bound, once it meets the power estimate (see
            # tomojax.core.operator_norm.normal_upper_bound).
            while not bounds_settled(steps, iterations, math.sqrt(norm), upper):
                for k in range(self.plan.count):
                    z0, z1 = self.plan.planes(k)
                    power.put(z0, z1, power.view(z0, z1) / np.float32(math.sqrt(norm) or 1))
                self.apply(power, normal)
                norm = upper = 0.0
                for k in range(self.plan.count):
                    z0, z1 = self.plan.planes(k)
                    part = normal.view(z0, z1)
                    norm += engine.norm2(part)
                    upper = max(upper, engine.max_ratio(part, power.view(z0, z1)))
                del part
                power, normal, steps = normal, power, steps + 1
            limit = upper * (1 + 8 * float(np.finfo(np.float32).eps)) or 1e-6
        for k in range(self.plan.count):
            z0, z1 = self.plan.planes(k)
            part = (
                xp.zeros((z1 - z0, *self.shape[1:]), xp.float32)
                if self.initial is None
                else xp.asarray(self.initial[z0:z1], xp.float32)
            )
            part = xp.clip(part, self.lower, self.upper)
            self.X.put(z0, z1, part)
            self.Z.put(z0, z1, part)
        return float(limit), value

    def update(self, limit, momentum, *, gradient=False):
        """Gradient step, global finite-step TV proximal, and the resident clipped momentum."""
        cfg, xp = self.cfg, self.xp
        metric = self.metric
        scales = (1.0, 1.0, 1.0) if metric is None else metric.gradient_scale_xyz
        if gradient:
            for k in range(self.plan.count):
                z0, z1 = self.plan.planes(k)
                step = self.Z.view(z0, z1) - np.float32(1 / limit) * self.G.view(z0, z1)
                self.G.put(z0, z1, step)
            del step

        def momentum_sink(z0, z1, new):
            old = self.X.view(z0, z1)
            self.X.put(z0, z1, new)
            self.Z.put(z0, z1, xp.clip(new + momentum * (new - old), self.lower, self.upper))

        proximal_host(
            self.G,
            momentum_sink,
            (0.0 if metric is None else float(metric.weight)) / limit,
            gradient_scale_xyz=scales,
            iterations=cfg.tv_prox_iterations,
            tau=1 / (gradient_norm_squared(self.shape, gradient_scale_xyz=scales) or 1),
            lower=self.lower,
            upper=self.upper,
        )

    def regularizer(self):
        """Isotropic TV with each slab's outgoing z edge, counted once."""
        if self.metric is None:
            return 0.0
        assert self.metric is not None
        xp, value = self.xp, 0.0
        for k in range(self.plan.count):
            z0, z1 = self.plan.planes(k)
            part = self.Z.view(z0, min(self.shape[0], z1 + 1))
            squares = xp.zeros_like(part)
            for axis, scale in zip((2, 1, 0), self.metric.gradient_scale_xyz, strict=True):
                slices = [slice(None)] * 3
                slices[axis] = slice(0, -1)
                squares[tuple(slices)] += (scale * xp.diff(part, axis=axis)) ** 2
            value += float(xp.sum(xp.sqrt(squares[: z1 - z0]), dtype=xp.float64))
            del part, squares
        return value

    def run(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Run exactly the same stopping rule and retain the padded objective history."""
        cfg = self.cfg
        limit, at_zero = self.initialize()
        losses, t, streak, done = [], np.float32(1), 0, False
        zero = self.initial is None and self.lower <= 0 <= self.upper
        for step in range(cfg.iterations):
            progress.check_cancelled()
            loss = 0.0 if self.metric is None else float(self.metric.weight) * self.regularizer()
            gradient = step == 0 and zero
            loss += (
                at_zero if gradient else self.apply(self.Z, self.G, residual=True, step=1 / limit)
            )
            tn = np.float32((1 + np.sqrt(1 + 4 * t * t)) / 2)
            self.update(limit, np.float32((t - 1) / tn), gradient=gradient)
            if cfg.tolerance and cfg.patience:
                small = (
                    bool(losses)
                    and abs(loss - losses[-1]) / max(abs(losses[-1]), 1e-6) <= cfg.tolerance
                )
                streak = streak + 1 if small else 0
                done = streak >= cfg.patience
            losses.append(loss)
            if (task := progress.for_unit("iterations")) is not None:
                self.problem.completed(task.advance, len(losses), {Metric.loss.name: loss})
            t = tn
            progress.preview(
                lambda: self.X.preview(self.plan.grid), final=done or step + 1 == cfg.iterations
            )
            if done:
                break
        count = len(losses)
        losses.extend([losses[-1]] * (cfg.iterations - count) if losses else [])
        return self.X.array, {
            "loss": losses,
            "lipschitz": limit,
            "effective_iterations": count,
            "early_stop": done,
            "regularizer": cfg.regularizer,
            "huber_delta": cfg.huber_delta,
            "host_slabs": True,
        }


def needs_host_state(
    grid: Grid,
    projections: np.ndarray | jax.Array,
    cfg: FISTAConfig,
    batch: int,
    *,
    zero_start: bool,
    stream: bool,
    joseph: bool,
) -> bool:
    """Check the resident FISTA workspace against the current device memory sample."""
    from tomojax.backends import device_free_memory_bytes

    from ._fista_owned import store_residuals

    free = device_free_memory_bytes()
    if free is None:
        return False
    owned = store_residuals(
        grid, projections, cfg, zero_start=zero_start, stream=stream, joseph=joseph
    )
    volumes = 2 if owned else 4
    need = volumes * 4 * grid.nx * grid.ny * grid.nz
    need += (0 if stream else 2) * projections.size * 4
    # Batch-sized measured, projected, residual and masked-residual storage.
    need += 4 * 4 * min(batch, projections.shape[0]) * projections.shape[1] * projections.shape[2]
    if cfg.iterations > 0 and cfg.tv_weight != 0 and cfg.tv_prox_iterations > 0:
        from ._tv_ops import cuda_tv_workspace_bytes

        scratch = cuda_tv_workspace_bytes(
            (grid.nz, grid.ny, grid.nx), np.dtype(np.float32), cfg.tv_prox_iterations
        )
        # TV and Huber TV use the same primal/dual proximal workspace.
        # The normalized float mask is already allocated before the free-memory
        # query; the proximal additionally keeps a boolean at its broadcast shape.
        if cfg.support is not None:
            need += math.prod(np.shape(cfg.support))
        # The reference proximal keeps three dual fields plus primal workspace.
        need += 4 * 4 * grid.nx * grid.ny * grid.nz if scratch is None else scratch
    return need > 0.85 * free


def host_objective_error(cfg: FISTAConfig) -> InvalidParameter | None:
    """Identify the field requiring a resident solve, if any."""
    if cfg.support is None and cfg.regularizer.lower() == "tv":
        return None
    field = "support" if cfg.support is not None else "regularizer"
    return InvalidParameter(
        f"config.{field}", "fista_tv: host volumes require TV and no support mask"
    )


def check_fista_storage(
    geometry: ScanGeometry,
    grid: Grid,
    projections: np.ndarray | jax.Array,
    cfg: FISTAConfig,
    *,
    warm_start: bool,
) -> None:
    """Check resident-only objectives against the same placement decision as the CUDA solver."""
    from tomojax.core.geometry.cone import is_cone_beam

    from ._backend import reconstruction_backend
    from .fista_tv import fista_view_batch, stream_fista_views

    if cfg.devices is not None or (error := host_objective_error(cfg)) is None:
        return
    cone = is_cone_beam(geometry)
    if reconstruction_backend(
        cfg.backend, cone=cone, context="fista_tv"
    ) == "cuda" and needs_host_state(
        grid,
        projections,
        cfg,
        fista_view_batch(cfg, projections.shape[0]),
        zero_start=not warm_start,
        stream=stream_fista_views(grid, projections, cfg),
        joseph=not cone,
    ):
        raise error


def solve_on_host(
    geometry, grid, detector, projections, runtime, *, out=None
) -> tuple[np.ndarray, dict[str, Any], Any] | None:
    """Select host state only when requested, or when the resident state cannot fit."""
    from tomojax.core.cone import ConeModel

    from ._cone_slabs import cone_problem, host_projections

    projector, cfg = runtime.projector, runtime.config
    if projector is None or projector[1] != "cuda" or runtime.split is not None:
        if out is not None:
            raise ValueError("fista_tv: out requires single-device CUDA Joseph projection")
        return None
    model, backend, batch = projector
    if model != "joseph" and not isinstance(model, ConeModel):
        if out is not None:
            raise ValueError("fista_tv: out requires single-device CUDA Joseph projection")
        return None
    if out is None and not needs_host_state(
        grid,
        projections,
        cfg,
        batch,
        zero_start=runtime.x0 is None,
        stream=runtime.stream,
        joseph=model == "joseph",
    ):
        return None
    if error := host_objective_error(cfg):
        raise error
    from ._parallel_slabs import parallel_problem

    data = host_projections(projections, "fista_tv")
    options = {"poses": runtime.poses, "batch": batch, "channels": 1}
    problem = (
        cone_problem(geometry, grid, detector, data, model=model, backend=backend, **options)
        if isinstance(model, ConeModel)
        else parallel_problem(geometry, grid, detector, data, **options)
    )
    initial = None if runtime.x0 is None else np.asarray(runtime.x0)
    solver = HostFista(
        problem,
        cfg,
        metric=runtime.metric,
        noise_scale=runtime.noise_scale,
        initial=initial,
        out=out,
    )
    with ExitStack() as resources:
        if solver.initial is not initial:
            resources.callback(close_work_file, solver.initial)
        resources.enter_context(problem.using(out, solver.X, solver.Z, solver.G))
        volume, info = solver.run()
    return problem.result(volume), info, solver.metric
