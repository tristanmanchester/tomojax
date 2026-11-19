"""A level's working memory beside its stacks must not grow faster than the number of views.

A level's plan measures its storage at setup and takes its largest transients from their
programs' compiles (``_solver._Level.place``); the level's other programs are not compiled
then, so the plan is only right if none of them holds a temporary that scales faster than the
views: K11 67208 (4000 views) failed in its Schur solve because conjugate gradients copied the
whole weighted-projection stack ``r`` out of a conditional on every iteration, a temporary
of 0.47 GiB at 1441 views and 1.29 GiB at 4000 that the plan had counted as volumes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.alignment._joint import _fit
from tomojax.geometry import stack_view_poses


def _scan(views, nu=96, nv=64, n=24, nz=16):
    grid, detector = tj.Grid(n, n, nz, 1, 1, 1), tj.Detector(nu, nv, 1, 1)
    angles = np.linspace(0, 360, views, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt=-30)
    rng = np.random.default_rng(3)
    volume = rng.uniform(0, 1, (nz, n, n)).astype(np.float32)
    projections = np.array(tj.project(geometry, volume))
    projections += rng.normal(0, 0.01, projections.shape).astype(np.float32)
    return projections, np.asarray(stack_view_poses(geometry, views)), grid, detector


ARGS = {}


def _abstract(tree):
    def shape(leaf):
        if isinstance(leaf, jax.Array):
            return jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, weak_type=leaf.weak_type)
        return leaf

    return jax.tree.map(shape, tree)


def captured_constants(compiled, *args, **kwargs):
    """The arrays a compiled function holds as constants of its program.

    ``trace``, not ``make_jaxpr``: a jitted closure's captured arrays are the consts of
    its own traced jaxpr; ``make_jaxpr`` of the jitted function shows none of them.
    """
    return compiled.trace(*args, **kwargs).jaxpr.consts


def level_memory(views, monkeypatch, nu=96, nv=64, n=24, nz=16):
    """Temporaries, outputs and captured constants (bytes) of each compiled level function.

    By name.
    """
    jit, seen = jax.jit, {}

    class Recorded:
        """A compiled function that keeps its first concrete call's arguments, by name."""

        def __init__(self, compiled, name):
            self.compiled, self.name = compiled, name

        def __getattr__(self, attr):  # lower, trace: the plan compiles some at setup
            return getattr(self.compiled, attr)

        def __call__(self, *args, **kwargs):
            leaves = jax.tree.leaves((args, kwargs))
            if self.name not in seen and not any(isinstance(a, jax.core.Tracer) for a in leaves):
                seen[self.name] = (self.compiled, _abstract(args), _abstract(kwargs))
            return self.compiled(*args, **kwargs)

    def recording(fun=None, **options):
        if fun is None:
            return lambda f: recording(f, **options)
        compiled = jit(fun, **options)
        name = getattr(fun, "__qualname__", repr(fun))
        if "_joint" not in getattr(fun, "__module__", "") or name in seen:
            return compiled
        return Recorded(compiled, name)

    operators, shapes = _fit.Operators, []

    def made(*args, **kwargs):
        ops = operators(*args, **kwargs)
        shapes.append((tuple(ops.shape), (ops.detector.nv, ops.detector.nu)))
        return ops

    monkeypatch.setattr(jax, "jit", recording)
    monkeypatch.setattr(_fit, "Operators", made)
    projections, nominal, grid, detector = _scan(views, nu, nv, n, nz)
    config = tj.AlignConfig(levels=(1,), iterations=2)
    _fit.fit_poses(projections, nominal, grid, detector, keep_grid=True, config=config)
    monkeypatch.setattr(jax, "jit", jit)
    monkeypatch.setattr(_fit, "Operators", operators)
    volume, image = shapes[-1]

    def bulky(c):
        """Shaped like the level's volume or like detector images: a volume or a stack."""
        shape = tuple(getattr(c, "shape", ()))
        return shape[-3:] == volume or shape[-2:] == image

    out = {}
    for name, (compiled, args, kwargs) in seen.items():
        ARGS[name] = (compiled, args, kwargs)
        with jax.enable_x64(True):  # as fit_poses runs them: its FP64 stays FP64
            analysis = compiled.lower(*args, **kwargs).compile().memory_analysis()
            consts = captured_constants(compiled, *args, **kwargs)
        out[name] = {
            "temp": analysis.temp_size_in_bytes,
            "output": analysis.output_size_in_bytes,
            "consts": sum(getattr(c, "nbytes", 0) for c in consts),
            "captured_shapes": [tuple(c.shape) for c in consts if bulky(c)],
        }
    return out


def level_temporaries(views, monkeypatch, nu=96, nv=64, n=24, nz=16):
    """XLA's temporary bytes of every function a one-level fit compiles, by name."""
    memory = level_memory(views, monkeypatch, nu, nv, n, nz)
    return {name: m["temp"] for name, m in memory.items()}


# Known to grow faster than the views, with the growth order measured from 300 to 900 views:
# the motion prior's factor, built from its dense spline design (local rows by about
# views / 10 knots). Its temporaries and the innovation solve's grow the same way at more
# views (5 to 6 copies of the design at 1000 to 4000), but a fixed part hides that at these
# counts. Item 64 makes the prior linear; remove the entry then.
SUPERLINEAR = {
    "MotionPrior._factor": 1.9,  # its output, measured in FP64 as the fit runs
}


def _order(small, large, ratio):
    """The growth exponent between two view counts, or 0 for what is too small to grow."""
    if large <= 256 * 1024 or small <= 0:
        return 0.0
    return float(np.log(large / small) / np.log(ratio))


@pytest.mark.gpu
def test_no_level_function_grows_faster_than_the_views(monkeypatch):
    # A reserve counted in volumes and a plan's prior term are only right if what each compiled
    # function holds grows no faster than the views. At 100, 300 and 900 views on one grid and
    # detector, each function's temporaries, outputs and captured constants are measured; the
    # growth order from 300 to 900 must be at most 1.3 (linear, with fixed parts), except the
    # motion prior's known quadratic functions, which must still need their exception.
    counts = (100, 300, 900)
    profiles = [level_memory(views, monkeypatch) for views in counts]
    ratio = counts[2] / counts[1]
    superlinear = {}
    for name, memory in profiles[2].items():
        for kind in ("temp", "output", "consts"):
            small = profiles[1].get(name, {}).get(kind, 0)
            order = _order(small, memory[kind], ratio)
            if order > 1.3:
                superlinear[name] = max(order, superlinear.get(name, 0.0))
    unexpected = {name: order for name, order in superlinear.items() if name not in SUPERLINEAR}
    assert not unexpected, f"growing faster than the views (order from 300 to 900): {unexpected}"
    gone = set(SUPERLINEAR) - set(superlinear)
    assert not gone, f"no longer superlinear, remove from SUPERLINEAR (item 64): {gone}"
    # No function captures a volume or a stack as a compiled constant: held by the program for
    # its life, beside the level's own arrays (the start's FISTA once held the shared images).
    captured = {
        name: m["captured_shapes"] for name, m in profiles[0].items() if m["captured_shapes"]
    }
    assert not captured, f"volume- or stack-shaped captured constants: {captured}"
    large = {name: m["consts"] for name, m in profiles[2].items() if m["consts"] > 256 * 1024}
    assert not large, f"large captured constants (bytes): {large}"


def _conditionals(jaxpr):
    """Every cond equation in a jaxpr and the jaxprs nested in it."""
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "cond":
            yield eqn
        for value in eqn.params.values():
            for item in value if isinstance(value, list | tuple) else [value]:
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    yield from _conditionals(inner)


def test_the_schur_solves_conditionals_carry_no_stack(monkeypatch):
    # Conjugate gradients chose between advancing and verifying in a conditional that also made
    # the product and returned the weighted-projection stack r it had written. XLA keeps a
    # conditional's results in buffers of their own, so at K11's size the compiled solve copied
    # r out of it on every iteration: a whole stack of temporaries (0.47 GiB at 1441 views,
    # 1.29 GiB at 4000) that the plan did not count. The product is made outside the
    # conditional; nothing the size of the views may pass through one.
    level_memory(48, monkeypatch)
    compiled, args, kwargs = next(v for k, v in ARGS.items() if k.endswith("schur"))
    views = args[1].shape[0]  # the weights sw: (padded views, rows, pixels)
    with jax.enable_x64(True):
        jaxpr = jax.make_jaxpr(compiled)(*args, **kwargs).jaxpr
    carried = [
        var.aval.shape
        for eqn in _conditionals(jaxpr)
        for var in eqn.outvars
        if len(var.aval.shape) >= 3 and var.aval.shape[0] == views
    ]
    assert not carried, f"stack-sized results of a conditional in the Schur solve: {carried}"


def test_the_constants_check_sees_a_captured_volume():
    # make_jaxpr of a jitted closure reported no constants, so the guard above could not see
    # a captured volume: the check must find one planted on purpose.
    volume = jnp.ones((16, 24, 24), jnp.float32)
    planted = jax.jit(lambda x: x + volume)
    assert jax.make_jaxpr(planted)(volume).consts == []  # what the guard used to read
    shapes = [tuple(c.shape) for c in captured_constants(planted, volume)]
    assert shapes == [(16, 24, 24)]
