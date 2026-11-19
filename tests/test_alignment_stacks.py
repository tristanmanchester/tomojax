"""Where a level's stacks live changes nothing but where they live.

A full-size synchrotron scan's stacks do not fit an 8 GB card (K11 67208, 4000
views, and I13 159815, 3001, raised "out of device memory"). Each level now plans
which stacks stay on the device, allocates them in order, holds the rest on the
host, and reads every stack a batch at a time through one set of scans.
"""

import jax
import numpy as np
import pytest

import tomojax as tj

# check-public-imports: allow-private
from tomojax.alignment._joint import _fit, _solver, _stacks
from tomojax.geometry import stack_view_poses

DEVICE = dict.fromkeys(_stacks.ORDER, "device")
HOST = dict.fromkeys(_stacks.ORDER, "host")
COLD = {**DEVICE, "y": "host", "precision": "host", "residual": "host"}


def _scan():
    """A small laminography scan with a slip, a wobble and unmeasured pixels (so a precision)."""
    grid, detector, views = tj.Grid(20, 20, 14, 1.0, 1.0, 1.0), tj.Detector(28, 22, 1.0, 1.0), 48
    angles = np.linspace(0, 360, views, endpoint=False)
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt=-30)
    z, y, x = np.mgrid[:14, :20, :20]
    volume = np.zeros((14, 20, 20), np.float32)
    for cz, cy, cx, r, mu in ((7, 7, 6, 4, 1.0), (6, 12, 13, 3, 0.6), (9, 13, 6, 2, 1.4)):
        volume[(z - cz) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r * r] = mu
    table = np.zeros((views, 6))
    table[24:, 3] = 1.0
    table[:, 0] = np.deg2rad(0.5) * np.sin(3 * np.deg2rad(angles))
    moved = tj.Scan(np.zeros((views, 22, 28), np.float32), geometry).with_poses(table).geometry
    projections = np.array(tj.project(moved, volume))
    rng = np.random.default_rng(5)
    projections += rng.normal(0, 0.01, projections.shape).astype(np.float32)
    projections[rng.uniform(size=projections.shape) < 0.02] = np.nan
    return projections, np.asarray(stack_view_poses(geometry, views)), grid, detector


def _aligned(fit, placement=None):
    projections, nominal, grid, detector = _scan()
    return _fit.fit_poses(projections, nominal, grid, detector, fit=fit, placement=placement)


@pytest.mark.slow
@pytest.mark.parametrize("fit", [("translations",), ("translations", "rotations")])
def test_every_placement_gives_the_same_result_bit_for_bit(fit):
    results = {
        name: _aligned(fit, placement)
        for name, placement in (("device", DEVICE), ("cold", COLD), ("host", HOST))
    }
    volume, _, table, record = results["device"]
    for name in ("cold", "host"):
        other_volume, _, other_table, other = results[name]
        np.testing.assert_array_equal(np.asarray(other_volume), np.asarray(volume), err_msg=name)
        np.testing.assert_array_equal(other_table, table, err_msg=name)
    # Each placement held what it was asked to, the precision included, at every level.
    for name, placement in (("device", DEVICE), ("cold", COLD), ("host", HOST)):
        levels = results[name][3]["levels"]
        for level in levels:
            assert level["stacks"]["placement"] == placement, (name, level["stacks"])
        stage = results[name][3]["refinement"]["stacks"]["placement"]
        assert stage == {n: placement[n] for n in ("y", "precision")}


def _stacks_under(monkeypatch, *, limit, pool, largest_free, transient):
    """Five small stacks placed under supplied allocator figures, and the allocations tried.

    The pool's largest free block is ``largest_free`` less what its device stacks hold.
    """
    tried, held = [], []
    device = _stacks.LevelStacks._device

    def allocate(shape, dtype, value):
        tried.append(shape)
        held.append(device(shape, dtype, value))
        return held[-1]

    def figures():
        taken = sum(a.nbytes for a in held if not a.is_deleted())
        free = largest_free - taken
        return {
            "bytes_limit": limit,
            "pool_bytes": pool,
            "largest_free_block_bytes": free,
            "bytes_in_use": taken,
        }

    monkeypatch.setattr(_stacks.LevelStacks, "_device", staticmethod(allocate))
    monkeypatch.setattr(_stacks, "_device_stats", figures)
    specs = dict.fromkeys(("sw", "r", "j", "y", "residual"), ((4, 2, 3), np.float32))  # 96 B
    stacks = _stacks.LevelStacks(specs, 4, data={}, room=None, reserve={"transient": transient})
    return stacks, tried


def test_a_stack_no_free_block_holds_goes_to_the_host_untried(monkeypatch):
    # The pool at its limit never returns a free region, so a stack larger than its largest
    # free block cannot be had however much is free in all: it is not tried (a refused
    # allocation waits 10 s in the allocator's retry), and it and every stack after it go to
    # the host.
    stacks, tried = _stacks_under(monkeypatch, limit=1000, pool=1000, largest_free=50, transient=0)
    assert tried == [] and set(stacks.placed.values()) == {"host"}
    stacks.close()


def test_a_stack_that_takes_the_transients_block_goes_to_the_host(monkeypatch):
    # The first stack fits, but once allocated no block is left for the level's largest
    # transient (one block too): it is released and it and every stack after it go to the host.
    stacks, tried = _stacks_under(
        monkeypatch, limit=1000, pool=1000, largest_free=150, transient=100
    )
    assert len(tried) == 1 and set(stacks.placed.values()) == {"host"}
    assert stacks.record()["raised"] == []
    stacks.close()


def test_a_pool_below_its_limit_can_grow_a_block_for_a_stack(monkeypatch):
    # Below its limit the pool can add a region: a stack larger than its largest free block
    # is tried, and here it is had.
    stacks, tried = _stacks_under(monkeypatch, limit=1000, pool=500, largest_free=10, transient=0)
    assert len(tried) == 5 and set(stacks.placed.values()) == {"device"}
    stacks.close()


@pytest.mark.slow
def test_a_stack_the_device_refuses_goes_to_the_host_with_every_stack_after_it(monkeypatch):
    # The allocator is the judge of a stack that fits by its figures: a refused allocation
    # sends that stack and the ones after it in device order to the host, the level names it,
    # and nothing else changes.
    expected_volume, _, expected, _ = _aligned(("translations",), DEVICE)
    device = _stacks.LevelStacks._device

    def refuse_the_columns(shape, dtype, value):
        if len(shape) == 4:  # j: (views, columns, rows, pixels)
            raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory (test)")
        return device(shape, dtype, value)

    monkeypatch.setattr(_stacks.LevelStacks, "_device", staticmethod(refuse_the_columns))
    volume, _, table, record = _aligned(("translations",), DEVICE)
    np.testing.assert_array_equal(table, expected)
    np.testing.assert_array_equal(np.asarray(volume), np.asarray(expected_volume))
    stacks = record["levels"][0]["stacks"]
    assert stacks["placement"] == {
        "sw": "device",
        "r": "device",
        **dict.fromkeys(("j", "y", "precision", "residual"), "host"),
    }
    assert stacks["raised"] == ["j"]


@pytest.mark.parametrize("where", ["linearize", "schur"])
def test_running_out_of_device_memory_inside_a_level_names_the_placement(monkeypatch, where):
    # Host stacks keep the level's own allocations bounded, but other work in the process
    # can still take the room: the linearisation or the matvec then fails with a plain
    # error that says what the level had placed where, not a bare allocator message.
    def exhausted(*_args, **_kwargs):
        raise jax.errors.JaxRuntimeError("RESOURCE_EXHAUSTED: Out of memory (test)")

    monkeypatch.setattr(_solver._Level, where, exhausted)
    with pytest.raises(MemoryError, match=r"y on the host.*residual on the host.*room at setup"):
        _aligned(("translations",), COLD)


def test_released_storage_leaves_nothing_of_the_level_on_the_device():
    # The stacks and every volume but the result (handed to the caller first) are released.
    projections, nominal, grid, detector = _scan()
    held = []
    original = _stacks.LevelStacks.close

    def close(self):
        stacks = [s.data for s in self.stacks.values() if not s.host]
        held.append([*stacks, *jax.tree.leaves(self.volumes)])
        original(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_stacks.LevelStacks, "close", close)
        _fit.fit_poses(projections, nominal, grid, detector, placement=COLD)
    assert held and all(array.is_deleted() for level in held for array in level)


def _watch_the_volumes(monkeypatch):
    """Each level, and the buffers and the device's bytes no array holds at setup and each solve."""
    seen = []
    place, schur = _solver._Level.place, _solver._Level.schur

    def state(level):
        buffers = {a.unsafe_buffer_pointer() for a in jax.tree.leaves(level.stacks.volumes)}
        stats = jax.devices()[0].memory_stats()
        if stats is None:
            return buffers, None
        live = sum(a.nbytes for a in jax.live_arrays() if not a.is_deleted())
        return buffers, stats["bytes_in_use"] - live

    def placed(self, *args, **kwargs):
        out = place(self, *args, **kwargs)
        seen.append((self, state(self), []))
        return out

    def solving(self, *args):
        jax.block_until_ready(args)
        seen[-1][2].append(state(self))
        return schur(self, *args)

    monkeypatch.setattr(_solver._Level, "place", placed)
    monkeypatch.setattr(_solver._Level, "schur", solving)
    return seen


def test_a_levels_volumes_stay_where_its_setup_put_them(monkeypatch):
    # A level allocates its volumes once and every step writes them in place, donated: one
    # reallocated mid-level could take a chunk of up to twice its size, and the plan, measured
    # at setup, would no longer hold. At every Schur solve each level holds the buffers its
    # setup allocated (the volume and the other trade places), and the solve that runs is the
    # one its setup compiled for the plan.
    seen = _watch_the_volumes(monkeypatch)
    _aligned(("translations", "rotations"), DEVICE)
    assert len(seen) == 2
    for level, (buffers, _), solves in seen:
        assert solves and all(entry[0] == buffers for entry in solves)
        assert level._schur._cache_size() == 1


def test_at_a_schur_solve_the_volumes_alive_are_the_levels_own(monkeypatch):
    # The second level starts from the first's volume and upsamples it into its own start
    # volume: at every Schur solve the live arrays of a level's volume shape are exactly the
    # volumes it owns. A copy resized outside the level and held beside them would be one no
    # plan owns. Arrays alive before the call (another test's) are held, so none of their
    # buffers is reused, and left out.
    extras = []
    schur = _solver._Level.schur
    before = [a for a in jax.live_arrays() if not a.is_deleted()]

    def solving(self, *args):
        jax.block_until_ready(args)

        def pointers(arrays):
            same = (a for a in arrays if a.shape == self.ops.shape and a.dtype == np.float32)
            return {a.unsafe_buffer_pointer() for a in same if not a.is_deleted()}

        owned = pointers(jax.tree.leaves(self.stacks.volumes))
        extras.append((self.ops.shape, pointers(jax.live_arrays()) - owned - pointers(before)))
        return schur(self, *args)

    monkeypatch.setattr(_solver._Level, "schur", solving)
    _aligned(("translations",), DEVICE)
    assert len({shape for shape, _ in extras}) == 2
    assert all(not extra for _, extra in extras), extras


@pytest.mark.gpu
def test_nothing_a_level_holds_beyond_its_arrays_grows_during_it(monkeypatch):
    # What the device holds that no array does (the allocator's rounding of its chunks) is
    # measured at a level's setup and counted in its plan. It is the same at every Schur solve:
    # nothing allocated during a step outlives it.
    seen = _watch_the_volumes(monkeypatch)
    _aligned(("translations", "rotations"), DEVICE)
    for _, (_, setup), solves in seen:
        for _, other in solves:
            assert abs(other - setup) <= 2**20, (other, setup)


def test_align_reports_each_levels_placement_and_time():
    projections, _, grid, detector = _scan()
    geometry = tj.LaminographyGeometry(
        grid, detector, np.linspace(0, 360, len(projections), endpoint=False), tilt=-30
    )
    info = tj.align(tj.Scan(projections, geometry)).info
    for level in [*info["levels"], info["refinement"]]:
        stacks = level["stacks"]
        assert set(stacks["placement"].values()) <= {"device", "host"}
        assert stacks["raised"] == []
        assert level["seconds"] > 0


_PEAK_PROBE = """
import json, sys
import jax, jax.numpy as jnp, numpy as np
import tomojax as tj
from tomojax.alignment._integration import cone_frames
from tomojax.alignment._joint import _fit, _solver
from tomojax.geometry import stack_view_poses

n, nz, nu, nv, views = (int(a) for a in sys.argv[1:6])
cone, stage = sys.argv[6] == "cone", sys.argv[7]
# One level, so the process's peak is the level's; for "fine", a short coarse level first,
# whose volume the fine level (measured, being the last and larger) starts from.
_fit._LEVELS = ((2, 2), (1, 8)) if stage == "fine" else ((1, 8),)
dev, seen = jax.devices()[0], {}
solve, refine = _solver.solve, _fit.refine


def measure(run, stacks_of, *args, **kwargs):
    stats = dev.memory_stats()
    base = stats["bytes_in_use"]
    out = run(*args, **kwargs)
    stacks = stacks_of(out)
    seen.update(
        earlier=stats["peak_bytes_in_use"] - base,
        used=dev.memory_stats()["peak_bytes_in_use"] - base,
        setup=stacks["in_use_bytes"] - base,
        free=stacks["free_bytes"],
        chunk=stacks["reserve_parts"]["chunk"],
        transients=stacks["transients"],
        volume=4 * int(np.prod(kwargs["ops"].shape)),
    )
    return out


def level(*args, **kwargs):
    return measure(solve, lambda out: out[3]["stacks"], *args, **kwargs)


def stand_in(y, precision, prior, *_args, ops, **_kwargs):
    # A level that holds nothing: the refinement after it sets the process's peak.
    return jnp.full(ops.shape, 0.5, jnp.float32), jnp.zeros(prior.size), y, {
        "translation_noise": None, "pose_fisher": 1.0
    }


def refinement(*args, **kwargs):
    measure(refine, lambda out: out[1]["stacks"], *args, **kwargs)
    print(json.dumps(seen))
    sys.exit(0)


if stage in ("level", "fine"):
    _solver.solve = _fit.solve = level
else:
    _fit.solve, _fit.refine = stand_in, refinement
grid, detector = tj.Grid(n, n, nz, 1, 1, 1), tj.Detector(nu, nv, 1, 1)
angles = np.linspace(0, 360, views, endpoint=False)
if cone:  # a circular orbit, magnification 1.5, the volume filling the beam
    geometry = tj.ConeGeometry(grid, detector, angles, tj.ConeBeam(3 * n, 4.5 * n))
else:
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt=-30)
rng = np.random.default_rng(1)
volume = rng.uniform(0, 1, (nz, n, n)).astype(np.float32)
with jax.default_device(jax.devices("cpu")[0]):
    projections = np.array(tj.project(geometry, volume))
projections += rng.normal(0, 0.01, projections.shape).astype(np.float32)
nominal = np.asarray(stack_view_poses(geometry, views))
frames = cone_frames(geometry)
seen["cone"] = frames is not None
_fit.fit_poses(projections, nominal, grid, detector, keep_grid=True, frames=frames)
print(json.dumps(seen))
"""


def _level_peak(*shape, geometry="laminography", stage="level"):
    """A one-level fit's own device peak and its plan, in a fresh process on the GPU.

    ``stage`` "fine" measures the second of two levels, started from the first's volume;
    "refinement", the pose refinement after a level that holds nothing.
    """
    import json
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONWARNINGS": "ignore", "JAX_PLATFORMS": "cuda,cpu"}
    argv = [sys.executable, "-c", _PEAK_PROBE, *map(str, shape), geometry, stage]
    run = subprocess.run(argv, check=True, capture_output=True, text=True, env=env)
    return json.loads(run.stdout.strip().splitlines()[-1])


def _assert_the_plan_is_the_peak(m):
    # Nothing is counted: a level's volumes and stacks are allocated at its setup and what the
    # device holds then is measured, waste included; beside it the plan keeps the largest
    # transient (from its program's compile) free, with the one chunk it lands in, which the
    # allocator hands out whole when it is under twice the request and wastes under 128 MiB.
    # A level's own peak (in a fresh process: the allocator's peak less what was in use before
    # it) must not exceed the plan, and the plan may exceed the peak by no more than that chunk.
    plan = m["setup"] + m["free"]
    assert m["earlier"] < m["used"], "the peak was set before the stage measured"
    assert m["used"] <= plan, (m["used"], plan)
    assert plan - m["used"] <= m["chunk"], (plan - m["used"], m["chunk"])


@pytest.mark.gpu
@pytest.mark.fresh_gpu  # the probe's process needs the device's room, not what this one holds
@pytest.mark.parametrize("geometry", ["laminography", "cone"])
def test_the_plan_covers_a_levels_own_peak_and_not_much_more(geometry):
    # Where the volume dominates the Schur solve is the largest transient; the cone projector's
    # temporaries are its own.
    m = _level_peak(192, 128, 256, 160, 480, geometry=geometry)
    assert m["cone"] == (geometry == "cone")
    assert max(m["transients"], key=m["transients"].get) == "schur"
    _assert_the_plan_is_the_peak(m)


@pytest.mark.gpu
@pytest.mark.fresh_gpu
def test_the_plan_takes_the_larger_transient_where_the_prior_dominates():
    # 3000 views on a small grid: the prior's factor outweighs the Schur solve, and they never
    # run together, so the plan takes the larger of the two, not their sum (summed, K11
    # 67208's fine level sent j to the host for nothing, +10%).
    m = _level_peak(48, 32, 64, 48, 3000)
    assert max(m["transients"], key=m["transients"].get) == "prior"
    _assert_the_plan_is_the_peak(m)


@pytest.mark.gpu
@pytest.mark.fresh_gpu
def test_a_level_started_from_a_coarser_levels_volume_plans_its_peak_the_same_way():
    # The level upsamples the coarser level's volume into its own start volume, a program
    # compiled into it whose temporaries the plan counts; the coarse volume is the caller's
    # and alive throughout, so nothing the setup measured is freed before the peak.
    m = _level_peak(192, 128, 256, 160, 480, stage="fine")
    assert m["transients"]["start"] > 0
    _assert_the_plan_is_the_peak(m)


@pytest.mark.gpu
@pytest.mark.fresh_gpu
@pytest.mark.parametrize("geometry", ["laminography", "cone"])
def test_the_refinement_plans_its_own_peak_the_same_way(geometry):
    # The pose refinement holds the data and its precision and runs its own programs on the
    # level's volume: its plan is what its setup measures in use, its largest compiled
    # transient and that one chunk, like a level's.
    m = _level_peak(192, 128, 256, 160, 480, geometry=geometry, stage="refinement")
    assert m["cone"] == (geometry == "cone")
    _assert_the_plan_is_the_peak(m)
