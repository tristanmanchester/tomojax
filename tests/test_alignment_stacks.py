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
    geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=-30)
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


@pytest.mark.slow
def test_a_level_with_little_device_room_plans_the_cold_stacks_onto_the_host(monkeypatch):
    # Room for the three stacks the matvec reads (sw, r, j) and the level's other arrays,
    # not for the data: the plan keeps the hot three and holds y, precision and residual
    # on the host, and the result is the one a device with room gives.
    expected_volume, _, expected, _ = _aligned(("translations",), DEVICE)
    plans = []
    original = _stacks.LevelStacks.__init__

    def room_for_the_hot_three(self, specs, batch, *, data, room, reserve, placement=None):
        sizes = {n: int(np.prod(s)) * np.dtype(d).itemsize for n, (s, d) in specs.items()}
        hot = sum(sizes[n] for n in ("sw", "r", "j") if n in sizes)
        total = sum(reserve.values())
        original(self, specs, batch, data=data, room=hot + total, reserve=reserve)
        plans.append(dict(self.planned))

    monkeypatch.setattr(_stacks.LevelStacks, "__init__", room_for_the_hot_three)
    volume, _, table, record = _aligned(("translations",))
    np.testing.assert_array_equal(table, expected)
    np.testing.assert_array_equal(np.asarray(volume), np.asarray(expected_volume))
    assert plans[0] == {**dict.fromkeys(("sw", "r", "j"), "device"), **COLD}
    assert record["levels"][0]["stacks"]["mispredicted"] == []


@pytest.mark.slow
def test_a_stack_the_device_refuses_goes_to_the_host_with_every_stack_after_it(monkeypatch):
    # The plan is a prediction from the total room; the allocator is the judge. A refused
    # allocation is a misprediction: that stack and the ones after it in device order go to
    # the host, the level says so, and nothing else changes.
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
    assert stacks["mispredicted"] == ["j", "y", "precision", "residual"]


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


def test_released_stacks_leave_nothing_of_the_level_on_the_device():
    projections, nominal, grid, detector = _scan()
    held = []
    original = _stacks.LevelStacks.close

    def close(self):
        held.append([s.data for s in self.stacks.values() if not s.host])
        original(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_stacks.LevelStacks, "close", close)
        _fit.fit_poses(projections, nominal, grid, detector, placement=COLD)
    assert held and all(array.is_deleted() for level in held for array in level)


def test_align_reports_each_levels_placement_and_time():
    projections, _, grid, detector = _scan()
    geometry = tj.LaminographyGeometry(
        grid, detector, np.linspace(0, 360, len(projections), endpoint=False), tilt_deg=-30
    )
    info = tj.align(tj.Scan(projections, geometry)).info
    for level in [*info["levels"], info["refinement"]]:
        stacks = level["stacks"]
        assert set(stacks["placement"].values()) <= {"device", "host"}
        assert stacks["planned"].keys() == stacks["placement"].keys()
        assert level["seconds"] > 0


_PEAK_PROBE = """
import json, sys
import jax, numpy as np
import tomojax as tj
from tomojax.alignment._joint import _fit, _solver
from tomojax.geometry import stack_view_poses

n, nz, nu, nv, views = (int(a) for a in sys.argv[1:6])
_fit._LEVELS = ((1, 8),)  # one level: the process's peak is the level's
dev, seen = jax.devices()[0], {}
solve = _solver.solve


def measured(*args, **kwargs):
    base = dev.memory_stats()["bytes_in_use"]
    out = solve(*args, **kwargs)
    stacks, prior = out[3]["stacks"], args[2]
    u = prior.matrices.U
    seen.update(
        used=dev.memory_stats()["peak_bytes_in_use"] - base,
        stacks=sum(stacks["stack_bytes"].values()),
        reserve=stacks["reserve_bytes"],
        volume=4 * int(np.prod(kwargs["ops"].shape)),
        design=u.dtype.itemsize * prior.local_size * 2 * u.shape[1],  # one copy of b
    )
    return out


_solver.solve = _fit.solve = measured
grid, detector = tj.Grid(n, n, nz, 1, 1, 1), tj.Detector(nu, nv, 1, 1)
angles = np.linspace(0, 360, views, endpoint=False)
geometry = tj.LaminographyGeometry(grid, detector, angles, tilt_deg=-30)
rng = np.random.default_rng(1)
volume = rng.uniform(0, 1, (nz, n, n)).astype(np.float32)
with jax.default_device(jax.devices("cpu")[0]):
    projections = np.array(tj.project(geometry, volume))
projections += rng.normal(0, 0.01, projections.shape).astype(np.float32)
nominal = np.asarray(stack_view_poses(geometry, views))
_fit.fit_poses(projections, nominal, grid, detector, keep_grid=True)
print(json.dumps(seen))
"""


def _level_peak(*shape):
    """A one-level fit's own device peak and its plan, in a fresh process on the GPU."""
    import json
    import os
    import subprocess
    import sys

    env = {**os.environ, "PYTHONWARNINGS": "ignore", "JAX_PLATFORMS": "cuda,cpu"}
    argv = [sys.executable, "-c", _PEAK_PROBE, *map(str, shape)]
    run = subprocess.run(argv, check=True, capture_output=True, text=True, env=env)
    return json.loads(run.stdout.strip().splitlines()[-1])


@pytest.mark.gpu
@pytest.mark.fresh_gpu  # the probe's process needs the device's room, not what this one holds
def test_the_plan_covers_a_levels_own_peak_and_not_much_more():
    # The reserve is counted (_stacks.level_reserve): the state, 9 volumes and 0.2 GiB that no
    # array holds; the prior's held bytes; and the larger of the Schur solve's temporaries
    # (9 volumes and 8 batches) and the prior factor's. It holds only because nothing a level
    # compiles grows faster than the views but the prior (test_alignment_memory). A level's own
    # peak, in a fresh process (the allocator's peak less what was in use at its setup), must
    # not exceed the plan, nor fall short of it by more than 6 volumes: on this level, where
    # volumes dominate, the plan's spare is what the two counts hold over this level's own
    # (they are set by the largest of three real scans, I13 159815).
    m = _level_peak(192, 128, 256, 160, 480)
    predicted = m["stacks"] + m["reserve"]
    assert m["used"] <= predicted, (m["used"], predicted)
    spare = (predicted - m["used"]) / m["volume"]
    assert spare <= 6, f"the plan reserves {spare:.1f} volumes more than the level used"


@pytest.mark.gpu
@pytest.mark.fresh_gpu
def test_the_plan_takes_the_larger_transient_where_the_prior_dominates():
    # 3000 views on a small grid: the prior's factor (about five copies of its design b at its
    # peak) outweighs the Schur solve, and they never run together, so the plan takes the
    # larger of the two, not their sum (summed, K11 67208's fine level sent j to the host for
    # nothing, +10%). The level's peak must not exceed the plan, and the plan's spare must be
    # no more than what it is made of: the state (9 volumes and 0.2 GiB, measured on real
    # scans), which a level this small does not reach; the transient's sixth copy of b
    # and its 8 MiB against 5 MB measured; and 2 volumes.
    m = _level_peak(48, 32, 64, 48, 3000)
    predicted = m["stacks"] + m["reserve"]
    assert m["used"] <= predicted, (m["used"], predicted)
    state = 9 * m["volume"] + 200 * 2**20
    margin = state + m["design"] + 8 * 2**20 + 2 * m["volume"]
    assert predicted - m["used"] <= margin, (predicted - m["used"], margin)
