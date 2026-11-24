"""Exercise worker ownership, native streams and admission over real stdio."""

import base64
from io import BytesIO
import json
import subprocess
import sys
import time

from PIL import Image
import pytest

from tests.test_mcp_stdio import connection

pytestmark = pytest.mark.surface


def python(c, code="", **kwargs):
    reply = c.call("python", code=code, **kwargs)
    assert not reply["isError"], reply
    return full_result(c, reply)["structuredContent"]


def full_result(c, reply):
    uri = reply["structuredContent"].get("continuation")
    if not uri:
        return reply
    chunks = []
    while uri:
        item = c.rpc("resources/read", {"uri": uri})["result"]["contents"][0]
        chunk = json.loads(item["text"])
        chunks.append(chunk["data"])
        uri = chunk["next"]
    return json.loads("".join(chunks))


def test_import_output_and_private_protocol():
    program = """
import os,sys
class Reporter:
    def find_spec(self, fullname, *args):
        if fullname == 'tomojax.cli.transport':
            os.write(1,b'native import log')
sys.meta_path.insert(0,Reporter())
from tomojax.cli.main import main
sys.exit(main(['serve']))
"""
    result = subprocess.run(
        [sys.executable, "-c", program], input=b"", capture_output=True, check=True
    )
    assert json.loads(result.stdout)["type"] == "ready"
    assert b"native import log" in result.stderr


def test_capture_reset_and_failed_registration(tmp_path):
    with connection(tmp_path) as c:
        out = python(
            c,
            """import ctypes, os, sys
x = np.arange(83)
print('python')
lib = ctypes.CDLL(None)
lib.printf(b'unflushed')
os.write(2,b'X'*12000+b'Y'*18000)
os.read(0,1) == b''""",
        )
        assert out["value"] and "python" in out["stdout"] and "unflushed" in out["stdout"]
        assert out["stderr"] == "X" * 10000 + "\n[10000 characters omitted]\n" + "Y" * 10000
        assert python(c, "print('next')")["stdout"] == "next\n"
        for error in ("SystemExit", "KeyboardInterrupt"):
            reply = c.call("python", code=f"survived=42\nregister(x)\nraise {error}('example')")
            assert reply["isError"]
            assert error in reply["content"][0]["text"]
            assert "traceback" in reply["structuredContent"]["next"]
        state = c.call("status")["structuredContent"]
        assert not state["handles"] and state["python_namespace_bytes"] >= 83 * 8
        assert python(c, "survived")["value"] == 42
        python(c, "class Notice:\n    def __del__(self): print('released')\nnotice=Notice()")
        reset = python(c, "'survived' in globals()", reset=True)
        assert reset["value"] is False and reset["stdout"] == "released\n"
        assert c.call("status")["structuredContent"]["python_namespace_bytes"] == 0


def test_namespace_views_private_snapshots_and_close(tmp_path):
    with connection(tmp_path) as c:
        out = python(
            c,
            """import jax
import jax.numpy as jnp
sim = tj.simulate(size=8,n_views=12)
g = sim.scan.geometry
x = np.ones((12,8,8),np.float32)
alias = x[:]
scan = register(tj.Scan(x,g))
alias[:] = 3
mm = np.memmap('buffer.raw',mode='w+',dtype='float32',shape=x.shape)
mm[:] = 7
disk = register(tj.Scan(mm,g))
other = np.memmap('buffer.raw',mode='r+',dtype='float32',shape=x.shape)
other[:] = 9
host = np.empty(8192,dtype='float32')
start = (-host.ctypes.data % 64)//4
z = host[start:start+x.size].reshape(x.shape)
z[:] = 4
j = jax.dlpack.from_dlpack(z,device=jax.devices('cpu')[0],copy=False)
assert j.unsafe_buffer_pointer() == z.ctypes.data
device = register(tj.Scan(j,g))
z[:] = 6
array = register(x)
[scan, disk, device, array]""",
        )
        refs = out["value"]
        assert set(refs) == set(out["handles"])
        check = python(
            c,
            """x[:] = 10
v = handles[scan].projections
assert not v.flags.writeable
[float(np.mean(handles[k].projections)) for k in (scan,disk,device)] + [int(handles[array].mean())]""",
        )
        assert check["value"] == [1, 7, 4, 3]
        derivative = python(c, "register(handles[scan].selected([1,3]))")["value"]
        c.call("control", action="close", target=refs[0])
        assert python(c, f"float(handles[{derivative!r}].projections.mean())")["value"] == 1
        c.call("control", action="restart")
        assert "restart_notice" in python(c, "'x' in globals()")


def test_late_child_and_close_during_queued_python(tmp_path):
    with connection(tmp_path) as c:
        ref = python(c, "register(np.arange(11))")["value"]
        python(
            c,
            """import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable,'-c', "import sys;sys.stdin.read(1);print('late' * 50000)"],stdin=subprocess.PIPE)
memory = np.zeros(128,np.uint8)""",
        )
        job = c.send(
            "tools/call",
            {
                "name": "python",
                "arguments": {
                    "code": """Path('running').touch()
while not Path('resume').exists(): time.sleep(.01)
child.stdin.close()
child.wait()"""
                },
            },
        )
        while not (tmp_path / "running").exists():
            assert c.process.poll() is None
            time.sleep(0.01)
        queued = c.send(
            "tools/call", {"name": "python", "arguments": {"code": f"int(handles[{ref!r}].sum())"}}
        )
        deadline = time.monotonic() + 10
        while True:
            state = c.call("status")["structuredContent"]
            if "queued" in state["jobs"].values():
                break
            assert time.monotonic() < deadline
        assert state["python_namespace_bytes"] >= 128
        c.call("control", action="close", target=ref)
        (tmp_path / "resume").touch()
        assert c.receive("id", job)["result"]["structuredContent"]["stdout"] == ""
        assert c.receive("id", queued)["result"]["structuredContent"]["value"] == 55
        assert python(c, f"{ref!r} in handles")["value"] is False


def test_second_publication_failure_rolls_back_group(tmp_path):
    with connection(tmp_path) as c:
        res = c.call(
            "python",
            code="""from tomojax.cli._staging import Stages
original = Stages.publish_array
calls = []
def injected(stages, token):
    calls.append(token)
    if len(calls) == 2:
        Stages.publish_array = original
        raise OSError('second publication')
    return original(stages,token)
Stages.publish_array = injected
scan = register(tj.simulate(size=8).scan)
a = register(np.arange(2))
b = register(np.arange(3))""",
        )
        assert res["isError"] and "second publication" in res["content"][0]["text"]
        assert not c.call("status")["structuredContent"]["handles"]
        assert python(c, "list(handles)")["value"] == []
        assert not list(tmp_path.rglob("*.stage.npy")) and not list(
            (tmp_path / "cache").rglob("*.npy")
        )


def test_python_progress_and_cooperative_cancel(tmp_path):
    with connection(tmp_path) as c:
        python(c, "state='before'\nsource=tj.simulate(size=8).scan")
        code = """from pathlib import Path
import time
with tj.progress.task('experiment','fit',None,''):
    while not Path('continue').exists():
        time.sleep(.01)
    register(tj.reconstruct(source,'fbp'))
state='completed'"""
        pending = c.send(
            "tools/call",
            {"name": "python", "arguments": {"code": code}, "_meta": {"progressToken": "fit"}},
        )
        notice = c.receive("method", "notifications/progress")
        assert notice["params"]["progressToken"] == "fit"
        jobs = c.call("status")["structuredContent"]["jobs"]
        (target,) = jobs
        assert not c.call("control", action="cancel", target=target)["isError"]
        (tmp_path / "continue").touch()
        terminal = c.receive("id", pending)["result"]
        assert terminal["isError"] and terminal["structuredContent"]["kind"] == "cancelled"
        assert python(c, "state")["value"] == "before"
        assert not c.call("status")["structuredContent"]["handles"]


def test_registration_copies_nested_metadata(tmp_path):
    with connection(tmp_path) as c:
        python(
            c,
            """from types import MappingProxyType
sim = tj.simulate(size=8,n_views=12)
metadata = np.arange(4)
info = {"pipeline": MappingProxyType({"coefficients":metadata})}
item = tj.Reconstruction(sim.phantom,sim.scan.geometry,"example",info,sim.scan)
saved = register(item)
metadata[:] = 0
info.clear()""",
        )
        got = python(
            c,
            """coeff = handles[saved].info["pipeline"]["coefficients"]
[coeff.tolist(),coeff.flags.writeable]""",
        )
        assert got["value"] == [[0, 1, 2, 3], False]


def test_registered_scan_retains_its_source_artifact(tmp_path):
    with connection(tmp_path) as c:
        array = python(c, "register(np.ones((24,8,8),np.float32))")["value"]
        registrations = python(
            c,
            f"""data=handles[{array!r}]
geometry=tj.simulate(size=8,n_views=12).scan.geometry
scan=register(tj.Scan(data[::2],geometry))
[str(data.filename), scan]""",
        )["value"]
        from pathlib import Path

        file, scan = Path(registrations[0]), registrations[1]
        assert file.exists()
        assert not c.call("control", action="close", target=array)["isError"]
        assert file.exists()
        assert python(c, f"float(handles[{scan!r}].projections.sum())")["value"] == 12 * 8**2
        assert not c.call("control", action="close", target=scan)["isError"]
        assert not file.exists()


def test_show_preserves_detail_up_to_panel_limit(tmp_path):
    with connection(tmp_path) as c:
        for width, height in ((1512, 1006), (700, 400)):
            reply = c.call("python", code=f"show(np.ones(({height}, {width})))")
            block = next(block for block in reply["content"] if block["type"] == "image")
            picture = Image.open(BytesIO(base64.b64decode(block["data"])))
            assert picture.width == min(width, 1024)
            assert picture.height == round(height * picture.width / width)


def test_python_subset_preserves_acquisition_path_across_calls(tmp_path):
    import tomojax as tj

    path = tmp_path / "scan.nxs"
    tj.save(path, tj.simulate(size=8, n_views=12).scan)
    with connection(tmp_path) as c:
        source = c.call("open", path=str(path), kind="scan")["structuredContent"]["handle"]
        python(c, f"subset = handles[{source!r}].selected(np.arange(0,12,2))")
        assert not c.call("control", action="close", target=source)["isError"]
        derived = python(c, "register(subset)")["value"]
        for dry_run in (True, False):
            reply = c.call("save", source=derived, path=str(path), force=True, dry_run=dry_run)
            assert reply["isError"] and "differ from input" in reply["content"][0]["text"]
        assert len(tj.load(path).angles) == 12


def test_registered_direct_load_protects_original_acquisition(tmp_path):
    import tomojax as tj

    path = tmp_path / "input.nxs"
    tj.save(path, tj.simulate(size=8, n_views=12).scan)
    with connection(tmp_path) as c:
        python(c, "subset = tj.load('input.nxs').selected(slice(0, 6))")
        derived = python(c, "import os\nos.chdir('..')\nregister(subset)")["value"]
        reply = c.call("save", source=derived, path=str(path), force=True)
        assert reply["isError"] and "differ from input" in reply["content"][0]["text"]
    assert len(tj.load(path).angles) == 12


def test_registered_volume_retains_acquisition_and_volume_paths(tmp_path):
    import tomojax as tj

    acquisition = tmp_path / "input.nxs"
    volume = tmp_path / "volume.nxs"
    tj.save(acquisition, tj.simulate(size=8, n_views=12).scan)
    tj.save(volume, tj.reconstruct(tj.load(acquisition), "fbp"))
    with connection(tmp_path) as c:
        handle = python(c, "register(tj.load_reconstruction('volume.nxs'))")["value"]
        for dest in (acquisition, volume):
            reply = c.call("save", source=handle, path=str(dest), force=True)
            assert reply["isError"] and "differ from input" in reply["content"][0]["text"]


def test_python_keeps_lazy_frames_offline_until_indexed(tmp_path):
    from tests._helpers import write_raw_nxtomo

    path = tmp_path / "raw.nxs"
    write_raw_nxtomo(path)
    with connection(tmp_path) as c:
        ref = c.call("open", path=str(path), kind="frames")["structuredContent"]["handle"]
        offline = path.rename(tmp_path / "offline.nxs")
        assert python(c, "40+2")["value"] == 42
        subset = python(
            c,
            f"""from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(1) as pool:
    subset = pool.submit(handles.__getitem__, {ref!r}).result().selected([1])
subset.views""",
        )
        assert subset["value"] == 1
        offline.rename(path)
        assert python(c, "subset.counts[0].tolist()")["value"] == [[9, 9], [9, 9]]
        registered = python(c, "register(subset)")["value"]
        path.unlink()
        assert python(c, f"handles[{registered!r}].counts[0].tolist()")["value"] == [[9, 9], [9, 9]]


def test_combined_scan_protects_both_python_inputs(tmp_path):
    import tomojax as tj

    scan = tj.simulate(size=8, n_views=12, geometry="cone").scan
    for name in ("a.nxs", "b.nxs"):
        tj.save(tmp_path / name, scan)
    with connection(tmp_path) as c:
        derived = python(c, "register(tj.Scan.combine([tj.load('a.nxs'), tj.load('b.nxs')]))")[
            "value"
        ]
        for name in ("a.nxs", "b.nxs"):
            reply = c.call("save", source=derived, path=name, force=True)
            assert reply["isError"] and "differ from input" in reply["content"][0]["text"]
            assert len(tj.load(tmp_path / name).angles) == 12


def test_directly_imported_loader_protects_registered_scan(tmp_path):
    import tomojax as tj

    tj.save(tmp_path / "data.npz", tj.simulate(size=8, n_views=12).scan)
    with connection(tmp_path) as c:
        python(c, "from tomojax.io import load_dataset as read")
        derived = python(
            c,
            """from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(1) as pool:
    pixels = pool.submit(read, 'data.npz').result().projections
geometry = tj.simulate(size=8, n_views=12).scan.geometry
register(tj.Scan(pixels * 2, geometry))""",
        )["value"]
        reply = c.call("save", source=derived, path="data.npz", force=True)
        assert reply["isError"] and "differ from input" in reply["content"][0]["text"]


@pytest.mark.parametrize("block_names", [False, True])
def test_save_protects_linked_volume_pixels(tmp_path, block_names):
    import h5py
    import numpy as np

    import tomojax as tj

    sim = tj.simulate(size=8, n_views=12)
    volume = tmp_path / "recon.nxs"
    tj.save(volume, tj.Reconstruction(sim.phantom, sim.scan.geometry, "test", {}, sim.scan))
    detector = tmp_path / "detector"
    detector.mkdir()
    dataset = "/entry/processing/tomojax/volume"
    with h5py.File(volume, "r+") as nexus, h5py.File(detector / "virtual.h5", "w") as virtual:
        pixels = nexus[dataset][...]
        layout = h5py.VirtualLayout(pixels.shape, pixels.dtype, maxshape=(None, *pixels.shape[1:]))
        parts = len(pixels) if block_names else 2
        for index, slab in enumerate(np.array_split(pixels, parts)):
            name = f"part{index}.h5"
            with h5py.File(detector / name, "w") as backing:
                backing["pixels"] = slab
            if not block_names:
                start = index * len(slab)
                layout[start : start + len(slab)] = h5py.VirtualSource(
                    name, "/pixels", shape=slab.shape
                )
        if block_names:
            layout[: h5py.h5s.UNLIMITED] = h5py.VirtualSource(
                "part%b.h5", "/pixels", shape=(1, *pixels.shape[1:])
            )
        virtual.create_virtual_dataset("volume", layout)
        del nexus[dataset]
        nexus[dataset] = h5py.ExternalLink("detector/virtual.h5", "/volume")
    with connection(tmp_path) as c:
        typed = c.call("open", path=volume.name, kind="reconstruction")["structuredContent"][
            "handle"
        ]
        registered = python(c, f"register(tj.load_reconstruction({volume.name!r}))")["value"]
        for handle in (typed, registered):
            for path in ("part0.h5", f"part{parts - 1}.h5", "virtual.h5"):
                reply = c.call("save", source=handle, path=str(detector / path), force=True)
                assert reply["isError"] and "differ from input" in reply["content"][0]["text"]
    with h5py.File(volume) as nexus:
        np.testing.assert_array_equal(nexus[dataset][...], pixels)


def test_large_python_images_and_value_keep_worker_alive(tmp_path):
    with connection(tmp_path) as c:
        saved = python(c, "register(np.arange(3))")["value"]
        reply = c.call(
            "python",
            code="rng=np.random.default_rng(1)\nfor i in range(14): show(rng.random((1024,1024)))",
        )
        assert not reply["isError"] and reply["structuredContent"]["continuation"]
        images = [item for item in full_result(c, reply)["content"] if item["type"] == "image"]
        assert len(images) == 14 and sum(len(item["data"]) for item in images) > 16 << 20
        reply = c.call("python", code="show(np.eye(3))\nregister(np.arange(7))\n'x' * (17 << 20)")
        assert reply["isError"] and "too large" in reply["content"][0]["text"]
        assert python(c, f"[list(handles), int(handles[{saved!r}].sum())]")["value"] == [[saved], 3]
        assert len(list((tmp_path / "cache").rglob("*.npy"))) == 1
