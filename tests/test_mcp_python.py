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
    data = reply["structuredContent"]
    if uri := data.get("continuation"):
        chunks = []
        while uri:
            item = c.rpc("resources/read", {"uri": uri})["result"]["contents"][0]
            chunk = json.loads(item["text"])
            chunks.append(chunk["data"])
            uri = chunk["next"]
        data = json.loads("".join(chunks))["structuredContent"]
    return data


def test_import_output_and_private_protocol():
    program = """
import os,sys
class Reporter:
    def find_spec(self, fullname, *args):
        if fullname == 'tomojax.cli._transport':
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
