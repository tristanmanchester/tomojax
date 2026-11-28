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
    assert not reply["isError"], reply["content"][0]["text"]
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
            """import sys
x = np.arange(83)
print('python')
print('X' * 12000 + 'Y' * 18000, end='', file=sys.stderr)
x.size""",
        )
        assert out["value"] == 83 and out["stdout"] == "python\n"
        assert (
            out["stderr"]
            == "X" * 10000
            + "\n[10000 characters omitted; write long output to a file and read it in parts]\n"
            + "Y" * 10000
        )
        assert python(c, "print('next')")["stdout"] == "next\n"
        for error in ("SystemExit", "KeyboardInterrupt"):
            reply = c.call("python", code=f"survived=42\nregister(x)\nraise {error}('example')")
            assert reply["isError"]
            assert error in reply["content"][0]["text"]
            assert "error" in reply["structuredContent"]["next"]
        state = c.call("status")["structuredContent"]
        assert not state["handles"] and state["python_namespace_bytes"] >= 83 * 8
        assert python(c, "survived")["value"] == 42
        python(c, "class Notice:\n    def __del__(self): print('released')\nnotice=Notice()")
        reset = python(c, "'survived' in globals()", reset=True)
        assert reset["value"] is False and reset["stdout"] == "released\n"
        assert c.call("status")["structuredContent"]["python_namespace_bytes"] == 0


def test_registered_objects_come_back_through_handles_until_closed(tmp_path):
    with connection(tmp_path) as c:
        out = python(
            c,
            """sim = tj.simulate(size=8, n_views=12)
x = np.ones((12, 8, 8), np.float32)
scan = register(tj.Scan(x, sim.scan.geometry))
array = register(np.full(5, 3))
[scan, array]""",
        )
        scan, array = out["value"]
        assert {scan, array} == set(out["handles"])
        check = python(c, "[float(handles[scan].projections.mean()), int(handles[array].mean())]")
        assert check["value"] == [1, 3]
        derivative = python(c, "register(handles[scan].selected([1, 3]))")["value"]
        c.call("control", action="close", target=scan)
        assert python(c, f"float(handles[{derivative!r}].projections.mean())")["value"] == 1
        assert python(c, "scan in handles")["value"] is False
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
        assert terminal["isError"] and terminal["structuredContent"]["kind"] == "canceled"
        assert python(c, "state")["value"] == "before"
        assert not c.call("status")["structuredContent"]["handles"]


def test_cancel_during_percentile_selection_preserves_worker_and_handle(tmp_path):
    import socket

    with connection(tmp_path) as c, socket.create_server(("127.0.0.1", 0)) as listener:
        listener.settimeout(10)
        source, pid = python(
            c,
            f"""import os, socket
live = register(tj.simulate(size=8, n_views=12).scan)
samples = register(np.arange((1 << 21) + 1, dtype=np.float32))
search = np.searchsorted
chunks = []
def gated_search(run, *args, **kwargs):
    if isinstance(run, np.memmap):
        chunks.append(run.size)
        if len(chunks) == 1:
            with socket.create_connection({listener.getsockname()!r}, timeout=10) as gate:
                gate.recv(1)
    return search(run, *args, **kwargs)
np.searchsorted = gated_search
[samples, os.getpid()]""",
        )["value"]
        request = c.send(
            "tools/call", {"name": "view", "arguments": {"source": source, "mode": "statistics"}}
        )
        gate, _ = listener.accept()
        with gate:
            (target,) = c.call("status")["structuredContent"]["jobs"]
            assert not c.call("control", action="cancel", target=target)["isError"]
        terminal = c.receive("id", request)["result"]
        assert terminal["isError"] and terminal["structuredContent"]["kind"] == "canceled"
        assert python(c, "[os.getpid(), len(chunks), len(handles[live].angles)]")["value"] == [
            pid,
            1,
            12,
        ]


def test_a_scan_built_on_a_registered_array_outlives_that_array(tmp_path):
    with connection(tmp_path) as c:
        array = python(c, "register(np.ones((24,8,8),np.float32))")["value"]
        scan = python(
            c,
            f"""geometry=tj.simulate(size=8,n_views=12).scan.geometry
register(tj.Scan(handles[{array!r}][::2],geometry))""",
        )["value"]
        assert not c.call("control", action="close", target=array)["isError"]
        assert python(c, f"float(handles[{scan!r}].projections.sum())")["value"] == 12 * 8**2


def test_show_preserves_detail_up_to_panel_limit(tmp_path):
    with connection(tmp_path) as c:
        for width, height in ((1512, 1006), (700, 400)):
            reply = c.call("python", code=f"show(np.ones(({height}, {width})))")
            block = next(block for block in reply["content"] if block["type"] == "image")
            picture = Image.open(BytesIO(base64.b64decode(block["data"])))
            assert picture.width == min(width, 1024)
            assert picture.height == round(height * picture.width / width)


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
        # Only the registered array survives the failed publication. The worker chooses
        # the session directory on the machine where it runs.
        count = "import tempfile\nfrom pathlib import Path\nlen(list(Path(tempfile.gettempdir()).rglob('*.npy')))"
        assert python(c, count)["value"] == 1
