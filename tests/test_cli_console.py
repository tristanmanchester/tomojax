"""The command's human output: usage errors with hints, warnings once, bug reports."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import warnings

import numpy as np
import pytest

from tomojax.cli.main import main, run

from ._helpers import write_projection_dataset

pytestmark = pytest.mark.surface


def _usage_error(capsys, argv):
    with pytest.raises(SystemExit) as error:
        main(argv)
    assert error.value.code == 2
    return capsys.readouterr()


def test_an_existing_output_names_force_and_the_one_line_usage(tmp_path, capsys):
    scan, out = tmp_path / "scan.nxs", tmp_path / "recon.nxs"
    write_projection_dataset(scan)
    out.write_bytes(b"")
    err = _usage_error(capsys, ["reconstruct", str(scan), "-o", str(out)]).err
    assert err.splitlines() == [
        f"tomojax reconstruct: error: output already exists: {out}",
        "  hint: pass --force to replace it",
        "usage: tomojax reconstruct INPUT -o OUTPUT [options]   (tomojax reconstruct --help)",
    ]
    assert "\x1b[" not in err  # no color off a terminal
    # JSON gets the message alone; the hint is for people.
    captured = _usage_error(
        capsys, ["reconstruct", str(scan), "-o", str(out), "--progress", "json"]
    )
    assert json.loads(captured.out)["message"] == f"output already exists: {out}"


@pytest.mark.parametrize(
    ("argv", "suggestion"),
    [
        (["align", "scan.nxs", "-o", "out.nxs", "--fit", "centre"], "did you mean center?"),
        (["reconstruct", "scan.nxs", "-o", "out.nxs", "--iteratons", "3"], "--iterations?"),
        (["reconstruct", "scan.nxs", "-o", "out.nxs", "--method", "cgsl"], "did you mean cgls?"),
        (["recon", "scan.nxs"], "did you mean reconstruct?"),
    ],
)
def test_a_mistyped_name_suggests_the_close_one(tmp_path, capsys, monkeypatch, argv, suggestion):
    monkeypatch.chdir(tmp_path)
    write_projection_dataset(tmp_path / "scan.nxs")
    err = _usage_error(capsys, argv).err
    assert f"  hint: {suggestion}" in err or suggestion in err.splitlines()[1]


@pytest.mark.parametrize("given", ["file", "argv"])
def test_a_mistyped_config_key_suggests_the_key_and_the_listing(tmp_path, capsys, given):
    scan, config = tmp_path / "scan.nxs", tmp_path / "run.toml"
    write_projection_dataset(scan)
    config.write_text("iteratons = 3\n")
    argv = ["reconstruct", str(scan), "-o", str(tmp_path / "r.nxs"), "--method", "cgls"]
    argv += ["--config", str(config)]
    if given == "argv":
        argv += ["--iterations", "4"]
    err = _usage_error(capsys, argv).err.splitlines()
    assert err[0] == f"tomojax reconstruct: error: unknown config key(s) in {config}: iteratons"
    assert err[1] == (
        "  hint: did you mean iterations? tomojax reconstruct --config-keys lists every key"
    )


def test_user_text_is_never_read_as_markup(tmp_path, capsys):
    missing = tmp_path / "[red]scan[/red].nxs"
    assert main(["reconstruct", str(missing), "-o", str(tmp_path / "r.nxs")]) == 1
    assert f"input not found: {missing}" in capsys.readouterr().err


def test_a_tomojax_bug_prints_a_traceback_and_where_to_report_it(monkeypatch, tmp_path, capsys):
    from tomojax._workflow import operations as _workflow

    scan = tmp_path / "scan.nxs"
    write_projection_dataset(scan)

    def broken(*_args, **_kwargs):
        raise TypeError("wrong internal call shape")

    monkeypatch.setattr(_workflow, "reconstruct_resolved", broken)
    argv = ["tomojax", "reconstruct", str(scan), "-o", str(tmp_path / "r.nxs")]
    monkeypatch.setattr(sys, "argv", argv)
    assert run() == 1
    err = capsys.readouterr().err
    assert "TypeError: wrong internal call shape" in err
    assert "tomojax reconstruct: internal error: this is a bug in TomoJAX" in err
    assert "github.com/tristanmanchester/tomojax/issues" in err


def _noisy_reconstruct(monkeypatch):
    """A solve that warns and logs from TomoJAX and from another package."""
    import tomojax
    from tomojax import progress
    from tomojax._workflow import operations as _workflow

    def noisy(plan, *, out=None):
        scan = plan.scan
        progress.warn("tomojax says check this", stacklevel=2)  # at the CLI's call
        logging.getLogger("tomojax.recon").info("tomojax info line")
        logging.getLogger("jax._src").warning("jax says watch out")
        warnings.warn_explicit("other package warns", UserWarning, "/elsewhere/other.py", 1)
        volume = np.zeros((scan.grid.nz, scan.grid.ny, scan.grid.nx), np.float32)
        return tomojax.Reconstruction(volume, scan.geometry, "fbp", {}, scan)

    monkeypatch.setattr(_workflow, "reconstruct_resolved", noisy)


@pytest.mark.parametrize("mode", ["bar", "auto", "none", "json"])
def test_every_warning_and_log_record_appears_once(monkeypatch, tmp_path, capsys, mode):
    _noisy_reconstruct(monkeypatch)

    def printed(message, category, filename, lineno, file=None, line=None):
        print(f"{filename}:{lineno}: {category.__name__}: {message}", file=sys.stderr)

    monkeypatch.setattr(warnings, "showwarning", printed)  # Python's own, not pytest's capture
    scan = tmp_path / "scan.nxs"
    meta = {"detector_center_status": "not_recorded"}
    write_projection_dataset(scan, geometry_metadata=meta)
    package, root = logging.getLogger("tomojax"), logging.getLogger()
    level, handlers = package.level, list(root.handlers)
    shown = warnings.showwarning
    for attempt in range(2):  # a second command in one process behaves the same
        out = tmp_path / f"recon{attempt}.nxs"
        argv = ["reconstruct", str(scan), "-o", str(out), "--progress", mode]
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            assert main(argv) == 0
        captured = capsys.readouterr()
        text = captured.err + captured.out
        for message in (
            "tomojax says check this",
            "jax says watch out",
            "other package warns",
            "records no rotation center",
        ):
            assert text.count(message) == 1, (mode, message, text)
        # Info lines are shown only with live rows on a terminal, and as JSON records.
        expected_info = 1 if mode == "json" else 0
        assert text.count("tomojax info line") == expected_info, (mode, text)
        if mode == "json":
            assert "tomojax says check this" in captured.out
            assert "other package warns" in captured.err
        assert (package.level, root.handlers, warnings.showwarning) == (level, handlers, shown)


def test_neither_import_tomojax_nor_the_worker_loads_rich():
    code = (
        "import sys, tomojax, tomojax.cli.serve; "
        "assert not {'rich', 'rich_argparse'} & set(sys.modules), sorted(sys.modules)"
    )
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
