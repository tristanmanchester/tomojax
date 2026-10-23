"""The public docstring gate validates objects, not similarly named modules."""

from pathlib import Path

from tools.check_docstrings import PUBLIC_DATA, docstring_errors, public_data, public_objects


def test_public_discovery_includes_methods_and_jit_functions():
    objects = dict(public_objects())
    assert "tomojax.Scan.with_poses" in objects
    assert "tomojax.Scan.grid" in objects
    assert "tomojax.corrections.unwrap_phase" in objects
    assert set(public_data()) <= PUBLIC_DATA.keys()
    assert "tomojax.geometry.Grid" not in objects  # Already exported at the root.


def test_missing_parameters_and_entrypoint_examples_are_rejected():
    def incomplete(value):
        """Return a value."""
        return value

    helper_errors = dict(docstring_errors("example.helper", incomplete))
    assert "PR01" in helper_errors
    assert "RT01" in helper_errors
    assert "EX01" not in helper_errors
    assert "EX01" in dict(docstring_errors("tomojax.load", incomplete))


def test_generated_reference_uses_public_volume_axes(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    from tools import build_api_docs

    monkeypatch.setattr(build_api_docs, "__file__", str(tmp_path / "tools" / "build_api_docs.py"))
    monkeypatch.setattr(build_api_docs, "build_main", lambda _: 0)
    assert build_api_docs.main([]) == 0
    index = (tmp_path / ".artifacts" / "api-source" / "index.rst").read_text()
    assert "Volumes use ``(z, y, x)``" in index
    assert "projections use ``(view, v, u)``" in index
