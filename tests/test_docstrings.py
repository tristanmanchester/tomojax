"""The public docstring gate validates objects, not similarly named modules."""

import ast
from pathlib import Path

from numpydoc.docscrape import NumpyDocString
from tools.check_docstrings import PUBLIC_DATA, docstring_errors, public_data, public_objects


def test_public_discovery_includes_methods_and_jit_functions():
    objects = dict(public_objects())
    assert "tomojax.Scan.with_poses" in objects
    assert "tomojax.Scan.grid" in objects
    assert "tomojax.corrections.unwrap_phase" in objects
    assert "tomojax.datasets.SimulationArtefacts.has_enabled" in objects
    assert "tomojax.forward.joseph_pose_normal_equations" in objects
    assert "tomojax.forward.api.residual_loss" in objects
    assert set(public_data()) <= PUBLIC_DATA.keys()
    assert "tomojax.geometry.Grid" not in objects  # Already exported at the root.


def test_simulation_and_forward_exports_have_complete_sections():
    modules = ("tomojax.datasets", "tomojax.forward", "tomojax.forward.api")
    errors = {
        name: problems
        for name, obj in public_objects(modules)
        if (problems := docstring_errors(name, obj))
    }
    assert not errors


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


def test_optional_torch_sections_without_importing_pytorch():
    source = Path(__file__).parents[1] / "src/tomojax/torch.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Projector"
    )
    methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
    for name in ("__init__", "forward", "adjoint"):
        method = methods[name]
        doc = NumpyDocString(ast.get_docstring(cls if name == "__init__" else method) or "")
        expected = {arg.arg for arg in [*method.args.args, *method.args.kwonlyargs]} - {"self"}
        assert expected == {param.name for param in doc["Parameters"]}
        if name != "__init__":
            assert doc["Returns"] and doc["Raises"]
