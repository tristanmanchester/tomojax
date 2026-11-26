from __future__ import annotations

import hashlib
import os
from pathlib import Path

import msgspec
import numpy as np
from PIL import Image
import pytest
from tools import verify_studio

import tomojax as tj

pytestmark = pytest.mark.surface
# A binary found on disk may be another checkout's or an older build, so name the one to test.
STUDIO_BINARY = os.environ.get("TOMOJAX_TEST_STUDIO")


def test_fixture_registers_a_scan_and_two_distinct_volumes(tmp_path: Path) -> None:
    projects = verify_studio.write_fixture(tmp_path / "fixture")
    assert set(projects) == {"prepare", "reconstruct", "explore", "compare", "jobs"}
    for label, project in projects.items():
        document = msgspec.json.decode(
            (project / "project.tomojax.json").read_bytes(), type=dict[str, object]
        )
        workspace = document["workspace"]
        assert isinstance(workspace, dict)
        assert workspace["workspace"] == label.title()
        if label == "compare":
            assert workspace["compare"] == ["ramp", "hann"]
        assert (project / "scan.nxs").is_file()
        assert (project / "ramp.nxs").is_file()
        assert (project / "hann.nxs").is_file()

    compare = projects["compare"]
    scan = tj.load(compare / "scan.nxs")
    assert len(scan.angles) == 64
    with pytest.warns(UserWarning, match="holds the reconstruction without its scan"):
        ramp = tj.load_reconstruction(compare / "ramp.nxs")
        hann = tj.load_reconstruction(compare / "hann.nxs")
    assert ramp.volume.shape == hann.volume.shape == (32, 32, 32)
    assert ramp.grid == hann.grid == scan.grid
    assert not np.allclose(ramp.volume, hann.volume)


def test_rgb_variance_is_spatial_and_uses_visible_pixels(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    image = Image.new("RGBA", (8, 4), (0, 0, 0, 255))
    image.paste((240, 240, 240, 255), (4, 0, 8, 4))
    image.save(path)
    assert verify_studio.validate_png(path, (8, 4)) == verify_studio.ImageStats(8, 4, 14400.0)

    # Transparent image data can vary without drawing a visible pixel.
    image.putalpha(0)
    image.save(path)
    with pytest.raises(ValueError, match="blank"):
        _ = verify_studio.validate_png(path, (8, 4))


def test_rejects_missing_uniform_wrong_size_and_non_png_images(tmp_path: Path) -> None:
    path = tmp_path / "frame.png"
    with pytest.raises(ValueError, match="was not produced"):
        _ = verify_studio.validate_png(path, (8, 4))

    image = Image.new("RGB", (8, 4), (220, 30, 92))
    image.save(path)
    with pytest.raises(ValueError, match="8x4, expected 4x8"):
        _ = verify_studio.validate_png(path, (4, 8))
    with pytest.raises(ValueError, match="blank"):
        _ = verify_studio.validate_png(path, (8, 4))

    image.save(path, format="BMP")
    with pytest.raises(ValueError, match="is not a PNG"):
        _ = verify_studio.validate_png(path, (8, 4))


def test_cargo_target_and_explicit_binary_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    built = target / "release" / "tomojax-studio"
    built.parent.mkdir(parents=True)
    built.touch(mode=0o700)
    monkeypatch.setenv("CARGO_TARGET_DIR", str(target))
    assert verify_studio.locate_binary() == built

    explicit = tmp_path / "studio"
    assert verify_studio.locate_binary(binary=explicit) is None
    explicit.touch(mode=0o700)
    assert verify_studio.locate_binary(binary=explicit) == explicit


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--size", "0x900"), ("--size", "1440"), ("--scale", "nan"), ("--scale", "0.00001")],
)
def test_cli_rejects_invalid_viewport(
    flag: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        _ = verify_studio.main([flag, value])
    assert exc.value.code == 2
    assert "positive WIDTHxHEIGHT" in capsys.readouterr().err


def test_cli_reports_missing_explicit_binary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert verify_studio.main(["--binary", str(tmp_path / "missing")]) == 1
    assert "Studio executable not found" in capsys.readouterr().err


def test_capture_requires_a_fresh_png_even_if_the_renderer_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    renderer = tmp_path / "renderer"
    _ = renderer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    renderer.chmod(0o700)
    output = tmp_path / "captures"
    output.mkdir()
    Image.new("RGB", (8, 4)).save(output / "start.png")
    assert (
        verify_studio.main(
            ["--binary", str(renderer), "--out-dir", str(output), "--size", "8x4", "--scale", "1"]
        )
        == 1
    )
    assert "capture failed for start: start.png was not produced" in capsys.readouterr().err
    assert {path.name for path in output.iterdir()} == {"start.png"}


@pytest.mark.skipif(STUDIO_BINARY is None, reason="set TOMOJAX_TEST_STUDIO to a Studio binary")
def test_native_studio_captures_all_six_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TOMOJAX_STUDIO_SETTINGS", str(tmp_path / "user-settings.json"))
    monkeypatch.setenv("TOMOJAX_STUDIO_INPUT", str(tmp_path / "user-project"))
    monkeypatch.setenv("TOMOJAX_STUDIO_DEMO", "1")
    output = tmp_path / "captures"
    assert (
        verify_studio.main(
            [
                "--binary",
                str(STUDIO_BINARY),
                "--out-dir",
                str(output),
                "--size",
                "1181x781",
                "--scale",
                "1.5",
            ]
        )
        == 0
    )
    assert not (tmp_path / "user-settings.json").exists()
    assert not (tmp_path / "user-project").exists()
    expected = {"start", "prepare", "reconstruct", "explore", "compare", "jobs"}
    assert {p.name for p in output.iterdir()} == {
        "manifest.json",
        *(f"{key}.png" for key in expected),
    }
    manifest = msgspec.json.decode((output / "manifest.json").read_bytes(), type=dict[str, object])
    assert manifest["size"] == "1181x781"
    assert manifest["scale"] == 1.5
    assert isinstance(manifest["views"], dict)
    assert manifest["views"].keys() == expected
    rendered: set[bytes] = set()
    for name in expected:
        path = output / f"{name}.png"
        with Image.open(path) as frame:
            assert frame.size == (1772, 1172)
            rendered.add(hashlib.sha256(frame.tobytes()).digest())
    assert len(rendered) == 6
