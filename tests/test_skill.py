"""The packaged Agent Skill follows the agentskills.io rules and installs where agents look."""

from pathlib import Path
import re
import subprocess
import sys

import pytest
import yaml

from tests.test_mcp_guides import guide
from tests.test_mcp_stdio import connection
import tomojax

pytestmark = pytest.mark.surface

SKILL = Path(tomojax.__file__).parent / "skill" / "tomojax"
ROOT = Path(__file__).resolve().parents[1]


def parts():
    text = (SKILL / "SKILL.md").read_text()
    _, front, body = text.split("---\n", 2)
    return yaml.safe_load(front), body


def test_frontmatter_follows_the_specification():
    # https://agentskills.io/specification
    meta, body = parts()
    assert set(meta) <= {"name", "description", "license", "compatibility", "metadata"}
    assert meta["name"] == SKILL.name == "tomojax"
    assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", meta["name"]) and len(meta["name"]) <= 64
    assert 1 <= len(meta["description"]) <= 1024 and "Use when" in meta["description"]
    assert meta["license"] == "GPL-3.0-only"
    assert len(meta["compatibility"]) <= 500 and "Python 3.12" in meta["compatibility"]
    assert len(body.splitlines()) < 500


def test_every_reference_is_linked_once_deep_and_exists():
    _, body = parts()
    links = set(re.findall(r"\]\(((?:references|scripts)/[^)#]+)\)", body))
    files = {str(p.relative_to(SKILL)) for p in SKILL.glob("references/*.md")}
    assert links >= files  # SKILL.md says when to read each reference
    for link in links:
        assert (SKILL / link).is_file(), link
    for reference in SKILL.glob("references/*.md"):
        for target in re.findall(r"\]\(([\w.-]+\.md)\)", reference.read_text()):
            assert (SKILL / "references" / target).is_file(), (reference.name, target)


def run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "tomojax.cli.main", "skill", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
        check=False,
    )


def test_skill_command_prints_and_installs_a_copy(tmp_path):
    shown = run()
    assert shown.returncode == 0 and str(SKILL) in shown.stdout
    assert "~/.claude/skills" in shown.stdout and "~/.codex/skills" in shown.stdout

    installed = run("-o", str(tmp_path / "skills"))
    assert installed.returncode == 0, installed.stderr
    copy = tmp_path / "skills" / "tomojax"
    assert (copy / "SKILL.md").read_text() == (SKILL / "SKILL.md").read_text()
    assert (copy / "scripts" / "feature_scores.py").is_file()
    assert not list(copy.rglob("__pycache__"))

    (copy / "SKILL.md").write_text("edited")
    refused = run("-o", str(tmp_path / "skills"))
    assert refused.returncode == 1 and "--force" in refused.stderr
    assert (copy / "SKILL.md").read_text() == "edited"
    assert run("-o", str(tmp_path / "skills"), "--force").returncode == 0
    assert (copy / "SKILL.md").read_text() == (SKILL / "SKILL.md").read_text()
    assert [p.name for p in (tmp_path / "skills").iterdir()] == ["tomojax"]


def test_the_plugin_and_the_server_read_the_same_skill(tmp_path):
    plugin = ROOT / "plugins" / "tomojax" / "skills" / "tomojax"
    assert plugin.resolve() == SKILL.resolve()
    with connection(tmp_path) as c:
        for topic in ("pointing-and-features", "tuning-without-truth", "reporting-bugs"):
            assert guide(c, topic) == (SKILL / "references" / f"{topic}.md").read_text().strip()
