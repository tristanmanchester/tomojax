"""Print where the TomoJAX Agent Skill is installed, or copy it where an agent reads skills."""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path
import shutil
import sys
import tempfile

# Where common agents look for skills; the Skill goes in a ``tomojax`` folder inside.
LOCATIONS = (
    ("~/.claude/skills", "Claude Code, every project"),
    (".claude/skills", "Claude Code, this project"),
    ("~/.codex/skills", "Codex"),
    (".agents/skills", "other agents that read agentskills.io skills"),
)


def build_parser() -> argparse.ArgumentParser:
    """Describe the skill command."""
    parser = argparse.ArgumentParser(
        prog="tomojax skill",
        description=(
            "Print where the TomoJAX Agent Skill is in this installation, or copy it into "
            "OUTPUT/tomojax so an agent can load it."
        ),
    )
    _ = parser.add_argument(
        "-o", "--output", type=Path, help="Skills directory to copy the Skill into"
    )
    _ = parser.add_argument(
        "--force", action="store_true", help="Replace an existing OUTPUT/tomojax"
    )
    return parser


def main() -> int:
    """Print the Skill's location, or install a copy of it."""
    args = build_parser().parse_args()
    with resources.as_file(resources.files("tomojax").joinpath("skill", "tomojax")) as source:
        if args.output is None:
            lines = [
                f"The TomoJAX Agent Skill is at {source}",
                "Copy it where your agent reads skills:",
            ]
            width = max(len(where) for where, _ in LOCATIONS)
            lines += [f"  tomojax skill -o {where:<{width}}  {who}" for where, who in LOCATIONS]
            sys.stdout.write("\n".join(lines) + "\n")
            return 0
        destination = args.output.expanduser().resolve() / "tomojax"
        if destination.exists() and not args.force:
            raise ValueError(f"{destination} exists; pass --force to replace it")
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Copy beside the destination, then swap it in, so a failed copy leaves the old one.
        staging = Path(tempfile.mkdtemp(prefix=".tomojax-skill-", dir=destination.parent))
        try:
            shutil.copytree(
                source, staging / "tomojax", ignore=shutil.ignore_patterns("__pycache__")
            )
            if destination.exists():
                shutil.rmtree(destination)
            (staging / "tomojax").rename(destination)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    sys.stdout.write(f"Installed the TomoJAX Agent Skill at {destination}\n")
    return 0
