"""Workflow prompts and resources backed by packaged guides and the live worker."""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING

from mcp.server.mcpserver.exceptions import ResourceNotFoundError

if TYPE_CHECKING:
    from collections.abc import Callable

    from mcp.server.mcpserver import MCPServer

    from tomojax.cli.mcp.supervisor import Object, Supervisor


WORKFLOWS = (
    "reconstruct-scan",
    "lab-ct-start-to-finish",
    "find-the-centre",
    "check-a-reconstruction",
)
PYTHON_TOPICS = ("python", "solvers", "geometry", "calibration")
TOPICS = (
    *WORKFLOWS,
    *PYTHON_TOPICS,
    "pointing-and-features",
    "tuning-without-truth",
    "large-scans",
    "reporting-bugs",
)


def guide(topic: str) -> str:
    """Read a reference of the packaged Agent Skill, the single source of the guides."""
    if topic not in TOPICS:
        raise ResourceNotFoundError(f"Choose a guide from {', '.join(TOPICS)}")
    directory = resources.files("tomojax").joinpath("skill", "tomojax", "references")
    topics = PYTHON_TOPICS if topic == "python" else (topic,)
    return "\n\n".join(directory.joinpath(name + ".md").read_text().strip() for name in topics)


def workflow_prompt(text: str) -> Callable[..., str]:
    """Bind one guide to an optional acquisition path or live handle."""

    def prompt(source: str = "") -> str:
        return f"{text}\n\nInput: {source}" if source else text

    return prompt


async def operation_parameters(
    supervisor: Supervisor, operation: str, method: str | None = None
) -> Object:
    """Select operation and method fields from the running worker's schema."""
    g = await supervisor.worker()
    definitions = (await g.request({"type": "describe"}))["parameters"]["$defs"]
    names = {
        d["properties"]["type"]["enum"][0]: k
        for k, d in definitions.items()
        if "type" in d["properties"] and not k.endswith("Settings")
    }
    if operation not in names:
        raise ValueError(f"Unknown operation. Choose from {', '.join(sorted(names))}")
    entry = definitions[names[operation]]
    if method:
        methods = entry["properties"].get("method", {}).get("x-choices", [])
        if method not in methods:
            raise ValueError(f"method must be one of {methods}")
        config = next((k for k in definitions if k.lower() == method.lower() + "config"), None)
        if operation == "align" and method == "joint":
            config = "AlignConfig"
        if config:
            entry = definitions[config]
    return {
        "operation": operation,
        "method": method,
        "parameters": {
            k: inline(v, definitions) for k, v in entry["properties"].items() if k != "type"
        },
    }


def inline(schema: object, definitions: Object, seen: frozenset[str] = frozenset()) -> object:
    """Replace each ``$ref`` with its definition so the reply stands on its own.

    A definition already being expanded on the current path stays a named reference,
    and so do solver configurations, which describe with a method lists in full.
    """
    if isinstance(schema, list):
        return [inline(item, definitions, seen) for item in schema]
    if not isinstance(schema, dict):
        return schema
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/$defs/"):
        name = ref.removeprefix("#/$defs/")
        if name in seen or name not in definitions:
            return {"title": name}
        if name.endswith("Config"):
            return {"title": name, "description": "Expert settings: describe with method"}
        expanded = inline(definitions[name], definitions, seen | {name})
        assert isinstance(expanded, dict)
        return {**expanded, **{k: v for k, v in schema.items() if k != "$ref"}}
    return {k: inline(v, definitions, seen) for k, v in schema.items()}


def install_guidance(server: MCPServer, supervisor: Supervisor) -> None:
    """Register stable prompts and resource templates for this connection."""
    server.resource("tomojax://guide/{topic}", mime_type="text/markdown")(guide)
    for topic in WORKFLOWS:
        text = guide(topic)
        server.prompt(name=topic, description=text.splitlines()[0].removeprefix("# "))(
            workflow_prompt(text)
        )

    @server.resource("tomojax://handle/{id}", mime_type="application/json")
    async def handle(id: str) -> dict[str, object]:  # noqa: A002 - SDK binds the URI variable by name.
        """Read a live handle's summary and provenance."""
        try:
            record = supervisor.ledger.lookup(id)
        except ValueError as err:
            raise ResourceNotFoundError(str(err)) from err
        return {"handle": id, **record["summary"], "manifest": record["manifest"]}

    @server.resource("tomojax://describe/{operation}", mime_type="application/json")
    async def describe(operation: str) -> dict[str, object]:
        """Read the parameters, choices and units for one worker operation."""
        try:
            return await operation_parameters(supervisor, operation)
        except ValueError as err:
            raise ResourceNotFoundError(str(err)) from err
