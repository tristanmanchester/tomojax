"""Serve tomography tools over MCP stdio."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Describe the MCP server command without loading the optional SDK."""
    return argparse.ArgumentParser(prog="tomojax mcp", description=__doc__)


def main() -> int:
    """Start the MCP server."""
    import asyncio

    build_parser().parse_args()
    try:
        from tomojax.cli.mcp.core import run
    except ModuleNotFoundError as err:
        if err.name == "mcp":
            raise ValueError("Install tomojax[mcp] to run this server") from err
        raise
    asyncio.run(run())
    return 0
