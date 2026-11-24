"""Request-scoped negotiation and process-lifetime MCP task results."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from mcp.server.extension import Extension, MethodBinding
from mcp.shared.exceptions import MCPError
from mcp_types import InputResponses, RequestParams
from pydantic import BaseModel, Field

from tomojax.cli.mcp.supervisor import Activity, activity

if TYPE_CHECKING:
    from mcp.server.context import CallNext, ServerRequestContext
    from mcp_types import CallToolRequestParams

    from tomojax.cli.mcp.supervisor import Object

TTL = 86_400
HEARTBEAT = 25.0
PROGRESS_INTERVAL = 1.0


async def notify_progress(job: Activity, ctx: ServerRequestContext[Any, Any]) -> None:
    """Sample the latest event independently of request admission and pipe draining."""
    loop = asyncio.get_running_loop()
    start = loop.time()
    while True:
        fresh = job.changed.is_set()
        job.changed.clear()
        elapsed = loop.time() - start
        message, eta = job.status()
        if not fresh:
            message, eta = f"waiting: {next(reversed(job.stages.values()), 'worker')}", None
        await ctx.session.report_progress(
            elapsed, elapsed + eta if eta is not None else None, message
        )
        await asyncio.sleep(PROGRESS_INTERVAL)
        with suppress(TimeoutError):
            await asyncio.wait_for(job.changed.wait(), max(0, HEARTBEAT - PROGRESS_INTERVAL))


class TaskParams(RequestParams):
    """Select one task by its opaque id."""

    task_id: str = Field(alias="taskId")


class UpdateParams(TaskParams):
    """Responses to outstanding input requests."""

    input_responses: InputResponses = Field(alias="inputResponses")


@dataclass
class Task:
    """Wire state, current job activity and execution lifetime."""

    fields: Object
    job: Activity
    runner: asyncio.Task[None]
    expiry: asyncio.TimerHandle


class Tasks(Extension):
    """Materialise eligible modern calls and serve task methods through the SDK."""

    identifier = "io.modelcontextprotocol/tasks"

    def __init__(self, parts: dict[str, list[str]]) -> None:
        self.parts = parts
        self.eligible: set[str] = set()
        self.store: dict[str, Task] = {}

    def methods(self) -> list[MethodBinding]:
        """Install task methods once, across all tool package reloads."""
        return [
            MethodBinding(f"tasks/{name}", cls, self.handle)
            for name, cls in (("get", TaskParams), ("cancel", TaskParams), ("update", UpdateParams))
        ]

    def supports(self, ctx: ServerRequestContext[Any, Any]) -> bool:
        """Check the capability on this request only."""
        caps = (ctx.meta or {}).get("io.modelcontextprotocol/clientCapabilities", {})
        return ctx.protocol_version == "2026-07-28" and self.identifier in caps.get(
            "extensions", {}
        )

    async def handle(self, ctx: ServerRequestContext[Any, Any], params: TaskParams) -> Object:
        """Poll, acknowledge stale input or request cooperative cancellation."""
        if not self.supports(ctx):
            raise MCPError(
                code=-32021,
                message="Declare the tasks extension on this request",
                data={"requiredCapabilities": {"extensions": {self.identifier: {}}}},
            )
        task = self.store.get(params.task_id)
        if task is None:
            raise MCPError(code=-32602, message="Unknown or expired task ID")
        if ctx.method == "tasks/get":
            fields = dict(task.fields)
            if fields["status"] == "working" and task.job.updated:
                fields["lastUpdatedAt"] = task.job.updated.isoformat()
                fields["statusMessage"] = task.job.status()[0]
            return {**fields, "resultType": "complete"}
        if ctx.method == "tasks/cancel" and task.fields["status"] == "working":
            task.job.request_cancel()
        return {"resultType": "complete"}

    async def intercept_tool_call(
        self,
        params: CallToolRequestParams,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> Any:
        """Register before returning the handle. Legacy and short calls block normally."""
        if (
            params.name not in self.eligible
            or (params.arguments or {}).get("dry_run") is True
            or not self.supports(ctx)
        ):
            job = Activity()
            token = activity.set(job)
            meta = ctx.meta or {}
            notify = (
                asyncio.create_task(notify_progress(job, ctx))
                if meta.get("progress_token", meta.get("progressToken")) is not None
                else None
            )
            try:
                return await call_next(ctx)
            finally:
                activity.reset(token)
                if notify:
                    notify.cancel()
                    await asyncio.gather(notify, return_exceptions=True)
        task_id = uuid4().hex
        now = datetime.now(UTC).isoformat()
        fields = {
            "taskId": task_id,
            "status": "working",
            "statusMessage": f"Starting {params.name}",
            "createdAt": now,
            "lastUpdatedAt": now,
            "ttlMs": TTL * 1000,
            "pollIntervalMs": 10000,
        }
        job = Activity()

        async def finish() -> None:
            token = activity.set(job)
            try:
                result = await call_next(ctx)
                if isinstance(result, BaseModel):
                    result = result.model_dump(mode="json", by_alias=True, exclude_none=True)
                if (
                    result is not None
                    and result.get("isError")
                    and (result.get("structuredContent") or {}).get("kind") == "cancelled"
                ):
                    fields.update(status="cancelled")
                else:
                    fields.update(status="completed", result=result)
            except MCPError as err:
                fields.update(
                    status="failed", error=err.error.model_dump(mode="json", exclude_none=True)
                )
            except Exception as err:
                fields.update(
                    status="failed",
                    error={"code": -32603, "message": str(err)},
                )
            finally:
                del fields["statusMessage"]
                fields["lastUpdatedAt"] = datetime.now(UTC).isoformat()
                activity.reset(token)

        def expire() -> None:
            task = self.store.pop(task_id)
            task.job.request_cancel()
            task.runner.cancel()
            content = task.fields.get("result", {}).get("structuredContent") or {}
            if uri := content.get("continuation"):
                self.parts.pop(uri.split("/")[-2], None)

        self.store[task_id] = Task(
            fields,
            job,
            asyncio.create_task(finish()),
            asyncio.get_running_loop().call_later(TTL, expire),
        )
        return {**fields, "resultType": "task"}

    async def close(self) -> None:
        """Stop retained work and TTL timers with the server."""
        for task in self.store.values():
            task.expiry.cancel()
            task.runner.cancel()
        await asyncio.gather(*(t.runner for t in self.store.values()), return_exceptions=True)
        self.store.clear()
