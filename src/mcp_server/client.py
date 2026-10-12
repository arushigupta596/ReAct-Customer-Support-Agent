"""The graph's tool executor in serve mode: one MCP call per approved tool call."""

import asyncio
import json
import os

from mcp import ClientSession
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client


class McpExecutor:
    def __init__(self, url: str, domain: str):
        self.url = url
        self.domain = domain

    async def _call(self, session_id: str, name: str, args: dict) -> tuple[str, bool]:
        headers = {"X-Session-Id": session_id, "X-Internal-Token": os.environ["MCP_TOKEN"]}
        async with (
            create_mcp_http_client(headers=headers) as http,
            streamable_http_client(self.url, http_client=http) as (read, write),
            ClientSession(read, write) as mcp,
        ):
            await mcp.initialize()
            result = await mcp.call_tool(f"{self.domain}.{name}", args)
        text = "".join(c.text for c in result.content if getattr(c, "type", None) == "text")
        return text, bool(result.is_error)

    def __call__(self, name: str, args: dict, session_id: str) -> tuple[str, bool]:
        # Graph nodes are sync and run in a worker thread with no event loop of its own.
        return asyncio.run(self._call(session_id, name, json.loads(json.dumps(args))))
