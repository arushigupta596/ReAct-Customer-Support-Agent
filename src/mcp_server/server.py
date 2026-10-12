"""MCP tool server (streamable HTTP) exposing the tau2 domain toolkits as `<domain>.<tool>`.

Every request carries `X-Session-Id` and `X-Internal-Token` headers. The session decides the
domain: list_tools returns only that domain's tools, and call_tool refuses anything else
(the third domain lock, after the API and guard_pre_tool).
"""

import hmac
import json
import os
from functools import cache

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from tau2.data_model.message import ToolCall

from mcp_server import session_store
from mcp_server.domains import DOMAINS, ENVIRONMENTS, fresh_environment, tool_specs


class McpAuthError(Exception):
    pass


def _session_for(ctx) -> dict:
    headers = ctx.request.headers if ctx.request is not None else {}
    token = headers.get("x-internal-token", "")
    if not hmac.compare_digest(token, os.environ["MCP_TOKEN"]):
        raise McpAuthError("invalid internal token")
    session = session_store.get_session(headers.get("x-session-id", ""))
    if session is None or session["domain"] not in DOMAINS:
        raise McpAuthError("unknown session")
    return session


@cache
def list_domain_tools(domain: str) -> list[types.Tool]:
    return [
        types.Tool(
            name=f"{domain}.{t.name}",
            description=t.openai_schema["function"].get("description", ""),
            input_schema=t.openai_schema["function"]["parameters"],
        )
        for t in fresh_environment(domain).get_tools()
    ]


async def on_list_tools(ctx, params) -> types.ListToolsResult:
    session = _session_for(ctx)
    return types.ListToolsResult(tools=list_domain_tools(session["domain"]))


def _error(text: str) -> types.CallToolResult:
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], is_error=True)


def execute(session: dict, name: str, arguments: dict) -> tuple[str, bool]:
    """Run one tool against the session's environment; log it if it changed the DB."""
    domain, _, tool = name.partition(".")
    if domain != session["domain"]:
        return f"Error: tool {name} is outside this session's domain ({session['domain']}).", True
    specs = tool_specs(domain)
    if tool not in specs:
        return f"Error: unknown tool {name}.", True
    sid = session["session_id"]
    with ENVIRONMENTS.lock(sid):
        env = ENVIRONMENTS.get(sid, domain)
        result = env.get_response(ToolCall(id="mcp", name=tool, arguments=arguments))
        if not result.error and specs[tool].mutates_state:
            session_store.append_action(sid, tool, arguments)
    return result.content or "", result.error


USER_TOOL = "call_discoverable_user_tool"


def given_user_actions(session: dict) -> list[dict]:
    """Actions the agent has handed to the customer (banking), with the agent's arguments."""
    env = ENVIRONMENTS.get(session["session_id"], session["domain"])
    state_fn = getattr(env.tools, "get_user_discoverable_tools_state", None)
    if state_fn is None:
        return []
    return [
        {
            "tool": name,
            "description": info["tool_info"].get("description", ""),
            "arguments": info["arguments"],
        }
        for name, info in state_fn().items()
    ]


def execute_user_action(session: dict, tool: str, arguments: dict) -> tuple[str, bool]:
    """Run a handed-over action as the customer; logged with requestor 'user' for replay."""
    args = {"discoverable_tool_name": tool, "arguments": json.dumps(arguments)}
    sid = session["session_id"]
    with ENVIRONMENTS.lock(sid):
        env = ENVIRONMENTS.get(sid, session["domain"])
        result = env.get_response(
            ToolCall(id="user", name=USER_TOOL, arguments=args, requestor="user")
        )
        if not result.error and not (result.content or "").startswith("Error"):
            session_store.append_action(sid, USER_TOOL, args, requestor="user")
    return result.content or "", result.error


async def on_call_tool(ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
    try:
        session = _session_for(ctx)
    except McpAuthError as e:
        return _error(f"Error: {e}")
    content, is_error = await anyio.to_thread.run_sync(
        execute, session, params.name, dict(params.arguments or {})
    )
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=content)], is_error=is_error
    )


server = Server(
    "react-support-tools",
    version="0.1.0",
    instructions="tau2-bench retail, airline and banking tools, locked to one domain per session.",
    on_list_tools=on_list_tools,
    on_call_tool=on_call_tool,
)
