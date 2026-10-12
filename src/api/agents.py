"""Serve-mode graphs: one per domain, sharing a Postgres checkpointer."""

import os
from functools import cache

from langgraph.checkpoint.postgres import PostgresSaver

from agent.context import GraphContext
from agent.graph import build_graph, recursion_limit
from agent.llm import llm_args, make_chat_model
from mcp_server.client import McpExecutor
from mcp_server.domains import fresh_environment, inner_tool_kinds, tool_specs
from mcp_server.session_store import pool


@cache
def checkpointer() -> PostgresSaver:
    # Tables are created once by scripts/migrate.py, never per request.
    return PostgresSaver(pool())


@cache
def chat_model():
    model = os.environ["AGENT_MODEL"]
    args = llm_args(
        model, os.environ["AGENT_PROVIDER"], cache=True, effort=os.getenv("AGENT_EFFORT")
    )
    return model, make_chat_model(model, args)


@cache
def serve_graph(domain: str):
    model, llm = chat_model()
    ctx = GraphContext(
        mode="serve",
        domain=domain,
        policy=fresh_environment(domain).get_policy(),
        llm=llm,
        model_name=model,
        tools=tool_specs(domain),
        discoverable_kinds=inner_tool_kinds(domain),
        approval_mode="human",
        max_steps=int(os.getenv("MAX_STEPS_PER_TURN", "30")),
        max_retries=int(os.getenv("MAX_RETRIES_PER_TURN", "2")),
        executor=McpExecutor(os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp/"), domain),
    )
    return build_graph(ctx, checkpointer=checkpointer()), ctx


def run_config(session_id: str, ctx: GraphContext) -> dict:
    return {"configurable": {"thread_id": session_id}, "recursion_limit": recursion_limit(ctx)}
