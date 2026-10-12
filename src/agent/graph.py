"""The ReAct graph shared by bench and serve mode. Only the tool-execution node differs."""

from functools import partial

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agent.context import GraphContext
from agent.nodes.approval import approval_gate, route_after_approval
from agent.nodes.emit import emit
from agent.nodes.execute import execute_tools
from agent.nodes.guard_input import guard_input
from agent.nodes.guard_output import guard_output, route_after_output
from agent.nodes.guard_pre_tool import guard_pre_tool, route_after_pre_tool
from agent.nodes.reason import reason
from agent.state import AgentState


def route_after_reason(state: AgentState) -> str:
    return "guard_pre_tool" if state["messages"][-1].tool_calls else "guard_output"


def build_graph(ctx: GraphContext, checkpointer: BaseCheckpointSaver | None = None):
    g = StateGraph(AgentState)

    g.add_node("guard_input", partial(guard_input, ctx=ctx))
    g.add_node("reason", partial(reason, ctx=ctx))
    g.add_node("guard_pre_tool", partial(guard_pre_tool, ctx=ctx))
    g.add_node("approval_gate", partial(approval_gate, ctx=ctx))
    g.add_node("guard_output", partial(guard_output, ctx=ctx))
    if ctx.mode == "serve":
        g.add_node("execute_tools", partial(execute_tools, ctx=ctx))
        g.add_edge("execute_tools", "reason")
        after_approval = ["execute_tools", "reason"]
    else:
        g.add_node("emit", partial(emit, ctx=ctx))
        g.add_edge("emit", END)
        after_approval = ["emit", "reason"]

    g.add_edge(START, "guard_input")
    g.add_edge("guard_input", "reason")
    g.add_conditional_edges("reason", route_after_reason, ["guard_pre_tool", "guard_output"])
    g.add_conditional_edges("guard_pre_tool", route_after_pre_tool, ["approval_gate", "reason"])
    g.add_conditional_edges("approval_gate", partial(route_after_approval, ctx=ctx), after_approval)
    g.add_conditional_edges("guard_output", route_after_output, {"end": END, "reason": "reason"})

    return g.compile(checkpointer=checkpointer)


def recursion_limit(ctx: GraphContext) -> int:
    # Worst case per step: reason, guard_pre_tool, approval_gate, execute_tools.
    return 4 * (ctx.max_steps + ctx.max_retries + 2) + 4
