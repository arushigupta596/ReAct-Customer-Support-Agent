from langchain_core.messages import SystemMessage

from agent.context import GraphContext
from agent.llm import usage_cost
from agent.nodes.common import text_of
from agent.state import AgentState


def reason(state: AgentState, ctx: GraphContext) -> dict:
    """One ReAct step: the model either calls one tool or writes one customer message."""
    schemas = [spec.schema for spec in ctx.tools.values()]
    tool_choice = "none" if state.get("force_reply") else "auto"
    llm = ctx.llm.bind_tools(schemas, tool_choice=tool_choice)

    ai = llm.invoke([SystemMessage(ctx.system_prompt), *state["messages"]])

    metadata = dict(ai.response_metadata)
    if metadata.get("cost_usd") is None:
        metadata["cost_usd"] = usage_cost(ctx.model_name, ai.usage_metadata or {})
    note = None
    update = {"response_metadata": metadata}
    if ai.tool_calls:
        # One tool per step. Any narration that came with the call is kept as the
        # display note, never sent to the customer (tau2 rejects text + tool call).
        note = text_of(ai).strip() or None
        update |= {"tool_calls": ai.tool_calls[:1], "content": ""}
    ai = ai.model_copy(update=update)

    return {
        "messages": [ai],
        "step_count": state.get("step_count", 0) + 1,
        "force_reply": False,
        "decision_note": note,
    }
