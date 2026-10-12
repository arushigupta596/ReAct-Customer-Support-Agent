from agent.context import GraphContext
from agent.state import AgentState


def emit(state: AgentState, ctx: GraphContext) -> dict:
    """Bench mode only: end the turn so tau2 executes the tool call in its own environment.

    tau2 sends the result back as the next input, which resumes the same thread.
    """
    return {"pending_tool_call": None}
