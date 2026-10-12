from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig

from agent.context import GraphContext
from agent.state import AgentState


def execute_tools(state: AgentState, config: RunnableConfig, ctx: GraphContext) -> dict:
    """Serve mode only: run the approved call (over MCP) and append the result.

    The thread id is the API session id, which picks the session's domain DB on the server.
    """
    call = state["pending_tool_call"]
    session_id = config["configurable"]["thread_id"]
    content, is_error = ctx.executor(call["name"], call["args"], session_id)
    return {
        "messages": [
            ToolMessage(
                content=content,
                tool_call_id=call["id"],
                name=call["name"],
                status="error" if is_error else "success",
            )
        ],
        "pending_tool_call": None,
    }
