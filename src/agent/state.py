from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

ToolKind = Literal["READ", "WRITE", "THINK", "GENERIC"]
Mode = Literal["bench", "serve"]
ApprovalMode = Literal["auto_approve", "rule_check", "llm_reviewer", "human"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    domain: str
    pending_tool_call: dict | None
    guard_feedback: list[str]  # reasons an action was blocked this turn
    retries_this_turn: int  # capped at MAX_RETRIES_PER_TURN
    approval: dict | None  # decision, edited_args, reviewer, reason
    step_count: int  # reason calls this turn, capped at MAX_STEPS_PER_TURN
    force_reply: bool  # set when a cap is hit: next reason call may not use tools
    decision_note: str | None  # one-line display summary for the timeline
    output_rewrites: int  # replies rewritten by guard_output this turn (max 1)
    guard_results: list[dict]  # checks run on the latest proposed tool call (shown at sign-off)
