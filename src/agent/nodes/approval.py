from langchain_core.messages import AIMessage
from langgraph.types import interrupt

from agent.context import GraphContext
from agent.nodes.common import is_customer, last_ai, text_of, tool_feedback
from agent.state import AgentState
from observability.tracing import approval_event


def _signoff_request(state: AgentState, call: dict) -> dict:
    """Payload the operator sees in the approvals console."""
    customer_turns = [m for m in state["messages"] if is_customer(m)]
    return {
        "tool": call["name"],
        "args": call["args"],
        "tool_call_id": call["id"],
        "summary": state.get("decision_note"),
        "policy_checks": state.get("guard_results", []),
        "customer_confirmation": text_of(customer_turns[-1]) if customer_turns else None,
    }


def _bench_decision(ctx: GraphContext) -> dict:
    if ctx.approval_mode in ("auto_approve", "rule_check"):
        # rule_check runs the deterministic policy rules once they exist (P3).
        return {"decision": "approve", "reviewer": ctx.approval_mode, "reason": None}
    raise NotImplementedError(f"approval_mode {ctx.approval_mode} is a stretch goal")


def approval_gate(state: AgentState, ctx: GraphContext) -> dict:
    """READ passes. WRITE needs sign-off: a human via interrupt() in serve, a reviewer in bench."""
    call = state["pending_tool_call"]
    if not ctx.needs_signoff(call):
        return {"approval": None}

    if ctx.approval_mode == "human":
        decision = interrupt(_signoff_request(state, call))
    else:
        decision = _bench_decision(ctx)

    approval = {**decision, "tool": call["name"], "args": call["args"]}
    approval_event(
        call["name"], decision["decision"], decision.get("reviewer", ""), decision.get("reason")
    )
    if decision["decision"] == "reject":
        return {
            "approval": approval,
            "pending_tool_call": None,
            "messages": [tool_feedback(call, "REJECTED by reviewer", decision.get("reason") or "")],
        }
    if decision["decision"] == "edit":
        edited = {**call, "args": decision["edited_args"]}
        ai = last_ai(state["messages"])
        # Same message id, so add_messages replaces the proposal with the edited call.
        return {
            "approval": approval,
            "pending_tool_call": edited,
            "messages": [AIMessage(id=ai.id, content=ai.content, tool_calls=[edited])],
        }
    return {"approval": approval}


def route_after_approval(state: AgentState, ctx: GraphContext) -> str:
    if state.get("pending_tool_call") is None:
        return "reason"
    return "execute_tools" if ctx.mode == "serve" else "emit"
