import re

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage

from agent.context import GraphContext
from agent.nodes.common import text_of
from agent.state import AgentState
from observability.tracing import guard_event

ACTIONS = (
    r"cancel+ed|processed|refunded|updated|submitted|modified|exchanged|returned|booked|"
    r"changed|applied|filed|sent|issued|credited|upgraded|resumed|suspended|added|removed"
)
# A statement that something was done, not a plan or a question about it.
CLAIM = re.compile(
    rf"\b(has|have) (now )?been ({ACTIONS})\b"
    rf"|\bI(?:'ve| have) (just )?({ACTIONS})\b"
    rf"|\bsuccessfully ({ACTIONS})\b"
    rf"|\bis now (cancel+ed|booked|confirmed|active|resumed|suspended|closed)\b",
    re.IGNORECASE,
)
HYPOTHETICAL = re.compile(r"\b(once|after|when|if|will|would|can|could)\b[^.!?]*$", re.IGNORECASE)
FEEDBACK_NAME = "output_guard"
# Real actions the agent can claim that tau2 doesn't class as WRITE.
NON_WRITE_ACTIONS = (
    "transfer_to_human_agents",
    "give_discoverable_user_tool",
)  # internal feedback turns carry this name and are never shown


def _claims(text: str) -> list[str]:
    hits = []
    for m in CLAIM.finditer(text):
        sentence_start = text.rfind(".", 0, m.start()) + 1
        if not HYPOTHETICAL.search(text[sentence_start : m.start()]):
            hits.append(m.group(0))
    return hits


def _any_successful_write(state: AgentState, ctx: GraphContext) -> bool:
    calls = {
        tc["id"]: tc for m in state["messages"] if isinstance(m, AIMessage) for tc in m.tool_calls
    }
    for m in state["messages"]:
        if isinstance(m, ToolMessage) and m.status != "error":
            call = calls.get(m.tool_call_id)
            acted = call and (ctx.is_write(call) or call["name"] in NON_WRITE_ACTIONS)
            if acted and not text_of(m).startswith(("Error", "BLOCKED", "REJECTED")):
                return True
        if isinstance(m, HumanMessage) and text_of(m).startswith("I ran "):
            return True  # the customer ran a handed-over action themselves (banking)
    return False


def guard_output(state: AgentState, ctx: GraphContext) -> dict:
    """Last check before a reply reaches the customer: no claimed action without a tool result.

    (Text and a tool call in one turn are already split in `reason`.) A failing reply is removed
    and the model is told why; one rewrite per turn, then the reply goes out as is.
    """
    reply = state["messages"][-1]
    claims = _claims(text_of(reply))
    if not claims or _any_successful_write(state, ctx):
        guard_event("false_claim", "pass", None)
        return {"output_rewrites": 0}
    reason = f"Reply claims {', '.join(claims)!r} but no action succeeded in this conversation."
    enforced = ctx.guard_active("false_claim") and state.get("output_rewrites", 0) < 1
    guard_event("false_claim", "block" if enforced else "would_block", reason)
    if not enforced:
        return {"output_rewrites": 0}
    return {
        "messages": [
            RemoveMessage(id=reply.id),
            HumanMessage(
                name=FEEDBACK_NAME,
                content=(
                    f"[Output guard, not from the customer] Your last reply said {', '.join(claims)!r}, "
                    "but no tool call has done that. Do not claim actions that have not happened. "
                    "Either call the tool (after the customer's explicit confirmation) or tell the "
                    "customer what is still needed."
                ),
            ),
        ],
        "output_rewrites": state.get("output_rewrites", 0) + 1,
    }


def route_after_output(state: AgentState) -> str:
    last = state["messages"][-1]
    return "reason" if isinstance(last, HumanMessage) and last.name == FEEDBACK_NAME else "end"
