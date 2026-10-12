import json
from functools import partial

import jsonschema
from langchain_core.messages import AIMessage

from agent.context import GraphContext
from agent.nodes.common import current_turn, last_ai, tool_feedback
from agent.policy_rules import rules_for
from agent.policy_rules.facts import collect
from agent.state import AgentState
from observability.tracing import guard_event

LOOP_LIMIT = 3  # same tool with same args this many times in one turn


def _check_step_cap(state: AgentState, ctx: GraphContext) -> str | None:
    if state.get("step_count", 0) > ctx.max_steps:
        return f"Step limit of {ctx.max_steps} reached this turn. Reply to the customer now."
    return None


def _check_known_tool(call: dict, ctx: GraphContext) -> str | None:
    # ctx.tools holds only the session domain's tools, so this is also the domain lock.
    if call["name"] not in ctx.tools:
        return f"'{call['name']}' is not a tool available for the {ctx.domain} domain."
    return None


def _check_schema(call: dict, ctx: GraphContext) -> str | None:
    try:
        jsonschema.validate(call["args"], ctx.tools[call["name"]].parameters)
    except jsonschema.ValidationError as e:
        return f"Invalid arguments for {call['name']}: {e.message}."
    return None


def _check_loop(call: dict, state: AgentState) -> str | None:
    key = (call["name"], json.dumps(call["args"], sort_keys=True))
    seen = sum(
        1
        for m in current_turn(state["messages"])
        if isinstance(m, AIMessage)
        for c in m.tool_calls
        if (c["name"], json.dumps(c["args"], sort_keys=True)) == key
    )
    if seen >= LOOP_LIMIT:
        return (
            f"{call['name']} has been called {seen} times with the same arguments this turn. "
            "Ask the customer for clarification or transfer to a human agent."
        )
    return None


def guard_pre_tool(state: AgentState, ctx: GraphContext) -> dict:
    """Deterministic checks on the proposed tool call. A block goes back to `reason`.

    Order: safety caps, structural checks, then (WRITE only) confirmation and domain policy.
    A guard that is switched off still runs in shadow mode and records "would_block",
    which gives graph-noguard its would-have-blocked counts and the ablations their data.
    """
    call = last_ai(state["messages"]).tool_calls[0]
    spec = ctx.tools.get(call["name"])

    # Caps and the domain lock are never switched off.
    checks = [
        ("step_cap", True, lambda: _check_step_cap(state, ctx)),
        ("tool_exists", True, lambda: _check_known_tool(call, ctx)),
    ]
    if spec is not None:
        checks += [
            ("schema", False, lambda: _check_schema(call, ctx)),
            ("loop", False, lambda: _check_loop(call, state)),
        ]
    if spec is not None and ctx.is_write(call):
        writes = frozenset(n for n, t in ctx.tools.items() if t.kind == "WRITE")
        facts = collect(state["messages"], writes)
        checks += [
            (rule.name, False, partial(rule.evaluate, call, facts, state["messages"]))
            for rule in rules_for(ctx.domain)
            if rule.applies(call["name"])
        ]

    results = []
    for name, always_on, check in checks:
        reason = check()
        enforced = always_on or ctx.guard_active(name)
        verdict = "pass" if not reason else "block" if enforced else "would_block"
        guard_event(name, verdict, reason, tool=call["name"])
        results.append({"name": name, "verdict": verdict, "reason": reason})
        if reason and enforced:
            retries = state.get("retries_this_turn", 0) + 1
            return {
                "messages": [tool_feedback(call, f"BLOCKED by guard '{name}'", reason)],
                "guard_feedback": [*state.get("guard_feedback", []), f"{name}: {reason}"],
                "retries_this_turn": retries,
                # Out of retries or steps: the next reason call must answer the customer.
                "force_reply": retries > ctx.max_retries
                or state.get("step_count", 0) >= ctx.max_steps,
                "pending_tool_call": None,
                "guard_results": results,
            }

    return {"pending_tool_call": call, "guard_results": results}


def route_after_pre_tool(state: AgentState) -> str:
    return "approval_gate" if state.get("pending_tool_call") else "reason"
