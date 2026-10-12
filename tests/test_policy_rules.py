"""Retail rules: each must block its violation and pass the compliant version.

False-positive rates on real trajectories are measured by scripts/replay_guards.py.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.policy_rules.facts import collect
from agent.policy_rules.retail import RULES
from tests.fakes import tool_call

RULE = {r.name: r for r in RULES}
WRITES = frozenset({"cancel_pending_order", "exchange_delivered_order_items"})
ORDER = {"order_id": "#W1", "user_id": "ann_1", "status": "pending"}


def tool(name, args, result, call_id):
    return [
        AIMessage(content="", tool_calls=[tool_call(name, args, call_id)]),
        ToolMessage(content=result, tool_call_id=call_id),
    ]


def conversation(
    *,
    auth=True,
    order=ORDER,
    summary="Cancel order #W1 because it is no longer needed?",
    reply="Yes, please.",
):
    msgs = [HumanMessage(content="I want to cancel an order")]
    if auth:
        msgs += tool("find_user_id_by_email", {"email": "a@x.com"}, "ann_1", "c1")
    if order:
        msgs += tool("get_order_details", {"order_id": order["order_id"]}, json.dumps(order), "c2")
    msgs += [AIMessage(content=summary), HumanMessage(content=reply)]
    return msgs


CANCEL = tool_call("cancel_pending_order", {"order_id": "#W1", "reason": "no longer needed"}, "c9")


def verdicts(msgs, call=CANCEL):
    facts = collect(msgs, WRITES)
    return {r.name: r.evaluate(call, facts, msgs) for r in RULES if r.applies(call["name"])}


def test_compliant_cancel_passes_every_rule():
    assert all(v is None for v in verdicts(conversation()).values())


def test_unauthenticated_is_blocked():
    assert verdicts(conversation(auth=False))["authenticated"]


def test_order_not_looked_up_is_blocked():
    assert verdicts(conversation(order=None))["order_checked"]


def test_other_users_order_is_blocked():
    assert verdicts(conversation(order={**ORDER, "user_id": "bob_2"}))["order_owner"]


def test_wrong_status_is_blocked():
    assert verdicts(conversation(order={**ORDER, "status": "delivered"}))["order_status"]


@pytest.mark.parametrize(
    "reply", ["Can you tell me more first?", "No, wait.", "Don't do that yet", "hmm"]
)
def test_no_explicit_yes_is_blocked(reply):
    assert verdicts(conversation(reply=reply))["confirmation"]


@pytest.mark.parametrize("reply", ["Yes", "yes please go ahead", "Confirmed.", "Fine, just do it."])
def test_explicit_yes_passes(reply):
    assert verdicts(conversation(reply=reply))["confirmation"] is None


def test_yes_to_a_different_order_is_blocked():
    msgs = conversation(summary="Shall I cancel order #W7?")
    assert "#W1" in verdicts(msgs)["confirmation"]


def test_one_yes_does_not_cover_a_later_action_on_another_order():
    msgs = conversation()
    msgs += tool("cancel_pending_order", CANCEL["args"], '{"status": "cancelled"}', "c9")
    other = tool_call(
        "cancel_pending_order", {"order_id": "#W2", "reason": "ordered by mistake"}, "c10"
    )
    msgs += tool(
        "get_order_details", {"order_id": "#W2"}, json.dumps({**ORDER, "order_id": "#W2"}), "c3"
    )
    assert verdicts(msgs, other)["confirmation"]


def test_second_exchange_on_same_order_is_blocked():
    order = {**ORDER, "status": "delivered"}
    exchange = tool_call(
        "exchange_delivered_order_items",
        {"order_id": "#W1", "item_ids": ["1"], "new_item_ids": ["2"], "payment_method_id": "gc_1"},
        "c11",
    )
    msgs = conversation(order=order, summary="Exchange items on order #W1?")
    msgs += tool(exchange["name"], exchange["args"], '{"status": "exchange requested"}', "c11")
    msgs += [AIMessage(content="Another exchange on #W1?"), HumanMessage(content="yes")]
    assert verdicts(msgs, {**exchange, "id": "c12"})["once_per_order"]


def test_rule_reasons_cite_the_policy():
    reason = verdicts(conversation(auth=False))["authenticated"]
    assert 'Policy: "At the beginning of the conversation' in reason


def test_airline_cancel_uses_reservation_state_after_an_upgrade():
    from agent.policy_rules.airline import RULES as AIRLINE

    rule = {r.name: r for r in AIRLINE}["cancel_eligible"]
    res = {
        "reservation_id": "X1",
        "user_id": "u1",
        "cabin": "basic_economy",
        "insurance": "no",
        "created_at": "2024-05-01T05:00:00",
        "flights": [{"flight_number": "HAT1", "date": "2024-05-20"}],
    }
    msgs = tool("get_reservation_details", {"reservation_id": "X1"}, json.dumps(res), "c1")
    cancel = tool_call("cancel_reservation", {"reservation_id": "X1"}, "c3")
    assert rule.evaluate(cancel, collect(msgs), msgs)  # basic economy, old booking: blocked

    upgraded = {**res, "cabin": "business"}
    msgs += tool(
        "update_reservation_flights",
        {"reservation_id": "X1", "cabin": "business"},
        json.dumps(upgraded),
        "c2",
    )
    assert rule.evaluate(cancel, collect(msgs), msgs) is None  # business can always cancel


def _rewards_rule():
    from agent.policy_rules.banking import RULES as BANKING

    return {r.name: r for r in BANKING}["rewards_after_dispute"]


UPDATE = tool_call(
    "call_discoverable_agent_tool",
    {
        "agent_tool_name": "update_transaction_rewards_3847",
        "arguments": '{"transaction_id": "txn_1", "new_rewards_earned": "369"}',
    },
    "u1",
)


def test_rewards_update_without_resolved_dispute_is_blocked():
    msgs = [HumanMessage(content="My cash back on txn_1 looks wrong, please fix it.")]
    reason = _rewards_rule().evaluate(UPDATE, collect(msgs), msgs)
    assert reason and "submit_cash_back_dispute_0589" in reason


def test_rewards_update_after_customer_relays_resolved_dispute_passes():
    msgs = [
        HumanMessage(
            content="I ran the dispute tool. Status: RESOLVED - The dispute has been reviewed and approved."
        ),
    ]
    assert _rewards_rule().evaluate(UPDATE, collect(msgs), msgs) is None


def test_agent_saying_resolved_is_not_evidence():
    msgs = [AIMessage(content="Your dispute is RESOLVED, I'll update the rewards now.")]
    assert _rewards_rule().evaluate(UPDATE, collect(msgs), msgs)


def test_other_discoverable_tools_are_unaffected():
    other = tool_call(
        "call_discoverable_agent_tool",
        {"agent_tool_name": "apply_statement_credit_8472", "arguments": "{}"},
        "u2",
    )
    msgs = [HumanMessage(content="hi")]
    assert _rewards_rule().evaluate(other, collect(msgs), msgs) is None


def test_customer_complaint_mentioning_resolved_is_not_evidence():
    msgs = [HumanMessage(content="This still hasn't been resolved, just fix my rewards.")]
    assert _rewards_rule().evaluate(UPDATE, collect(msgs), msgs)


def test_knowledge_base_text_is_not_evidence():
    msgs = [HumanMessage(content="my cash back is wrong")]
    msgs += tool(
        "KB_search",
        {"query": "cash back dispute"},
        "After a cash back dispute is resolved and approved, you must update the affected transaction(s).",
        "k1",
    )
    assert _rewards_rule().evaluate(UPDATE, collect(msgs), msgs)
