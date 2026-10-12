"""Retail policy rules, transcribed from tau2 retail policy.md.

Only rules the retail tools do not already enforce themselves (they reject wrong status,
invalid cancel reasons, unavailable items and wrong refund methods with an error).
"""

from agent.policy_rules.base import Rule
from agent.policy_rules.common import CONFIRMATION

ORDER_WRITES = (
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "return_delivered_order_items",
    "exchange_delivered_order_items",
)
REQUIRED_STATUS = {
    "cancel_pending_order": "pending",
    "modify_pending_order_address": "pending",
    "modify_pending_order_items": "pending",
    "modify_pending_order_payment": "pending",
    "return_delivered_order_items": "delivered",
    "exchange_delivered_order_items": "delivered",
}
ONCE_PER_ORDER = ("modify_pending_order_items", "exchange_delivered_order_items")


def _authenticated(call, facts, messages):
    if facts.authenticated_user is None:
        return "The customer is not authenticated yet. Find their user id by email or name + zip first."
    return None


def _order_checked(call, facts, messages):
    order_id = call["args"].get("order_id")
    if order_id not in facts.orders:
        return f"Order {order_id} has not been looked up. Check its details and status first."
    return None


def _order_owner(call, facts, messages):
    order = facts.orders.get(call["args"].get("order_id"))
    if order and facts.authenticated_user and order.get("user_id") != facts.authenticated_user:
        return (
            f"Order {call['args']['order_id']} belongs to another user, not the authenticated "
            f"customer {facts.authenticated_user}."
        )
    return None


def _status_matches(call, facts, messages):
    order = facts.orders.get(call["args"].get("order_id"))
    required = REQUIRED_STATUS[call["name"]]
    if order and order.get("status") != required:
        return (
            f"Order {call['args']['order_id']} is '{order.get('status')}'; "
            f"{call['name']} needs '{required}'."
        )
    return None


def _once_per_order(call, facts, messages):
    order_id = call["args"].get("order_id")
    if any(
        name in ONCE_PER_ORDER and args.get("order_id") == order_id
        for name, args in facts.succeeded
    ):
        return f"Items on order {order_id} were already modified or exchanged once."
    return None


def _own_address(call, facts, messages):
    if facts.authenticated_user and call["args"].get("user_id") != facts.authenticated_user:
        return "Only the authenticated customer's own address can be changed."
    return None


RULES = [
    CONFIRMATION,
    Rule(
        "authenticated",
        None,
        _authenticated,
        "At the beginning of the conversation, you have to authenticate the user identity by "
        "locating their user id via email, or via name + zip code.",
    ),
    Rule(
        "order_checked",
        ORDER_WRITES,
        _order_checked,
        "You should check its status before taking the action.",
    ),
    Rule(
        "order_owner",
        ORDER_WRITES,
        _order_owner,
        "You can only help one user per conversation and must deny any requests for tasks "
        "related to any other user.",
    ),
    Rule(
        "order_status",
        ORDER_WRITES,
        _status_matches,
        "Generally, you can only take action on pending or delivered orders.",
    ),
    Rule(
        "once_per_order",
        ONCE_PER_ORDER,
        _once_per_order,
        "Exchange or modify order tools can only be called once per order.",
    ),
    Rule(
        "own_address",
        ("modify_user_address",),
        _own_address,
        "You can only help one user per conversation.",
    ),
]
