"""Rules shared by every domain that requires confirmation before a WRITE."""

import re

from langchain_core.messages import AIMessage, AnyMessage

from agent.nodes.common import is_customer, text_of
from agent.policy_rules.base import Rule
from agent.policy_rules.facts import Facts

AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|yup|confirm(ed|s)?|go ahead|proceed|please do|do it|correct|"
    r"that'?s right|sounds (good|great|perfect)|ok(ay)?|sure|absolutely|definitely|"
    r"fine|just do it|go for it)\b",
    re.IGNORECASE,
)
NEGATIVE_START = re.compile(
    r"^\s*(no\b|nope|don'?t|do not|wait|hold on|not yet|stop)", re.IGNORECASE
)

# The record the customer is agreeing to change. Item, product and payment ids are usually
# described by name ("the blue t-shirt", "Visa ending 1234"), and the tools themselves verify
# them against the database, so requiring them verbatim blocks correct actions.
# (Replay over 120 real WRITE calls: requiring every *_id blocked 79 correct actions.)
CONFIRM_KEYS = ("order_id", "reservation_id")


def _mentioned(value: str, text: str) -> bool:
    return value in text or value.lstrip("#") in text


def _since_last_write(messages: list[AnyMessage], facts: Facts) -> list[AnyMessage]:
    """The conversation since the previous successful WRITE (one confirmation per action)."""
    write_ids = {
        tc["id"]
        for m in messages
        if isinstance(m, AIMessage)
        for tc in m.tool_calls
        if (tc["name"], tc["args"]) in facts.succeeded_writes
    }
    start = 0
    for i, m in enumerate(messages):
        if getattr(m, "tool_call_id", None) in write_ids:
            start = i + 1
    return messages[start:]


def check_confirmation(call: dict, facts: Facts, messages: list[AnyMessage]) -> str | None:
    last_human = max((i for i, m in enumerate(messages) if is_customer(m)), default=None)
    if last_human is None:
        return "No customer message to confirm this action."
    reply = text_of(messages[last_human])
    if NEGATIVE_START.search(reply) or not AFFIRMATIVE.search(reply):
        return (
            "The customer has not explicitly confirmed. List the exact action details and ask "
            "for a clear yes before calling this tool."
        )
    # The agent must have listed the record before the yes ("you must list the action
    # details"): a customer saying "cancel #W1, yes" in one breath is not a confirmation.
    # Tool results don't count as shown to the customer.
    window = " ".join(
        text_of(m)
        for m in _since_last_write(messages[:last_human], facts)
        if isinstance(m, AIMessage)
    )
    missing = [
        str(call["args"][k])
        for k in CONFIRM_KEYS
        if k in call["args"] and not _mentioned(str(call["args"][k]), window)
    ]
    if missing:
        return (
            f"You did not state {', '.join(missing)} to the customer before they said yes. List "
            "the exact action details, including which record it changes, and ask again."
        )
    return None


CONFIRMATION = Rule(
    name="confirmation",
    tools=None,
    check=check_confirmation,
    citation=(
        "Before taking any action that updates the database, you must list the action "
        "details and obtain explicit user confirmation (yes) to proceed."
    ),
)
