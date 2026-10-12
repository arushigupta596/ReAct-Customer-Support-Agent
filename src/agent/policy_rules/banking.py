"""Banking (banking_knowledge) rules, from the tau2 Rho-Bank policy and knowledge base.

Most procedures live in the knowledge base and vary per scenario, so only hard rules are
enforced here. There is no blanket confirmation rule: unlike retail and airline, the banking
policy does not require a "yes" before every action.
"""

import re

from agent.nodes.common import is_customer, text_of
from agent.policy_rules.base import Rule

REWARDS_UPDATE = "update_transaction_rewards_3847"
CASH_BACK_DISPUTE = "submit_cash_back_dispute_0589"
# The tool's own status line is uppercase; "this hasn't been resolved" must not count.
RESOLVED = re.compile(r"\bRESOLVED\b|(?i:resolved and approved|been reviewed and approved)")


def _verified(call, facts, messages):
    if call["name"] == "log_verification":
        return None
    if not facts.verified:
        return (
            "The customer's identity has not been verified and logged. Confirm 2 of: date of "
            "birth, email, phone number, address; then call log_verification."
        )
    return None


def _inner_tool(call) -> str | None:
    return call["args"].get("agent_tool_name")


def _rewards_after_resolved_dispute(call, facts, messages):
    """Rewards are corrected only after the customer's cash-back dispute comes back approved.

    The dispute is the customer's own action (they run the tool the agent hands them); a
    valid one returns "Status: RESOLVED ... approved" at once. Correcting rewards without it
    skips the customer's step, which is what failed train task_021 (debug journal entry 13).
    """
    if _inner_tool(call) != REWARDS_UPDATE:
        return None
    # Evidence is the dispute result the customer relayed. Not the agent's words, and not tool
    # results: the knowledge-base article itself says "after a dispute is resolved and approved".
    seen = [text_of(m) for m in messages if is_customer(m) and RESOLVED.search(text_of(m))]
    if seen:
        return None
    return (
        "No resolved, approved cash-back dispute in this conversation. For a cash-back "
        f"discrepancy, give the customer `{CASH_BACK_DISPUTE}` with give_discoverable_user_tool "
        "so they submit the dispute themselves; only update rewards after it comes back RESOLVED."
    )


RULES = [
    Rule(
        "verified_first",
        None,
        _verified,
        "For any scenario involving accessing customer information in internal databases, you "
        "must first verify their identify before proceeding... After verification, you must "
        "call the verification logging tool.",
    ),
    Rule(
        "rewards_after_dispute",
        ("call_discoverable_agent_tool",),
        _rewards_after_resolved_dispute,
        "After a cash back dispute is resolved and approved, you must update the affected "
        "transaction(s) with the correct rewards value. (Knowledge base: Applying Resolved Cash "
        "Back Dispute Corrections)",
    ),
]
