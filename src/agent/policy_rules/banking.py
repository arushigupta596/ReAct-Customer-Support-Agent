"""Banking (banking_knowledge) rules, from the tau2 Rho-Bank policy.

Most procedures live in the knowledge base and vary per scenario, so only the policy's
hard, scenario-independent rule is enforced here. There is no blanket confirmation rule:
unlike retail and airline, the banking policy does not require a "yes" before every action.
"""

from agent.policy_rules.base import Rule


def _verified(call, facts, messages):
    if call["name"] == "log_verification":
        return None
    if not facts.verified:
        return (
            "The customer's identity has not been verified and logged. Confirm 2 of: date of "
            "birth, email, phone number, address; then call log_verification."
        )
    return None


RULES = [
    Rule(
        "verified_first",
        None,
        _verified,
        "For any scenario involving accessing customer information in internal databases, you "
        "must first verify their identify before proceeding... After verification, you must "
        "call the verification logging tool.",
    ),
]
