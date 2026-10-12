"""Domain policy rules, applied to WRITE tool calls by guard_pre_tool."""

from agent.policy_rules import airline, banking, retail, telecom
from agent.policy_rules.base import Rule

RULES: dict[str, list[Rule]] = {
    "retail": retail.RULES,
    "airline": airline.RULES,
    "banking_knowledge": banking.RULES,
    "telecom": telecom.RULES,
}


def rules_for(domain: str) -> list[Rule]:
    return RULES.get(domain, [])
