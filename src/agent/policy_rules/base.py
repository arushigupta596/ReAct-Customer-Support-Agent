from collections.abc import Callable
from dataclasses import dataclass

from langchain_core.messages import AnyMessage

from agent.policy_rules.facts import Facts

# (tool call, facts, messages) -> reason it fails, or None if it passes
Check = Callable[[dict, Facts, list[AnyMessage]], str | None]


@dataclass(frozen=True)
class Rule:
    name: str
    tools: tuple[str, ...] | None  # None: every WRITE tool
    check: Check
    citation: str  # the policy text this rule enforces

    def applies(self, tool: str) -> bool:
        return self.tools is None or tool in self.tools

    def evaluate(self, call: dict, facts: Facts, messages: list[AnyMessage]) -> str | None:
        reason = self.check(call, facts, messages)
        return f'{reason} Policy: "{self.citation}"' if reason else None
