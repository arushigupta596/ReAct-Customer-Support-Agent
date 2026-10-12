from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from agent.state import ApprovalMode, Mode
from agent.tools import DISCOVERABLE_WRAPPER, ToolSpec

# WRITE tools that only record what the agent did (no customer-facing effect): the guards
# still check them, but they don't wait for an operator.
SIGNOFF_EXEMPT = {"banking_knowledge": ("log_verification",)}

PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"

# (tool_name, args, session_id) -> (content, is_error). Serve mode only.
ToolExecutor = Callable[[str, dict, str], tuple[str, bool]]


@dataclass
class GraphContext:
    """Everything the nodes need that is fixed for the life of one graph."""

    mode: Mode
    domain: str
    policy: str
    llm: BaseChatModel
    model_name: str
    tools: dict[str, ToolSpec]
    approval_mode: ApprovalMode = "rule_check"
    guards_enabled: bool = True
    disabled_guards: frozenset[str] = frozenset()  # ablations: these run in shadow mode
    discoverable_kinds: dict[str, str] = field(default_factory=dict)  # banking inner tools
    max_steps: int = 30
    max_retries: int = 2
    executor: ToolExecutor | None = None
    system_prompt: str = field(init=False)

    def kind_of(self, call: dict) -> str | None:
        """READ/WRITE/... of a call; a discoverable-tool wrapper takes its inner tool's kind."""
        spec = self.tools.get(call["name"])
        if spec is None:
            return None
        if call["name"] == DISCOVERABLE_WRAPPER:
            inner = call["args"].get("agent_tool_name")
            return self.discoverable_kinds.get(inner, spec.kind)
        return spec.kind

    def is_write(self, call: dict) -> bool:
        return self.kind_of(call) == "WRITE"

    def needs_signoff(self, call: dict) -> bool:
        """Human sign-off: every WRITE except internal audit records."""
        return self.is_write(call) and call["name"] not in SIGNOFF_EXEMPT.get(self.domain, ())

    def guard_active(self, name: str) -> bool:
        return self.guards_enabled and name not in self.disabled_guards

    def __post_init__(self):
        self.system_prompt = PROMPT_PATH.read_text().format(policy=self.policy)
        if self.mode == "serve" and self.executor is None:
            raise ValueError("serve mode needs a tool executor")
