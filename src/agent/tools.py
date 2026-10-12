"""Tool metadata the graph needs, independent of where tools execute.

READ/WRITE classification comes from tau2's own `@is_tool(ToolType...)` decorators,
never from a hardcoded list (spec 4.3).
"""

from dataclasses import dataclass

from agent.state import ToolKind

TOOL_TYPE_ATTR = "__tool_type__"
MUTATES_STATE_ATTR = "__mutates_state__"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema: dict  # OpenAI function schema: {"type": "function", "function": {...}}
    kind: ToolKind
    mutates_state: bool

    @property
    def parameters(self) -> dict:
        return self.schema["function"].get("parameters", {})


def spec_from_tau2_tool(tool) -> ToolSpec:
    """Build a ToolSpec from a tau2 `Tool`. Unknown types default to WRITE (fail safe)."""
    func = getattr(tool, "_func", None)
    tool_type = getattr(func, TOOL_TYPE_ATTR, None)
    kind = tool_type.value.upper() if tool_type is not None else "WRITE"
    mutates = getattr(func, MUTATES_STATE_ATTR, kind == "WRITE")
    return ToolSpec(name=tool.name, schema=tool.openai_schema, kind=kind, mutates_state=mutates)


DISCOVERABLE_WRAPPER = "call_discoverable_agent_tool"


def discoverable_kinds(tau2_tools) -> dict[str, ToolKind]:
    """Kinds of a toolkit's discoverable tools (banking), read from the same decorators.

    Discoverable tools are not in the tool list; the agent runs them through
    `call_discoverable_agent_tool`, so the wrapper call takes the kind of the tool it runs.
    """
    for tool in tau2_tools:
        toolkit = getattr(getattr(tool, "_func", None), "__self__", None)
        if toolkit is not None and hasattr(toolkit, "get_discoverable_tools"):
            return {
                name: getattr(fn, TOOL_TYPE_ATTR).value.upper()
                for name, fn in toolkit.get_discoverable_tools().items()
            }
    return {}
