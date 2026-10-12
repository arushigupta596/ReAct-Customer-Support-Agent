from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage


def text_of(message: AnyMessage) -> str:
    """Plain text of a message whose content may be a string or a list of blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") if isinstance(block, dict) else str(block) for block in content
    )


INTERNAL_NAMES = ("output_guard",)  # graph-internal feedback sent in the human role


def is_customer(message: AnyMessage) -> bool:
    """A real customer turn (not internal guard feedback that uses the human role)."""
    return isinstance(message, HumanMessage) and message.name not in INTERNAL_NAMES


def last_ai(messages: list[AnyMessage]) -> AIMessage:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    raise ValueError("no AIMessage in state")


def current_turn(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Messages since the latest customer message (inclusive)."""
    for i in range(len(messages) - 1, -1, -1):
        if is_customer(messages[i]):
            return messages[i:]
    return messages


def tool_feedback(call: dict, verdict: str, reason: str) -> ToolMessage:
    """Tool result the model sees when an action did not run (blocked or rejected)."""
    return ToolMessage(
        content=f"{verdict}: {reason} The action did not run.",
        tool_call_id=call["id"],
        name=call["name"],
        status="error",
    )
