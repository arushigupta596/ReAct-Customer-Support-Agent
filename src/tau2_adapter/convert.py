"""tau2 <-> LangChain message conversion.

Bugs here fail silently (a dropped tool-call id just makes the model behave oddly),
so every branch is covered by tests/test_convert.py round trips.
"""

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
)
from langchain_core.messages import SystemMessage as LCSystemMessage
from langchain_core.messages import ToolMessage as LCToolMessage
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from agent.nodes.common import text_of


def to_lc(message) -> list[AnyMessage]:
    """One tau2 message -> LangChain messages (a MultiToolMessage expands to several)."""
    if isinstance(message, MultiToolMessage):
        return [m for tm in message.tool_messages for m in to_lc(tm)]
    if isinstance(message, UserMessage):
        return [HumanMessage(content=message.content or "")]
    if isinstance(message, ToolMessage):
        return [
            LCToolMessage(
                content=message.content or "",
                tool_call_id=message.id,
                status="error" if message.error else "success",
            )
        ]
    if isinstance(message, AssistantMessage):
        tool_calls = [
            {"id": tc.id, "name": tc.name, "args": tc.arguments, "type": "tool_call"}
            for tc in message.tool_calls or []
        ]
        return [AIMessage(content=message.content or "", tool_calls=tool_calls)]
    if isinstance(message, SystemMessage):
        return [LCSystemMessage(content=message.content or "")]
    raise TypeError(f"Unsupported tau2 message type: {type(message).__name__}")


def to_lc_history(messages) -> list[AnyMessage]:
    return [m for message in messages for m in to_lc(message)]


def to_tau2_assistant(message: AIMessage) -> AssistantMessage:
    """The graph's final AIMessage of a turn -> the AssistantMessage tau2 records and scores."""
    tool_calls = [
        ToolCall(id=tc["id"], name=tc["name"], arguments=tc["args"], requestor="assistant")
        for tc in message.tool_calls
    ]
    usage = message.usage_metadata
    return AssistantMessage(
        role="assistant",
        content=None if tool_calls else text_of(message),
        tool_calls=tool_calls or None,
        cost=message.response_metadata.get("cost_usd"),
        usage=(
            {
                "prompt_tokens": usage["input_tokens"],
                "completion_tokens": usage["output_tokens"],
            }
            if usage
            else None
        ),
    )
