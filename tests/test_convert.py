from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import ToolMessage as LCToolMessage
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

from tau2_adapter.convert import to_lc, to_lc_history, to_tau2_assistant


def test_user_message():
    [m] = to_lc(UserMessage(role="user", content="I want to return my order"))
    assert isinstance(m, HumanMessage)
    assert m.content == "I want to return my order"


def test_assistant_tool_call_round_trip_keeps_id_and_args():
    original = AssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ToolCall(id="toolu_01", name="get_order_details", arguments={"order_id": "#W1"})
        ],
    )
    [lc] = to_lc(original)
    assert isinstance(lc, AIMessage)
    assert lc.tool_calls[0]["id"] == "toolu_01"

    back = to_tau2_assistant(lc)
    assert back.content is None
    assert back.tool_calls == original.tool_calls
    assert back.is_tool_call()


def test_assistant_text_round_trip():
    original = AssistantMessage(role="assistant", content="Your order is on its way.")
    back = to_tau2_assistant(to_lc(original)[0])
    assert back.content == original.content
    assert back.tool_calls is None


def test_tool_message_keeps_call_id_and_error_flag():
    [ok] = to_lc(ToolMessage(id="toolu_01", role="tool", content="{}", requestor="assistant"))
    [err] = to_lc(
        ToolMessage(id="toolu_02", role="tool", content="Error", requestor="assistant", error=True)
    )
    assert isinstance(ok, LCToolMessage)
    assert ok.tool_call_id == "toolu_01" and ok.status == "success"
    assert err.tool_call_id == "toolu_02" and err.status == "error"


def test_multi_tool_message_expands_in_order():
    multi = MultiToolMessage(
        role="tool",
        tool_messages=[
            ToolMessage(id="a", role="tool", content="1", requestor="assistant"),
            ToolMessage(id="b", role="tool", content="2", requestor="assistant"),
        ],
    )
    out = to_lc(multi)
    assert [m.tool_call_id for m in out] == ["a", "b"]


def test_history_with_greeting():
    history = [
        AssistantMessage(role="assistant", content="Hi! How can I help you today?"),
        UserMessage(role="user", content="Cancel my flight"),
    ]
    out = to_lc_history(history)
    assert [type(m) for m in out] == [AIMessage, HumanMessage]


def test_usage_and_cost_carry_over():
    ai = AIMessage(
        content="Done.",
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        response_metadata={"cost_usd": 0.0002},
    )
    msg = to_tau2_assistant(ai)
    assert msg.cost == 0.0002
    assert msg.usage == {"prompt_tokens": 100, "completion_tokens": 20}


def test_list_content_blocks_flatten_to_text():
    ai = AIMessage(content=[{"type": "text", "text": "Hello "}, {"type": "text", "text": "there"}])
    assert to_tau2_assistant(ai).content == "Hello there"
