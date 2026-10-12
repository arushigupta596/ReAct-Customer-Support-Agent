"""The tau2 adapter against real tau2 domain tools, with a scripted model."""

from langchain_core.messages import AIMessage
from tau2.data_model.message import AssistantMessage, ToolMessage, UserMessage
from tau2.runner.build import build_environment

from agent.tools import spec_from_tau2_tool
from tau2_adapter.agent import LangGraphReActAgent
from tests.fakes import ScriptedChatModel, tool_call


def retail_agent(script):
    env = build_environment("retail")
    llm = ScriptedChatModel(script=list(script), calls=[])
    agent = LangGraphReActAgent(
        tools=env.get_tools(),
        domain_policy=env.get_policy(),
        llm="scripted",
        domain="retail",
        chat_model=llm,
    )
    return agent, llm


def test_read_write_kinds_come_from_tau2_decorators():
    env = build_environment("retail")
    kinds = {t.name: spec_from_tau2_tool(t).kind for t in env.get_tools()}
    assert kinds["get_order_details"] == "READ"
    assert kinds["cancel_pending_order"] == "WRITE"
    assert kinds["transfer_to_human_agents"] == "GENERIC"


def test_banking_discoverable_wrapper_is_write():
    env = build_environment("banking_knowledge", env_kwargs={"retrieval_variant": "bm25"})
    kinds = {t.name: spec_from_tau2_tool(t).kind for t in env.get_tools()}
    assert kinds["call_discoverable_agent_tool"] == "WRITE"
    assert kinds["log_verification"] == "WRITE"


def test_full_bench_turn_cycle():
    agent, llm = retail_agent(
        [
            AIMessage(
                content="",
                tool_calls=[tool_call("get_order_details", {"order_id": "#W1"}, "toolu_1")],
                response_metadata={"cost_usd": 0.001},
            ),
            AIMessage(content="Your order is pending.", response_metadata={"cost_usd": 0.002}),
        ]
    )
    greeting = AssistantMessage(role="assistant", content="Hi! How can I help you today?")
    state = agent.get_init_state(message_history=[greeting])

    msg, state = agent.generate_next_message(
        UserMessage(role="user", content="Where is order #W1?"), state
    )
    assert msg.is_tool_call() and msg.tool_calls[0].id == "toolu_1"
    assert msg.cost == 0.001
    # The greeting from tau2's history reached the model before the user turn.
    sent = llm.calls[0]["messages"]
    assert [m.type for m in sent] == ["system", "ai", "human"]

    msg, state = agent.generate_next_message(
        ToolMessage(
            id="toolu_1", role="tool", content='{"status": "pending"}', requestor="assistant"
        ),
        state,
    )
    assert msg.content == "Your order is pending." and not msg.is_tool_call()


def test_blocked_attempt_cost_is_counted():
    agent, _ = retail_agent(
        [
            AIMessage(
                content="",
                tool_calls=[tool_call("not_a_tool", {})],
                response_metadata={"cost_usd": 0.001},
            ),
            AIMessage(
                content="Could you share your order id?", response_metadata={"cost_usd": 0.002}
            ),
        ]
    )
    state = agent.get_init_state()
    msg, _ = agent.generate_next_message(UserMessage(role="user", content="help"), state)
    assert msg.content == "Could you share your order id?"
    assert abs(msg.cost - 0.003) < 1e-9
