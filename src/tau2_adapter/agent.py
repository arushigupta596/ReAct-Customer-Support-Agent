"""tau2 HalfDuplexAgent that delegates every turn to the LangGraph ReAct graph (bench mode)."""

import uuid

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import AssistantMessage, Message
from tau2.environment.tool import Tool

from agent.context import GraphContext
from agent.graph import build_graph, recursion_limit
from agent.llm import make_chat_model
from agent.state import ApprovalMode
from agent.tools import discoverable_kinds, spec_from_tau2_tool
from tau2_adapter.convert import to_lc, to_lc_history, to_tau2_assistant


class GraphAgentState(BaseModel):
    thread_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    # History tau2 hands us at init; sent with the first input so the graph sees it.
    pending_history: list = Field(default_factory=list)


class LangGraphReActAgent(LLMConfigMixin, HalfDuplexAgent[GraphAgentState]):
    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str,
        llm_args: dict | None = None,
        *,
        domain: str,
        approval_mode: ApprovalMode = "rule_check",
        guards_enabled: bool = True,
        max_steps: int = 30,
        max_retries: int = 2,
        chat_model=None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self._graph_options = {
            "domain": domain,
            "approval_mode": approval_mode,
            "guards_enabled": guards_enabled,
            "max_steps": max_steps,
            "max_retries": max_retries,
        }
        self._chat_model = chat_model  # injected in tests; otherwise built from llm/llm_args
        self._graph = None
        self._ctx = None

    def set_seed(self, seed: int):
        # Anthropic models do not accept `seed`, and OpenRouter's require_parameters
        # would then refuse to route the request. The baseline drops it the same way
        # (see factory.NoSeedLLMAgent), so both sides stay comparable.
        pass

    def _build(self):
        # Built lazily so the LLM picks up the final llm_args.
        chat = self._chat_model or make_chat_model(self.llm, self.llm_args)
        self._ctx = GraphContext(
            mode="bench",
            policy=self.domain_policy,
            llm=chat,
            model_name=self.llm,
            tools={t.name: spec_from_tau2_tool(t) for t in self.tools},
            discoverable_kinds=discoverable_kinds(self.tools),
            **self._graph_options,
        )
        self._graph = build_graph(self._ctx, checkpointer=InMemorySaver())

    def _turn_ai_messages(self, config: dict, before: set) -> list:
        """AI messages written during this invocation, from the checkpoint history."""
        seen, out = set(before), []
        for snapshot in self._graph.get_state_history(config):
            for m in snapshot.values.get("messages", []):
                if m.id not in seen and isinstance(m, AIMessage):
                    seen.add(m.id)
                    out.append(m)
            if snapshot.values.get("messages") and all(
                m.id in before for m in snapshot.values["messages"]
            ):
                break
        return out

    def get_init_state(self, message_history: list[Message] | None = None) -> GraphAgentState:
        return GraphAgentState(pending_history=list(message_history or []))

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: GraphAgentState
    ) -> tuple[AssistantMessage, GraphAgentState]:
        if self._graph is None:
            self._build()
        new_messages = to_lc_history(state.pending_history) + to_lc(message)
        state.pending_history = []

        config = {
            "configurable": {"thread_id": state.thread_id},
            "recursion_limit": recursion_limit(self._ctx),
        }
        before = {m.id for m in self._graph.get_state(config).values.get("messages", [])}
        out = self._graph.invoke({"messages": new_messages}, config=config)
        # Every model call this turn, including replies the output guard removed.
        produced = [m for m in self._turn_ai_messages(config, before) if isinstance(m, AIMessage)]
        reply = to_tau2_assistant(out["messages"][-1])
        # tau2 only sees the final message, but blocked attempts cost money too.
        costs = [m.response_metadata.get("cost_usd") for m in produced]
        if any(c is not None for c in costs):
            reply.cost = sum(c or 0.0 for c in costs)
        return reply, state
