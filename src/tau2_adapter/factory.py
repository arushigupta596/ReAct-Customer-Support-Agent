"""Agent factories registered with tau2 at runtime (no fork; PLAN.md D4)."""

import os

from tau2.agent.llm_agent import LLMAgent
from tau2.registry import registry

from tau2_adapter.agent import LangGraphReActAgent


class NoSeedLLMAgent(LLMAgent):
    """tau2's built-in llm_agent, unchanged except that it drops `seed` like our agent does."""

    def set_seed(self, seed: int):
        pass


def make_langgraph_factory(domain: str, approval_mode: str, guards_enabled: bool):
    def create_agent(tools, domain_policy, **kwargs):
        return LangGraphReActAgent(
            tools=tools,
            domain_policy=domain_policy,
            llm=kwargs.get("llm"),
            llm_args=kwargs.get("llm_args"),
            domain=domain,
            approval_mode=approval_mode,
            guards_enabled=guards_enabled,
            max_steps=int(os.getenv("MAX_STEPS_PER_TURN", "30")),
            max_retries=int(os.getenv("MAX_RETRIES_PER_TURN", "2")),
        )

    return create_agent


def create_baseline_agent(tools, domain_policy, **kwargs):
    return NoSeedLLMAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=kwargs.get("llm_args"),
    )


# Run-matrix configs (spec 7.2). Each name is what tau2 records as the agent.
CONFIGS = {
    "baseline": None,
    "graph-noguard": ("auto_approve", False),
    "graph-full": ("rule_check", True),
}


def register(config: str, domain: str) -> str:
    """Register the agent for `config` with tau2 and return its registry name."""
    name = f"{config}:{domain}"
    if registry.get_agent_factory(name) is not None:
        return name
    if config == "baseline":
        registry.register_agent_factory(create_baseline_agent, name)
    else:
        approval_mode, guards = CONFIGS[config]
        registry.register_agent_factory(make_langgraph_factory(domain, approval_mode, guards), name)
    return name
