"""OpenRouter access through LiteLLM with one pinned provider per model.

Every model call (our agent, the tau2 baseline, the user simulator and the judge)
uses `llm_args(model, provider)`, so all of them make identical requests.
"""

import litellm
from langchain_litellm import ChatLiteLLM

# USD per token (input, output, cache read), from https://openrouter.ai/api/v1/models
# on 2026-10-12. Anthropic cache writes cost 1.25x input.
# LiteLLM 1.82 has no prices for these OpenRouter ids, so without this tau2
# would record a cost of 0 for every simulation (ours and the baseline's).
OPENROUTER_PRICES = {
    "openrouter/anthropic/claude-haiku-4.5": (1.0e-6, 5.0e-6, 1.0e-7),
    "openrouter/anthropic/claude-sonnet-5.5": (2.0e-6, 1.0e-5, 1.0e-7),
    "openrouter/openai/gpt-4.1": (2.0e-6, 8.0e-6, 5.0e-7),
    "openrouter/openai/gpt-4.1-mini": (4.0e-7, 1.6e-6, 1.0e-7),
}


def register_prices() -> None:
    litellm.register_model(
        {
            model: {
                "input_cost_per_token": inp,
                "output_cost_per_token": out,
                "cache_read_input_token_cost": cached,
                "cache_creation_input_token_cost": inp * 1.25 if "anthropic/" in model else inp,
                "litellm_provider": "openrouter",
                "mode": "chat",
                "supports_function_calling": True,
            }
            for model, (inp, out, cached) in OPENROUTER_PRICES.items()
        }
    )


def provider_routing(provider: str) -> dict:
    """OpenRouter provider block: no load balancing, no fallbacks."""
    return {
        "order": [provider],
        "allow_fallbacks": False,
        "require_parameters": True,
    }


# Cache the tools + system prefix and the growing conversation. Anthropic needs explicit
# markers; LiteLLM injects them per call, so tau2's baseline gets exactly the same caching.
CACHE_POINTS = [{"location": "message", "role": "system"}, {"location": "message", "index": -1}]


# Claude Sonnet 5.5 rejects sampling parameters, so temperature is only sent where the
# model accepts it; with require_parameters, sending it would make OpenRouter refuse to route.
NO_SAMPLING_PARAMS = ("claude-sonnet-5.5",)
# Thinking-capable models: effort replaces temperature as the main control.
EFFORT_MODELS = ("claude-sonnet-5.5",)


def llm_args(model: str, provider: str, cache: bool = False, effort: str | None = None) -> dict:
    """Arguments shared by every LiteLLM call routed through OpenRouter."""
    args: dict = {"extra_body": {"provider": provider_routing(provider)}}
    if not any(m in model for m in NO_SAMPLING_PARAMS):
        args["temperature"] = 0.0
    if effort and any(m in model for m in EFFORT_MODELS):
        args["reasoning_effort"] = effort
    if cache:
        args["cache_control_injection_points"] = CACHE_POINTS
    return args


def make_chat_model(model: str, args: dict) -> ChatLiteLLM:
    args = dict(args)
    return ChatLiteLLM(
        model=model,
        temperature=args.pop("temperature", None),
        model_kwargs=args,
        max_retries=3,
    )


def usage_cost(model: str, usage: dict) -> float | None:
    """USD for one call from LangChain usage_metadata, pricing cache reads and writes."""
    details = usage.get("input_token_details") or {}
    try:
        prompt_cost, completion_cost = litellm.cost_per_token(
            model=model,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            cache_read_input_tokens=details.get("cache_read") or 0,
            cache_creation_input_tokens=details.get("cache_creation") or 0,
        )
    except Exception:  # noqa: BLE001  unknown model: cost is reported as missing, not 0
        return None
    return prompt_cost + completion_cost
