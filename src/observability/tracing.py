"""Phoenix tracing (spec section 8).

Local Phoenix for benchmark runs, Phoenix Cloud for the demo; the endpoint and key come
from PHOENIX_COLLECTOR_ENDPOINT / PHOENIX_API_KEY. Every guard and approval decision is a
span event, so block rates can be compared with task outcomes.
"""

import os

from openinference.instrumentation.langchain import LangChainInstrumentor
from openinference.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry import trace
from phoenix.otel import (
    PROJECT_NAME,
    BatchSpanProcessor,
    HTTPSpanExporter,
    Resource,
    SimpleSpanProcessor,
    TracerProvider,
)

from observability.masking import SanitizingExporter

_provider = None


def init_tracing(project: str, batch: bool = True):
    """Register Phoenix as the global tracer provider. Safe to call once per process.

    batch=False exports each span immediately (serverless may freeze after a response).
    """
    global _provider
    if _provider is not None:
        return _provider
    base = os.environ.setdefault("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
    headers = {"authorization": f"Bearer {key}"} if (key := os.getenv("PHOENIX_API_KEY")) else None
    # HTTP export (gRPC caps messages at 4 MB) through the PII-masking, size-capping wrapper.
    exporter = SanitizingExporter(
        HTTPSpanExporter(endpoint=f"{base.rstrip('/')}/v1/traces", headers=headers)
    )
    _provider = TracerProvider(resource=Resource({PROJECT_NAME: project}), verbose=False)
    _provider.add_span_processor(
        BatchSpanProcessor(span_exporter=exporter) if batch else SimpleSpanProcessor(exporter)
    )
    trace.set_tracer_provider(_provider)
    # LangChain covers our graph (node spans + its LLM calls).
    LangChainInstrumentor().instrument(tracer_provider=_provider)
    _trace_tau2_llm_calls(_provider)
    return _provider


def _trace_tau2_llm_calls(provider) -> None:
    """Trace tau2's own LLM calls (baseline agent, user simulator, judge) exactly once.

    tau2 binds `litellm.completion` at import, so global LiteLLM instrumentation misses it,
    while instrumenting globally would also double-record every ChatLiteLLM call from our graph.
    So: take the instrumented function, give it to tau2 only, and put LiteLLM's originals back.
    (`uninstrument()` would also clear the originals the traced wrapper still calls.)
    """
    import litellm
    import tau2.utils.llm_utils as tau2_llm

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    tau2_llm.completion = litellm.completion
    for name, original in instrumentor.original_litellm_funcs.items():
        setattr(litellm, name, original)


def flush() -> None:
    if _provider is not None:
        _provider.force_flush()


def tracing_enabled() -> bool:
    return _provider is not None


def guard_event(name: str, verdict: str, reason: str | None = None, **extra) -> None:
    """Record one guard decision on the current span (a no-op when tracing is off)."""
    attrs = {"guard.name": name, "guard.verdict": verdict, "guard.reason": reason or ""}
    attrs |= {f"guard.{k}": v for k, v in extra.items()}
    trace.get_current_span().add_event("guard", attrs)


def approval_event(tool: str, decision: str, reviewer: str, reason: str | None = None) -> None:
    trace.get_current_span().add_event(
        "approval",
        {
            "approval.tool": tool,
            "approval.decision": decision,
            "approval.reviewer": reviewer,
            "approval.reason": reason or "",
        },
    )
