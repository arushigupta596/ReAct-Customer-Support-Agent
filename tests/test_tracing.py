from opentelemetry.sdk.trace import TracerProvider

from observability.tau2_spans import GuardEventCollector


def test_collector_keeps_non_pass_guard_decisions_per_trace():
    provider = TracerProvider()
    collector = GuardEventCollector()
    provider.add_span_processor(collector)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("node") as span:
        span.add_event("guard", {"guard.name": "schema", "guard.verdict": "pass"})
        span.add_event(
            "guard",
            {
                "guard.name": "confirmation",
                "guard.verdict": "would_block",
                "guard.tool": "cancel",
                "guard.reason": "no yes",
            },
        )
        span.add_event("approval", {"approval.decision": "approve"})
        trace_id = span.get_span_context().trace_id
    assert collector.by_trace[trace_id] == [
        {"guard": "confirmation", "verdict": "would_block", "tool": "cancel", "reason": "no yes"}
    ]
