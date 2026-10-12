"""One trace per tau2 simulation, without modifying tau2.

`tau2.runner.batch.run_tasks` calls `run_single_task` by its module-level name, so wrapping
that attribute puts every simulation (agent, user simulator, judge calls) under one root span.
"""

import threading
from collections import defaultdict

from openinference.instrumentation import using_metadata
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor
from tau2.runner import batch


class GuardEventCollector(SpanProcessor):
    """Keeps every non-pass guard decision per trace, so results don't depend on querying Phoenix."""

    def __init__(self):
        self.by_trace: dict[int, list[dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def on_end(self, span) -> None:
        for event in span.events:
            attrs = event.attributes or {}
            if event.name == "guard" and attrs.get("guard.verdict") != "pass":
                with self._lock:
                    self.by_trace[span.context.trace_id].append(
                        {
                            "guard": attrs.get("guard.name"),
                            "verdict": attrs.get("guard.verdict"),
                            "tool": attrs.get("guard.tool"),
                            "reason": attrs.get("guard.reason"),
                        }
                    )


class SimulationSpans:
    """Wraps tau2's per-task runner and remembers which trace belongs to which simulation."""

    def __init__(self, run_attrs: dict):
        self.run_attrs = run_attrs  # config.name, git.sha, model, provider, tau2.domain
        self.spans: dict[str, dict] = {}  # simulation id -> {span_id, trace_id, task_id, seed}
        self._pending: dict[str, int] = {}  # simulation id -> raw trace id
        self._lock = threading.Lock()
        self._original = None
        self.guards = GuardEventCollector()

    def install(self) -> "SimulationSpans":
        self._original = batch.run_single_task
        batch.run_single_task = self._traced
        trace.get_tracer_provider().add_span_processor(self.guards)
        return self

    def uninstall(self) -> None:
        if self._original is not None:
            batch.run_single_task = self._original

    def _traced(self, config, task, **kwargs):
        tracer = trace.get_tracer("tau2-bench")
        attrs = {
            **self.run_attrs,
            "tau2.task_id": task.id,
            "tau2.seed": kwargs.get("seed") or -1,
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.AGENT.value,
        }
        with tracer.start_as_current_span(f"simulation task={task.id}", attributes=attrs) as span:
            with using_metadata({"tau2.task_id": task.id, **self.run_attrs}):
                sim = self._original(config, task, **kwargs)
            reward = (sim.reward_info.reward if sim.reward_info else None) or 0.0
            span.set_attributes(
                {
                    "tau2.simulation_id": sim.id,
                    "tau2.reward": reward,
                    "tau2.termination_reason": str(sim.termination_reason),
                    "tau2.agent_cost_usd": sim.agent_cost or 0.0,
                    "tau2.user_cost_usd": sim.user_cost or 0.0,
                }
            )
            ctx = span.get_span_context()
            with self._lock:
                self.spans[sim.id] = {
                    "span_id": format(ctx.span_id, "016x"),
                    "trace_id": format(ctx.trace_id, "032x"),
                    "task_id": task.id,
                    "seed": kwargs.get("seed"),
                }
                self._pending[sim.id] = ctx.trace_id
        return sim

    def report(self) -> dict[str, dict]:
        """Span ids per simulation plus its guard blocks / would-blocks (call after flush)."""
        trace_of = {sim_id: tid for sim_id, tid in self._pending.items()}
        return {
            sim_id: {**info, "guard_events": self.guards.by_trace.get(trace_of[sim_id], [])}
            for sim_id, info in self.spans.items()
        }
