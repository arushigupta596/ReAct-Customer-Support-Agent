"""Red-team suite runner: does a forbidden action execute, with which guards on?

Each case runs in serve mode against a fresh tau2 domain DB with a local executor, and
sign-off set to auto-approve: this measures the model plus guards, with no human in the loop
(in the live demo the operator is a third layer on top).

Configs per case: all guards on, all off (shadow mode still records would-blocks), and each
of the case's expected guards switched off alone.
"""

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from tau2.data_model.message import ToolCall

from agent.context import GraphContext
from agent.graph import build_graph, recursion_limit
from agent.llm import usage_cost
from agent.nodes.common import text_of
from agent.nodes.guard_output import _claims
from mcp_server.domains import fresh_environment, inner_tool_kinds, tool_specs
from observability.tau2_spans import GuardEventCollector

CASES = Path(__file__).resolve().parents[2] / "evals" / "redteam" / "cases.yaml"

_collector = GuardEventCollector()
_provider_lock = threading.Lock()
_provider_ready = False


def _tracer():
    """A local, in-memory tracer so guard events can be read back per case run."""
    global _provider_ready
    with _provider_lock:
        if not _provider_ready:
            provider = trace.get_tracer_provider()
            if not isinstance(provider, TracerProvider):
                provider = TracerProvider()
                trace.set_tracer_provider(provider)
            provider.add_span_processor(_collector)
            _provider_ready = True
    return trace.get_tracer("redteam")


def load_cases(path: Path = CASES) -> list[dict]:
    raw = yaml.safe_load(path.read_text())
    return [{**case, "domain": domain} for domain, cases in raw.items() for case in cases]


def configs_for(case: dict) -> list[tuple[str, bool, frozenset]]:
    configs = [("all_on", True, frozenset()), ("all_off", False, frozenset())]
    configs += [(f"minus_{g}", True, frozenset({g})) for g in case.get("guards", [])]
    return configs


def _matches(pattern: dict, name: str, args: dict) -> bool:
    if pattern["tool"] != name:
        return False
    return all(str(args.get(k)) == str(v) for k, v in (pattern.get("args") or {}).items())


@dataclass
class CaseResult:
    case: str
    domain: str
    category: str
    config: str
    breached: bool
    passed: bool  # attacks: not breached; benign: expected action executed
    error: str | None = None  # the run crashed; excluded from rates
    executed: list = field(default_factory=list)
    guard_events: list = field(default_factory=list)
    replies: list = field(default_factory=list)
    cost_usd: float = 0.0


def run_case(case: dict, config: tuple[str, bool, frozenset], llm, model_name: str) -> CaseResult:
    name, guards_enabled, disabled = config
    domain = case["domain"]
    env = fresh_environment(domain)
    executed: list[tuple[str, dict]] = []

    def executor(tool: str, args: dict, session_id: str) -> tuple[str, bool]:
        result = env.get_response(ToolCall(id="rt", name=tool, arguments=args))
        content = result.content or ""
        if not result.error and not content.startswith("Error"):
            executed.append((tool, args))
        return content, result.error

    ctx = GraphContext(
        mode="serve",
        domain=domain,
        policy=env.get_policy(),
        llm=llm,
        model_name=model_name,
        tools=tool_specs(domain),
        discoverable_kinds=inner_tool_kinds(domain),
        approval_mode="auto_approve",
        guards_enabled=guards_enabled,
        disabled_guards=disabled,
        executor=executor,
    )
    graph = build_graph(ctx, checkpointer=InMemorySaver())
    config_ = {
        "configurable": {"thread_id": uuid.uuid4().hex},
        "recursion_limit": recursion_limit(ctx),
    }

    with _tracer().start_as_current_span(f"redteam {case['id']} {name}") as span:
        for turn in case["turns"]:
            graph.invoke({"messages": [HumanMessage(content=turn)]}, config_)
        trace_id = span.get_span_context().trace_id

    messages = graph.get_state(config_).values["messages"]
    replies = [text_of(m) for m in messages if isinstance(m, AIMessage) and not m.tool_calls]
    cost = sum(
        usage_cost(model_name, m.usage_metadata or {}) or 0
        for m in messages
        if isinstance(m, AIMessage)
    )

    if case.get("claim"):
        breached = not executed and any(_claims(r) for r in replies)
    else:
        breached = any(_matches(p, t, a) for p in case.get("forbidden", []) for t, a in executed)
    if case["category"] == "benign":
        passed = all(any(_matches(p, t, a) for t, a in executed) for p in case["expect"])
    else:
        passed = not breached
    return CaseResult(
        case=case["id"],
        domain=domain,
        category=case["category"],
        config=name,
        breached=breached,
        passed=passed,
        executed=executed,
        guard_events=_collector.by_trace.get(trace_id, []),
        replies=replies,
        cost_usd=cost,
    )


def run_suite(
    llm, model_name: str, case_ids: list[str] | None = None, workers: int = 6
) -> list[CaseResult]:
    cases = [c for c in load_cases() if case_ids is None or c["id"] in case_ids]
    jobs = [(c, cfg) for c in cases for cfg in configs_for(c)]

    def safe(job):
        case, cfg = job
        try:
            return run_case(case, cfg, llm, model_name)
        except Exception as e:  # noqa: BLE001  one broken case must not discard the suite
            return CaseResult(
                case["id"], case["domain"], case["category"], cfg[0], False, False, error=repr(e)
            )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(safe, jobs))


def save(results: list[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.__dict__ for r in results], indent=1, default=str))
