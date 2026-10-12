"""Replay the WRITE-time policy rules over saved trajectories, offline and free.

For every WRITE call in every simulation, evaluate each rule on the conversation up to that
call. A block is a false positive if the task wanted that action (it is in the task's expected
actions) or the task passed; a block of an unexpected action in a failed task is a true catch.

    uv run python scripts/replay_guards.py results/*/retail/*.json --domain retail
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from tau2.runner.build import build_environment

from agent.policy_rules import rules_for
from agent.policy_rules.facts import collect
from agent.tools import DISCOVERABLE_WRAPPER, discoverable_kinds, spec_from_tau2_tool
from evals.metrics import _args_match, load


def to_lc(messages: list[dict]) -> list:
    out = []
    for m in messages:
        if m["role"] == "user" and not m.get("tool_calls"):
            out.append(HumanMessage(content=m.get("content") or ""))
        elif m["role"] == "assistant":
            calls = [
                {"id": tc["id"], "name": tc["name"], "args": tc["arguments"], "type": "tool_call"}
                for tc in m.get("tool_calls") or []
            ]
            out.append(AIMessage(content=m.get("content") or "", tool_calls=calls))
        elif m["role"] == "tool" and m.get("requestor", "assistant") == "assistant":
            out.append(
                ToolMessage(
                    content=m.get("content") or "",
                    tool_call_id=m["id"],
                    status="error" if m.get("error") else "success",
                )
            )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--domain", required=True)
    p.add_argument("--show", type=int, default=5, help="print this many false-positive examples")
    args = p.parse_args()

    env_kwargs = {"retrieval_variant": "bm25"} if args.domain == "banking_knowledge" else None
    tools = build_environment(args.domain, env_kwargs=env_kwargs).get_tools()
    kinds = {t.name: spec_from_tau2_tool(t).kind for t in tools}
    inner = discoverable_kinds(tools)

    def kind_of(call):
        if call["name"] == DISCOVERABLE_WRAPPER:
            return inner.get(call["args"].get("agent_tool_name"), kinds.get(call["name"]))
        return kinds.get(call["name"])

    rules = rules_for(args.domain)
    writes = frozenset(n for n, k in kinds.items() if k == "WRITE")
    stats: dict[str, Counter] = defaultdict(Counter)
    examples = []
    n_writes = 0

    for f in args.files:
        if f.endswith((".spans.json", "summary.json")):
            continue
        tasks, sims = load(Path(f))
        for sim in sims:
            expected = [
                a
                for a in (tasks[sim["task_id"]].get("evaluation_criteria") or {}).get("actions")
                or []
                if a["requestor"] == "assistant"
            ]
            ok = ((sim.get("reward_info") or {}).get("reward") or 0) >= 1 - 1e-6
            msgs = to_lc(sim["messages"])
            errored = {
                m.tool_call_id for m in msgs if isinstance(m, ToolMessage) and m.status == "error"
            }
            for i, m in enumerate(msgs):
                if not isinstance(m, AIMessage):
                    continue
                for call in m.tool_calls:
                    if kind_of(call) != "WRITE":
                        continue
                    n_writes += 1
                    history = msgs[: i + 1]
                    facts = collect(history, writes)
                    for rule in rules:
                        if not rule.applies(call["name"]):
                            continue
                        reason = rule.evaluate(call, facts, history)
                        is_expected = any(
                            a["name"] == call["name"] and _args_match(a, call["args"])
                            for a in expected
                        )
                        if not reason:
                            outcome = "pass"
                        elif call["id"] in errored and not is_expected:
                            outcome = "block_failed_anyway"  # the tool rejected it too
                        elif is_expected or ok:
                            outcome = "false_positive"  # blocked an action the task wanted
                        else:
                            outcome = "true_catch"  # an unexpected WRITE in a failed task
                        stats[rule.name][outcome] += 1
                        if (
                            outcome in ("false_positive", "true_catch")
                            and len(examples) < args.show
                        ):
                            examples.append(
                                (
                                    f"{Path(f).parent.parent.name}/{Path(f).stem}",
                                    sim["task_id"],
                                    rule.name,
                                    reason,
                                )
                            )

    print(f"\n{n_writes} WRITE calls replayed from {len(args.files)} files\n")
    print(f"{'rule':22}{'pass':>7}{'false +':>9}{'failed anyway':>15}{'true catch':>12}")
    for name, c in stats.items():
        print(
            f"{name:22}{c['pass']:>7}{c['false_positive']:>9}"
            f"{c['block_failed_anyway']:>15}{c['true_catch']:>12}"
        )
    for run, task, rule, reason in examples:
        print(f"\n[{run} task {task}] {rule}: {reason[:300]}")


if __name__ == "__main__":
    main()
