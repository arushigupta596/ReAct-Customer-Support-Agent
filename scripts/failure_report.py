"""Group a run's failed simulations into clusters, to pick what to fix first.

    uv run python scripts/failure_report.py --run-id final-0517462 --config graph-full

For each failure: which part of tau2's reward failed, whether the agent missed an expected
action, made an unexpected write, transferred, or was blocked by a guard. Tasks that fail on
every trial are listed separately from flaky ones (some trials pass).
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evals.metrics import SUCCESS, load


def classify(sim: dict, task: dict, guard_events: list[dict]) -> list[str]:
    info = sim.get("reward_info") or {}
    tags = []
    for part, score in (info.get("reward_breakdown") or {}).items():
        if score < SUCCESS:
            tags.append(f"failed:{part}")
    checks = info.get("action_checks") or []
    missed = [
        c["action"]["name"]
        for c in checks
        if not c["action_match"] and c["action"]["requestor"] == "assistant"
    ]
    if missed:
        tags.append("missed:" + ",".join(sorted(set(missed))))
    if any(e["verdict"] == "block" for e in guard_events):
        tags.append(
            "guard_blocked:"
            + ",".join(sorted({e["guard"] for e in guard_events if e["verdict"] == "block"}))
        )
    made = [
        tc["name"]
        for m in sim["messages"]
        if m["role"] == "assistant"
        for tc in m.get("tool_calls") or []
    ]
    if "transfer_to_human_agents" in made:
        tags.append("transferred")
    if str(sim.get("termination_reason", "")).lower() not in (
        "user_stop",
        "terminationreason.user_stop",
    ):
        tags.append(f"ended:{sim.get('termination_reason')}")
    return tags or ["unclassified"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--config", default="graph-full")
    p.add_argument(
        "--domains", nargs="*", default=["retail", "airline", "telecom", "banking_knowledge"]
    )
    args = p.parse_args()

    for domain in args.domains:
        path = ROOT / "results" / args.run_id / domain / f"{args.config}.json"
        if not path.exists():
            continue
        tasks, sims = load(path)
        spans_path = path.with_suffix(".spans.json")
        spans = json.loads(spans_path.read_text()) if spans_path.exists() else {}

        by_task: dict[str, list[bool]] = defaultdict(list)
        clusters: Counter = Counter()
        examples: dict[str, list[str]] = defaultdict(list)
        for sim in sims:
            ok = ((sim.get("reward_info") or {}).get("reward") or 0) >= SUCCESS
            by_task[sim["task_id"]].append(ok)
            if ok:
                continue
            for tag in classify(
                sim, tasks[sim["task_id"]], spans.get(sim["id"], {}).get("guard_events", [])
            ):
                clusters[tag] += 1
                if len(examples[tag]) < 4 and sim["task_id"] not in examples[tag]:
                    examples[tag].append(sim["task_id"])

        always = sorted(t for t, v in by_task.items() if not any(v))
        flaky = sorted(t for t, v in by_task.items() if any(v) and not all(v))
        fails = sum(not ok for v in by_task.values() for ok in v)
        print(f"\n== {domain} / {args.config}: {fails} failed of {len(sims)} sims")
        print(
            f"   always fail ({len(always)}): {', '.join(always[:12])}{' …' if len(always) > 12 else ''}"
        )
        print(
            f"   flaky       ({len(flaky)}): {', '.join(flaky[:12])}{' …' if len(flaky) > 12 else ''}"
        )
        for tag, n in clusters.most_common(10):
            print(f"   {n:4}  {tag:55} e.g. tasks {', '.join(examples[tag])}")


if __name__ == "__main__":
    main()
