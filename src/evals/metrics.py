"""Metrics from a tau2 results.json (spec 7.1).

Everything is computed from the saved simulations, so any past run can be re-scored.
"""

import json
from collections import defaultdict
from math import comb
from pathlib import Path

TRANSFER_TOOL = "transfer_to_human_agents"
SUCCESS = 1.0 - 1e-6  # tau2 rewards are 0/1 but stored as floats


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Probability that k i.i.d. trials of a task all succeed, estimated from c of n."""
    if n < k:
        raise ValueError(f"need at least k={k} trials, got n={n}")
    return comb(c, k) / comb(n, k)


def _args_match(expected: dict, actual_args: dict) -> bool:
    keys = expected.get("compare_args")
    exp_args = expected.get("arguments") or {}
    if keys is None:
        keys = exp_args.keys()
    return all(actual_args.get(k) == exp_args.get(k) for k in keys)


def _tool_calls(sim: dict) -> list[dict]:
    return [
        tc for m in sim["messages"] if m["role"] == "assistant" for tc in m.get("tool_calls") or []
    ]


def _tool_results(sim: dict) -> list[dict]:
    return [m for m in sim["messages"] if m["role"] == "tool" and m.get("requestor") == "assistant"]


def score_simulation(sim: dict, task: dict, tool_kinds: dict[str, str]) -> dict:
    """Per-simulation scores; `summarize` aggregates these and report.py writes them to traces."""
    reward = (sim.get("reward_info") or {}).get("reward") or 0.0
    expected = [
        a
        for a in (task.get("evaluation_criteria") or {}).get("actions") or []
        if a["requestor"] == "assistant"
    ]
    checks = [
        c
        for c in (sim.get("reward_info") or {}).get("action_checks") or []
        if c["action"]["requestor"] == "assistant"
    ]
    made = _tool_calls(sim)
    write_calls = [tc for tc in made if tool_kinds.get(tc["name"]) == "WRITE"]
    results = _tool_results(sim)

    transferred = any(tc["name"] == TRANSFER_TOOL for tc in made)
    should = any(a["name"] == TRANSFER_TOOL for a in expected)
    escalation = (
        "correct"
        if transferred and should
        else "unnecessary"
        if transferred
        else "missed"
        if should
        else None
    )
    return {
        "success": reward >= SUCCESS,
        "reward": reward,
        "recall_hits": sum(c["action_match"] for c in checks),
        "recall_total": len(checks),
        "writes": len(write_calls),
        "writes_expected": sum(
            any(a["name"] == tc["name"] and _args_match(a, tc["arguments"]) for a in expected)
            for tc in write_calls
        ),
        "calls": len(results),
        "errors": sum(1 for r in results if r.get("error")),
        "escalation": escalation,
        "cost_usd": (sim.get("agent_cost") or 0.0) + (sim.get("user_cost") or 0.0),
        "steps": sum(1 for m in sim["messages"] if m["role"] == "assistant"),
    }


def load(results_path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(Path(results_path).read_text())
    return {t["id"]: t for t in data["tasks"]}, data["simulations"]


def summarize(results_path: Path, tool_kinds: dict[str, str]) -> dict:
    """Aggregate one config on one domain. `tool_kinds` maps tool name -> READ/WRITE/..."""
    tasks, sims = load(results_path)
    scores = [score_simulation(sim, tasks[sim["task_id"]], tool_kinds) for sim in sims]

    by_task: dict[str, list[bool]] = defaultdict(list)
    for sim, sc in zip(sims, scores, strict=True):
        by_task[sim["task_id"]].append(sc["success"])

    def total(key):
        return sum(sc[key] for sc in scores)

    escalation = {"correct": 0, "unnecessary": 0, "missed": 0}
    for sc in scores:
        if sc["escalation"]:
            escalation[sc["escalation"]] += 1

    trials = min(len(v) for v in by_task.values())
    n_sims = len(sims)
    return {
        "num_tasks": len(by_task),
        "num_sims": n_sims,
        "trials": trials,
        "success_rate": total("success") / n_sims,
        "pass_k": {
            k: sum(pass_hat_k(len(v), sum(v), k) for v in by_task.values()) / len(by_task)
            for k in range(1, trials + 1)
        },
        "action_recall": total("recall_hits") / total("recall_total")
        if total("recall_total")
        else None,
        "write_precision": total("writes_expected") / total("writes") if total("writes") else None,
        "write_calls": total("writes"),
        "invalid_call_rate": total("errors") / total("calls") if total("calls") else None,
        "escalation": escalation,
        "cost_usd_per_sim": total("cost_usd") / n_sims,
        "cost_usd_total": total("cost_usd"),
        "steps_per_sim": total("steps") / n_sims,
    }
