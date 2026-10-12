import json

from evals.report import build_annotations


def _sim(sim_id, task_id, reward, calls=()):
    return {
        "id": sim_id,
        "task_id": task_id,
        "reward_info": {"reward": reward, "action_checks": []},
        "messages": [{"role": "assistant", "tool_calls": [c]} for c in calls],
        "agent_cost": 0.01,
        "user_cost": 0.0,
    }


def test_annotations_per_simulation(tmp_path):
    cancel = {"name": "cancel", "arguments": {"id": "1"}}
    tasks = [
        {"id": "a", "evaluation_criteria": {"actions": [{**cancel, "requestor": "assistant"}]}},
        {"id": "b", "evaluation_criteria": {"actions": []}},
    ]
    sims = [
        _sim("s1", "a", 1.0, [cancel]),
        _sim("s2", "a", 0.0, [{"name": "cancel", "arguments": {"id": "9"}}]),
        _sim("s3", "b", 1.0),
        _sim("s4", "b", 1.0, [{"name": "transfer_to_human_agents", "arguments": {}}]),
    ]
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"tasks": tasks, "simulations": sims}))
    spans = {f"s{i}": {"span_id": f"{i:016x}"} for i in range(1, 4)}  # s4 ran untraced

    anns = build_annotations(path, spans, {"cancel": "WRITE"})
    by = {(a["span_id"], a["name"]): a["result"] for a in anns}

    assert {sid for sid, _ in by} == {f"{i:016x}" for i in range(1, 4)}  # s4 skipped
    assert by[("0000000000000001", "reward")] == {"score": 1.0, "label": "pass"}
    assert by[("0000000000000002", "reward")] == {"score": 0.0, "label": "fail"}
    assert by[("0000000000000001", "write_precision")]["score"] == 1.0
    assert by[("0000000000000002", "write_precision")]["score"] == 0.0
    # Task a failed one of two trials, so neither of its runs counts toward pass^k.
    assert by[("0000000000000001", "passes_all_trials")]["label"] == "no"
    assert by[("0000000000000003", "passes_all_trials")]["label"] == "yes"
    assert all(a["annotator_kind"] == "CODE" for a in anns)
