import json

import pytest

from evals.metrics import pass_hat_k, summarize


def test_pass_hat_k_hand_computed():
    assert pass_hat_k(4, 4, 4) == 1.0
    assert pass_hat_k(4, 3, 1) == 0.75
    assert pass_hat_k(4, 3, 2) == pytest.approx(3 / 6)  # C(3,2)/C(4,2)
    assert pass_hat_k(4, 3, 4) == 0.0
    assert pass_hat_k(4, 0, 1) == 0.0


def test_pass_hat_k_needs_enough_trials():
    with pytest.raises(ValueError):
        pass_hat_k(2, 2, 3)


def _sim(task_id, reward, calls, errors=0, agent_cost=0.01, user_cost=0.002):
    messages = [{"role": "assistant", "tool_calls": [c]} for c in calls]
    messages += [
        {"role": "tool", "requestor": "assistant", "error": i < errors} for i in range(len(calls))
    ]
    return {
        "task_id": task_id,
        "reward_info": {"reward": reward, "action_checks": []},
        "messages": messages,
        "agent_cost": agent_cost,
        "user_cost": user_cost,
    }


def test_summarize(tmp_path):
    cancel = {"name": "cancel", "arguments": {"id": "1"}}
    tasks = [
        {"id": "a", "evaluation_criteria": {"actions": [{**cancel, "requestor": "assistant"}]}},
        {
            "id": "b",
            "evaluation_criteria": {
                "actions": [
                    {"name": "transfer_to_human_agents", "arguments": {}, "requestor": "assistant"}
                ]
            },
        },
    ]
    sims = [
        _sim("a", 1.0, [cancel]),
        _sim("a", 0.0, [{"name": "cancel", "arguments": {"id": "2"}}], errors=1),
        _sim("b", 1.0, [{"name": "transfer_to_human_agents", "arguments": {}}]),
        _sim("b", 0.0, []),
    ]
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"tasks": tasks, "simulations": sims}))

    out = summarize(path, {"cancel": "WRITE", "transfer_to_human_agents": "GENERIC"})
    assert out["trials"] == 2
    assert out["success_rate"] == 0.5
    assert out["pass_k"][1] == 0.5 and out["pass_k"][2] == 0.0
    assert out["write_precision"] == 0.5  # one of two cancels had the expected args
    assert out["invalid_call_rate"] == 1 / 3
    assert out["escalation"] == {"correct": 1, "unnecessary": 0, "missed": 1}
    assert out["cost_usd_per_sim"] == pytest.approx(0.012)
