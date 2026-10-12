"""Rehearse the demo scenarios against a running API (local or deployed).

    uv run python scripts/rehearse.py                          # all, against localhost:8000
    uv run python scripts/rehearse.py retail_exchange --base https://<space>.hf.space

Each scenario plays its customer turns, makes the scripted operator decisions when an
approval appears, then checks the decision timeline and the 5-minute budget.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
SCENARIOS = ROOT / "demo" / "scenarios"
TIME_BUDGET_S = 300


def run(client: httpx.Client, name: str) -> bool:
    sc = yaml.safe_load((SCENARIOS / f"{name}.yaml").read_text())
    op = {"Authorization": f"Bearer {os.getenv('OPERATOR_TOKEN', '')}"}  # ignored when unset
    decisions = list(sc.get("operator", []))
    start = time.monotonic()
    sid = client.post("/sessions", json={"domain": sc["domain"]}, headers=op).json()["session_id"]
    print(f"\n== {sc['title']}  (session {sid[:8]})")

    def settle(result: dict) -> dict:
        while result.get("status") == "awaiting_approval":
            approval = result["approval"]
            if not decisions or decisions[0]["tool"] != approval["tool"]:
                raise AssertionError(f"unexpected approval for {approval['tool']}")
            step = decisions.pop(0)
            body = {"decision": step["decision"], "reason": step.get("reason")}
            if step["decision"] == "edit":
                body["edited_args"] = {**approval["args"], **step["edited_args"]}
            print(f"   operator: {step['decision']} {approval['tool']}")
            result = client.post(f"/approvals/{approval['id']}", json=body, headers=op).json()
        return result

    for turn in sc["turns"]:
        print(f"   customer: {turn[:90]}")
        result = settle(client.post(f"/sessions/{sid}/messages", json={"content": turn}).json())
        reply = (result.get("reply") or "").replace("\n", " ")
        print(f"   agent:    {reply[:110]}")

    events = client.get(f"/sessions/{sid}/events").json()
    elapsed = time.monotonic() - start
    types = {e["type"] for e in events}
    done = {
        e["title"].removeprefix("Done: ")
        for e in events
        if e["type"] == "act" and e["verdict"] == "done"
    }
    expect = sc["expect"]
    problems = [f"missing timeline step '{t}'" for t in expect.get("events", []) if t not in types]
    problems += [f"{t} did not run" for t in expect.get("executed", []) if t not in done]
    problems += [f"{t} ran but must not" for t in expect.get("not_executed", []) if t in done]
    problems += [f"operator step for {d['tool']} never happened" for d in decisions]
    if elapsed > TIME_BUDGET_S:
        problems.append(f"took {elapsed:.0f}s, over the {TIME_BUDGET_S}s budget")

    print(f"   timeline: {' > '.join(e['type'] for e in events)}")
    print(
        f"   {'PASS' if not problems else 'FAIL'} in {elapsed:.0f}s  replay: {client.base_url}/chat/{sid}"
    )
    for p in problems:
        print(f"     - {p}")
    return not problems


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("names", nargs="*", default=sorted(f.stem for f in SCENARIOS.glob("*.yaml")))
    p.add_argument("--base", default="http://localhost:8000")
    args = p.parse_args()
    with httpx.Client(base_url=args.base, timeout=300) as client:
        results = [run(client, n) for n in args.names]
    print(f"\n{sum(results)}/{len(results)} scenarios passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
