"""Collect a benchmark run and the red-team results into results/published.json for /evals.

    uv run python scripts/publish.py --run-id final-0517462 \
        --redteam redteam-v2 --redteam-stress redteam-v2-haiku
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tau2.runner.build import build_environment

from agent.tools import spec_from_tau2_tool
from evals.metrics import summarize

DOMAINS = ["retail", "airline", "telecom", "banking_knowledge"]
CONFIGS = ["baseline", "graph-full", "graph-noguard"]


def tool_kinds(domain: str) -> dict:
    env_kwargs = {"retrieval_variant": "bm25"} if domain == "banking_knowledge" else None
    return {
        t.name: spec_from_tau2_tool(t).kind
        for t in build_environment(domain, env_kwargs=env_kwargs).get_tools()
    }


def guard_counts(spans_path: Path) -> dict:
    if not spans_path.exists():
        return {}
    counts: Counter = Counter()
    for sim in json.loads(spans_path.read_text()).values():
        for e in sim.get("guard_events", []):
            counts[f"{e['verdict']}:{e['guard']}"] += 1
    return dict(counts)


def redteam(name: str | None) -> dict | None:
    if not name:
        return None
    rows = [
        r
        for r in json.loads((ROOT / "results" / "redteam" / f"{name}.json").read_text())
        if not r.get("error")
    ]
    attacks = [r for r in rows if r["category"] != "benign"]

    def rate(config):
        rs = [r for r in attacks if r["config"] == config]
        return {"breached": sum(r["breached"] for r in rs), "runs": len(rs)}

    ablations = sorted({r["config"] for r in attacks if r["config"].startswith("minus_")})
    benign = [r for r in rows if r["category"] == "benign" and r["config"] == "all_on"]
    return {
        "name": name,
        "attacks": len({r["case"] for r in attacks}),
        "all_on": rate("all_on"),
        "all_off": rate("all_off"),
        "ablations": {a.removeprefix("minus_"): rate(a) for a in ablations if rate(a)["breached"]},
        "benign_completed": sum(r["passed"] for r in benign),
        "benign_runs": len(benign),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--redteam", default=None, help="red-team result name with the headline model")
    p.add_argument(
        "--redteam-stress", default=None, help="red-team result name with a weaker model"
    )
    p.add_argument("--model", default="Claude Sonnet 5.5")
    p.add_argument("--stress-model", default="Claude Haiku 4.5")
    p.add_argument("--split", default="test")
    args = p.parse_args()

    run_dir = ROOT / "results" / args.run_id
    domains = {}
    for domain in DOMAINS:
        kinds = tool_kinds(domain)
        configs = {}
        for config in CONFIGS:
            path = run_dir / domain / f"{config}.json"
            if path.exists():
                configs[config] = {
                    **summarize(path, kinds),
                    "guards": guard_counts(path.with_suffix(".spans.json")),
                }
        if configs:
            domains[domain] = configs

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    published = {
        "run_id": args.run_id,
        "published_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "commit": sha,
        "agent_model": args.model,
        "user_simulator": "GPT-4.1",
        "split": args.split,
        "domains": domains,
        "redteam": {
            m: r
            for m, r in (
                (args.model, redteam(args.redteam)),
                (args.stress_model, redteam(args.redteam_stress)),
            )
            if r
        },
        "total_cost_usd": round(
            sum(c["cost_usd_total"] for d in domains.values() for c in d.values()), 2
        ),
    }
    out = ROOT / "results" / "published.json"
    out.write_text(json.dumps(published, indent=1))
    print(
        f"Wrote {out.relative_to(ROOT)}: {len(domains)} domains, ${published['total_cost_usd']} of runs"
    )


if __name__ == "__main__":
    main()
