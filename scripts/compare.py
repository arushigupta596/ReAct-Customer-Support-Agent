"""Side-by-side metrics for every config of one run and domain.

uv run python scripts/compare.py --run-id smoke-retail10 --domain retail
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tau2.runner.build import build_environment

from agent.tools import spec_from_tau2_tool
from evals.metrics import summarize

ROWS = [
    ("success_rate", "Success rate (pass^1)", "{:.0%}"),
    ("action_recall", "Action recall", "{:.0%}"),
    ("write_precision", "Write precision", "{:.0%}"),
    ("write_calls", "Write calls", "{}"),
    ("invalid_call_rate", "Invalid call rate", "{:.1%}"),
    ("steps_per_sim", "Agent messages / sim", "{:.1f}"),
    ("cost_usd_per_sim", "USD / sim (agent + user)", "${:.4f}"),
    ("cost_usd_total", "USD total", "${:.2f}"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--domain", required=True)
    args = p.parse_args()

    env_kwargs = {"retrieval_variant": "bm25"} if args.domain == "banking_knowledge" else None
    env = build_environment(args.domain, env_kwargs=env_kwargs)
    kinds = {t.name: spec_from_tau2_tool(t).kind for t in env.get_tools()}

    run_dir = ROOT / "results" / args.run_id / args.domain
    summaries = {
        f.stem: summarize(f, kinds)
        for f in sorted(run_dir.glob("*.json"))
        if f.stem != "summary" and not f.name.endswith(".spans.json")
    }
    (run_dir / "summary.json").write_text(json.dumps(summaries, indent=2))

    configs = list(summaries)
    print(f"\n{args.run_id} / {args.domain}  ({summaries[configs[0]]['num_tasks']} tasks)\n")
    print(f"{'':28}" + "".join(f"{c:>16}" for c in configs))
    for key, label, fmt in ROWS:
        cells = [summaries[c][key] for c in configs]
        print(
            f"{label:28}"
            + "".join(f"{(fmt.format(v) if v is not None else '-'):>16}" for v in cells)
        )
    for k in range(2, max(summaries[c]["trials"] for c in configs) + 1):
        cells = [summaries[c]["pass_k"].get(k) for c in configs]
        print(
            f"{f'pass^{k}':28}"
            + "".join(f"{(f'{v:.2f}' if v is not None else '-'):>16}" for v in cells)
        )
    esc = [summaries[c]["escalation"] for c in configs]
    cells = [f"{e['correct']}/{e['unnecessary']}/{e['missed']}" for e in esc]
    print(f"{'Escalation ok/extra/missed':28}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
