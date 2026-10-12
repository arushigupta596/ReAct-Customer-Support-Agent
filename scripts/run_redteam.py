"""Run the red-team suite and print breach rates per config.

uv run python scripts/run_redteam.py                       # all cases
uv run python scripts/run_redteam.py --cases rt-preconfirm al-cancel-ineligible
"""

import argparse
import os
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from agent.llm import llm_args, make_chat_model, register_prices
from evals.redteam import run_suite, save


def table(results) -> None:
    errors = [r for r in results if r.error]
    for r in errors:
        print(f"ERROR {r.case} / {r.config}: {r.error[:200]}")
    results = [r for r in results if not r.error]
    attacks = [r for r in results if r.category != "benign"]
    benign = [r for r in results if r.category == "benign"]
    by_config = defaultdict(list)
    for r in attacks:
        by_config[r.config].append(r)

    print(f"\nAttacks: {len({r.case for r in attacks})} cases\n")
    print(f"{'config':28}{'runs':>6}{'breached':>10}{'rate':>8}")
    for cfg in ["all_on", "all_off"] + sorted(c for c in by_config if c.startswith("minus_")):
        rs = by_config.get(cfg, [])
        if rs:
            n = sum(r.breached for r in rs)
            print(f"{cfg:28}{len(rs):>6}{n:>10}{n / len(rs):>8.0%}")

    print(f"\n{'category':20}{'all_on breached':>17}{'all_off breached':>18}")
    cats = defaultdict(lambda: {"all_on": [0, 0], "all_off": [0, 0]})
    for r in attacks:
        if r.config in ("all_on", "all_off"):
            cats[r.category][r.config][0] += r.breached
            cats[r.category][r.config][1] += 1
    for cat, v in sorted(cats.items()):
        on, off = v["all_on"], v["all_off"]
        print(f"{cat:20}{f'{on[0]}/{on[1]}':>17}{f'{off[0]}/{off[1]}':>18}")

    # Which layer stopped each attack with all guards on.
    print("\nWith all guards on, attacks were stopped by:")
    stopped = defaultdict(int)
    for r in by_config.get("all_on", []):
        blocks = {e["guard"] for e in r.guard_events if e["verdict"] == "block"}
        stopped[
            "breached"
            if r.breached
            else ("guard: " + ", ".join(sorted(blocks)) if blocks else "model refused")
        ] += 1
    for k, v in sorted(stopped.items(), key=lambda kv: -kv[1]):
        print(f"  {v:3}  {k}")

    on = [r for r in benign if r.config == "all_on"]
    blocked = [r for r in on if any(e["verdict"] == "block" for e in r.guard_events)]
    print(
        f"\nBenign controls (all guards on): {sum(r.passed for r in on)}/{len(on)} completed; "
        f"{len(blocked)} had a guard block along the way"
    )
    for r in on:
        if not r.passed:
            print(f"  NOT completed: {r.case}: last reply: {(r.replies or [''])[-1][:160]!r}")
    print(f"\nCost: ${sum(r.cost_usd for r in results):.2f} (agent only)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases", nargs="*", default=None)
    p.add_argument("--run-id", default=f"{date.today().isoformat()}")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()

    register_prices()
    model = os.environ["AGENT_MODEL"]
    llm = make_chat_model(
        model,
        llm_args(model, os.environ["AGENT_PROVIDER"], cache=True, effort=os.getenv("AGENT_EFFORT")),
    )
    results = run_suite(llm, model, args.cases, workers=args.workers)
    out = ROOT / "results" / "redteam" / f"{args.run_id}.json"
    save(results, out)
    table(results)
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
