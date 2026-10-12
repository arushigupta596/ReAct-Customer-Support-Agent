"""Run one config on one domain through tau2 and copy the results into results/.

    uv run python scripts/run_bench.py --domain retail --config graph-full --num-tasks 10

Configs: baseline (tau2 llm_agent), graph-noguard, graph-full. See PLAN.md section 6.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env")

from tau2.data_model.simulation import TextRunConfig
from tau2.runner.batch import run_domain
from tau2.runner.build import build_environment
from tau2.utils.utils import DATA_DIR

from agent.llm import llm_args, register_prices
from agent.tools import spec_from_tau2_tool
from evals.report import annotate
from tau2_adapter.factory import CONFIGS, register

# Free, local retrieval for banking (tau2's default needs OpenAI embeddings + a sandbox).
BANKING_RETRIEVAL = "bm25"


def point_judge_at_openrouter() -> None:
    """tau2 hard-codes its NL-assertion judge (retail) to gpt-4.1 on OpenAI's own API."""
    import tau2.evaluator.evaluator_nl_assertions as nl

    nl.DEFAULT_LLM_NL_ASSERTIONS = os.environ["JUDGE_MODEL"]
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = llm_args(
        os.environ["JUDGE_MODEL"], os.environ["JUDGE_PROVIDER"]
    )


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, check=False
    )
    return out.stdout.decode().strip() or "nogit"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--domain", required=True, choices=["retail", "airline", "telecom", "banking_knowledge"]
    )
    p.add_argument("--config", required=True, choices=list(CONFIGS))
    # Develop on train only; headline runs pass --split test explicitly (PLAN.md D5).
    # banking_knowledge has no official split, so ours is in data/splits (seed 2026, 60/37).
    p.add_argument("--split", default="train", help="train | test | base")
    p.add_argument("--num-tasks", type=int, default=None)
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--model", default=os.getenv("DEV_MODEL") or os.getenv("AGENT_MODEL"))
    p.add_argument("--run-id", default=None)
    p.add_argument("--no-trace", action="store_true", help="metrics only (scheduled CI runs)")
    args = p.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        sys.exit("OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill it in.")

    register_prices()
    point_judge_at_openrouter()
    agent_name = register(args.config, args.domain)
    run_id = args.run_id or f"{date.today().isoformat()}-{git_sha()}"
    save_to = f"{run_id}_{args.domain}_{args.config}"

    spans = None
    if not args.no_trace:
        from observability.tau2_spans import SimulationSpans
        from observability.tracing import init_tracing

        init_tracing(project=f"bench-{args.domain}-{args.config}")
        spans = SimulationSpans(
            {
                "tau2.domain": args.domain,
                "config.name": args.config,
                "run.id": run_id,
                "git.sha": git_sha(),
                "model": args.model,
                "provider": os.environ["AGENT_PROVIDER"],
            }
        ).install()

    task_ids = args.task_ids
    if args.domain == "banking_knowledge" and args.split in ("train", "test"):
        split_ids = json.loads((ROOT / "data/splits/banking_knowledge.json").read_text())[
            args.split
        ]
        task_ids = [t for t in (task_ids or split_ids) if t in split_ids]

    config = TextRunConfig(
        domain=args.domain,
        agent=agent_name,
        llm_agent=args.model,
        llm_args_agent=llm_args(
            args.model, os.environ["AGENT_PROVIDER"], cache=True, effort=os.getenv("AGENT_EFFORT")
        ),
        llm_user=os.environ["USER_SIM_MODEL"],
        llm_args_user=llm_args(os.environ["USER_SIM_MODEL"], os.environ["USER_SIM_PROVIDER"]),
        task_split_name=args.split,
        task_ids=task_ids,
        num_tasks=args.num_tasks,
        num_trials=args.trials,
        max_concurrency=args.concurrency,
        save_to=save_to,
        retrieval_config=BANKING_RETRIEVAL if args.domain == "banking_knowledge" else None,
    )
    run_domain(config)

    src = DATA_DIR / "simulations" / save_to / "results.json"
    dest = ROOT / "results" / run_id / args.domain / f"{args.config}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    print(f"\nResults copied to {dest.relative_to(ROOT)}")

    if spans is not None:
        from observability.tracing import flush

        flush()
        spans_path = dest.with_suffix(".spans.json")
        spans_path.write_text(json.dumps(spans.report(), indent=1))
        env_kwargs = (
            {"retrieval_variant": BANKING_RETRIEVAL} if args.domain == "banking_knowledge" else None
        )
        env = build_environment(args.domain, env_kwargs=env_kwargs)
        kinds = {t.name: spec_from_tau2_tool(t).kind for t in env.get_tools()}
        endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
        n = annotate(dest, spans_path, kinds, endpoint)
        print(f"Wrote {n} eval annotations to Phoenix project bench-{args.domain}-{args.config}")


if __name__ == "__main__":
    main()
