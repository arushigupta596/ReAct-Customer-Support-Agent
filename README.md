# ReAct Support Agent on tau2-bench

A LangGraph ReAct customer-support agent (reason, call one tool, read the result, decide again) with guardrails and human sign-off before irreversible actions, scored on tau2-bench against tau2's built-in baseline. Full plan: [PLAN.md](PLAN.md).

## Setup

```bash
brew install uv
git submodule update --init          # tau2-bench pinned at v1.0.1
uv sync
cp .env.example .env                 # then add OPENROUTER_API_KEY
```

## Tracing (Phoenix)

```bash
docker compose up -d                 # Phoenix + Postgres on http://localhost:6006
# or, without Docker:
uvx --from arize-phoenix==20.21.0 phoenix serve
```

Every benchmark run traces to project `bench-<domain>-<config>` and writes per-simulation scores (reward, action recall, write precision, escalation, passes-all-trials) back onto each trace. Add `--no-trace` for metrics-only runs.

## Run

```bash
uv run pytest -q                                                              # no API calls
uv run python scripts/run_bench.py --domain retail --config baseline --num-tasks 10
uv run python scripts/run_bench.py --domain retail --config graph-full --num-tasks 10
uv run python scripts/compare.py --run-id <run-id> --domain retail                  # side-by-side table
```

Runs use the train split by default; pass `--split test` for headline numbers.

Configs: `baseline` (tau2 `llm_agent`), `graph-noguard` (our graph, guards off, auto-approve), `graph-full` (all guards, rule-check reviewer). Results are copied to `results/<date>-<sha>/<domain>/<config>.json`.

## Live demo (local)

```bash
brew services start postgresql@16 && createdb react_agent    # once
uv run python scripts/migrate.py                              # once
uv run uvicorn api.main:app --app-dir src --port 8000
```

Open `http://localhost:8000/console`, start a session and open the customer chat link. With `OPERATOR_TOKEN` unset (the local default) the console needs no sign-in; set it in `.env` to require one, and always set it before any public deploy. Every action that changes data waits in the console for approve, edit or reject.

The chat shows suggested messages for each domain (defined in `src/mcp_server/domains.py`), tagged by what they demonstrate: **Sign-off** (reaches the approvals console), **Question**, **Policy test** (the agent should refuse) and **Guard test** (an injection attempt the input guard flags). `uv run python scripts/rehearse.py` plays the three scripted demo scenarios end to end.

## Layout

| Path | What |
|---|---|
| `src/agent/graph.py` | The graph: `guard_input -> reason -> guard_pre_tool -> approval_gate -> execute_tools / emit`, `reason -> guard_output` |
| `src/agent/nodes/` | One file per node |
| `src/agent/llm.py` | OpenRouter provider pinning and price registration |
| `src/tau2_adapter/` | tau2 `HalfDuplexAgent` wrapper, message conversion, runtime registration |
| `scripts/run_bench.py` | Runs one config on one domain |
| `vendor/tau2-bench` | tau2-bench v1.0.1 (submodule, unmodified) |
