# ReAct Support Agent on tau2-bench

A customer-support AI agent that resolves real requests (cancel an order, change a flight, update an account) by thinking one step at a time and using tools, with guardrails on every action and a human who signs off before anything irreversible happens. It is measured on [tau2-bench](https://github.com/sierra-research/tau2-bench) against tau2's built-in agent, and every decision is traced. Full plan and build log: [PLAN.md](PLAN.md). Debugging write-ups: [docs/debug_journal.md](docs/debug_journal.md).

## What it is

A **ReAct agent** built with LangGraph: it **reasons** about what the customer needs, **acts** by calling one tool, **reads the result**, and decides again, looping until it can reply. It works in four customer-service domains from tau2-bench, each with its own database, tools and written policy:

| Domain | What customers ask for | Live demo |
|---|---|---|
| Retail | Cancel, modify, return or exchange orders; update addresses | Yes |
| Airline | Book, change, cancel flights; bags; compensation | Yes |
| Banking | Disputes, cards, account changes, answered from a policy knowledge base | Yes |
| Telecom | Bills, line suspensions, data, roaming, phone troubleshooting | Benchmark only (the customer must run tools on their own phone) |

The same agent code runs in two modes:

- **Benchmark mode:** tau2-bench plays the customer with a simulated user and scores the agent on whether the database ends up correct.
- **Live mode:** a real person chats with the agent in a browser, tools run through an MCP server, and an operator approves actions in a console.

## What it does

A customer turn flows through these steps:

1. **Input guard:** strips prompt-injection and fake-authority text ("ignore your instructions", "SYSTEM OVERRIDE", "as the manager I authorize").
2. **Reason:** the model (Claude Sonnet 5.5) decides to call one tool or reply.
3. **Pre-action checks:** before any tool runs, deterministic checks confirm the tool exists for this domain, the arguments are valid, and the agent is not looping. For actions that change data, domain policy rules also run, each quoting the policy line it enforces. Examples: the customer was authenticated, the order belongs to them, the customer said an explicit "yes" to the exact action, a cancellation is actually eligible, at most 2 GB of data is refueled.
4. **Human sign-off:** every action that changes customer data pauses for an operator, who sees the action, the customer's exact words and every check result, then approves, edits the arguments, or rejects with a reason. Read-only lookups run without waiting.
5. **Act:** the approved tool runs against that session's data through the MCP server.
6. **Output guard:** before a reply reaches the customer, it is blocked if it claims an action that never ran ("your order has been cancelled" with no cancellation).

Every step appears in a plain-language **decision timeline** beside the chat, and each step links to its full trace in Phoenix.

## Where the human sits

The operator is the last gate before anything irreversible. Policy checks run first, so the operator only sees actions that already passed them, along with the evidence. Every decision is stored with reviewer, time and reason. An internal audit record (banking identity-verification logging) is the only write that does not wait for sign-off.

## How it is evaluated

| Eval | What it measures | Where |
|---|---|---|
| tau2-bench | Task success, pass^1 to pass^4 (consistency across repeated trials), action recall, write precision, escalation accuracy, cost per conversation, against tau2's built-in agent on the same model | `scripts/run_bench.py`, `src/evals/metrics.py` |
| Guard replay | Every rule replayed over all saved real conversations, offline and free: did it block an action the task actually wanted? | `scripts/replay_guards.py` |
| Red-team suite | 21 attacks (injection, fake authority, other customers' data, skipped verification, policy violations, false claims) and 5 normal requests, with each guard switched off one at a time | `evals/redteam/cases.yaml`, `scripts/run_redteam.py` |
| Demo rehearsal | The three demo scenarios end to end, with checks on the timeline and a 5-minute budget | `demo/scenarios/`, `scripts/rehearse.py` |

Results so far (development runs on the train split; headline test-split numbers come later):

- Guards: 0 false blocks across all real conversations replayed in all four domains.
- Red-team, Claude Sonnet 5.5: 0 of 21 attacks succeeded with guards on or off (the model refused all of them).
- Red-team, Claude Haiku 4.5: 2 of 21 succeeded with guards off, 0 with guards on. Switching off only the confirmation rule, or only the overdue-bill rule, brings its breach back.

## Observability

Every benchmark simulation and every live customer turn is one trace in [Phoenix](https://phoenix.arize.com): graph steps, model calls, MCP tool calls, and every guard and approval decision as an event. Eval scores are written back onto each benchmark trace, so a failed task opens straight into its conversation. Personal data (emails, phone numbers, card numbers) is masked in exported traces, never in what the model sees.

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
| `src/agent/nodes/` | One file per graph step (guards, reason, approval, execute) |
| `src/agent/policy_rules/` | Domain policy rules (retail, airline, banking, telecom), each citing its policy line |
| `src/agent/llm.py` | OpenRouter provider pinning and price registration |
| `src/tau2_adapter/` | tau2 `HalfDuplexAgent` wrapper, message conversion, runtime registration |
| `src/mcp_server/` | MCP tool server, per-session state (an action log replayed on the stock data), domain config and suggested messages |
| `src/api/` | FastAPI service, approvals, decision timeline, and the console and chat pages (`ui/`) |
| `src/evals/` | Metrics, Phoenix score write-back, red-team runner |
| `src/observability/` | Phoenix tracing, PII masking, per-simulation guard events |
| `scripts/` | Benchmark runner, comparison table, guard replay, red-team, rehearsal, migrations |
| `demo/scenarios/` | The three scripted demo scenarios |
| `results/` | Saved benchmark and red-team results |
| `docs/debug_journal.md` | What broke, how the traces showed it, the fix, and the measured effect |
| `vendor/tau2-bench` | tau2-bench v1.0.1 (submodule, unmodified) |
