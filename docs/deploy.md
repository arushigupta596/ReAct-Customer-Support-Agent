# Deploying the demo to a Hugging Face Space

The live demo runs as one Docker container (API, MCP tool server and console) on a free Hugging Face Space, with Neon Postgres for state and Phoenix Cloud for traces. GitHub holds every secret and pushes the app to the Space through the **Deploy to Hugging Face Space** workflow.

Cost: $0 for hosting. Model calls bill your OpenRouter key per conversation (about $0.05 to $0.20 each).

## 1. Create the accounts (once)

| Service | What to create | What you'll need from it |
|---|---|---|
| [Hugging Face](https://huggingface.co/join) | An account. The workflow creates the Space itself. | An access token with **write** permission (Settings → Access Tokens → Create new token → Write) |
| [Neon](https://neon.tech) | A free project, any region close to you | Two connection strings from the project dashboard (Connect): the **pooled** one (host contains `-pooler`) and the **direct** one. Keep `?sslmode=require` at the end. |
| [Phoenix Cloud](https://app.phoenix.arize.com) | An account and a space | An **API key** (Settings → API Keys) and your **collector endpoint / hostname** (Settings), for example `https://app.phoenix.arize.com/s/<your-space>` |

Pick a Space name, for example `arushigupta596/react-support-agent`.

## 2. Add secrets and variables to GitHub (once)

Each `gh secret set` command prompts for the value, so nothing lands in your shell history. Run them from the project folder while logged in to GitHub as the repo owner.

Secrets:

```bash
gh secret set OPENROUTER_API_KEY
```

```bash
gh secret set HF_TOKEN
```

```bash
gh secret set DATABASE_URL           # Neon pooled connection string
```

```bash
gh secret set DATABASE_URL_DIRECT    # Neon direct connection string (used only for table setup)
```

```bash
gh secret set PHOENIX_API_KEY
```

The operator token is the console password for the public URL. Generate a random one and keep a copy somewhere safe (a password manager), because you'll paste it into the console to sign in:

```bash
openssl rand -base64 24
```

```bash
gh secret set OPERATOR_TOKEN
```

Variables (not secret):

```bash
gh variable set HF_SPACE --body "arushigupta596/react-support-agent"
```

```bash
gh variable set PHOENIX_COLLECTOR_ENDPOINT --body "https://app.phoenix.arize.com/s/<your-space>"
```

Optional: `gh variable set AGENT_MODEL --body "openrouter/anthropic/claude-sonnet-5.5"` (this is already the default).

## 3. Deploy

GitHub → **Actions** → **Deploy to Hugging Face Space** → **Run workflow**. It takes about a minute, then Hugging Face builds the image (about 5 to 10 minutes the first time; watch the Space's **Logs** tab).

The app is at `https://<owner>-<space-name>.hf.space` (the workflow log prints the exact URL). It opens the console; sign in with your operator token. On first start the container creates its tables in Neon.

Run the workflow again whenever you want the Space to pick up new code.

## 4. Rehearse against the deployed URL

```bash
OPERATOR_TOKEN=<your token> uv run python scripts/rehearse.py --base https://<owner>-<space-name>.hf.space
```

All three scenarios should pass, as they do locally.

## Before a client demo

- **Wake everything about 5 minutes early.** Free Spaces sleep after about 48 hours without visits, and Neon scales to zero when idle: open the URL and send one message.
- **Have the fallback ready:** the local stack (`README.md`, "Live demo (local)"), or replay a finished session from its decision timeline.
- **Phoenix Cloud** is a single-developer account: show traces from your own screen, never share the login.

## What the deploy does (and doesn't) upload

`scripts/deploy_space.py` uploads only what the image needs: `Dockerfile`, `pyproject.toml`, `uv.lock`, `src/`, `scripts/migrate.py` and tau2-bench without its 576 MB of published leaderboard results, voice data or tests. It writes the Space's README header (Docker SDK, port 7860) and copies the secrets above into the Space's settings. It refuses to deploy without `OPERATOR_TOKEN`, so the public console is never open to everyone.

## Other workflows

- **CI** (`.github/workflows/ci.yml`, automatic on every push and pull request): lint, all tests against a Postgres service, and a build of the Docker image. No API calls, no cost.
- **Benchmark** (`.github/workflows/bench.yml`, manual): Actions → Benchmark → Run workflow, then choose domains, configs, split, task count and trials. Needs only the `OPENROUTER_API_KEY` secret. Writes a comparison table to the run's summary page and commits the results to `results/ci-<run id>/`. A 10-task smoke run on all four domains with `baseline graph-full` costs about $6.
