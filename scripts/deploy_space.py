"""Deploy the serve app to a Hugging Face Space (Docker SDK). Run by .github/workflows/deploy-space.yml.

GitHub is the source of truth: this uploads only what the image needs (tau2 comes from the
checked-out submodule, which Spaces cannot fetch themselves), writes the Space's README header,
and copies runtime secrets from the environment into the Space. Hugging Face then builds the
Dockerfile.

    HF_TOKEN=... HF_SPACE=user/space-name OPERATOR_TOKEN=... python scripts/deploy_space.py
"""

import os
import secrets
import sys
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]

UPLOAD = [
    "Dockerfile",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    "src/**",
    "scripts/migrate.py",
    "vendor/tau2-bench/**",
]
SKIP = [
    "**/__pycache__/**",
    "**/.DS_Store",
    "vendor/tau2-bench/.git*",
    "vendor/tau2-bench/data/tau2/results/**",  # 576 MB of leaderboard results the app never reads
    "vendor/tau2-bench/data/simulations/**",
    "vendor/tau2-bench/data/voice/**",
    "vendor/tau2-bench/web/**",
    "vendor/tau2-bench/tests/**",
]
SECRETS = [
    "OPENROUTER_API_KEY",
    "DATABASE_URL",
    "DATABASE_URL_DIRECT",
    "OPERATOR_TOKEN",
    "MCP_TOKEN",
    "PHOENIX_API_KEY",
]
VARIABLES = {
    "AGENT_MODEL": "openrouter/anthropic/claude-sonnet-5.5",
    "AGENT_PROVIDER": "anthropic",
    "AGENT_EFFORT": "medium",
    "PHOENIX_COLLECTOR_ENDPOINT": None,
}

SPACE_README = """---
title: ReAct Support Agent
emoji: 🛎️
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Guarded customer-support agent with human sign-off
---

# ReAct Support Agent

A LangGraph customer-support agent that reasons, calls one tool at a time, passes policy
guardrails, and waits for an operator's sign-off before any action that changes customer data.
Every decision is shown in a live timeline with a link to its trace.

- Operator console: `/console` (sign-in required)
- Source, evaluations and debug journal: {repo}
"""


def upload_code(api: HfApi, space: str, repo: str, sha: str) -> None:
    api.upload_file(
        path_or_fileobj=SPACE_README.format(repo=repo).encode(),
        path_in_repo="README.md",
        repo_id=space,
        repo_type="space",
        commit_message=f"Space README ({sha})",
    )
    api.upload_folder(
        folder_path=ROOT,
        repo_id=space,
        repo_type="space",
        allow_patterns=UPLOAD,
        ignore_patterns=SKIP,
        delete_patterns=[
            "src/**",
            "scripts/**",
            "vendor/**",
        ],  # drop files removed since last deploy
        commit_message=f"Deploy {sha}",
    )
    print(f"Deployed {sha} to https://huggingface.co/spaces/{space}")
    print(f"App URL: https://{space.replace('/', '-').replace('_', '-').lower()}.hf.space/console")


def main() -> None:
    space = os.environ.get("HF_SPACE")
    if not os.environ.get("HF_TOKEN") or not space:
        sys.exit("Set HF_TOKEN and HF_SPACE (for example 'your-name/react-support-agent').")
    repo = os.environ.get(
        "GITHUB_REPO_URL", "https://github.com/arushigupta596/ReAct-Customer-Support-Agent"
    )
    sha = os.environ.get("GITHUB_SHA", "local")[:7]

    if "--code-only" in sys.argv:
        # Space created and its secrets set on the Hugging Face website: only push the code.
        upload_code(HfApi(), space, repo, sha)
        return

    if not os.environ.get("OPERATOR_TOKEN"):
        sys.exit(
            "Refusing to deploy a public Space without OPERATOR_TOKEN: the console would be open to anyone."
        )
    missing = [k for k in ("OPENROUTER_API_KEY", "DATABASE_URL") if not os.environ.get(k)]
    if missing:
        sys.exit(f"Missing required secrets: {', '.join(missing)}")

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)

    env = dict(os.environ)
    env.setdefault("MCP_TOKEN", secrets.token_urlsafe(24))  # internal API <-> MCP secret
    for key in SECRETS:
        if env.get(key):
            api.add_space_secret(space, key, env[key])
    for key, default in VARIABLES.items():
        value = env.get(key) or default
        if value:
            api.add_space_variable(space, key, value)
    if env.get("PHOENIX_COLLECTOR_ENDPOINT"):
        api.add_space_variable(space, "PHOENIX_UI_URL", env["PHOENIX_COLLECTOR_ENDPOINT"])

    upload_code(api, space, repo, sha)


if __name__ == "__main__":
    main()
