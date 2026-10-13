"""Write Cloud Run's environment (YAML) from .env, for scripts/deploy_cloudrun.sh.

Only the settings the live app needs are copied. OPERATOR_TOKEN is required for a public URL;
it is read from the shell first (so .env can keep local mode), then from .env.
"""

import json
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "OPENROUTER_API_KEY",
    "DATABASE_URL",
    "DATABASE_URL_DIRECT",
    "MCP_TOKEN",
    "OPERATOR_TOKEN",
]
OPTIONAL = ["PHOENIX_COLLECTOR_ENDPOINT", "PHOENIX_API_KEY"]
DEFAULTS = {
    "AGENT_MODEL": "openrouter/anthropic/claude-sonnet-5.5",
    "AGENT_PROVIDER": "anthropic",
    "AGENT_EFFORT": "medium",
}


def main() -> None:
    env = {**dotenv_values(ROOT / ".env"), **{k: v for k, v in os.environ.items() if v}}
    if "localhost" in (env.get("DATABASE_URL") or "") or "127.0.0.1" in (
        env.get("DATABASE_URL") or ""
    ):
        sys.exit("DATABASE_URL points at this Mac; Cloud Run needs the hosted (Supabase) URL.")
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing:
        sys.exit(
            f"Missing: {', '.join(missing)}. Set them in .env (OPERATOR_TOKEN can be exported in the shell)."
        )
    out = {k: env[k] for k in REQUIRED}
    out |= {k: env[k] for k in OPTIONAL if env.get(k)}
    out |= {k: env.get(k) or v for k, v in DEFAULTS.items()}
    if env.get("PHOENIX_COLLECTOR_ENDPOINT"):
        out["PHOENIX_UI_URL"] = env["PHOENIX_COLLECTOR_ENDPOINT"]
    # JSON is valid YAML, and quoting every value avoids YAML type surprises.
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
