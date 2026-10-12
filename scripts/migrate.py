"""One-off schema setup: LangGraph checkpoint tables + app tables. Never run per request.

uv run python scripts/migrate.py            # uses DATABASE_URL_DIRECT or DATABASE_URL
"""

import os
import sys
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

APP_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    domain              TEXT NOT NULL,
    customer_profile_id TEXT,
    status              TEXT NOT NULL DEFAULT 'active',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Successful WRITE calls per session. The session's domain DB is the stock tau2 DB with
-- these replayed in order, so storage is bytes per session instead of a 3-7 MB DB copy.
CREATE TABLE IF NOT EXISTS session_actions (
    session_id  TEXT NOT NULL REFERENCES sessions(session_id),
    seq         INTEGER NOT NULL,
    tool        TEXT NOT NULL,
    args        JSONB NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (session_id, seq)
);

-- 'user' rows are actions the customer ran themselves (banking tools the agent handed over).
ALTER TABLE session_actions ADD COLUMN IF NOT EXISTS requestor TEXT NOT NULL DEFAULT 'assistant';

CREATE TABLE IF NOT EXISTS approvals (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL REFERENCES sessions(session_id),
    thread_id    TEXT NOT NULL,
    domain       TEXT NOT NULL,
    tool         TEXT NOT NULL,
    args         JSONB NOT NULL,
    request      JSONB NOT NULL,          -- what the operator saw: summary, checks, confirmation
    edited_args  JSONB,
    status       TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | edited | rejected
    reviewer     TEXT,
    reason       TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    decided_at   TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS approvals_status ON approvals(status, created_at);

CREATE TABLE IF NOT EXISTS decision_events (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id),
    seq         INTEGER NOT NULL,
    type        TEXT NOT NULL,
    node        TEXT,
    title       TEXT NOT NULL,
    detail      JSONB,
    verdict     TEXT,
    trace_id    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, seq)
);
"""


def main() -> None:
    url = os.getenv("DATABASE_URL_DIRECT") or os.getenv("DATABASE_URL")
    if not url:
        sys.exit("Set DATABASE_URL (or DATABASE_URL_DIRECT) first.")
    with psycopg.connect(url, autocommit=True, prepare_threshold=None) as conn:
        PostgresSaver(conn).setup()
        conn.execute(APP_SCHEMA)
    print("Schema ready.")


if __name__ == "__main__":
    main()
