"""Approvals: one row per WRITE the graph paused on, and the operator's decision."""

import json

from psycopg.types.json import Jsonb

from mcp_server.session_store import pool


def record_pending(approval_id: str, session: dict, request: dict) -> dict:
    with pool().connection() as conn:
        conn.execute(
            """INSERT INTO approvals (id, session_id, thread_id, domain, tool, args, request)
               VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (id) DO NOTHING""",
            (
                approval_id,
                session["session_id"],
                session["session_id"],
                session["domain"],
                request["tool"],
                Jsonb(request["args"]),
                Jsonb(request),
            ),
        )
    return get(approval_id)


def get(approval_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute("SELECT * FROM approvals WHERE id = %s", (approval_id,)).fetchone()


def list_approvals(status: str | None, domain: str | None) -> list[dict]:
    query, params = "SELECT * FROM approvals WHERE TRUE", []
    if status:
        query += " AND status = %s"
        params.append(status)
    if domain:
        query += " AND domain = %s"
        params.append(domain)
    with pool().connection() as conn:
        return conn.execute(query + " ORDER BY created_at", params).fetchall()


def decide(
    approval_id: str, status: str, reviewer: str, reason: str | None, edited_args: dict | None
) -> bool:
    """Record the decision. Returns False if it was already decided (double click, two operators)."""
    with pool().connection() as conn:
        row = conn.execute(
            """UPDATE approvals SET status = %s, reviewer = %s, reason = %s, edited_args = %s,
                      decided_at = now()
               WHERE id = %s AND status = 'pending' RETURNING id""",
            (status, reviewer, reason, Jsonb(edited_args) if edited_args else None, approval_id),
        ).fetchone()
    return row is not None


def to_json(row: dict) -> dict:
    return json.loads(json.dumps(row, default=str))
