"""Session rows and the per-session log of successful WRITE calls, in Postgres."""

import json
import os
import uuid
from functools import lru_cache

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


@lru_cache(maxsize=1)
def pool() -> ConnectionPool:
    # prepare_threshold=None turns prepared statements off, which poolers (Neon, Supabase)
    # in transaction mode require. (0 would mean "prepare everything".)
    return ConnectionPool(
        os.environ["DATABASE_URL"],
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": None, "row_factory": dict_row},
        open=True,
    )


def create_session(domain: str, customer_profile_id: str | None = None) -> str:
    session_id = uuid.uuid4().hex
    with pool().connection() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, domain, customer_profile_id) VALUES (%s, %s, %s)",
            (session_id, domain, customer_profile_id),
        )
    return session_id


def get_session(session_id: str) -> dict | None:
    with pool().connection() as conn:
        return conn.execute(
            "SELECT * FROM sessions WHERE session_id = %s", (session_id,)
        ).fetchone()


def list_actions(session_id: str) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            "SELECT tool, args, requestor FROM session_actions WHERE session_id = %s ORDER BY seq",
            (session_id,),
        ).fetchall()


def append_action(session_id: str, tool: str, args: dict, requestor: str = "assistant") -> None:
    with pool().connection() as conn:
        conn.execute(
            """INSERT INTO session_actions (session_id, seq, tool, args, requestor)
               SELECT %s, COALESCE(MAX(seq), 0) + 1, %s, %s, %s FROM session_actions
               WHERE session_id = %s""",
            (session_id, tool, json.dumps(args), requestor, session_id),
        )


def recent_sessions(limit: int = 20) -> list[dict]:
    with pool().connection() as conn:
        return conn.execute(
            """SELECT s.session_id, s.domain, s.created_at,
                      (SELECT count(*) FROM approvals a
                       WHERE a.session_id = s.session_id AND a.status = 'pending') AS pending,
                      (SELECT max(created_at) FROM decision_events e
                       WHERE e.session_id = s.session_id) AS last_activity
               FROM sessions s ORDER BY COALESCE(
                   (SELECT max(created_at) FROM decision_events e WHERE e.session_id = s.session_id),
                   s.created_at) DESC
               LIMIT %s""",
            (limit,),
        ).fetchall()
