"""Decision timeline: plain-language events from LangGraph `stream_mode="updates"` (spec 4.7).

Business viewers read the timeline; technical viewers click through to the Phoenix trace.
Events are stored in decision_events so any session can be replayed in a meeting.
"""

import json
import os
import re
import time

import httpx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from opentelemetry import trace
from psycopg.types.json import Jsonb

from agent.context import GraphContext
from agent.nodes.common import text_of
from agent.nodes.guard_output import FEEDBACK_NAME
from mcp_server.session_store import pool

TRANSFER = "transfer_to_human_agents"


def _plain(text: str) -> str:
    """Markdown to one plain line for the timeline."""
    text = re.sub(r"\*\*|__|`", "", str(text))
    return re.sub(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+", r"\1", text)


def _short(text: str, limit: int = 220) -> str:
    text = " ".join(_plain(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fact(content: str) -> str:
    """The one-line gist of a tool result for the timeline."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return _short(content, 160)
    if isinstance(data, dict):
        keys = [
            k
            for k in (
                "status",
                "order_id",
                "reservation_id",
                "user_id",
                "cabin",
                "name",
                "full_name",
            )
            if k in data
        ]
        if keys:
            return ", ".join(
                f"{k}: {data[k] if not isinstance(data[k], dict) else '…'}" for k in keys
            )
        return _short(", ".join(list(data)[:6]), 160)
    if isinstance(data, list):
        return f"{len(data)} results"
    return _short(str(data), 160)


class TimelineBuilder:
    """Turns one graph run's updates into timeline events."""

    def __init__(self, ctx: GraphContext):
        self.ctx = ctx
        self.pending: dict | None = None

    def from_update(self, node: str, update: dict | None) -> list[dict]:
        update = update or {}
        msgs = update.get("messages", [])
        if node == "guard_input" and msgs:
            return [
                _event(
                    "guard",
                    node,
                    "Input guard",
                    "Removed text that tried to change the agent's instructions.",
                    "block",
                )
            ]
        if node == "reason":
            ai = next((m for m in msgs if isinstance(m, AIMessage)), None)
            if ai is not None and ai.tool_calls:
                call = ai.tool_calls[0]
                args = ", ".join(f"{k}: {v}" for k, v in call["args"].items())
                detail = _short(update.get("decision_note") or args, 160)
                return [
                    _event(
                        "think",
                        node,
                        f"Decides to use {call['name']}",
                        detail,
                        None,
                        {"args": call["args"]},
                    )
                ]
            return []
        if node == "guard_pre_tool":
            results = update.get("guard_results", [])
            blocked = [r for r in results if r["verdict"] == "block"]
            if update.get("pending_tool_call"):
                self.pending = update["pending_tool_call"]
            name = (self.pending or {}).get("name", "tool")
            if blocked:
                return [
                    _event(
                        "check",
                        node,
                        f"Policy check blocked {name}",
                        blocked[0]["reason"],
                        "block",
                        {"checks": results},
                    )
                ]
            if self.pending and self.ctx.is_write(self.pending):
                passed = ", ".join(r["name"] for r in results)
                return [
                    _event(
                        "check",
                        node,
                        f"Policy checks passed for {name}",
                        passed,
                        "pass",
                        {"checks": results},
                    )
                ]
            return []
        if node == "approval_gate":
            approval = update.get("approval")
            if approval and approval.get("reviewer") not in (None, "auto_approve", "rule_check"):
                verdict = {"approve": "approved", "edit": "edited", "reject": "rejected"}[
                    approval["decision"]
                ]
                detail = approval.get("reason") or (
                    f"new arguments {approval.get('edited_args')}" if verdict == "edited" else ""
                )
                return [
                    _event(
                        "signoff", node, f"Operator {verdict} {approval['tool']}", detail, verdict
                    )
                ]
            return []
        if node == "execute_tools":
            tm = next((m for m in msgs if isinstance(m, ToolMessage)), None)
            if tm is None:
                return []
            call = self.pending or {"name": tm.name, "args": {}}
            failed = tm.status == "error" or text_of(tm).startswith("Error")
            if call["name"] == TRANSFER:
                return [
                    _event(
                        "escalate",
                        node,
                        "Transferred to a human agent",
                        call["args"].get("summary", ""),
                        "escalated",
                    )
                ]
            if self.ctx.is_write(call):
                title = f"{'Failed' if failed else 'Done'}: {call['name']}"
                return [
                    _event(
                        "act",
                        node,
                        title,
                        _fact(text_of(tm)),
                        "error" if failed else "done",
                        {"args": call["args"]},
                    )
                ]
            return [
                _event(
                    "lookup",
                    node,
                    f"Looked up {call['name']}",
                    _fact(text_of(tm)),
                    "error" if failed else None,
                )
            ]
        if node == "guard_output":
            if any(isinstance(m, HumanMessage) and m.name == FEEDBACK_NAME for m in msgs):
                return [
                    _event(
                        "guard",
                        node,
                        "Output guard",
                        "Stopped a reply claiming an action that never ran; the agent rewrote it.",
                        "block",
                    )
                ]
            return []
        return []

    def interrupt(self, request: dict) -> dict:
        return _event(
            "propose",
            "approval_gate",
            f"Waiting for operator sign-off: {request['tool']}",
            request.get("summary") or "",
            "waiting",
            {"args": request["args"]},
        )


def reply_event(text: str) -> dict:
    return _event("reply", "guard_output", "Replied to the customer", _short(text), None)


def _event(
    type_: str, node: str, title: str, detail: str, verdict: str | None, extra: dict | None = None
) -> dict:
    ctx = trace.get_current_span().get_span_context()
    return {
        "type": type_,
        "node": node,
        "title": title,
        "detail": detail,
        "verdict": verdict,
        "extra": extra or {},
        "trace_id": format(ctx.trace_id, "032x") if ctx.is_valid else None,
    }


def persist(session_id: str, event: dict) -> dict:
    with pool().connection() as conn:
        row = conn.execute(
            """INSERT INTO decision_events (session_id, seq, type, node, title, detail, verdict, trace_id)
               SELECT %s, COALESCE(MAX(seq), 0) + 1, %s, %s, %s, %s, %s, %s
               FROM decision_events WHERE session_id = %s
               RETURNING seq, created_at""",
            (
                session_id,
                event["type"],
                event["node"],
                event["title"],
                Jsonb({"text": event["detail"], **event["extra"]}),
                event["verdict"],
                event["trace_id"],
                session_id,
            ),
        ).fetchone()
    return {**event, "seq": row["seq"], "ts": row["created_at"].isoformat()}


def list_events(session_id: str, after: int = 0) -> list[dict]:
    with pool().connection() as conn:
        rows = conn.execute(
            """SELECT seq, type, node, title, detail, verdict, trace_id, created_at
               FROM decision_events WHERE session_id = %s AND seq > %s ORDER BY seq""",
            (session_id, after),
        ).fetchall()
    return [
        {
            "seq": r["seq"],
            "type": r["type"],
            "node": r["node"],
            "title": r["title"],
            "detail": (r["detail"] or {}).get("text", ""),
            "extra": {k: v for k, v in (r["detail"] or {}).items() if k != "text"},
            "verdict": r["verdict"],
            "trace_id": r["trace_id"],
            "ts": r["created_at"].isoformat(),
        }
        for r in rows
    ]


def phoenix_ui() -> str | None:
    """Base URL for links the viewer's browser opens."""
    return os.getenv("PHOENIX_UI_URL") or os.getenv("PHOENIX_COLLECTOR_ENDPOINT")


_project = {"id": None, "checked_at": 0.0}
RETRY_AFTER_S = 60


def phoenix_project_id(name: str = "serve") -> str | None:
    """Phoenix's id for the serve project, for trace deep links.

    A miss (Phoenix down, or no trace exported yet) is remembered for a minute, so an
    unreachable Phoenix costs one short timeout a minute instead of one per timeline event.
    """
    if _project["id"] or time.monotonic() - _project["checked_at"] < RETRY_AFTER_S:
        return _project["id"]
    _project["checked_at"] = time.monotonic()
    # Looked up from inside the deployment, which may reach Phoenix at a different address.
    base = os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or phoenix_ui()
    if not base:
        return None
    headers = {"authorization": f"Bearer {k}"} if (k := os.getenv("PHOENIX_API_KEY")) else {}
    try:
        resp = httpx.get(f"{base.rstrip('/')}/v1/projects", headers=headers, timeout=3)
        projects = resp.json()["data"]
    except Exception:  # noqa: BLE001  the timeline still works without the deep link
        return None
    _project["id"] = next((p["id"] for p in projects if p["name"] == name), None)
    return _project["id"]


def trace_url(trace_id: str) -> str | None:
    project = phoenix_project_id()
    base = phoenix_ui()
    if not (project and base):
        return None
    return f"{base.rstrip('/')}/projects/{project}/traces/{trace_id}"
