"""FastAPI service: customer chat, operator console and approvals, plus the MCP server at /mcp.

One request runs the graph to the next reply or interrupt; nothing waits in memory for an
approval. POST /approvals/{id} resumes from the Postgres checkpoint in a new request.
"""

import contextvars
import hmac
import json
import os
import queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from agent.llm import register_prices
from agent.nodes.common import is_customer, text_of
from api import approvals, timeline
from api.agents import run_config, serve_graph
from mcp_server import session_store
from mcp_server.domains import DOMAINS
from mcp_server.server import execute_user_action, given_user_actions
from mcp_server.server import server as mcp_server

UI = Path(__file__).parent / "ui"

# The MCP endpoint authenticates every call with MCP_TOKEN, so host checks are not needed.
mcp_app = mcp_server.streamable_http_app(
    streamable_http_path="/",
    stateless_http=True,
    json_response=True,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    register_prices()
    if os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        from observability.tracing import init_tracing

        init_tracing(project="serve", batch=False)  # export at once; hosts may freeze idle work
    async with mcp_server.session_manager.run():
        yield


app = FastAPI(title="ReAct Support Agent", lifespan=lifespan)
app.mount("/mcp", mcp_app)
app.mount("/ui", StaticFiles(directory=UI), name="ui")


def operator_auth_required() -> bool:
    """Sign-in is on only when OPERATOR_TOKEN is set (leave it unset for a local demo)."""
    return bool(os.getenv("OPERATOR_TOKEN"))


def operator(authorization: str = Header(default="")) -> str:
    if not operator_auth_required():
        return "operator"
    token = authorization.removeprefix("Bearer ").strip()
    if not token or not hmac.compare_digest(token, os.environ["OPERATOR_TOKEN"]):
        raise HTTPException(401, "operator token required")
    return "operator"


@app.get("/config")
def ui_config():
    return {"operator_auth": operator_auth_required()}


def load_session(session_id: str) -> dict:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(404, "unknown session")
    return session


def run_turn(session: dict, graph_input, emit) -> dict:
    """Run the session's graph to its next stop (a reply or a pending approval).

    Every timeline event is persisted and passed to `emit` as it happens; returns the outcome.
    """
    sid = session["session_id"]
    graph, ctx = serve_graph(session["domain"])
    config = run_config(sid, ctx)
    builder = timeline.TimelineBuilder(ctx)
    builder.pending = graph.get_state(config).values.get("pending_tool_call")  # resuming
    with _session_trace(sid):
        for chunk in graph.stream(graph_input, config, stream_mode="updates"):
            for node, update in chunk.items():
                if node != "__interrupt__":
                    for event in builder.from_update(node, update):
                        emit(timeline.persist(sid, event))
        snapshot = graph.get_state(config)
        if snapshot.interrupts:
            interrupt = snapshot.interrupts[0]
            row = approvals.record_pending(interrupt.id, session, interrupt.value)
            emit(timeline.persist(sid, builder.interrupt(interrupt.value)))
            return {"status": "awaiting_approval", "approval": approvals.to_json(row)}
        reply = text_of(snapshot.values["messages"][-1])
        emit(timeline.persist(sid, timeline.reply_event(reply)))
        return {"status": "replied", "reply": reply}


def advance(session: dict, graph_input) -> dict:
    events: list[dict] = []
    outcome = run_turn(session, graph_input, events.append)
    return {**outcome, "events": [_json(e) for e in events]}


def stream_turn(session: dict, graph_input) -> StreamingResponse:
    """Server-sent events: one `event` per timeline step, then `done` with the outcome.

    The turn runs on one worker thread (so trace context stays intact) and hands events to
    the response through a queue.
    """
    q: queue.Queue = queue.Queue()

    def work():
        try:
            q.put(("done", run_turn(session, graph_input, lambda e: q.put(("event", e)))))
        except Exception as e:  # noqa: BLE001  surfaced to the client as an error event
            q.put(("error", {"detail": str(e)}))

    threading.Thread(target=contextvars.copy_context().run, args=(work,), daemon=True).start()

    async def events():
        while True:
            kind, payload = await anyio.to_thread.run_sync(q.get)
            yield f"event: {kind}\ndata: {json.dumps(_json(payload))}\n\n"
            if kind != "event":
                return

    return StreamingResponse(events(), media_type="text/event-stream")


def _json(value):
    return json.loads(json.dumps(value, default=str))


def _session_trace(session_id: str):
    from openinference.instrumentation import using_session

    return using_session(session_id)  # Phoenix Sessions view groups the conversation


# ---------- operator ----------


class NewSession(BaseModel):
    domain: str
    customer_profile_id: str | None = None


@app.get("/domains")
def list_domains():
    return [
        {
            "key": k,
            "label": v["label"],
            "description": v["description"],
            "suggestions": v["suggestions"],
        }
        for k, v in DOMAINS.items()
    ]


@app.post("/sessions")
def create_session(body: NewSession, who: str = Depends(operator)):
    if body.domain not in DOMAINS:
        raise HTTPException(422, f"unknown domain {body.domain!r}; use one of {list(DOMAINS)}")
    session_id = session_store.create_session(body.domain, body.customer_profile_id)
    return {"session_id": session_id, "domain": body.domain, "chat_url": f"/chat/{session_id}"}


@app.get("/sessions")
def recent_sessions(who: str = Depends(operator)):
    return [approvals.to_json(r) for r in session_store.recent_sessions()]


@app.get("/approvals")
def list_approvals(
    status: str | None = "pending", domain: str | None = None, who: str = Depends(operator)
):
    return [approvals.to_json(r) for r in approvals.list_approvals(status, domain)]


class Decision(BaseModel):
    decision: str  # approve | edit | reject
    edited_args: dict | None = None
    reason: str | None = None


@app.post("/approvals/{approval_id}")
def decide(approval_id: str, body: Decision, who: str = Depends(operator)):
    row = approvals.get(approval_id)
    if row is None:
        raise HTTPException(404, "unknown approval")
    if body.decision not in ("approve", "edit", "reject"):
        raise HTTPException(422, "decision must be approve, edit or reject")
    if body.decision == "edit" and not body.edited_args:
        raise HTTPException(422, "edit needs edited_args")
    if body.decision == "reject" and not body.reason:
        raise HTTPException(422, "a rejection needs a reason for the agent and the audit log")
    status = {"approve": "approved", "edit": "edited", "reject": "rejected"}[body.decision]
    if not approvals.decide(approval_id, status, who, body.reason, body.edited_args):
        raise HTTPException(409, "already decided")
    resume = {
        "decision": body.decision,
        "edited_args": body.edited_args,
        "reviewer": who,
        "reason": body.reason,
    }
    return advance(load_session(row["session_id"]), Command(resume=resume))


# ---------- customer ----------


class CustomerMessage(BaseModel):
    content: str


@app.post("/sessions/{session_id}/messages")
def send_message(session_id: str, body: CustomerMessage, accept: str = Header(default="")):
    session = load_session(session_id)
    graph, ctx = serve_graph(session["domain"])
    if graph.get_state(run_config(session_id, ctx)).interrupts:
        raise HTTPException(409, "an action is waiting for operator approval")
    graph_input = {"messages": [HumanMessage(content=body.content)]}
    if "text/event-stream" in accept:
        return stream_turn(session, graph_input)
    return advance(session, graph_input)


@app.get("/sessions/{session_id}/events")
def session_events(session_id: str, after: int = 0):
    """Decision timeline for live polling and replay, each event with its Phoenix trace link."""
    load_session(session_id)
    events = timeline.list_events(session_id, after)
    for e in events:
        e["trace_url"] = timeline.trace_url(e["trace_id"]) if e["trace_id"] else None
    return events


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = load_session(session_id)
    graph, ctx = serve_graph(session["domain"])
    snapshot = graph.get_state(run_config(session_id, ctx))
    transcript = [
        {"role": "customer" if is_customer(m) else "agent", "text": text_of(m)}
        for m in snapshot.values.get("messages", [])
        if (is_customer(m) or isinstance(m, AIMessage)) and text_of(m).strip()
    ]
    return {
        "session_id": session_id,
        "domain": session["domain"],
        "awaiting_approval": bool(snapshot.interrupts),
        "transcript": transcript,
        "user_actions": given_user_actions(session),
    }


class UserAction(BaseModel):
    tool: str
    arguments: dict | None = None  # defaults to the arguments the agent handed over


@app.post("/sessions/{session_id}/user-actions")
def run_user_action(session_id: str, body: UserAction):
    """The customer runs an action the agent handed to them; the result is their next message."""
    session = load_session(session_id)
    given = {a["tool"]: a for a in given_user_actions(session)}
    if body.tool not in given:
        raise HTTPException(404, "the agent has not given you this action")
    graph, ctx = serve_graph(session["domain"])
    if graph.get_state(run_config(session_id, ctx)).interrupts:
        raise HTTPException(409, "an action is waiting for operator approval")
    arguments = body.arguments if body.arguments is not None else given[body.tool]["arguments"]
    content, is_error = execute_user_action(session, body.tool, arguments)
    note = "It failed" if is_error else "Result"
    message = f"I ran {body.tool} myself. {note}: {content}"
    return advance(session, {"messages": [HumanMessage(content=message)]})


# ---------- pages ----------


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/console")
def console_page():
    return FileResponse(UI / "console.html")


@app.get("/chat/{session_id}")
def chat_page(session_id: str):
    load_session(session_id)
    return FileResponse(UI / "chat.html")
