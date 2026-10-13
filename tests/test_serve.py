"""Serve mode end to end: real API + MCP server + Postgres + tau2 retail data, scripted LLM.

Runs against TEST_DATABASE_URL (default: the local Postgres database react_agent) with
scripts/migrate.py applied; skipped otherwise. Never the DATABASE_URL in .env, which may point
at the hosted demo's database.
"""

import json
import os
import socket
import threading
import time

import httpx
import psycopg
import pytest
import uvicorn
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from tests.fakes import ScriptedChatModel, tool_call

load_dotenv()
os.environ["DATABASE_URL"] = os.getenv(
    "TEST_DATABASE_URL", "postgresql://localhost:5432/react_agent"
)


def _db_ready() -> bool:
    try:
        with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=2) as c:
            return c.execute("SELECT to_regclass('approvals')").fetchone()[0] is not None
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _db_ready(), reason="local Postgres not migrated")

MODEL = ScriptedChatModel(script=[], calls=[])  # each test sets MODEL.script before chatting


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def api():
    port = _free_port()
    os.environ["MCP_URL"] = f"http://127.0.0.1:{port}/mcp/"
    os.environ.pop("PHOENIX_COLLECTOR_ENDPOINT", None)

    from api import agents
    from api.main import app

    agents.chat_model.cache_clear()
    agents.serve_graph.cache_clear()
    agents.chat_model = lambda: ("scripted", MODEL)

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
        yield client
    server.should_exit = True
    thread.join(timeout=5)


# Sign-in tests need a token; local mode (no token) is tested separately.
os.environ["OPERATOR_TOKEN"] = "test-operator-token"
OP = {"Authorization": "Bearer test-operator-token"}
ORDER = "#W2974929"  # pending, Fatima Anderson, zip 32100


def _script_cancel_flow(final: str = "Done. Your order is cancelled."):
    MODEL.script = [
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    "find_user_id_by_name_zip",
                    {"first_name": "Fatima", "last_name": "Anderson", "zip": "32100"},
                    "t1",
                )
            ],
        ),
        AIMessage(
            content="", tool_calls=[tool_call("get_order_details", {"order_id": ORDER}, "t2")]
        ),
        AIMessage(content=f"Order {ORDER} is pending. Cancel it because it is no longer needed?"),
        AIMessage(
            content="",
            tool_calls=[
                tool_call(
                    "cancel_pending_order", {"order_id": ORDER, "reason": "no longer needed"}, "t3"
                )
            ],
        ),
        AIMessage(content=final),
    ]


def _start(api, domain="retail") -> str:
    r = api.post("/sessions", json={"domain": domain}, headers=OP)
    assert r.status_code == 200, r.text
    return r.json()["session_id"]


def _pending_approval(api, sid):
    r1 = api.post(
        f"/sessions/{sid}/messages",
        json={"content": "I'm Fatima Anderson, 32100. Cancel my order."},
    )
    assert r1.json()["status"] == "replied"
    r2 = api.post(f"/sessions/{sid}/messages", json={"content": "Yes, please cancel it."})
    body = r2.json()
    assert body["status"] == "awaiting_approval", body
    return body["approval"]


def _order_status(sid) -> str:
    from mcp_server.domains import ENVIRONMENTS
    from mcp_server.server import execute
    from mcp_server.session_store import get_session

    ENVIRONMENTS._envs.pop(sid, None)  # force a rebuild from the Postgres action log
    content, _ = execute(get_session(sid), "retail.get_order_details", {"order_id": ORDER})
    return json.loads(content)["status"]


def test_session_creation_needs_token_and_known_domain(api):
    assert api.post("/sessions", json={"domain": "retail"}).status_code == 401
    assert api.post("/sessions", json={"domain": "telecom"}, headers=OP).status_code == 422


def test_write_waits_for_operator_then_runs_on_approve(api):
    _script_cancel_flow()
    sid = _start(api)
    approval = _pending_approval(api, sid)

    assert approval["tool"] == "cancel_pending_order"
    assert approval["request"]["customer_confirmation"] == "Yes, please cancel it."
    checks = {c["name"]: c["verdict"] for c in approval["request"]["policy_checks"]}
    assert checks["confirmation"] == "pass" and checks["order_owner"] == "pass"
    assert _order_status(sid) == "pending"  # nothing ran before sign-off
    assert api.post(f"/sessions/{sid}/messages", json={"content": "hello?"}).status_code == 409

    r = api.post(f"/approvals/{approval['id']}", json={"decision": "approve"}, headers=OP).json()
    assert (r["status"], r["reply"]) == ("replied", "Done. Your order is cancelled.")
    assert [e["type"] for e in r["events"]] == ["signoff", "act", "reply"]
    assert r["events"][0]["verdict"] == "approved"

    timeline = api.get(f"/sessions/{sid}/events").json()  # the whole session, for replay
    types = [e["type"] for e in timeline]
    assert types[:3] == ["think", "lookup", "think"]
    assert "check" in types and "propose" in types and types[-1] == "reply"
    assert [e["seq"] for e in timeline] == list(range(1, len(timeline) + 1))
    assert api.get(f"/sessions/{sid}/events?after={len(timeline) - 1}").json()[0]["type"] == "reply"
    assert _order_status(sid) == "cancelled"  # rebuilt from the action log
    again = api.post(f"/approvals/{approval['id']}", json={"decision": "approve"}, headers=OP)
    assert again.status_code == 409  # a double click cannot run it twice


def test_reject_does_not_run_and_needs_reason(api):
    _script_cancel_flow(final="Sorry, I can't cancel that order right now.")
    sid = _start(api)
    approval = _pending_approval(api, sid)

    no_reason = api.post(f"/approvals/{approval['id']}", json={"decision": "reject"}, headers=OP)
    assert no_reason.status_code == 422
    r = api.post(
        f"/approvals/{approval['id']}",
        json={"decision": "reject", "reason": "Customer is disputing a charge on this order."},
        headers=OP,
    )
    assert r.json()["reply"] == "Sorry, I can't cancel that order right now."
    assert _order_status(sid) == "pending"


def test_edit_runs_with_operator_args(api):
    _script_cancel_flow()
    sid = _start(api)
    approval = _pending_approval(api, sid)
    edited = {"order_id": ORDER, "reason": "ordered by mistake"}
    api.post(
        f"/approvals/{approval['id']}", json={"decision": "edit", "edited_args": edited}, headers=OP
    )

    from mcp_server.session_store import list_actions

    assert list_actions(sid) == [
        {"tool": "cancel_pending_order", "args": edited, "requestor": "assistant"}
    ]
    row = api.get("/approvals?status=edited", headers=OP).json()
    assert any(a["id"] == approval["id"] and a["reviewer"] == "operator" for a in row)


def test_mcp_lists_exact_tau2_schemas_and_refuses_other_domains(api):
    from mcp_server.domains import fresh_environment
    from mcp_server.server import execute, list_domain_tools
    from mcp_server.session_store import get_session

    expected = {
        f"retail.{t.name}": t.openai_schema["function"]["parameters"]
        for t in fresh_environment("retail").get_tools()
    }
    assert {t.name: t.input_schema for t in list_domain_tools("retail")} == expected

    sid = _start(api)
    content, is_error = execute(
        get_session(sid), "airline.get_reservation_details", {"reservation_id": "X"}
    )
    assert is_error and "outside this session's domain" in content


def test_banking_customer_runs_a_handed_over_action(api):
    from mcp_server.domains import ENVIRONMENTS
    from mcp_server.session_store import list_actions

    uid, account = "lj82d4f1a9", "cc_lj82d4f1a9_silver"
    verification = {
        "name": "Liang Jinhai",
        "user_id": uid,
        "address": "1420 Lake Shore Drive, Chicago, IL 60610",
        "email": "jinhai.liang@techventures.com",
        "phone_number": "312-555-0847",
        "date_of_birth": "04/12/2002",
        "time_verified": "2025-01-01 10:00:00",
    }
    give = {
        "discoverable_tool_name": "get_card_last_4_digits",
        "arguments": json.dumps({"credit_card_account_id": account}),
    }
    MODEL.script = [
        AIMessage(
            content="", tool_calls=[tool_call("get_user_information_by_id", {"user_id": uid}, "b1")]
        ),
        AIMessage(content="", tool_calls=[tool_call("log_verification", verification, "b2")]),
        AIMessage(content="", tool_calls=[tool_call("give_discoverable_user_tool", give, "b3")]),
        AIMessage(content="I've given you a button to see your card's last 4 digits."),
        AIMessage(content="Great, that's your card."),
    ]
    sid = _start(api, "banking_knowledge")
    r = api.post(
        f"/sessions/{sid}/messages",
        json={"content": "I'm Liang, DOB 04/12/2002, email jinhai.liang@techventures.com"},
    )
    assert r.json()["status"] == "replied"  # log_verification is exempt from operator sign-off

    actions = api.get(f"/sessions/{sid}").json()["user_actions"]
    assert [a["tool"] for a in actions] == ["get_card_last_4_digits"]
    r = api.post(f"/sessions/{sid}/user-actions", json={"tool": "get_card_last_4_digits"})
    assert r.json()["reply"] == "Great, that's your card."
    assert "I ran get_card_last_4_digits myself" in MODEL.calls[-1]["messages"][-1].content

    assert [a["requestor"] for a in list_actions(sid)] == ["assistant", "assistant", "user"]
    ENVIRONMENTS._envs.pop(sid, None)  # rebuilt from the log: the hand-over survives
    assert [a["tool"] for a in api.get(f"/sessions/{sid}").json()["user_actions"]] == [
        "get_card_last_4_digits"
    ]
    assert (
        api.post(f"/sessions/{sid}/user-actions", json={"tool": "deposit_check_3847"}).status_code
        == 404
    )


def test_turn_streams_timeline_events_then_done(api):
    _script_cancel_flow()
    sid = _start(api)
    body = {"content": "I'm Fatima Anderson, 32100. Cancel my order."}
    with api.stream(
        "POST", f"/sessions/{sid}/messages", json=body, headers={"Accept": "text/event-stream"}
    ) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        frames = [f for f in r.read().decode().split("\n\n") if f.strip()]
    kinds = [f.split("\n")[0].removeprefix("event: ") for f in frames]
    payloads = [json.loads(f.split("\n")[1].removeprefix("data: ")) for f in frames]
    assert kinds[-1] == "done" and payloads[-1]["status"] == "replied"
    assert [p["type"] for p in payloads[:-1]] == ["think", "lookup", "think", "lookup", "reply"]


def test_local_mode_without_operator_token(api, monkeypatch):
    assert api.get("/config").json() == {"operator_auth": True}
    monkeypatch.delenv("OPERATOR_TOKEN")
    assert api.get("/config").json() == {"operator_auth": False}
    r = api.post("/sessions", json={"domain": "retail"})  # no Authorization header
    assert r.status_code == 200
    assert api.get("/approvals").status_code == 200
