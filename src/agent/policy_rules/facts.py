"""What the conversation has established so far, read from tool calls and their results.

Policy rules decide only from facts a tool actually returned, never from what the model
claims, so a rule can't be talked past.
"""

import json
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage

from agent.nodes.common import text_of

AUTH_TOOLS = ("find_user_id_by_email", "find_user_id_by_name_zip")


@dataclass
class Facts:
    authenticated_user: str | None = None
    orders: dict[str, dict] = field(default_factory=dict)  # order_id -> latest details
    succeeded: list[tuple[str, dict]] = field(default_factory=list)  # (tool, args) that ran OK
    succeeded_writes: list[tuple[str, dict]] = field(default_factory=list)
    # airline
    users: dict[str, dict] = field(default_factory=dict)
    reservations: dict[str, dict] = field(default_factory=dict)
    flight_status: dict[tuple[str, str], str] = field(default_factory=dict)
    # banking
    verified: bool = False
    # telecom
    customers: dict[str, dict] = field(default_factory=dict)
    bills: dict[str, dict] = field(default_factory=dict)
    lines: dict[str, dict] = field(default_factory=dict)


def _parse(content: str):
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return content


def collect(messages: list[AnyMessage], write_tools: frozenset[str] = frozenset()) -> Facts:
    calls = {tc["id"]: tc for m in messages if isinstance(m, AIMessage) for tc in m.tool_calls}
    facts = Facts()
    for m in messages:
        if not isinstance(m, ToolMessage) or m.status == "error":
            continue
        call = calls.get(m.tool_call_id)
        if call is None:
            continue
        content = text_of(m)
        if content.startswith(("Error", "BLOCKED", "REJECTED")):
            continue  # tau2 reports tool failures as text
        result = _parse(content)
        facts.succeeded.append((call["name"], call["args"]))
        if call["name"] in write_tools:
            facts.succeeded_writes.append((call["name"], call["args"]))
        if call["name"] in AUTH_TOOLS and isinstance(result, str):
            facts.authenticated_user = result.strip().strip('"')
        elif isinstance(result, dict) and "order_id" in result and "status" in result:
            # get_order_details, and every retail write (they return the updated order)
            facts.orders[result["order_id"]] = result
        elif call["name"] == "get_user_details" and isinstance(result, dict):
            facts.users[result.get("user_id", call["args"].get("user_id"))] = result
        elif isinstance(result, dict) and "reservation_id" in result and "flights" in result:
            # get_reservation_details, and airline writes (they return the updated reservation),
            # so an upgrade followed by a cancel is judged on the upgraded cabin
            facts.reservations[result["reservation_id"]] = result
        elif call["name"] == "get_flight_status":
            key = (call["args"].get("flight_number"), call["args"].get("date"))
            facts.flight_status[key] = str(result).strip('"').lower()
        elif call["name"] == "log_verification":
            facts.verified = True
        elif call["name"].startswith("get_customer_by") and isinstance(result, dict):
            facts.customers[result.get("customer_id")] = result
        elif call["name"] in ("get_bills_for_customer", "get_details_by_id"):
            for item in result if isinstance(result, list) else [result]:
                if isinstance(item, dict) and "bill_id" in item:
                    facts.bills[item["bill_id"]] = item
                elif isinstance(item, dict) and "line_id" in item:
                    facts.lines[item["line_id"]] = item
                elif isinstance(item, dict) and "customer_id" in item and "full_name" in item:
                    facts.customers[item["customer_id"]] = item
    return facts
