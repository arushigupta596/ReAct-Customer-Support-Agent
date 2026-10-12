"""Airline policy rules, transcribed from tau2 airline policy.md.

The airline policy says twice that "the API does not check" its rules, so these carry more
weight than in retail: cancellation eligibility and basic-economy changes are only enforced here.
"""

from datetime import datetime, timedelta

from agent.policy_rules.base import Rule
from agent.policy_rules.common import CONFIRMATION

NOW = datetime(2024, 5, 15, 15, 0)  # "The current time is 2024-05-15 15:00:00 EST."
RESERVATION_WRITES = (
    "cancel_reservation",
    "update_reservation_flights",
    "update_reservation_baggages",
    "update_reservation_passengers",
)
FLOWN_STATUSES = ("flying", "landed")


def _reservation(call, facts):
    return facts.reservations.get(call["args"].get("reservation_id"))


def _flown(reservation, facts) -> list[str]:
    flown = []
    for f in reservation.get("flights", []):
        status = facts.flight_status.get((f["flight_number"], f["date"]))
        if f["date"] < NOW.date().isoformat() or status in FLOWN_STATUSES:
            flown.append(f"{f['flight_number']} on {f['date']}")
    return flown


def _user_known(call, facts, messages):
    user_id = call["args"].get("user_id")
    if user_id is None:
        res = _reservation(call, facts)
        user_id = res.get("user_id") if res else None
    if user_id is not None and user_id not in facts.users:
        return f"User {user_id} has not been looked up. Get the user id from the customer and check their profile first."
    return None


def _reservation_checked(call, facts, messages):
    if _reservation(call, facts) is None:
        return f"Reservation {call['args'].get('reservation_id')} has not been looked up. Check its details first."
    return None


def _cancel_eligible(call, facts, messages):
    res = _reservation(call, facts)
    if res is None:
        return None  # reservation_checked reports this
    flown = _flown(res, facts)
    if flown:
        return f"Part of this trip has already been flown ({', '.join(flown)}). Transfer to a human agent instead."
    created = datetime.fromisoformat(res["created_at"])
    airline_cancelled = any(
        facts.flight_status.get((f["flight_number"], f["date"])) == "cancelled"
        for f in res["flights"]
    )
    if (
        NOW - created <= timedelta(hours=24)
        or res.get("cabin") == "business"
        or res.get("insurance") == "yes"
        or airline_cancelled
    ):
        return None
    return (
        "None of the cancellation conditions hold: booked more than 24 hours ago, not business "
        "class, no travel insurance, and no flight confirmed as cancelled by the airline."
    )


def _no_basic_economy_flight_change(call, facts, messages):
    res = _reservation(call, facts)
    if res is None or res.get("cabin") != "basic_economy":
        return None
    old = {(f["flight_number"], f["date"]) for f in res["flights"]}
    new = {(f.get("flight_number"), f.get("date")) for f in call["args"].get("flights", [])}
    if new and new != old:
        return "This is a basic economy reservation, so its flights cannot be changed (only its cabin)."
    return None


def _no_cabin_change_after_flown(call, facts, messages):
    res = _reservation(call, facts)
    if res is None or call["args"].get("cabin") in (None, res.get("cabin")):
        return None
    flown = _flown(res, facts)
    if flown:
        return f"The cabin cannot change because {', '.join(flown)} has already been flown."
    return None


def _bags_not_removed(call, facts, messages):
    res = _reservation(call, facts)
    total = call["args"].get("total_baggages")
    if res is not None and total is not None and total < res.get("total_baggages", 0):
        return f"Checked bags can be added but not removed (currently {res['total_baggages']}, requested {total})."
    return None


def _max_passengers(call, facts, messages):
    if len(call["args"].get("passengers", [])) > 5:
        return "A reservation can have at most five passengers."
    return None


def _compensation_allowed(call, facts, messages):
    user = facts.users.get(call["args"].get("user_id"))
    if user is None:
        return None  # user_known reports this
    reservations = [
        r for r in facts.reservations.values() if r.get("user_id") == user.get("user_id")
    ]
    eligible = user.get("membership") in ("silver", "gold") or any(
        r.get("insurance") == "yes" or r.get("cabin") == "business" for r in reservations
    )
    if not eligible:
        return (
            "No compensation for a regular member without travel insurance flying (basic) economy."
        )
    amount = call["args"].get("amount")
    allowed = {per * len(r.get("passengers", [])) for r in reservations for per in (50, 100)}
    if reservations and amount not in allowed:
        return f"Compensation must be $100 (cancelled) or $50 (delayed) per passenger; {amount} matches neither for the reservations looked up."
    return None


RULES = [
    CONFIRMATION,
    Rule(
        "user_known",
        None,
        _user_known,
        "The agent must first obtain the user id from the user.",
    ),
    Rule(
        "reservation_checked",
        RESERVATION_WRITES,
        _reservation_checked,
        "First, the agent must obtain the user id and reservation id.",
    ),
    Rule(
        "cancel_eligible",
        ("cancel_reservation",),
        _cancel_eligible,
        "The API does not check that cancellation rules are met, so the agent must make sure "
        "the rules apply before calling the API!",
    ),
    Rule(
        "basic_economy_flights",
        ("update_reservation_flights",),
        _no_basic_economy_flight_change,
        "Basic economy flights cannot be modified.",
    ),
    Rule(
        "cabin_after_flown",
        ("update_reservation_flights",),
        _no_cabin_change_after_flown,
        "Cabin cannot be changed if any flight in the reservation has already been flown.",
    ),
    Rule(
        "bags_not_removed",
        ("update_reservation_baggages",),
        _bags_not_removed,
        "The user can add but not remove checked bags.",
    ),
    Rule(
        "max_passengers",
        ("book_reservation",),
        _max_passengers,
        "Each reservation can have at most five passengers.",
    ),
    Rule(
        "compensation",
        ("send_certificate",),
        _compensation_allowed,
        "Only compensate if the user is a silver/gold member or has travel insurance or flies business.",
    ),
]
