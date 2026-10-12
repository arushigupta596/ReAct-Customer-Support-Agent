"""Telecom rules, from tau2 telecom main_policy.md (benchmark only; not in the live demo).

No blanket confirmation rule: most telecom turns are guided troubleshooting, where the
customer acts on their own phone, and the policy asks for confirmation only of prices.
"""

from datetime import date

from agent.policy_rules.base import Rule

TODAY = date(2025, 2, 25)  # "The current time is 2025-02-25 12:08:00 EST."
MAX_REFUEL_GB = 2.0


def _identified(call, facts, messages):
    if not facts.customers:
        return "The customer has not been identified. Look them up by phone, id, or name + date of birth."
    return None


def _line_owned(call, facts, messages):
    line_id = call["args"].get("line_id")
    if line_id is None or not facts.customers:
        return None
    if not any(line_id in c.get("line_ids", []) for c in facts.customers.values()):
        return f"Line {line_id} does not belong to the identified customer."
    return None


def _refuel_limit(call, facts, messages):
    amount = float(call["args"].get("gb_amount", 0))
    already = float(
        (facts.lines.get(call["args"].get("line_id")) or {}).get("data_refueling_gb", 0) or 0
    )
    if amount + already > MAX_REFUEL_GB:
        return f"At most {MAX_REFUEL_GB:g} GB can be refueled (requested {amount:g}, already {already:g})."
    return None


def _bill_overdue(call, facts, messages):
    bill = facts.bills.get(call["args"].get("bill_id"))
    if bill is None:
        return (
            f"Bill {call['args'].get('bill_id')} has not been checked. Confirm it is overdue first."
        )
    if str(bill.get("status", "")).lower() != "overdue":
        return f"Bill {bill.get('bill_id')} is '{bill.get('status')}', not overdue."
    return None


def _can_resume(call, facts, messages):
    line = facts.lines.get(call["args"].get("line_id"))
    if line is None:
        return (
            f"Line {call['args'].get('line_id')} has not been checked. Look up its details first."
        )
    end = line.get("contract_end_date")
    if end and date.fromisoformat(end) < TODAY:
        return f"The line's contract ended on {end}, so the suspension cannot be lifted."
    overdue = [
        b["bill_id"] for b in facts.bills.values() if str(b.get("status", "")).lower() == "overdue"
    ]
    if overdue:
        return f"Overdue bills remain ({', '.join(overdue)}); they must be paid before resuming the line."
    return None


RULES = [
    Rule(
        "customer_identified",
        None,
        _identified,
        "You must first identify the customer.",
    ),
    Rule(
        "line_owned",
        None,
        _line_owned,
        "Apply the refueled data to the line associated with the phone number the user provided.",
    ),
    Rule(
        "refuel_limit",
        ("refuel_data",),
        _refuel_limit,
        "The maximum amount of data that can be refueled is 2GB.",
    ),
    Rule(
        "bill_overdue",
        ("send_payment_request",),
        _bill_overdue,
        "The send payement request tool will not check if the bill is overdue. You should always "
        "check that the bill is overdue before sending a payment request.",
    ),
    Rule(
        "can_resume",
        ("resume_line",),
        _can_resume,
        "You are allowed to lift the suspension after the user has paid all their overdue bills. "
        "You are not allowed to lift the suspension if the line's contract end date is in the past.",
    ),
]
