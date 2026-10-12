"""Write eval scores back onto each simulation's root span in Phoenix (spec section 8).

After a run, a failed pass^k task opens straight into its trajectory: filter the Phoenix
project on the `reward` or `passes_all_trials` annotation and click through.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import httpx
from phoenix.client import Client

from evals.metrics import load, score_simulation


def build_annotations(
    results_path: Path, spans: dict[str, dict], tool_kinds: dict[str, str]
) -> list[dict]:
    tasks, sims = load(results_path)
    scores = {s["id"]: score_simulation(s, tasks[s["task_id"]], tool_kinds) for s in sims}

    # pass^k membership: did this task succeed on every trial of this run?
    all_trials: dict[str, bool] = defaultdict(lambda: True)
    for sim in sims:
        all_trials[sim["task_id"]] &= scores[sim["id"]]["success"]

    annotations = []
    for sim in sims:
        span = spans.get(sim["id"])
        if span is None:
            continue  # simulation ran without tracing
        sc = scores[sim["id"]]

        def ann(name, score=None, label=None, explanation=None, _span_id=span["span_id"]):
            annotations.append(
                {
                    "span_id": _span_id,
                    "name": name,
                    "annotator_kind": "CODE",
                    "result": {
                        k: v
                        for k, v in {
                            "score": score,
                            "label": label,
                            "explanation": explanation,
                        }.items()
                        if v is not None
                    },
                }
            )

        ann("reward", score=sc["reward"], label="pass" if sc["success"] else "fail")
        if sc["recall_total"]:
            ann("action_recall", score=sc["recall_hits"] / sc["recall_total"])
        if sc["writes"]:
            ann(
                "write_precision",
                score=sc["writes_expected"] / sc["writes"],
                explanation=f"{sc['writes_expected']} of {sc['writes']} WRITE calls were expected",
            )
        if sc["escalation"]:
            ann("escalation", label=sc["escalation"])
        passed_all = all_trials[sim["task_id"]]
        ann("passes_all_trials", score=float(passed_all), label="yes" if passed_all else "no")
    return annotations


def annotate(
    results_path: Path, spans_path: Path, tool_kinds: dict[str, str], base_url: str
) -> int:
    spans = json.loads(Path(spans_path).read_text())
    annotations = build_annotations(results_path, spans, tool_kinds)
    if not annotations:
        return 0
    client = Client(base_url=base_url)
    # Phoenix ingests spans asynchronously; right after a run the root spans may not exist yet.
    for attempt in range(10):
        try:
            client.spans.log_span_annotations(span_annotations=annotations, sync=True)
            return len(annotations)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404 or attempt == 9:
                raise
            time.sleep(3)
    return 0
