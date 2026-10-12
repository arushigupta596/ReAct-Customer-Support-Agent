"""Span exporter wrapper: masks PII and caps attribute size before spans leave the process.

PII is masked in traces only, never in the model's context (spec 5.1). Size capping exists
because LangGraph node spans carry the whole conversation state: a long banking conversation
with knowledge-base documents produced 5-14 MB export batches, over the 4 MB gRPC limit, and
Phoenix dropped them whole (debug journal entry 7).
"""

import re
from collections.abc import Sequence

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Phones need separators so 10-digit product / item ids stay readable in traces.
PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
MAX_CHARS = 16_000


def mask(text: str) -> str:
    text = EMAIL.sub("[email]", text)
    text = CARD.sub("[card]", text)
    return PHONE.sub("[phone]", text)


def cap(text: str, limit: int = MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    head, tail = limit * 2 // 3, limit // 3
    return (
        f"{text[:head]}\n...[{len(text) - head - tail} chars truncated in trace]...\n{text[-tail:]}"
    )


def _clean(value):
    if isinstance(value, str):
        return cap(mask(value))
    if isinstance(value, (list, tuple)):
        return type(value)(_clean(v) for v in value)
    return value


def _clean_attrs(attrs) -> dict:
    return {k: _clean(v) for k, v in (attrs or {}).items()}


def sanitize(span: ReadableSpan) -> ReadableSpan:
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_clean_attrs(span.attributes),
        events=[Event(e.name, _clean_attrs(e.attributes), e.timestamp) for e in span.events],
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class SanitizingExporter(SpanExporter):
    def __init__(self, inner: SpanExporter):
        self.inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self.inner.export([sanitize(s) for s in spans])

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self.inner.force_flush(timeout_millis)
