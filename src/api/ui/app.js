// Shared helpers for the chat and console pages.

const escapeHtml = (t) => String(t ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));

// Minimal, safe markdown for agent text: escape everything first, then add a few tags.
function inlineMd(t) {
  return t
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
}
function markdown(text) {
  const out = []; let list = null;
  for (const raw of escapeHtml(text).split('\n')) {
    const item = raw.match(/^\s*(?:[-*•]|(\d+)[.)])\s+(.*)$/);
    if (item) {
      const kind = item[1] ? 'ol' : 'ul';
      if (list !== kind) { if (list) out.push(`</${list}>`); out.push(`<${kind}>`); list = kind; }
      out.push(`<li>${inlineMd(item[2])}</li>`);
      continue;
    }
    if (list) { out.push(`</${list}>`); list = null; }
    const h = raw.match(/^#{1,3}\s+(.*)$/);
    if (h) out.push(`<p><strong>${inlineMd(h[1])}</strong></p>`);
    else if (raw.trim()) out.push(`<p>${inlineMd(raw)}</p>`);
  }
  if (list) out.push(`</${list}>`);
  return out.join('');
}

const ICON = { think: '…', lookup: 'L', check: '✓', propose: '?', signoff: 'S', act: '!', escalate: 'H', guard: 'G', reply: 'R' };
const LABEL = {
  think: 'Understand', lookup: 'Look up', check: 'Check policy', propose: 'Propose action',
  signoff: 'Sign-off', act: 'Act', escalate: 'Escalate', guard: 'Guard', reply: 'Reply',
};

function timelineItem(e) {
  const li = document.createElement('li');
  li.className = `tl ${e.type} ${e.verdict || ''}`;
  const time = e.ts ? new Date(e.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '';
  const trace = e.trace_url ? `<a href="${escapeHtml(e.trace_url)}" target="_blank" rel="noopener">Open trace in Phoenix</a>` : '';
  li.innerHTML = `
    <div class="dot" aria-hidden="true">${ICON[e.type] || '•'}</div>
    <div>
      <div class="title"><span>${escapeHtml(e.title)}</span>${e.verdict ? `<span class="chip ${escapeHtml(e.verdict)}">${escapeHtml(e.verdict)}</span>` : ''}</div>
      ${e.detail ? `<div class="detail">${escapeHtml(e.detail)}</div>` : ''}
      <div class="meta"><span>${LABEL[e.type] || e.type}</span><span>${time}</span>${trace}</div>
    </div>`;
  return li;
}

// Polls /sessions/{id}/events and appends new ones; returns {refresh, reset}.
function timeline(listEl, sessionId) {
  let last = 0;
  async function refresh() {
    const r = await fetch(`/sessions/${sessionId}/events?after=${last}`);
    if (!r.ok) return;
    const events = await r.json();
    if (events.length && listEl.querySelector('.empty')) listEl.innerHTML = '';
    for (const e of events) { listEl.append(timelineItem(e)); last = e.seq; }
    if (events.length) listEl.lastElementChild?.scrollIntoView({ block: 'nearest' });
  }
  function reset(id) { sessionId = id; last = 0; listEl.innerHTML = '<li class="empty">No decisions yet.</li>'; }
  reset(sessionId);
  return { refresh, reset };
}

// POST a turn and read server-sent events as they arrive.
async function streamTurn(url, body, onEvent) {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' }, body: JSON.stringify(body) });
  if (!r.ok || !r.body) return { status: 'error', detail: (await r.json().catch(() => ({}))).detail };
  const reader = r.body.getReader(); const dec = new TextDecoder(); let buf = ''; let outcome = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, i); buf = buf.slice(i + 2);
      const kind = (frame.match(/^event: (.*)$/m) || [])[1];
      const data = JSON.parse((frame.match(/^data: (.*)$/m) || [])[1] || '{}');
      if (kind === 'event') onEvent(data); else outcome = { kind, ...data };
    }
  }
  return outcome;
}
