// Tiny UI helpers — formatters and DOM utilities shared across pages.
// No framework; vanilla DOM. Keep this file lean.

/** Format an ISO timestamp → "Apr 21 19:07" (machine-local time). */
export function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

/** Format milliseconds → "12.3s" / "1m 24s" / "2h 13m". */
export function fmtDuration(ms) {
  if (ms == null) return '—';
  const s = Math.round(ms / 1000);
  if (s < 60)   return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

/** Format a [0..1] rate → "57%" with a sign lead if negative. */
export function fmtPct(x) {
  if (x == null || Number.isNaN(x)) return '—';
  return `${(x * 100).toFixed(1)}%`;
}

/** Escape a string for safe innerHTML insertion. Tiny, enough for our use. */
export function esc(s) {
  if (s == null) return '';
  return String(s).replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]));
}

/** Small status pill (colored) for runs.status values. */
export function statusPill(status) {
  const tone = {
    queued:    '#888',
    running:   '#2563eb',
    done:      '#16a34a',
    failed:    '#dc2626',
    discarded: '#6b7280',
  }[status] || '#555';
  return `<span style="background:${tone}; color:#fff; padding:2px 8px; border-radius:10px; font-size:11px; letter-spacing:.3px; text-transform:uppercase;">${esc(status)}</span>`;
}

/** Build a <tr> of <td> cells from an array of innerHTML strings. */
export function tr(cells, { className } = {}) {
  return `<tr${className ? ` class="${className}"` : ''}>${cells.map(c => `<td>${c}</td>`).join('')}</tr>`;
}

/** games/min derived from games_played + wall_ms; '—' if either missing. */
export function gamesPerMin(gamesPlayed, wallMs) {
  if (!gamesPlayed || !wallMs) return '—';
  return (gamesPlayed / (wallMs / 60000)).toFixed(0);
}

/** Games/sec for compact display. Prefer result.games_per_sec (captured during
 *  training only); fallback to games_played / wall_ms (includes admission). */
export function gamesPerSec(result, gamesPlayed, wallMs) {
  if (result && typeof result.games_per_sec === 'number') {
    return result.games_per_sec.toFixed(1);
  }
  if (!gamesPlayed || !wallMs) return '—';
  return (gamesPlayed / (wallMs / 1000)).toFixed(1);
}

/** Parameter count → "185k" / "2.3M". */
export function fmtParams(n) {
  if (n == null) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return Math.round(n / 1e3) + 'k';
  return String(n);
}

/** Compact bottleneck pill. */
export function bottleneckPill(b) {
  const tone = {
    cpu:                '#dc2626',
    gpu:                '#f59e0b',
    balanced:           '#16a34a',
    neither_saturated:  '#6b7280',
  }[b] || '#6b7280';
  if (!b) return '';
  return `<span style="background:${tone}; color:#fff; padding:1px 6px; border-radius:8px; font-size:10px;">${esc(b)}</span>`;
}
