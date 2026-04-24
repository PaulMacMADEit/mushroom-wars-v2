/**
 * Show a "build {sha7} · {ago}" chip next to the header h1 so we can see at a
 * glance what version is deployed. Reads the latest commit on `main` from
 * GitHub's public API. No auth needed — rate-limited to 60/hr per IP, plenty
 * for a personal dev dashboard.
 */
const REPO = "PaulMacMADEit/mushroom-wars-v2";

function timeAgo(iso) {
  const seconds = (Date.now() - new Date(iso).getTime()) / 1000;
  if (seconds < 60)        return `${Math.round(seconds)}s ago`;
  if (seconds < 3600)      return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400)     return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

function inject(text, title) {
  const h1 = document.querySelector("header h1");
  if (!h1) return;
  const chip = document.createElement("span");
  chip.className = "build-chip";
  chip.textContent = text;
  if (title) chip.title = title;
  chip.style.cssText = `
    margin-left: 10px;
    padding: 2px 8px;
    font-size: 11px;
    font-family: var(--mono, ui-monospace, monospace);
    color: var(--fg-muted, #8b92a4);
    background: var(--bg-card, #1f2433);
    border: 1px solid var(--border, #2a2f3d);
    border-radius: 6px;
    vertical-align: middle;
  `;
  h1.appendChild(chip);
}

fetch(`https://api.github.com/repos/${REPO}/commits/main`)
  .then(r => r.ok ? r.json() : Promise.reject(r.status))
  .then(d => {
    const sha = (d.sha || "").slice(0, 7);
    const date = d.commit?.committer?.date || d.commit?.author?.date;
    const ago = date ? timeAgo(date) : "";
    inject(`build ${sha}${ago ? " · " + ago : ""}`, date ? `Deployed commit ${sha} (${date})` : "");
  })
  .catch(() => inject("build ?", "Could not reach GitHub API"));
