"""
Write the last-24h Supabase done-runs to `docs/runs_summary.md` + commit + push.

Fired daily before 11am Pacific by `mushroom-summary.timer` on PaulLinux. The
cloud routine `Mushroom Wars v2 — daily batch scheduler` reads this file at
11am to plan the next batch — without it, the cloud agent has no live data.

Idempotent: if the file content is unchanged, skips the commit.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import PROJECT, connect


SUMMARY_PATH = ROOT / "docs" / "runs_summary.md"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _read_recent_done(conn, hours: int = 24) -> list[dict]:
    since = _utc_now() - timedelta(hours=hours)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, label, status, hyperparams::text, result::text,
                   weights_url, started_at, finished_at,
                   model_id, simulator_id, seed, budget_ms
              FROM runs
             WHERE project = %s
               AND launch_at >= %s
             ORDER BY launch_at ASC
            """,
            (PROJECT, int(since.timestamp() * 1000)),
        )
        rows = cur.fetchall()
    out = []
    for row in rows:
        result = json.loads(row[4]) if row[4] else None
        hp = json.loads(row[3]) if row[3] else {}
        out.append({
            "id":            str(row[0]),
            "label":         row[1],
            "status":        row[2],
            "hp":            hp,
            "result":        result,
            "weights_url":   row[5],
            "started_at":    row[6],
            "finished_at":   row[7],
            "model_id":      row[8],
            "simulator_id":  row[9],
            "seed":          row[10],
            "budget_ms":     row[11],
        })
    return out


def _format_markdown(runs: list[dict]) -> str:
    now = _utc_now().isoformat(timespec="seconds")
    done = [r for r in runs if r["status"] == "done"]
    failed = [r for r in runs if r["status"] == "failed"]
    discarded = [r for r in runs if r["status"] == "discarded"]
    running = [r for r in runs if r["status"] == "running"]
    queued = [r for r in runs if r["status"] == "queued"]

    lines = []
    lines.append("# Mushroom Wars v2 — last-24h Supabase summary")
    lines.append("")
    lines.append(f"_Generated: {now} (UTC). Auto-written by `scripts/write_runs_summary.py`._")
    lines.append("")
    lines.append(f"## Counts")
    lines.append(f"- done: {len(done)}")
    lines.append(f"- failed: {len(failed)}")
    lines.append(f"- discarded: {len(discarded)}")
    lines.append(f"- running: {len(running)}")
    lines.append(f"- queued: {len(queued)}")
    lines.append("")

    if done:
        lines.append("## Done runs")
        lines.append("")
        lines.append("| label | status | rate | updates | wall (s) | seed | opponent | sim | weights |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for r in done:
            res = r["result"] or {}
            rate = res.get("rate", "")
            rate_str = f"{rate:.3f}" if isinstance(rate, (int, float)) else ""
            updates = res.get("updates", "")
            if r["started_at"] and r["finished_at"]:
                wall = f"{(r['finished_at'] - r['started_at']).total_seconds():.0f}"
            else:
                wall = ""
            opp = r["hp"].get("opponent_name", "?")
            opp_kw = r["hp"].get("opponent_kwargs") or {}
            opp_id = opp_kw.get("opponent_run_id")
            if opp_id:
                opp = f"{opp} ({opp_id[:8]})"
            wkey = "yes" if r["weights_url"] else "no"
            lines.append(
                f"| {r['label']} | {r['status']} | {rate_str} | {updates} | {wall} | "
                f"{r['seed']} | {opp} | {r['simulator_id']} | {wkey} |"
            )
        lines.append("")

        # Strongest checkpoint candidates (rate >= 0.55, weights present)
        lines.append("## Strongest checkpoints (rate >= 0.55, weights present)")
        lines.append("")
        candidates = sorted(
            (r for r in done
             if r["weights_url"]
             and r["result"]
             and (r["result"].get("rate") or 0) >= 0.55),
            key=lambda r: -(r["result"].get("updates") or 0),
        )
        if candidates:
            lines.append("| run id | label | rate | updates | opponent |")
            lines.append("|---|---|---|---|---|")
            for r in candidates[:10]:
                res = r["result"] or {}
                opp = r["hp"].get("opponent_name", "?")
                opp_kw = r["hp"].get("opponent_kwargs") or {}
                opp_id = opp_kw.get("opponent_run_id")
                if opp_id:
                    opp = f"{opp} ({opp_id[:8]})"
                lines.append(
                    f"| `{r['id']}` | {r['label']} | "
                    f"{res.get('rate', 0):.3f} | {res.get('updates', '')} | {opp} |"
                )
        else:
            lines.append("_None met the threshold._")
        lines.append("")

    if failed:
        lines.append("## Failed runs")
        for r in failed:
            lines.append(f"- `{r['id'][:8]}` {r['label']}")
        lines.append("")

    if running:
        lines.append("## Currently running")
        for r in running:
            lines.append(f"- `{r['id'][:8]}` {r['label']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def _git(*args: str) -> tuple[int, str]:
    res = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return res.returncode, (res.stdout + res.stderr).strip()


def _commit_and_push() -> bool:
    """Returns True if a commit was made and pushed."""
    code, out = _git("status", "--porcelain", "docs/runs_summary.md")
    if code != 0:
        print(f"git status failed: {out}")
        return False
    if not out.strip():
        print("no change in docs/runs_summary.md — skipping commit")
        return False

    _git("add", "docs/runs_summary.md")
    code, out = _git(
        "-c", "user.name=PaulMacMADEit",
        "-c", "user.email=paul@madeit.tech",
        "commit", "-m", f"runs_summary: auto-update {_utc_now().date().isoformat()}",
    )
    if code != 0:
        print(f"commit failed: {out}")
        return False
    code, out = _git("push", "origin", "main")
    if code != 0:
        print(f"push failed: {out}")
        return False
    print("committed + pushed")
    return True


def main() -> None:
    with connect() as conn:
        runs = _read_recent_done(conn)
    md = _format_markdown(runs)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(md)
    print(f"wrote {SUMMARY_PATH} ({len(md)} bytes, {len(runs)} runs)")
    _commit_and_push()


if __name__ == "__main__":
    main()
