#!/usr/bin/env python
"""Quick game-replay sanity check for the most recent rated karp- run.

Pulls 1 win + 1 loss replay from bench_eval matches, computes behavior
signals (action type distribution, no-op rate, entropy stats, value-trajectory),
and prints a markdown-table summary suitable for pasting into KARPATHY_LOG.md.

Usage:
  python scripts/karp_review_games.py                  # most recent rated karp- run
  python scripts/karp_review_games.py --label karp-... # specific run
  python scripts/karp_review_games.py --n-games 4      # sample more games
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from workers.storage import client


TYPE_NAMES = {0: "25%", 1: "50%", 2: "75%", 3: "100%", 4: "noop"}


def _latest_rated_karp_label(sb) -> str | None:
    res = (
        sb.table("runs")
        .select("label,queued_at")
        .like("label", "karpv2-%")
        .eq("elo_status", "rated")
        .order("queued_at", desc=True)
        .limit(1)
        .execute()
    )
    return res.data[0]["label"] if res.data else None


def _games_for_run(sb, run_id: str, limit: int = 12):
    # Bench_eval queues all match rows in one transaction, so created_at
    # ties are common — fetch ALL matches for the run, then filter to
    # the ones that have actually been populated with games.
    matches = (
        sb.table("matches")
        .select("id,model_a_run_id,model_b_run_id,summary")
        .or_(f"model_a_run_id.eq.{run_id},model_b_run_id.eq.{run_id}")
        .order("created_at", desc=True)
        .limit(20)
        .execute()
    ).data
    if not matches:
        return [], None
    match_ids = [m["id"] for m in matches]
    games = (
        sb.table("games")
        .select(
            "id,match_id,winner,duration_ms,actions_url,stats,seed,map_name,"
            "player_1_run_id,player_2_run_id"
        )
        .in_("match_id", match_ids)
        .limit(limit)
        .execute()
    ).data
    return games, matches


def _label_outcome(g, run_id: str) -> str:
    if g["winner"] == run_id:
        return "win"
    if g["winner"] is None:
        return "draw"
    return "loss"


def _player_for_run(g, run_id: str) -> int | None:
    if g["player_1_run_id"] == run_id:
        return 1
    if g["player_2_run_id"] == run_id:
        return 2
    return None


def _fetch_replay(sb, actions_url: str) -> dict | None:
    if not actions_url or not actions_url.startswith("replays/"):
        return None
    key = actions_url[len("replays/") :]
    try:
        raw = sb.storage.from_("replays").download(key)
    except Exception:
        return None
    return json.loads(raw)


def _analyze(replay: dict, our_player: int) -> dict:
    decisions = [d for d in replay.get("decisions", []) if d.get("player") == our_player]
    events = replay.get("events", [])
    duration_ticks = replay.get("duration_ticks", 0)
    if not decisions:
        return {"n_decisions": 0}

    type_counter = Counter()
    src_counter = Counter()
    tgt_counter = Counter()
    repeated_picks = 0
    last_pick = None
    streak = 0
    max_streak = 0
    entropies = []
    values = []
    for d in decisions:
        p = d["picked"]
        t = p.get("type", -1)
        type_counter[t] += 1
        src_counter[p.get("src", -1)] += 1
        tgt_counter[p.get("tgt", -1)] += 1
        entropies.append(d.get("entropy", 0.0))
        values.append(d.get("value", 0.0))
        pick_tuple = (p.get("src"), p.get("type"), p.get("tgt"))
        if pick_tuple == last_pick:
            streak += 1
            max_streak = max(max_streak, streak)
            if streak >= 3:
                repeated_picks += 1
        else:
            streak = 0
        last_pick = pick_tuple

    n = len(decisions)
    noop_rate = type_counter[4] / n if n else 0.0
    type_dist = {TYPE_NAMES.get(t, f"?{t}"): c / n for t, c in type_counter.items()}

    # Send hygiene: count outgoing sends with count > 0 (real action) vs garrison_after
    our_sends = [
        e for e in events
        if e.get("kind") == "send" and e.get("owner") == our_player
    ]
    weak_sends = sum(1 for e in our_sends if (e.get("count") or 0) < 10)
    full_sends = sum(1 for e in our_sends if (e.get("src_garrison_after") or 0) == 0)

    # Value trajectory (sign tells confidence direction)
    value_first = values[0] if values else 0.0
    value_last = values[-1] if values else 0.0
    value_drop = value_first - value_last  # +ve means lost confidence

    return {
        "n_decisions": n,
        "duration_ticks": duration_ticks,
        "noop_rate": noop_rate,
        "type_dist": type_dist,
        "n_sends": len(our_sends),
        "weak_sends": weak_sends,
        "full_garrison_sends": full_sends,
        "max_repeat_streak": max_streak,
        "ge3_repeats": repeated_picks,
        "entropy_mean": mean(entropies) if entropies else 0.0,
        "entropy_std": stdev(entropies) if len(entropies) > 1 else 0.0,
        "value_first": value_first,
        "value_last": value_last,
        "value_drop": value_drop,
    }


def _flag_anomalies(stats: dict) -> list[str]:
    flags = []
    if stats.get("n_decisions", 0) == 0:
        flags.append("no decisions for our player (replay/role mismatch?)")
        return flags
    if stats["noop_rate"] > 0.5:
        flags.append(f"high noop rate {stats['noop_rate']:.0%}")
    if stats["entropy_mean"] < 0.3:
        flags.append(f"low entropy mean {stats['entropy_mean']:.2f} (deterministic)")
    if stats["max_repeat_streak"] >= 5:
        flags.append(f"repeat-pick streak {stats['max_repeat_streak']}")
    type_dist = stats.get("type_dist", {})
    dominant = max(type_dist.values(), default=0)
    if dominant >= 0.85 and stats["n_decisions"] >= 8:
        top_type = max(type_dist, key=type_dist.get)
        flags.append(f"type-collapse: {top_type} is {dominant:.0%}")
    if stats["n_sends"] and stats["weak_sends"] / stats["n_sends"] > 0.3:
        flags.append(f"weak sends (count<10) {stats['weak_sends']}/{stats['n_sends']}")
    return flags


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", default=None, help="karp run label (default: latest rated)")
    ap.add_argument("--n-games", type=int, default=2, help="# games to sample (default 2)")
    args = ap.parse_args()

    sb = client()

    label = args.label or _latest_rated_karp_label(sb)
    if not label:
        print("[review] no rated karp- runs found")
        return
    run = sb.table("runs").select("id,label,elo_score,elo_status,result").eq("label", label).execute().data
    if not run:
        print(f"[review] run {label} not found")
        return
    run = run[0]
    run_id = run["id"]
    print(f"[review] {run['label']}  elo={run.get('elo_score'):.1f}  status={run.get('elo_status')}")

    games, matches = _games_for_run(sb, run_id, limit=24)
    if not games:
        print("[review] no bench games found for this run")
        return

    # Bucket by outcome
    wins = [g for g in games if _label_outcome(g, run_id) == "win"]
    losses = [g for g in games if _label_outcome(g, run_id) == "loss"]
    samples = []
    if wins:
        samples.append(("WIN", wins[0]))
    if losses:
        samples.append(("LOSS", losses[0]))
    # Top up if requested
    extras = [g for g in games if g not in [s[1] for s in samples]]
    while len(samples) < args.n_games and extras:
        samples.append(("EXTRA", extras.pop(0)))

    print(f"[review] sampling {len(samples)} games from {len(games)} bench games "
          f"(wins={len(wins)} losses={len(losses)})")
    print()

    rows = []
    all_flags = []
    for tag, g in samples:
        replay = _fetch_replay(sb, g["actions_url"])
        if not replay:
            print(f"  [{tag}] no replay available for {g['id']}")
            continue
        our_player = _player_for_run(g, run_id)
        stats = _analyze(replay, our_player)
        flags = _flag_anomalies(stats)
        all_flags.extend(flags)

        type_str = " ".join(
            f"{name}={pct:.0%}"
            for name, pct in sorted(
                stats.get("type_dist", {}).items(), key=lambda x: -x[1]
            )[:3]
        )
        flag_str = "; ".join(flags) if flags else "ok"
        rows.append({
            "tag": tag,
            "game_id": g["id"],
            "ticks": stats.get("duration_ticks", 0),
            "decisions": stats.get("n_decisions", 0),
            "noop": stats.get("noop_rate", 0),
            "ent": stats.get("entropy_mean", 0),
            "value_drop": stats.get("value_drop", 0),
            "types": type_str,
            "flags": flag_str,
            "url": g["actions_url"],
        })

    # Markdown summary
    print("| game | tag | ticks | decisions | noop% | entropy | value drop | top types | flags |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| `{r['game_id'][:8]}` | {r['tag']} | {r['ticks']} | "
            f"{r['decisions']} | {r['noop']:.0%} | {r['ent']:.2f} | "
            f"{r['value_drop']:+.2f} | {r['types']} | {r['flags']} |"
        )
    print()
    if all_flags:
        print(f"🚩 anomalies: {'; '.join(set(all_flags))}")
    else:
        print("✓ behavior looks normal across sampled games")

    # Replay URLs for browser drag-drop
    print()
    print("Replay URLs (drag-drop into dashboard/game.html):")
    for r in rows:
        print(f"  - {r['tag']}: {r['url']}")


if __name__ == "__main__":
    main()
