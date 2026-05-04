"""Rerate a single run against a targeted opponent set.

Modes:
  - 'top':  play the top N rated runs (excluding self)
  - 'near': play the N runs closest in Elo (excluding self), N above + N below
            when possible. Useful for tightening a rank position.

Each match writes its Elo delta back via tournament.update_elo_from_match,
so the leaderboard updates incrementally.
"""
import json
import time

from cli.db import connect, PROJECT


def handle(job: dict, mark_done_fn, mark_failed_fn) -> None:
    hp = job.get("hyperparams") or {}
    target_run_id = hp.get("target_run_id")
    mode  = str(hp.get("mode", "near"))
    n     = int(hp.get("n", 5))
    games = int(hp.get("games", 64))
    level = str(hp.get("level", "random_close_4_5"))

    if not target_run_id:
        raise ValueError("rerate_one job missing hyperparams.target_run_id")

    # Imports here so we don't pay torch/jax cost on worker startup if no
    # rerate_one job is ever picked up.
    from scripts.tournament import run_match, update_elo_from_match
    import torch

    device = torch.device("cuda" if torch.cuda.is_available()
                          else ("mps" if torch.backends.mps.is_available() else "cpu"))

    print(f"[job:rerate_one] target={target_run_id} mode={mode} n={n} "
          f"games={games} level={level}", flush=True)

    # Resolve target + opponent set up front in one connection.
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT label, elo_score FROM runs WHERE id = %s",
            (target_run_id,),
        )
        row = cur.fetchone()
        if row is None:
            raise ValueError(f"target run {target_run_id} not found")
        target_label, target_elo = row

        if mode == "top":
            cur.execute(
                """
                SELECT id, label, elo_score FROM runs
                WHERE project = %s
                  AND status = 'done'
                  AND elo_n_matches >= 1
                  AND id <> %s
                ORDER BY elo_score DESC
                LIMIT %s
                """,
                (PROJECT, target_run_id, n),
            )
        elif mode == "near":
            cur.execute(
                """
                SELECT id, label, elo_score FROM runs
                WHERE project = %s
                  AND status = 'done'
                  AND elo_n_matches >= 1
                  AND id <> %s
                ORDER BY ABS(elo_score - %s) ASC
                LIMIT %s
                """,
                (PROJECT, target_run_id, target_elo or 1000.0, 2 * n),
            )
        else:
            raise ValueError(f"unknown mode {mode!r} (expected 'top' or 'near')")

        opponents = [(str(r[0]), r[1], r[2]) for r in cur.fetchall()]

    if not opponents:
        raise RuntimeError("no rated opponents found — nothing to rerate against")

    print(f"[job:rerate_one] {target_label} (Elo {target_elo:.0f}) vs "
          f"{len(opponents)} opponents", flush=True)

    log_lines = [
        f"target: {target_label}  (Elo {target_elo:.0f}, id {target_run_id[:8]})",
        f"mode: {mode}  n: {n}  games/match: {games}  level: {level}",
        f"opponents ({len(opponents)}):",
    ]
    for _, lbl, elo in opponents:
        log_lines.append(f"  - {lbl}  Elo {elo:.0f}")
    log_lines.append("")

    per_opponent = []
    started_elo = target_elo
    t0 = time.time()
    for i, (opp_id, opp_label, opp_elo) in enumerate(opponents, 1):
        seed = i  # deterministic per-pair seed
        try:
            res = run_match(
                p1=target_run_id, p2=opp_id,
                games=games, level=level, seed=seed,
                device=device, verbose=False,
            )
            with connect() as conn:
                p1_new, _ = update_elo_from_match(
                    conn, p1_run_id=target_run_id, p2_run_id=opp_id,
                    result=res, k=32,
                )
            wr = (res["p1_wins"] + 0.5 * res["draws"]) / max(res["total"] - res.get("timeouts", 0), 1)
            line = (f"  [{i}/{len(opponents)}] vs {opp_label[:40]:40s} "
                    f"opp_elo={opp_elo:.0f}  rate={wr:.3f}  "
                    f"target_elo→{p1_new:.0f}  ({(time.time()-t0)/60:.1f}min)")
            log_lines.append(line)
            print(line, flush=True)
            per_opponent.append({
                "opp_id": opp_id, "opp_label": opp_label,
                "opp_elo_before": opp_elo,
                "wins": res["p1_wins"], "losses": res["p2_wins"],
                "draws": res["draws"], "timeouts": res.get("timeouts", 0),
                "rate": wr,
            })
        except Exception as e:
            line = f"  [{i}/{len(opponents)}] vs {opp_label[:40]} FAILED: {e}"
            log_lines.append(line)
            print(line, flush=True)

        # Periodic log flush so the dashboard can tail.
        if (i % 3 == 0) or i == len(opponents):
            _flush_log(target_run_id, job["id"], "\n".join(log_lines), int((time.time() - t0) * 1000))

    wall_ms = int((time.time() - t0) * 1000)

    # Read the final Elo so the result row reflects the post-rerate state.
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT elo_score FROM runs WHERE id = %s", (target_run_id,))
        final_elo = cur.fetchone()[0]

    log_lines.append("")
    log_lines.append(f"done: target Elo {started_elo:.0f} → {final_elo:.0f} "
                     f"(Δ {final_elo - (started_elo or 0):+.0f}) "
                     f"over {len(per_opponent)} opponents in {wall_ms/60000:.1f}min")

    result = {
        "kind": "rerate_one",
        "params": {"target_run_id": target_run_id, "mode": mode,
                   "n": n, "games": games, "level": level},
        "target_label": target_label,
        "elo_before": started_elo,
        "elo_after":  final_elo,
        "elo_delta":  (final_elo - (started_elo or 0)) if final_elo is not None else None,
        "opponents":  per_opponent,
        "log":        "\n".join(log_lines),
        "wall_s":     round(wall_ms / 1000, 1),
    }

    matches_done = len(per_opponent)
    with connect() as conn:
        mark_done_fn(
            conn, job["id"], result,
            games_played=matches_done * games,
            wall_ms=wall_ms,
        )


def _flush_log(target_run_id, job_id, log: str, wall_ms: int) -> None:
    """Stream incremental log into the JOB row (not the target run)."""
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                UPDATE runs
                   SET result = jsonb_build_object(
                                  'kind', 'rerate_one',
                                  'target_run_id', %s::text,
                                  'log', %s::text,
                                  'wall_s', %s::float,
                                  'in_progress', true),
                       wall_ms = %s
                 WHERE id = %s
                """,
                (target_run_id, log, wall_ms / 1000.0, wall_ms, job_id),
            )
            conn.commit()
    except Exception as e:
        print(f"[job:rerate_one] log flush failed (non-fatal): {e}", flush=True)
