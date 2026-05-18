"""Audit Supabase storage + DB row footprint for the mushroom-wars project.

Bypasses the REST egress meter by talking to Postgres direct via the pooler.
Prints two tables:
  1. Storage bucket sizes + object counts (from storage.objects)
  2. Per-status run-artifact summary (weights/optimizer/obs_norm/log) + age

Run:
    .venv/bin/python scripts/storage_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.db import connect, PROJECT


def fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


def table(headers: list[str], rows: list[list[str]], aligns: list[str] | None = None) -> str:
    aligns = aligns or ["l"] * len(headers)
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]

    def fmt_row(cells: list[str]) -> str:
        out = []
        for c, w, a in zip(cells, widths, aligns):
            s = str(c)
            out.append(s.rjust(w) if a == "r" else s.ljust(w))
        return "  ".join(out)

    sep = "  ".join("-" * w for w in widths)
    return "\n".join([fmt_row(headers), sep, *[fmt_row(r) for r in rows]])


def main() -> None:
    with connect() as c, c.cursor() as cur:
        print("\n━━━ 1. STORAGE BUCKETS ━━━\n")
        cur.execute(
            """
            SELECT bucket_id,
                   count(*) AS n_objects,
                   sum( (metadata->>'size')::bigint ) AS total_bytes
            FROM storage.objects
            GROUP BY bucket_id
            ORDER BY total_bytes DESC NULLS LAST
            """
        )
        bucket_rows = cur.fetchall()
        rows = [
            [b, f"{n:,}", fmt_bytes(s)]
            for (b, n, s) in bucket_rows
        ]
        total_objects = sum(r[1] for r in bucket_rows)
        total_bytes = sum((r[2] or 0) for r in bucket_rows)
        rows.append(["TOTAL", f"{total_objects:,}", fmt_bytes(total_bytes)])
        print(table(["bucket", "objects", "size"], rows, ["l", "r", "r"]))

        print("\n━━━ 2. STORAGE BREAKDOWN BY TOP-LEVEL PREFIX ━━━\n")
        cur.execute(
            """
            SELECT bucket_id,
                   split_part(name, '/', 1) AS prefix,
                   count(*) AS n_objects,
                   sum((metadata->>'size')::bigint) AS total_bytes
            FROM storage.objects
            GROUP BY bucket_id, split_part(name, '/', 1)
            ORDER BY total_bytes DESC NULLS LAST
            LIMIT 30
            """
        )
        rows = [
            [b, p or "(root)", f"{n:,}", fmt_bytes(s)]
            for (b, p, n, s) in cur.fetchall()
        ]
        print(table(["bucket", "prefix", "objects", "size"], rows, ["l", "l", "r", "r"]))

        print("\n━━━ 3. RUNS TABLE — STATUS + AGE ━━━\n")
        cur.execute(
            f"""
            SELECT status,
                   count(*) AS n,
                   count(*) FILTER (WHERE weights_url IS NOT NULL)    AS w,
                   count(*) FILTER (WHERE optimizer_url IS NOT NULL)  AS o,
                   count(*) FILTER (WHERE obs_norm_url IS NOT NULL)   AS n2,
                   count(*) FILTER (WHERE log_url IS NOT NULL)        AS l,
                   min(queued_at) AS oldest,
                   max(queued_at) AS newest
            FROM runs
            WHERE project = '{PROJECT}'
            GROUP BY status
            ORDER BY n DESC
            """
        )
        rows = []
        total_runs = 0
        for status, n, w, o, n2, l, oldest, newest in cur.fetchall():
            total_runs += n
            rows.append([
                status,
                f"{n:,}",
                f"{w:,}",
                f"{o:,}",
                f"{n2:,}",
                f"{l:,}",
                oldest.strftime("%Y-%m-%d") if oldest else "—",
                newest.strftime("%Y-%m-%d") if newest else "—",
            ])
        print(table(
            ["status", "runs", "weights", "opt", "obsN", "log", "oldest", "newest"],
            rows, ["l", "r", "r", "r", "r", "r", "l", "l"],
        ))
        print(f"\nTOTAL runs in '{PROJECT}': {total_runs:,}")

        print("\n━━━ 4. TOP-20 RUNS BY result.rate (potential 'keep' list) ━━━\n")
        cur.execute(
            f"""
            SELECT id, label, status,
                   (result->>'rate')::float AS rate,
                   queued_at,
                   (weights_url IS NOT NULL) AS has_w
            FROM runs
            WHERE project = '{PROJECT}'
              AND status  = 'done'
              AND result ? 'rate'
            ORDER BY (result->>'rate')::float DESC NULLS LAST
            LIMIT 20
            """
        )
        rows = [
            [str(rid)[:8], (lbl or "")[:48], st, f"{rate:.3f}" if rate is not None else "—",
             qa.strftime("%Y-%m-%d"), "✓" if has_w else ""]
            for (rid, lbl, st, rate, qa, has_w) in cur.fetchall()
        ]
        print(table(
            ["id8", "label", "status", "rate", "queued", "w?"],
            rows, ["l", "l", "l", "r", "l", "l"],
        ))

        print("\n━━━ 5. STORAGE SIZE — ALL TABLES (Postgres on-disk) ━━━\n")
        cur.execute(
            """
            SELECT relname,
                   pg_size_pretty(pg_total_relation_size(relid)) AS total,
                   pg_total_relation_size(relid)                  AS bytes
            FROM pg_catalog.pg_statio_user_tables
            ORDER BY pg_total_relation_size(relid) DESC
            LIMIT 15
            """
        )
        rows = [[name, size] for (name, size, _) in cur.fetchall()]
        print(table(["table", "size"], rows, ["l", "r"]))

        print("\n━━━ 6. DATABASE TOTAL ━━━\n")
        cur.execute("SELECT pg_size_pretty(pg_database_size(current_database()))")
        db_size = cur.fetchone()[0]
        print(f"Database on-disk: {db_size}")
        print()


if __name__ == "__main__":
    main()
