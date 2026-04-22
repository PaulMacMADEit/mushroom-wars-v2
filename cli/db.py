"""Shared Supabase Postgres helpers.

One place to import the connection string + common settings. Every CLI
script + the worker uses this so credentials live in .env, not code.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import psycopg
from dotenv import load_dotenv


PROJECT = "mushroom-wars"  # project name used across all rows

# Load .env from repo root no matter where the script is run from.
_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


def db_url() -> str:
    """Prefer the pooler URL when set — it's IPv4-available and the direct
    connection is IPv6-only on current Supabase, which isn't routable from
    every network.
    """
    url = os.environ.get("SUPABASE_POOL_URL") or os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError(
            "Neither SUPABASE_POOL_URL nor SUPABASE_DB_URL is set — fill in .env."
        )
    return url


@contextmanager
def connect():
    """Short-lived connection. Autocommit off; caller commits explicitly.

    Disables client-side prepared statements because Supabase's transaction
    pooler (PgBouncer) multiplexes client connections over a smaller pool of
    backends — a prepared statement named by one client collides when the
    same backend later serves another. Symptom without this: the second
    connection to reuse a backend errors with
    `DuplicatePreparedStatement: prepared statement "_pg3_0" already exists`.
    """
    with psycopg.connect(db_url()) as conn:
        conn.prepare_threshold = None
        yield conn
