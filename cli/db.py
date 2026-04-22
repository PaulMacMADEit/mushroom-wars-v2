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
    url = os.environ.get("SUPABASE_DB_URL")
    if not url:
        raise RuntimeError(
            "SUPABASE_DB_URL not set — fill it in .env (see .env.example)."
        )
    return url


@contextmanager
def connect():
    """Short-lived connection. Autocommit off; caller commits explicitly."""
    with psycopg.connect(db_url()) as conn:
        yield conn
