"""
Smoke-test the Supabase connection.

Verifies:
  1. .env is loaded and all required keys are present
  2. Supabase REST API responds (via supabase-py) with the service role key
  3. Direct Postgres connection works (via psycopg)
  4. We can create a temp table, insert, select, drop — full write access confirmed

Run:
    pip install -r requirements.txt
    python scripts/check_supabase.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


OK = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
ARROW = "\033[36m→\033[0m"


def check_env() -> dict[str, str]:
    """Load .env and verify all required vars are present and not placeholders."""
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    load_dotenv(env_path)

    required = [
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "SUPABASE_DB_URL",
    ]
    missing = []
    placeholders = []
    values: dict[str, str] = {}
    for key in required:
        v = os.environ.get(key, "").strip()
        if not v:
            missing.append(key)
        elif v.startswith("PASTE_") or v.startswith("[YOUR"):
            placeholders.append(key)
        else:
            values[key] = v

    print(f"{ARROW} Loading {env_path}")
    if missing:
        print(f"{FAIL} Missing env vars: {', '.join(missing)}")
        sys.exit(1)
    if placeholders:
        print(f"{FAIL} Placeholder values in: {', '.join(placeholders)}")
        sys.exit(1)
    print(f"{OK} All 4 required env vars present")
    print(f"    URL:              {values['SUPABASE_URL']}")
    print(f"    anon key:         {values['SUPABASE_ANON_KEY'][:16]}…")
    print(f"    service role key: {values['SUPABASE_SERVICE_ROLE_KEY'][:16]}…")
    print(f"    DB URL:           postgresql://postgres:***@{values['SUPABASE_DB_URL'].split('@')[1]}")
    print()
    return values


def check_rest(values: dict[str, str]) -> None:
    """Hit the REST API with the service-role key to confirm auth + connectivity."""
    print(f"{ARROW} Testing Supabase REST API (supabase-py)")
    try:
        from supabase import create_client
    except ImportError:
        print(f"{FAIL} supabase package not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    try:
        client = create_client(values["SUPABASE_URL"], values["SUPABASE_SERVICE_ROLE_KEY"])
        # List storage buckets — always works, even on a fresh project.
        buckets = client.storage.list_buckets()
        print(f"{OK} REST connected. Buckets: {[b.name for b in buckets] or '(none yet)'}")
    except Exception as e:
        print(f"{FAIL} REST call failed: {type(e).__name__}: {e}")
        sys.exit(1)
    print()


def check_postgres(values: dict[str, str]) -> None:
    """Open a direct Postgres connection, do a round-trip, and a create/drop dance."""
    print(f"{ARROW} Testing direct Postgres connection (psycopg)")
    try:
        import psycopg
    except ImportError:
        print(f"{FAIL} psycopg not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    try:
        with psycopg.connect(values["SUPABASE_DB_URL"]) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                print(f"{OK} Connected: {version.split(',')[0]}")

                # Round-trip on a temp table to verify write access.
                cur.execute("CREATE TEMP TABLE _smoketest (n int)")
                cur.execute("INSERT INTO _smoketest (n) VALUES (42)")
                cur.execute("SELECT n FROM _smoketest")
                row = cur.fetchone()
                if row and row[0] == 42:
                    print(f"{OK} Write + read round-trip successful (SELECT returned 42)")
                else:
                    print(f"{FAIL} Unexpected round-trip result: {row}")
                    sys.exit(1)
                # Temp table auto-drops on connection close.
    except Exception as e:
        print(f"{FAIL} Postgres connection failed: {type(e).__name__}: {e}")
        sys.exit(1)
    print()


def main() -> None:
    print()
    print("━━━ Supabase smoke test ━━━\n")
    values = check_env()
    check_rest(values)
    check_postgres(values)
    print(f"{OK} All checks passed. You're connected to Supabase.")
    print()


if __name__ == "__main__":
    main()
