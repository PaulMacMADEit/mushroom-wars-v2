"""Supabase Storage upload/download helpers.

Buckets are provisioned in Phase 0:
  - models   — private — weights.pt, optimizer.pt, obs_norm.pt
  - logs     — private — metrics.json, full log files
  - replays  — private — action log artifacts (future)

We store **relative paths** (not signed URLs) in `runs.{weights_url, ...}`.
Signed URLs are short-lived; paths are permanent. The dashboard mints signed
URLs on demand from the path. Callers do:

    storage.upload("models", f"{run_id}/weights.pt", local_path)
    url = storage.signed_url("models", f"{run_id}/weights.pt")   # dashboard-side
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client


_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


_client_cache: Client | None = None


def client() -> Client:
    """Lazy, cached service-role client. Workers need service-role to write."""
    global _client_cache
    if _client_cache is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        _client_cache = create_client(url, key)
    return _client_cache


def upload(bucket: str, key: str, local_path: str | Path, content_type: str | None = None) -> str:
    """Upload local file to `bucket/key`. Overwrites if it already exists.

    Returns the stored `bucket/key` reference (what goes in runs.*_url columns).
    """
    local_path = Path(local_path)
    if not local_path.is_file():
        raise FileNotFoundError(local_path)

    options: dict = {"upsert": "true"}
    if content_type:
        options["content-type"] = content_type

    with local_path.open("rb") as fh:
        data = fh.read()

    # supabase-py raises on error; no need to check the response here.
    client().storage.from_(bucket).upload(path=key, file=data, file_options=options)
    return f"{bucket}/{key}"


def signed_url(bucket: str, key: str, expires_in: int = 3600) -> str:
    """Mint a short-lived signed URL. Used by dashboard/download CLIs."""
    resp = client().storage.from_(bucket).create_signed_url(path=key, expires_in=expires_in)
    return resp["signedURL"] if isinstance(resp, dict) else resp.signed_url
