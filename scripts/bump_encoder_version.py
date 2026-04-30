"""Automate the encoder version bump workflow.

The 5-step migration documented in `training/encoders/__init__.py` is
mechanical: snapshot the current live encoder into a versioned module,
register it, bump the constant, then edit `training/encoder.py` freely.
This script does the first three steps. After it runs, edit the live
encoder, re-run tests, and commit.

Why this exists: every time you forget step 1 (snapshot BEFORE editing
the live encoder), you've lost the prior version forever — the registry
falls back to importing it from `training.encoder`, but `training.encoder`
is now the new version. The script removes the chance of getting that
order wrong.

Usage:
  python scripts/bump_encoder_version.py v11
  python scripts/bump_encoder_version.py v11 --dry-run

After it runs:
  1. Edit `training/encoder.py` and `training/encoder_jax.py` to add
     whatever the v11 change is (e.g. new feature columns).
  2. Update `OBS_DIM` accordingly inside `training/encoder.py`.
  3. Run `pytest tests/test_encoder_registry.py tests/test_encoder_parity.py`
     to confirm the registry still serves both versions and the new
     encoder's numpy/JAX paths agree.
  4. Register `v11-1024` (or whatever model_id the new shape requires)
     in the `models` table via a small `INSERT … ON CONFLICT DO NOTHING`
     migration script (see `tmp/register_v10_model.py` for the v10 example).
  5. Commit. The pre-bump version remains forever loadable through
     `training.encoders.get_encoder("v{prev}")`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

ENCODER_PY      = ROOT / "training" / "encoder.py"
ENCODER_JAX_PY  = ROOT / "training" / "encoder_jax.py"
ENCODERS_DIR    = ROOT / "training" / "encoders"
INIT_PY         = ENCODERS_DIR / "__init__.py"


_CURRENT_RE = re.compile(r'^CURRENT_ENCODER_VERSION\s*=\s*"([^"]+)"', re.MULTILINE)
_VERSION_LABEL_RE = re.compile(r"^v\d+(?:\.\d+)?$")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write(path: Path, content: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would write {path} ({len(content)} bytes)")
        return
    path.write_text(content, encoding="utf-8")
    print(f"[wrote] {path}")


def _current_version() -> str:
    """Read the value of `CURRENT_ENCODER_VERSION` from the live registry."""
    text = _read(INIT_PY)
    m = _CURRENT_RE.search(text)
    if not m:
        raise SystemExit(
            f"Could not find CURRENT_ENCODER_VERSION in {INIT_PY}. "
            f"The script's regex needs updating — check the line manually."
        )
    return m.group(1)


def _snapshot_numpy_encoder(prev_version: str, dry_run: bool) -> Path:
    """Copy the live `training/encoder.py` to `training/encoders/v{prev}.py`.

    No path-rewriting needed — the live numpy encoder doesn't import from
    `training.encoder` (that would be self-import); it's already self-
    contained.
    """
    src = _read(ENCODER_PY)
    dst_path = ENCODERS_DIR / f"{prev_version}.py"
    if dst_path.exists():
        raise SystemExit(
            f"{dst_path} already exists — refusing to overwrite. "
            f"Either the bump already happened or the version label is stale."
        )
    _write(dst_path, src, dry_run)
    return dst_path


def _snapshot_jax_encoder(prev_version: str, dry_run: bool) -> Path:
    """Copy `training/encoder_jax.py` to `training/encoders/v{prev}_jax.py`,
    redirecting its imports.

    The live `encoder_jax.py` imports its sibling constants (OBS_DIM,
    BUILDING_FEATS, …) from `training.encoder`. After the snapshot, those
    constants point at the *new* live encoder, not the snapshotted one,
    so the import has to be rewritten to read from
    `training.encoders.v{prev}` (the just-snapshotted numpy file).
    """
    src = _read(ENCODER_JAX_PY)
    src = src.replace(
        "from training.encoder import",
        f"from training.encoders.{prev_version} import",
    )
    dst_path = ENCODERS_DIR / f"{prev_version}_jax.py"
    if dst_path.exists():
        raise SystemExit(
            f"{dst_path} already exists — refusing to overwrite."
        )
    _write(dst_path, src, dry_run)
    return dst_path


def _redirect_prev_factory_to_snapshot(text: str, prev_version: str) -> str:
    """The existing `_build_v{prev}` reads from `training.encoder` /
    `training.encoder_jax` (the *live* encoder). After the bump those
    names will point at the new version, so we have to re-target the
    factory to the just-snapshotted module under `training.encoders.`.

    Idempotent: if the factory already imports from `training.encoders`,
    the script bails (a previous bump already redirected it).
    """
    safe_prev = prev_version.replace(".", "_")
    # Find the function header and the two `from training import …` lines,
    # tolerating any non-`from`-line junk (comments, blank lines, type hints)
    # in between.
    fn_re = re.compile(
        rf"(def _build_{safe_prev}\(\) -> EncoderEntry:\s*\n)"
        rf"((?:[^\n]*\n)*?)"  # tolerated body up to first `from` line
        rf"(    from training import encoder as[^\n]*\n)"
        rf"((?:[^\n]*\n)*?)"  # any (more) lines between the two imports
        rf"(    from training import encoder_jax as[^\n]*\n)",
        re.MULTILINE,
    )
    m = fn_re.search(text)
    if m is None:
        # Already-redirected (or hand-edited) — leave it alone.
        return text
    head, mid1, _live_np, mid2, _live_jax = m.groups()
    redirect = (
        f"{head}"
        f"{mid1}"
        f"    from training.encoders import {prev_version} as enc_prev\n"
        f"{mid2}"
        f"    from training.encoders import {prev_version}_jax as enc_prev_jax\n"
    )
    text = text[:m.start()] + redirect + text[m.end():]

    # The original factory body still references `enc_v{N}` / `enc_v{N}_jax`
    # local names. Rewrite those too — they're the only consumers of the
    # imported modules within the function. Standardising on `enc_prev` /
    # `enc_prev_jax` keeps the script generic across version names.
    safe_alias = prev_version.replace(".", "_")
    text = re.sub(
        rf"\benc_{safe_alias}\b", "enc_prev", text,
    )
    text = re.sub(
        rf"\benc_{safe_alias}_jax\b", "enc_prev_jax", text,
    )
    return text


def _add_builder_to_registry(prev_version: str, new_version: str, dry_run: bool) -> None:
    """Two coordinated edits in `training/encoders/__init__.py`:

    1. Redirect the existing `_build_v{prev}` factory from `training.encoder`
       (the *live* encoder, about to become v{new}) to the just-snapshotted
       `training.encoders.v{prev}` module — otherwise after Paul edits
       `training/encoder.py`, the v{prev} entry silently returns the v{new}
       encoder.
    2. Add a new `_build_v{new}` factory that reads from `training.encoder`
       (live), register it in `_BUILDERS`, and bump `CURRENT_ENCODER_VERSION`.
    """
    text = _read(INIT_PY)

    # ---- Step 1: redirect the old factory ------------------------------
    text = _redirect_prev_factory_to_snapshot(text, prev_version)

    # ---- Step 2a: insert the new factory above the _BUILDERS dict ------
    safe_new = new_version.replace(".", "_")
    new_factory = f'''

def _build_{safe_new}() -> EncoderEntry:
    # The active live encoder always lives at `training.encoder`. The
    # bump script will redirect this factory to a snapshot the next time
    # the version is bumped.
    from training import encoder as enc_live
    from training import encoder_jax as enc_live_jax
    return EncoderEntry(
        version="{new_version}",
        obs_dim=enc_live.OBS_DIM,
        encode=enc_live.encode_obs,
        encode_jax=enc_live_jax.encode_obs_batched_jit,
        description="Live encoder. Edit `training/encoder.py` to define {new_version}.",
    )
'''
    builders_anchor_re = re.compile(r"^_BUILDERS:[^=]*=\s*\{", re.MULTILINE)
    m = builders_anchor_re.search(text)
    if not m:
        raise SystemExit(f"Could not find `_BUILDERS:` declaration in {INIT_PY}.")
    text = text[:m.start()] + new_factory.rstrip() + "\n\n\n" + text[m.start():]

    # ---- Step 2b: add the new key to the _BUILDERS dict ----------------
    builders_dict_re = re.compile(
        r"(_BUILDERS:[^=]*=\s*\{)([^}]*)(\})",
        re.DOTALL,
    )
    m = builders_dict_re.search(text)
    if not m:
        raise SystemExit("Could not parse _BUILDERS dict shape.")
    head, body, tail = m.groups()
    new_entry = f'    "{new_version}":  _build_{safe_new},\n'
    if new_entry.strip() not in body:
        body = body.rstrip() + "\n" + new_entry
    text = text[:m.start()] + head + body + tail + text[m.end():]

    # ---- Step 3: bump CURRENT_ENCODER_VERSION --------------------------
    text, n = _CURRENT_RE.subn(
        f'CURRENT_ENCODER_VERSION = "{new_version}"', text,
    )
    if n != 1:
        raise SystemExit("Could not bump CURRENT_ENCODER_VERSION (regex matched != 1).")

    _write(INIT_PY, text, dry_run)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("new_version", help="The version label this bump promotes to (e.g. 'v11').")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without writing.")
    args = ap.parse_args()

    new_version = args.new_version.strip()
    if not _VERSION_LABEL_RE.match(new_version):
        raise SystemExit(
            f"Refusing to use {new_version!r} as a version label — expected e.g. 'v11' or 'v10.1'."
        )

    prev_version = _current_version()
    if prev_version == new_version:
        raise SystemExit(
            f"CURRENT_ENCODER_VERSION is already {new_version}; nothing to bump."
        )

    print(f"[bump] snapshotting current encoder ({prev_version}) into the registry, "
          f"bumping CURRENT_ENCODER_VERSION → {new_version}")

    np_path  = _snapshot_numpy_encoder(prev_version, args.dry_run)
    jax_path = _snapshot_jax_encoder(prev_version, args.dry_run)
    _add_builder_to_registry(prev_version, new_version, args.dry_run)

    print()
    print("[bump] done. Next:")
    print(f"  1. Edit `training/encoder.py` (and `encoder_jax.py`) — they're now the live {new_version} encoder.")
    print(f"  2. Bump `OBS_DIM` inside `training/encoder.py` if the shape changed.")
    print(f"  3. Run `pytest tests/test_encoder_registry.py tests/test_encoder_parity.py`.")
    print(f"  4. Register the new model_id (e.g. {new_version}-1024) in the `models` table — see")
    print(f"     `tmp/register_v10_model.py` for a worked example.")
    print(f"  5. Commit; the registry now keeps {prev_version} loadable forever via")
    print(f"     `training.encoders.get_encoder({prev_version!r})`.")


if __name__ == "__main__":
    main()
