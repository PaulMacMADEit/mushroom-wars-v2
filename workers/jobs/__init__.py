"""Admin job handlers (rerate, future bench_eval, etc.).

Admin jobs ride the existing `runs` table with simulator_id='admin'. The
discriminator inside hyperparams is `kind` ('rerate' for now). The worker
dispatches via `dispatch(job, ...)`.
"""
from . import rerate
from . import rerate_one
from . import rerate_full

HANDLERS = {
    "rerate":      rerate.handle,
    "rerate_one":  rerate_one.handle,
    "rerate_full": rerate_full.handle,
}


def dispatch(job: dict, mark_done_fn, mark_failed_fn) -> None:
    """Look up the handler by hyperparams.kind and run it.

    Raises KeyError if the kind is unknown — caller should mark_failed().
    """
    kind = (job.get("hyperparams") or {}).get("kind")
    if kind not in HANDLERS:
        raise KeyError(f"unknown job kind: {kind!r} (known: {list(HANDLERS)})")
    HANDLERS[kind](job, mark_done_fn, mark_failed_fn)
