"""Long-running daemons that execute queued Supabase runs.

Each worker:
  1. Calls `claim_next_run(project, machine)` to atomically grab a queued row.
  2. Constructs the training stack from the row's (model_id, simulator_id, hyperparams).
  3. Trains for the requested budget.
  4. Writes result + metrics back into the row and sets status='done'/'failed'.

Phase-3-minimum scope: no Storage uploads, no eval against champion pool, no
self-play. The worker writes summary metrics into `runs.result` (JSONB) and
that's enough to see the queue round-trip in the dashboard later.
"""
