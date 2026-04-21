-- mushroom-wars-v2 RPCs
-- Source of truth: ARCHITECTURE.md §7 (training pipeline)
-- Apply with: psql "$SUPABASE_DB_URL" -f infra/rpc.sql

BEGIN;

-- Atomically claim the oldest queued run for a given project/machine.
-- Returns the claimed row, or zero rows if nothing is queued.
-- Uses SKIP LOCKED so concurrent workers never fight over the same row.
CREATE OR REPLACE FUNCTION claim_next_run(
  p_project TEXT,
  p_machine TEXT
) RETURNS SETOF runs AS $$
  UPDATE runs
     SET status     = 'running',
         machine    = p_machine,
         started_at = now()
   WHERE id = (
     SELECT id FROM runs
      WHERE project = p_project
        AND status  = 'queued'
      ORDER BY queued_at
      FOR UPDATE SKIP LOCKED
      LIMIT 1
   )
   RETURNING *;
$$ LANGUAGE sql;

COMMIT;
