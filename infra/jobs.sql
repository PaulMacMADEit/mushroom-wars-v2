-- mushroom-wars-v2 admin jobs
-- Apply with: psql "$SUPABASE_DB_URL" -f infra/jobs.sql
--
-- Admin jobs (rerate, future bench_eval, etc.) ride the existing `runs` table
-- so they show up in the Active / Upcoming / Completed sections of the
-- dashboard like any other run. Discriminator: simulator_id='admin'.
-- hyperparams.kind names the specific job ('rerate' for now).
--
-- Sentinel rows in `models` and `simulators` (id='admin') are pre-seeded.

BEGIN;

-- ---------------------------------------------------------------------------
-- claim_next_job: like claim_next_run but filtered to admin jobs only.
-- Used by the worker when matches_only=true so it can pick up rerate work
-- without touching real training queues.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION claim_next_job(
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
        AND simulator_id = 'admin'
      ORDER BY queued_at
      FOR UPDATE SKIP LOCKED
      LIMIT 1
   )
   RETURNING *;
$$ LANGUAGE sql;

-- ---------------------------------------------------------------------------
-- Anon INSERT policy on runs — admin jobs only.
-- The dashboard's anon key can queue rerate / bench_eval jobs but cannot
-- queue real training runs (those still go through service_role).
-- ---------------------------------------------------------------------------
DROP POLICY IF EXISTS anon_insert_admin_jobs ON runs;
CREATE POLICY anon_insert_admin_jobs ON runs FOR INSERT TO anon, authenticated
  WITH CHECK (
    simulator_id = 'admin'
    AND model_id = 'admin'
    AND status = 'queued'
    AND project = 'mushroom-wars'
  );

GRANT INSERT ON runs TO anon, authenticated;

COMMIT;
