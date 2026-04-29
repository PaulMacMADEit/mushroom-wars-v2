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

-- Filter admin jobs out of the training-claim path so an OLD worker (one
-- not yet upgraded to the job dispatcher) can't accidentally claim a
-- rerate row, fail to build_net_for_model('admin'), and mark it failed.
-- New workers go through claim_one then fall back to claim_one_job, so
-- full-mode workers still pick up admin work.
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
        AND simulator_id <> 'admin'
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

-- The runs_bump_counts and runs_update_best_rate triggers UPDATE the
-- models / simulators tables on INSERT. Anon doesn't have UPDATE on
-- those tables (RLS revoke for safety), so without SECURITY DEFINER
-- the trigger fires as anon and fails with "permission denied for
-- table models". Run them as the postgres owner instead.
ALTER FUNCTION runs_bump_counts()      SECURITY DEFINER;
ALTER FUNCTION runs_update_best_rate() SECURITY DEFINER;

COMMIT;
