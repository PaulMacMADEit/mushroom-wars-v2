-- Attribution jobs queue + RPCs.
--
-- Dashboard inserts via SECURITY-DEFINER RPC (queue_attribution_job).
-- Workers claim atomically via claim_next_attribution_job. Same shape as
-- claim_next_run.
--
-- Apply with: psql "$SUPABASE_DB_URL" -f infra/attribution_jobs.sql

BEGIN;

CREATE TABLE IF NOT EXISTS attribution_jobs (
  id           UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  project      TEXT        NOT NULL DEFAULT 'mushroom-wars',
  run_id       UUID        NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  status       TEXT        NOT NULL DEFAULT 'queued',
  level        TEXT        NOT NULL,
  n_games      INTEGER     NOT NULL DEFAULT 16,
  ig_steps     INTEGER     NOT NULL DEFAULT 50,
  max_states   INTEGER     NOT NULL DEFAULT 2000,
  machine      TEXT,
  queued_at    TIMESTAMPTZ DEFAULT now(),
  started_at   TIMESTAMPTZ,
  finished_at  TIMESTAMPTZ,
  error        TEXT,
  n_states     INTEGER,
  CONSTRAINT attribution_jobs_status_check CHECK (
    status IN ('queued', 'running', 'done', 'failed')
  )
);

CREATE INDEX IF NOT EXISTS attribution_jobs_run_idx
  ON attribution_jobs(run_id);
CREATE INDEX IF NOT EXISTS attribution_jobs_queued_idx
  ON attribution_jobs(status, queued_at);

ALTER TABLE attribution_jobs ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  CREATE POLICY anon_read_attribution_jobs
    ON attribution_jobs FOR SELECT TO anon, authenticated USING (true);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

-- ---------------------------------------------------------------------------
-- Atomic claim — same shape as claim_next_run. Workers poll this; first
-- claimant wins, others see no rows.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION claim_next_attribution_job(
  p_project TEXT,
  p_machine TEXT
) RETURNS SETOF attribution_jobs AS $$
  UPDATE attribution_jobs
     SET status     = 'running',
         machine    = p_machine,
         started_at = now()
   WHERE id = (
     SELECT id FROM attribution_jobs
      WHERE project = p_project
        AND status  = 'queued'
      ORDER BY queued_at
      FOR UPDATE SKIP LOCKED
      LIMIT 1
   )
   RETURNING *;
$$ LANGUAGE sql;

-- ---------------------------------------------------------------------------
-- Anon-callable queue RPC. Validates run exists + level matches the
-- whitelist used by queue_interactive_play.
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION queue_attribution_job(
  p_run_id  UUID,
  p_level   TEXT,
  p_n_games INTEGER
) RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_job_id UUID;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM runs WHERE id = p_run_id) THEN
    RAISE EXCEPTION 'run % not found', p_run_id USING ERRCODE = 'P0002';
  END IF;

  IF p_level !~ '^(crossroads_6|random_\d+_\d+|random_close_\d+_\d+|asym_\d+_\d+)$' THEN
    RAISE EXCEPTION 'invalid level: %', p_level USING ERRCODE = '22023';
  END IF;

  IF p_n_games < 1 OR p_n_games > 256 THEN
    RAISE EXCEPTION 'n_games out of range (1-256): %', p_n_games USING ERRCODE = '22023';
  END IF;

  INSERT INTO attribution_jobs (run_id, level, n_games)
  VALUES (p_run_id, p_level, p_n_games)
  RETURNING id INTO v_job_id;

  RETURN v_job_id;
END;
$$;

GRANT EXECUTE ON FUNCTION queue_attribution_job(UUID, TEXT, INTEGER)
  TO anon, authenticated;

COMMIT;
