-- mushroom-wars-v2 Row Level Security
-- Apply with: psql "$SUPABASE_DB_URL" -f infra/rls.sql
--
-- Shape:
--   - anon          → SELECT only, via policies below
--   - authenticated → same as anon (we don't use auth for this project)
--   - service_role  → bypasses RLS entirely, as always in Supabase
--
-- Workers authenticate with SERVICE_ROLE_KEY so they keep writing freely.
-- The dashboard uses ANON_KEY so it's effectively read-only.

BEGIN;

-- ---------------------------------------------------------------------------
-- Enable RLS
-- ---------------------------------------------------------------------------

ALTER TABLE models         ENABLE ROW LEVEL SECURITY;
ALTER TABLE simulators     ENABLE ROW LEVEL SECURITY;
ALTER TABLE runs           ENABLE ROW LEVEL SECURITY;
ALTER TABLE matches        ENABLE ROW LEVEL SECURITY;
ALTER TABLE games          ENABLE ROW LEVEL SECURITY;
ALTER TABLE host_telemetry ENABLE ROW LEVEL SECURITY;

-- ---------------------------------------------------------------------------
-- Read policies — anon + authenticated get SELECT on everything.
-- ---------------------------------------------------------------------------

DROP POLICY IF EXISTS anon_read_models          ON models;
DROP POLICY IF EXISTS anon_read_simulators      ON simulators;
DROP POLICY IF EXISTS anon_read_runs            ON runs;
DROP POLICY IF EXISTS anon_read_matches         ON matches;
DROP POLICY IF EXISTS anon_read_games           ON games;
DROP POLICY IF EXISTS anon_read_host_telemetry  ON host_telemetry;
DROP POLICY IF EXISTS anon_delete_queued_runs   ON runs;
DROP POLICY IF EXISTS anon_delete_runs          ON runs;
DROP POLICY IF EXISTS anon_insert_queued_runs   ON runs;

CREATE POLICY anon_read_models         ON models         FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY anon_read_simulators     ON simulators     FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY anon_read_runs           ON runs           FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY anon_read_matches        ON matches        FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY anon_read_games          ON games          FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY anon_read_host_telemetry ON host_telemetry FOR SELECT TO anon, authenticated USING (true);

-- ---------------------------------------------------------------------------
-- Delete policy — anon may delete a run if its status is queued, running, or
-- failed. This wires up the dashboard's "×" buttons (index.html, both the
-- Upcoming tab and the Active/Completed table) without giving anon any other
-- write access. 'done' is excluded so finished, scored runs cannot be wiped
-- from the browser even by accident.
--
-- Worst-case blast radius if someone scrapes the public anon key: they can
-- drop pending/running/failed runs. Pending runs re-queue cheaply from
-- queue_karp_sweep.py / the cron-agent; failed rows are debris anyway; a
-- live running row only loses the in-flight worker's state — annoying, but
-- the worker writes a fresh run on next start. They still cannot touch any
-- 'done' row, and still cannot UPDATE/INSERT (revoked at the GRANT layer
-- below).
-- ---------------------------------------------------------------------------

CREATE POLICY anon_delete_runs ON runs
  FOR DELETE TO anon, authenticated
  USING (status IN ('queued', 'running', 'failed'));

-- ---------------------------------------------------------------------------
-- Insert policy — anon may queue a new run from the dashboard launch form.
-- The WITH CHECK constraints prevent abuse: only 'queued' status, only for
-- this project. Worst-case blast radius if anon key is scraped: someone
-- queues bogus runs. Workers will fail to load them and mark 'failed' —
-- annoying, not catastrophic.
-- ---------------------------------------------------------------------------

CREATE POLICY anon_insert_queued_runs ON runs
  FOR INSERT TO anon, authenticated
  WITH CHECK (
    status  = 'queued'
    AND project = 'mushroom-wars'
  );

-- ---------------------------------------------------------------------------
-- Revoke the dangerous default grants on the anon role.
--
-- Supabase's default is to grant all operations to anon + authenticated on
-- every table in `public`; RLS then filters SELECT/etc. Since we don't want
-- anon to INSERT/UPDATE/DELETE/TRUNCATE AT ALL, it's belt-and-braces to
-- revoke the grants even though there are no matching policies. If a future
-- RLS bypass bug or misconfiguration hits, we're still safe.
-- ---------------------------------------------------------------------------

REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON models         FROM anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON simulators     FROM anon, authenticated;
-- runs: keep UPDATE/TRUNCATE revoked, but leave DELETE + INSERT granted so
-- the anon_delete_queued_runs + anon_insert_queued_runs RLS policies above
-- can actually fire.
REVOKE UPDATE, TRUNCATE                 ON runs           FROM anon, authenticated;
GRANT  DELETE, INSERT                   ON runs           TO   anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON matches        FROM anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON games          FROM anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON host_telemetry FROM anon, authenticated;

COMMIT;
