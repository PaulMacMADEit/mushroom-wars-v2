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

CREATE POLICY anon_read_models         ON models         FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY anon_read_simulators     ON simulators     FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY anon_read_runs           ON runs           FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY anon_read_matches        ON matches        FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY anon_read_games          ON games          FOR SELECT TO anon, authenticated USING (true);
CREATE POLICY anon_read_host_telemetry ON host_telemetry FOR SELECT TO anon, authenticated USING (true);

-- ---------------------------------------------------------------------------
-- Delete policy — anon may delete a run ONLY if it's still queued. This wires
-- up the dashboard's "× delete queued run" button (index.html → sb.from('runs')
-- .delete().eq('id',…).eq('status','queued')) without giving anon any other
-- write access. Worst-case blast radius if someone scrapes the public anon
-- key: they can drop pending sweep runs, which are easily re-queued via
-- queue_karp_sweep.py / the cron-agent. They CANNOT touch running/done runs
-- (USING filter blocks them at the RLS layer) and still cannot UPDATE/INSERT
-- (revoked at the GRANT layer below).
-- ---------------------------------------------------------------------------

CREATE POLICY anon_delete_queued_runs ON runs
  FOR DELETE TO anon, authenticated
  USING (status = 'queued');

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
-- runs: keep INSERT/UPDATE/TRUNCATE revoked, but leave DELETE granted so the
-- anon_delete_queued_runs RLS policy above can actually fire.
REVOKE INSERT, UPDATE, TRUNCATE         ON runs           FROM anon, authenticated;
GRANT  DELETE                           ON runs           TO   anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON matches        FROM anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON games          FROM anon, authenticated;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON host_telemetry FROM anon, authenticated;

COMMIT;
