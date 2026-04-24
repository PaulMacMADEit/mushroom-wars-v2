-- Per-machine pause flag. Worker polls this each idle tick; when paused,
-- it skips claim_one and keeps sleeping. Lets the dashboard turn workers
-- on/off without SSH/webhooks/keys.

BEGIN;

CREATE TABLE IF NOT EXISTS worker_state (
    machine      TEXT PRIMARY KEY,
    paused       BOOLEAN NOT NULL DEFAULT false,
    matches_only BOOLEAN NOT NULL DEFAULT false,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE worker_state ADD COLUMN IF NOT EXISTS matches_only BOOLEAN NOT NULL DEFAULT false;

-- RLS: let anon read + write (no secrets here; same posture as the rest of the
-- project's public dashboards).
ALTER TABLE worker_state ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS worker_state_read ON worker_state;
CREATE POLICY worker_state_read ON worker_state FOR SELECT USING (true);

DROP POLICY IF EXISTS worker_state_write ON worker_state;
CREATE POLICY worker_state_write ON worker_state FOR ALL USING (true) WITH CHECK (true);

-- Seed both known machines as unpaused so nothing breaks on first deploy.
INSERT INTO worker_state (machine, paused) VALUES
  ('PaulLinux',             false),
  ('Pauls-MacBook-Air.local', false)
ON CONFLICT (machine) DO NOTHING;

COMMIT;
