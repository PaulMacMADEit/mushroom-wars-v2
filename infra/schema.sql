-- mushroom-wars-v2 schema
-- Source of truth: ARCHITECTURE.md §4 (data model)
-- Apply with: psql "$SUPABASE_DB_URL" -f infra/schema.sql
-- Idempotent: safe to re-run. Drops are explicit and ordered by dependency.

BEGIN;

-- ---------------------------------------------------------------------------
-- Tables
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS models (
  id             TEXT        PRIMARY KEY,
  project        TEXT        NOT NULL,
  name           TEXT,
  created_at     TIMESTAMPTZ DEFAULT now(),
  created_by     TEXT,
  parent_model   TEXT        REFERENCES models(id),
  what_changed   TEXT,
  obs_size       INTEGER     NOT NULL,
  num_actions    INTEGER     NOT NULL,
  obs_encoder    TEXT,
  action_decoder TEXT,
  layers         JSONB       NOT NULL,
  total_params   INTEGER,
  keep_weights   BOOLEAN     DEFAULT false,
  run_count      INTEGER     DEFAULT 0,
  best_rate      REAL,
  best_run_id    UUID
);

CREATE TABLE IF NOT EXISTS simulators (
  id             TEXT        PRIMARY KEY,
  project        TEXT        NOT NULL,
  name           TEXT,
  created_at     TIMESTAMPTZ DEFAULT now(),
  parent_sim     TEXT        REFERENCES simulators(id),
  what_changed   TEXT,
  git_sha        TEXT,
  features       JSONB,
  benchmark      JSONB,
  run_count      INTEGER     DEFAULT 0
);

CREATE TABLE IF NOT EXISTS runs (
  id                   UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id             TEXT        NOT NULL REFERENCES models(id),
  simulator_id         TEXT        NOT NULL REFERENCES simulators(id),
  project              TEXT        NOT NULL,
  label                TEXT        NOT NULL,
  description          TEXT,
  status               TEXT        NOT NULL,
  result               JSONB,
  error                TEXT,
  budget_ms            INTEGER     NOT NULL,
  seed                 TEXT,
  hyperparams          JSONB,

  parent_run_id        UUID        REFERENCES runs(id),
  is_continuation      BOOLEAN     DEFAULT false,
  cumulative_budget_ms BIGINT,

  games_played         INTEGER,
  training_games       INTEGER,
  eval_games           INTEGER,
  ticks_played         BIGINT,
  machine              TEXT        NOT NULL,
  launch_at            BIGINT      NOT NULL,
  queued_at            TIMESTAMPTZ DEFAULT now(),
  started_at           TIMESTAMPTZ,
  finished_at          TIMESTAMPTZ,
  wall_ms              INTEGER,
  weights_url          TEXT,
  optimizer_url        TEXT,
  obs_norm_url         TEXT,
  log_url              TEXT,
  keep_weights         BOOLEAN     DEFAULT false,

  CONSTRAINT runs_status_check CHECK (
    status IN ('queued', 'running', 'done', 'discarded', 'failed')
  )
);

-- Forward-declared FK from models.best_run_id → runs.id (models is created first, so
-- we add it after runs exists). NOT VALID avoids a full table scan on re-apply.
DO $$ BEGIN
  ALTER TABLE models
    ADD CONSTRAINT models_best_run_id_fkey
    FOREIGN KEY (best_run_id) REFERENCES runs(id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS matches (
  id             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  project        TEXT        NOT NULL,
  description    TEXT,
  model_a_run_id UUID        NOT NULL REFERENCES runs(id),
  model_b_run_id UUID        NOT NULL REFERENCES runs(id),
  simulator_id   TEXT        NOT NULL REFERENCES simulators(id),
  games_planned  INTEGER     NOT NULL,
  status         TEXT        NOT NULL,
  created_at     TIMESTAMPTZ DEFAULT now(),
  finished_at    TIMESTAMPTZ,
  summary        JSONB
);

CREATE TABLE IF NOT EXISTS games (
  id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
  match_id        UUID        REFERENCES matches(id) ON DELETE CASCADE,
  game_index      INTEGER     NOT NULL,
  seed            INTEGER     NOT NULL,
  map_name        TEXT,
  player_1_run_id UUID        NOT NULL REFERENCES runs(id),
  player_2_run_id UUID        NOT NULL REFERENCES runs(id),
  winner          TEXT,
  duration_ms     INTEGER,
  stats           JSONB,
  actions_url     TEXT,
  log_url         TEXT
);

-- Host-level resource samples, independent of runs. Written once a minute by
-- a systemd timer on each training box; the dashboard reads it as a sanity
-- "is the box being used at all" timeseries.
CREATE TABLE IF NOT EXISTS host_telemetry (
  ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
  machine         TEXT        NOT NULL,
  cpu_pct         REAL,
  mem_used_gib    REAL,
  mem_total_gib   REAL,
  load1           REAL,
  gpu_sm_pct      REAL,
  gpu_mem_pct     REAL,
  vram_used_gib   REAL,
  vram_total_gib  REAL,
  gpu_power_w     REAL,
  gpu_temp_c      REAL,
  PRIMARY KEY (machine, ts)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS runs_project_status_queued_idx ON runs(project, status, queued_at);
CREATE INDEX IF NOT EXISTS runs_model_status_idx          ON runs(model_id, status);
CREATE INDEX IF NOT EXISTS runs_simulator_idx             ON runs(simulator_id);
CREATE INDEX IF NOT EXISTS runs_machine_status_idx        ON runs(machine, status);
CREATE INDEX IF NOT EXISTS runs_parent_run_idx            ON runs(parent_run_id);

CREATE INDEX IF NOT EXISTS matches_project_status_idx     ON matches(project, status, created_at);
CREATE INDEX IF NOT EXISTS games_match_index_idx          ON games(match_id, game_index);

CREATE INDEX IF NOT EXISTS host_telemetry_machine_ts_idx  ON host_telemetry(machine, ts DESC);

-- ---------------------------------------------------------------------------
-- Triggers: maintain denormalized counts/bests on models + simulators
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION runs_bump_counts() RETURNS TRIGGER AS $$
BEGIN
  IF (TG_OP = 'INSERT') THEN
    UPDATE models     SET run_count = run_count + 1 WHERE id = NEW.model_id;
    UPDATE simulators SET run_count = run_count + 1 WHERE id = NEW.simulator_id;
    RETURN NEW;
  ELSIF (TG_OP = 'DELETE') THEN
    UPDATE models     SET run_count = GREATEST(run_count - 1, 0) WHERE id = OLD.model_id;
    UPDATE simulators SET run_count = GREATEST(run_count - 1, 0) WHERE id = OLD.simulator_id;
    RETURN OLD;
  END IF;
  RETURN NULL;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS runs_bump_counts_trg ON runs;
CREATE TRIGGER runs_bump_counts_trg
  AFTER INSERT OR DELETE ON runs
  FOR EACH ROW EXECUTE FUNCTION runs_bump_counts();

-- When a run finishes with a rate, update models.best_rate / best_run_id if higher.
CREATE OR REPLACE FUNCTION runs_update_best_rate() RETURNS TRIGGER AS $$
DECLARE
  new_rate REAL;
BEGIN
  IF NEW.status <> 'done' OR NEW.result IS NULL THEN
    RETURN NEW;
  END IF;

  new_rate := (NEW.result->>'rate')::REAL;
  IF new_rate IS NULL THEN
    RETURN NEW;
  END IF;

  UPDATE models
    SET best_rate   = new_rate,
        best_run_id = NEW.id
    WHERE id = NEW.model_id
      AND (best_rate IS NULL OR new_rate > best_rate);

  RETURN NEW;
END $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS runs_update_best_rate_trg ON runs;
CREATE TRIGGER runs_update_best_rate_trg
  AFTER INSERT OR UPDATE OF status, result ON runs
  FOR EACH ROW EXECUTE FUNCTION runs_update_best_rate();

COMMIT;
