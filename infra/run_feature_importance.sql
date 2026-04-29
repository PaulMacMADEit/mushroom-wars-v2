-- Per-run feature attribution: which named obs features drive the value head.
-- Computed offline by scripts/compute_attributions.py via Integrated Gradients
-- against a sample of states played by the run's policy.
--
-- Apply with: psql "$SUPABASE_DB_URL" -f infra/run_feature_importance.sql

BEGIN;

CREATE TABLE IF NOT EXISTS run_feature_importance (
  run_id          UUID        NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  feature_block   TEXT        NOT NULL,   -- 'globals' | 'building' | 'group'
  feature_index   INTEGER     NOT NULL,   -- ordering within block
  feature_name    TEXT        NOT NULL,
  ig_mean_abs     REAL        NOT NULL,   -- mean |IG| over sampled states (V head)
  ig_mean_signed  REAL        NOT NULL,   -- mean signed IG (direction)
  ig_std          REAL        NOT NULL,
  weight_l2       REAL        NOT NULL,   -- layer-1 weight L2-norm aggregated to feature
  n_states        INTEGER     NOT NULL,
  computed_at     TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (run_id, feature_block, feature_index)
);

CREATE INDEX IF NOT EXISTS run_feature_importance_run_idx
  ON run_feature_importance(run_id);

-- Anon dashboard read access (matches pattern in infra/rls.sql for runs table).
ALTER TABLE run_feature_importance ENABLE ROW LEVEL SECURITY;

DO $$ BEGIN
  CREATE POLICY run_feature_importance_anon_read
    ON run_feature_importance FOR SELECT
    USING (true);
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

COMMIT;
