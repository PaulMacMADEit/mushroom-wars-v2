-- Interactive "Play" RPC - lets the anon dashboard queue a 1-game match
-- without exposing the matches table for general INSERT.
--
-- Apply with: psql "$SUPABASE_DB_URL" -f infra/rpc_interactive_play.sql

BEGIN;

CREATE OR REPLACE FUNCTION queue_interactive_play(
  p_run_a      UUID,
  p_run_b      UUID,
  p_level_name TEXT
) RETURNS UUID
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
  v_sim_id TEXT;
  v_match_id UUID;
BEGIN
  -- Validate both runs exist.
  SELECT simulator_id INTO v_sim_id FROM runs WHERE id = p_run_a;
  IF v_sim_id IS NULL THEN
    RAISE EXCEPTION 'run A % not found', p_run_a USING ERRCODE = 'P0002';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM runs WHERE id = p_run_b) THEN
    RAISE EXCEPTION 'run B % not found', p_run_b USING ERRCODE = 'P0002';
  END IF;

  -- Whitelist level names - prevents abuse via weird generated names.
  IF p_level_name !~ '^(crossroads_6|random_\d+_\d+|asym_\d+_\d+)$' THEN
    RAISE EXCEPTION 'invalid level_name: %', p_level_name USING ERRCODE = '22023';
  END IF;

  INSERT INTO matches (
    project, description, model_a_run_id, model_b_run_id,
    simulator_id, games_planned, status, summary
  ) VALUES (
    'mushroom-wars', 'interactive-play',
    p_run_a, p_run_b, v_sim_id, 1, 'queued',
    jsonb_build_object('level_name', p_level_name)
  ) RETURNING id INTO v_match_id;

  RETURN v_match_id;
END;
$$;

-- Anon + authenticated can call this RPC. The function is SECURITY DEFINER
-- so it runs with the function owner's privileges and bypasses RLS on INSERT.
GRANT EXECUTE ON FUNCTION queue_interactive_play(UUID, UUID, TEXT) TO anon, authenticated;

COMMIT;
