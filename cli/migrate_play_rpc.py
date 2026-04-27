"""Update queue_interactive_play RPC: accept games + target_machine.

Old signature: queue_interactive_play(p_run_a, p_run_b, p_level_name) -> uuid
              Always queued 1 game with no machine routing.

New signature: queue_interactive_play(p_run_a, p_run_b, p_level_name,
                                      p_games int default 5,
                                      p_target_machine text default null) -> uuid
              Queues N games; optionally pins to a specific worker hostname
              via summary.target_machine (workers filter by this).

Idempotent — CREATE OR REPLACE.

Usage:
    python cli/migrate_play_rpc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import connect


SQL = r"""
CREATE OR REPLACE FUNCTION public.queue_interactive_play(
    p_run_a uuid,
    p_run_b uuid,
    p_level_name text,
    p_games int DEFAULT 5,
    p_target_machine text DEFAULT NULL
)
RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = 'public'
AS $function$
DECLARE
  v_sim_id TEXT;
  v_match_id UUID;
  v_summary JSONB;
BEGIN
  -- Validate both runs exist.
  SELECT simulator_id INTO v_sim_id FROM runs WHERE id = p_run_a;
  IF v_sim_id IS NULL THEN
    RAISE EXCEPTION 'run A % not found', p_run_a USING ERRCODE = 'P0002';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM runs WHERE id = p_run_b) THEN
    RAISE EXCEPTION 'run B % not found', p_run_b USING ERRCODE = 'P0002';
  END IF;

  -- Whitelist level names — prevents abuse via weird generated names.
  IF p_level_name !~ '^(crossroads_6|random_close_\d+_\d+|random_\d+_\d+|asym_\d+_\d+)$' THEN
    RAISE EXCEPTION 'invalid level_name: %', p_level_name USING ERRCODE = '22023';
  END IF;

  IF p_games < 1 OR p_games > 100 THEN
    RAISE EXCEPTION 'p_games out of range (1..100): %', p_games USING ERRCODE = '22023';
  END IF;

  v_summary := jsonb_build_object('level_name', p_level_name);
  IF p_target_machine IS NOT NULL THEN
    v_summary := v_summary || jsonb_build_object('target_machine', p_target_machine);
  END IF;

  INSERT INTO matches (
    project, description, model_a_run_id, model_b_run_id,
    simulator_id, games_planned, status, summary
  ) VALUES (
    'mushroom-wars', 'interactive-play',
    p_run_a, p_run_b, v_sim_id, p_games, 'queued',
    v_summary
  ) RETURNING id INTO v_match_id;

  RETURN v_match_id;
END;
$function$;

-- Make sure the anon/authenticated roles can call it (matching the original grant).
GRANT EXECUTE ON FUNCTION public.queue_interactive_play(uuid, uuid, text, int, text) TO anon, authenticated;
"""


def main():
    print("[migrate_play_rpc] applying queue_interactive_play overload (5 args)…")
    with connect() as c:
        with c.cursor() as cur:
            cur.execute(SQL)
        c.commit()
    print("[migrate_play_rpc] done")


if __name__ == "__main__":
    main()
