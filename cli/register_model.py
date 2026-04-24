"""Register the current model into Supabase `models`.

Packages a descriptive `layers` JSON blob from the live ActorCritic
definition. The worker reconstructs the net from code (training/net.py) — the
layers JSON here is metadata for the dashboard, not a serialized graph.

Usage:
    python cli/register_model.py --id v9.0-smoke --what-changed "flat-head smoke net"
    python cli/register_model.py --id v9.0-smoke --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cli.db import PROJECT, connect
from sim.actions import ACTION_SPACE_SIZE
from training.encoder import OBS_DIM
from training.net import ActorCritic


def layers_blob(net: ActorCritic) -> dict:
    """Shape description of the net. Not load-bearing — documentation only."""
    layers = []
    for name, module in net.named_modules():
        cls = type(module).__name__
        if cls == "Linear":
            layers.append({"name": name, "type": cls,
                           "in_features": module.in_features,
                           "out_features": module.out_features})
        elif cls == "ReLU":
            layers.append({"name": name, "type": cls})
        elif cls == "Embedding":
            layers.append({"name": name, "type": cls,
                           "num_embeddings": module.num_embeddings,
                           "embedding_dim":  module.embedding_dim})
    return {
        "kind": "actor-critic-chained-heads",
        "layers": layers,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", default="v9.0-full")
    ap.add_argument("--name", default="v9.0 full encoder, chained src/type/tgt heads")
    ap.add_argument("--what-changed",
                    default="Full v9.0 encoder (1002 dims) + chained heads (ARCHITECTURE §9.4): source (32) → type (5, incl. noop) | src → target (32) | src. ~17x smaller policy-head param count than flat 4097.")
    ap.add_argument("--parent-model", default="v9.0-enc-full")
    ap.add_argument("--obs-encoder", default="training.encoder.encode_obs (v9.0 full, 1002 dims)")
    ap.add_argument("--action-decoder", default="sim.actions.decode (flat 4097 env-side; factored src/type/tgt on the net side)")
    ap.add_argument("--keep-weights", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--body-dim", type=int, default=128,
                    help="ActorCritic trunk width. Must match build_net_for_model's dispatch.")
    args = ap.parse_args()

    net = ActorCritic(body_dim=args.body_dim)
    layers = layers_blob(net)
    total_params = sum(p.numel() for p in net.parameters())

    row = (
        args.id, PROJECT, args.name, args.parent_model, args.what_changed,
        OBS_DIM, ACTION_SPACE_SIZE, args.obs_encoder, args.action_decoder,
        json.dumps(layers), total_params, args.keep_weights,
    )

    with connect() as conn:
        with conn.cursor() as cur:
            if args.force:
                cur.execute("""
                    INSERT INTO models (id, project, name, parent_model, what_changed,
                                        obs_size, num_actions, obs_encoder, action_decoder,
                                        layers, total_params, keep_weights)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        name           = EXCLUDED.name,
                        parent_model   = EXCLUDED.parent_model,
                        what_changed   = EXCLUDED.what_changed,
                        obs_size       = EXCLUDED.obs_size,
                        num_actions    = EXCLUDED.num_actions,
                        obs_encoder    = EXCLUDED.obs_encoder,
                        action_decoder = EXCLUDED.action_decoder,
                        layers         = EXCLUDED.layers,
                        total_params   = EXCLUDED.total_params,
                        keep_weights   = EXCLUDED.keep_weights
                """, row)
            else:
                cur.execute("""
                    INSERT INTO models (id, project, name, parent_model, what_changed,
                                        obs_size, num_actions, obs_encoder, action_decoder,
                                        layers, total_params, keep_weights)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, row)
            affected = cur.rowcount
        conn.commit()

    if affected == 0 and not args.force:
        print(f"model {args.id!r} already exists (use --force to overwrite).")
    else:
        print(f"registered model {args.id!r}: obs={OBS_DIM}, actions={ACTION_SPACE_SIZE}, params={total_params:,}")


if __name__ == "__main__":
    main()
