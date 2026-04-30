"""PyTorch RL code. Depends on `sim/`; never the other way around.

v10 encoder (1008 dims): drops dead `type_oh` block, adds prod_rate /
wasted_prod / total_alive / share_live / reward_delta globals plus
HISTORY_K=5 own/opponent action history, plus per-building event-explicit
features (hostile_landed / friendly_landed / ownership_changed_this_interval)
to fix close-map signal loss when MIN_TRAVEL_TICKS < DECISION_INTERVAL_TICKS.
See `training/encoder.py` docstring for the full diff vs v9.0.
"""
