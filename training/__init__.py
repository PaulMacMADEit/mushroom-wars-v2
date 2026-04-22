"""PyTorch RL code. Depends on `sim/`; never the other way around.

Phase 2 smoke scope is intentionally minimal — encoder is ~200 dims (not the
full v9.0 1150 spec) and the policy uses a flat 4097-way action head instead
of the chained source/type/target heads. Both upgrade paths are architectural
no-ops; this just gets the loop closed end-to-end.
"""
