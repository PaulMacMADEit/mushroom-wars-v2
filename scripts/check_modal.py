"""
Smoke-test Modal: define one function, run it remotely, verify it comes back.

Run:
    modal run scripts/check_modal.py
"""

from __future__ import annotations

import platform
import sys

import modal

app = modal.App("mushroom-wars-v2-smoke")


@app.function()
def remote_add(a: int, b: int) -> dict:
    import platform as p
    import sys as s
    return {
        "sum": a + b,
        "python": s.version.split()[0],
        "platform": p.platform(),
    }


@app.local_entrypoint()
def main():
    print(f"local  python={sys.version.split()[0]} platform={platform.platform()}")
    result = remote_add.remote(40, 2)
    print(f"remote python={result['python']} platform={result['platform']}")
    print(f"remote returned: 40 + 2 = {result['sum']}")
    assert result["sum"] == 42, f"expected 42, got {result['sum']}"
    print("OK: Modal round-trip works.")
