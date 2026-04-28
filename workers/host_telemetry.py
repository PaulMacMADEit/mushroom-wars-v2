"""One-shot host-level telemetry sample → Supabase host_telemetry.

Fired every minute by a systemd timer. Independent of any worker process or
training run — purely a "is this box being used at all?" sanity check that
records host CPU/RAM and (if NVML is available) GPU 0 stats.

Usage (from a systemd timer or cron):
    .venv/bin/python -m workers.host_telemetry

Reads SUPABASE_URL + SUPABASE_SERVICE_ROLE_KEY from process env (the systemd
unit sources the repo's .env via `EnvironmentFile=`).

Designed to be cheap and crash-tolerant:
  - All fields except `machine` and `ts` are nullable; missing data → NULL
  - Network/DB failure prints to stderr and exits non-zero (systemd will log it)
  - CPU sampling uses a 1.0s window, so the script takes ~1.5s wall total
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import psutil
from dotenv import load_dotenv

# Load repo-root .env so SUPABASE_* are available the same way as the worker.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

try:
    import pynvml
    _NVML_OK = True
except Exception:
    pynvml = None
    _NVML_OK = False


def _round(v, n=1):
    return None if v is None else round(float(v), n)


def collect() -> dict:
    """Take one host-level sample. All fields nullable except machine."""
    sample: dict = {"machine": socket.gethostname()}

    # CPU: 1s window so we get a real reading (not the priming-call 0.0).
    sample["cpu_pct"] = _round(psutil.cpu_percent(interval=1.0), 1)

    vm = psutil.virtual_memory()
    sample["mem_used_gib"] = _round(vm.used / (1024 ** 3), 2)
    sample["mem_total_gib"] = _round(vm.total / (1024 ** 3), 2)

    try:
        sample["load1"] = _round(os.getloadavg()[0], 2)
    except Exception:
        pass

    if _NVML_OK:
        try:
            pynvml.nvmlInit()
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                sample["gpu_sm_pct"] = _round(util.gpu, 1)
                sample["gpu_mem_pct"] = _round(util.memory, 1)
                sample["vram_used_gib"] = _round(mem.used / (1024 ** 3), 2)
                sample["vram_total_gib"] = _round(mem.total / (1024 ** 3), 2)
                try:
                    sample["gpu_power_w"] = _round(
                        pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 1
                    )
                except Exception:
                    pass
                try:
                    sample["gpu_temp_c"] = _round(
                        pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                        1,
                    )
                except Exception:
                    pass
            finally:
                pynvml.nvmlShutdown()
        except Exception as e:
            print(f"[host_telemetry] nvml failed: {e}", file=sys.stderr)

    return sample


def insert(sample: dict) -> None:
    """POST one row to Supabase via PostgREST + service-role key."""
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

    import urllib.error
    import urllib.request
    import json

    endpoint = f"{url.rstrip('/')}/rest/v1/host_telemetry"
    body = json.dumps(sample).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status not in (201, 204):
                raise RuntimeError(f"unexpected status {resp.status}: {resp.read()[:200]!r}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from e


def main() -> int:
    sample = collect()
    insert(sample)
    print(
        f"[host_telemetry] {sample['machine']} cpu={sample.get('cpu_pct')}% "
        f"gpu={sample.get('gpu_sm_pct')}% vram={sample.get('vram_used_gib')}/"
        f"{sample.get('vram_total_gib')}GiB"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"[host_telemetry] error: {e}", file=sys.stderr)
        sys.exit(1)
