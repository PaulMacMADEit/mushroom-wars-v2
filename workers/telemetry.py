"""Per-run resource telemetry — CPU, RAM, GPU, VRAM, games/sec.

Starts a background polling thread at training start, stops at end, returns
a compact summary ready to embed in the run's result JSON. Graceful if
pynvml isn't available (CPU-only hosts).

Usage:
    from workers.telemetry import ResourceSampler
    sampler = ResourceSampler(worker_pid=os.getpid(), interval_s=2.0)
    sampler.start()
    ... training ...
    summary = sampler.stop()   # dict, safe to json.dumps
"""

from __future__ import annotations

import os
import threading
import time
from statistics import mean, median
from typing import Optional

import psutil

try:
    import pynvml
    _NVML_OK = True
except Exception:
    pynvml = None
    _NVML_OK = False


def _percentile(values: list[float], p: float) -> float:
    """Cheap percentile without numpy. p in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


class ResourceSampler:
    """Background polling thread that captures CPU/RAM/GPU stats periodically.

    Counts the main worker process + all children (async vec envs live in
    subprocs, so reading only the main proc misses ~95% of sim CPU load).
    """

    def __init__(self, worker_pid: Optional[int] = None, interval_s: float = 2.0):
        self.worker_pid = worker_pid or os.getpid()
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: list[dict] = []
        self._started_at: Optional[float] = None

        # NVML setup: one handle for GPU 0 (worker is pinned to a single GPU
        # in all current deployments). Fail silently if unavailable.
        self._nvml_handle = None
        if _NVML_OK:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            except Exception:
                self._nvml_handle = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ResourceSampler")
        self._thread.start()

    def stop(self) -> dict:
        if self._thread is None:
            return {"error": "sampler not started", "samples": 0}
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        if self._nvml_handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return self._summarize()

    # --- internals ---

    def _run(self) -> None:
        try:
            main_proc = psutil.Process(self.worker_pid)
        except psutil.NoSuchProcess:
            return

        # psutil.cpu_percent() returns 0 on first call — prime it for the
        # main proc and warmed-up children.
        _ = main_proc.cpu_percent(interval=None)

        while not self._stop_event.is_set():
            ts = time.time() - (self._started_at or time.time())  # seconds since start
            sample: dict = {"t": round(ts, 2)}

            # CPU / RAM aggregated across main proc + living children.
            try:
                procs = [main_proc] + main_proc.children(recursive=True)
                cpu_total = 0.0
                rss_total = 0
                for p in procs:
                    try:
                        cpu_total += p.cpu_percent(interval=None)
                        rss_total += p.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                # cpu_percent() returns per-core % summed — divide by logical
                # core count to get 0-100 scale for "all cores combined".
                n_cores = psutil.cpu_count(logical=True) or 1
                sample["cpu_pct_sum"] = round(cpu_total, 1)         # e.g. 800 = 8 cores @ 100%
                sample["cpu_pct_norm"] = round(cpu_total / n_cores, 1)  # 0-100 aggregate
                sample["rss_gib"] = round(rss_total / (1024 ** 3), 2)
                sample["n_procs"] = len(procs)
            except psutil.NoSuchProcess:
                break

            # Host-wide CPU + RAM for cross-reference (load avg etc).
            try:
                sample["host_load1"] = round(os.getloadavg()[0], 1)
                sample["host_mem_used_gib"] = round(psutil.virtual_memory().used / (1024 ** 3), 2)
            except Exception:
                pass

            # GPU: best-effort via NVML.
            if self._nvml_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    sample["gpu_sm_pct"] = int(util.gpu)
                    sample["gpu_mem_pct"] = int(util.memory)
                    sample["vram_used_gib"] = round(mem.used / (1024 ** 3), 2)
                    sample["vram_total_gib"] = round(mem.total / (1024 ** 3), 2)
                    try:
                        sample["gpu_power_w"] = round(
                            pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0, 1
                        )
                    except Exception:
                        pass
                except Exception:
                    # NVML can flake mid-run — don't crash the sampler.
                    pass

            self._samples.append(sample)

            # Sleep in 0.2s chunks so stop() is responsive.
            slept = 0.0
            while slept < self.interval_s and not self._stop_event.is_set():
                time.sleep(0.2)
                slept += 0.2

    def _summarize(self) -> dict:
        """Compact statistical summary of the sample stream."""
        if not self._samples:
            return {"samples": 0}

        def _agg(key: str) -> dict:
            vals = [s[key] for s in self._samples if key in s]
            if not vals:
                return {}
            return {
                "mean": round(mean(vals), 2),
                "median": round(median(vals), 2),
                "max": round(max(vals), 2),
                "p95": round(_percentile(vals, 95), 2),
            }

        summary: dict = {
            "samples": len(self._samples),
            "duration_s": round(time.time() - (self._started_at or 0), 1),
            "interval_s": self.interval_s,
            "cpu_pct_sum": _agg("cpu_pct_sum"),
            "cpu_pct_norm": _agg("cpu_pct_norm"),
            "rss_gib": _agg("rss_gib"),
            "n_procs": _agg("n_procs"),
            "host_load1": _agg("host_load1"),
            "host_mem_used_gib": _agg("host_mem_used_gib"),
        }
        if any("gpu_sm_pct" in s for s in self._samples):
            summary["gpu_sm_pct"] = _agg("gpu_sm_pct")
            summary["gpu_mem_pct"] = _agg("gpu_mem_pct")
            summary["vram_used_gib"] = _agg("vram_used_gib")
            summary["gpu_power_w"] = _agg("gpu_power_w")
            # Static — first sample's reading is fine.
            if "vram_total_gib" in self._samples[0]:
                summary["vram_total_gib"] = self._samples[0]["vram_total_gib"]

        # Bottleneck heuristic: if CPU norm > 80% and GPU SM < 25%, we're CPU-bound.
        cpu_mean = summary["cpu_pct_norm"].get("mean", 0) if summary["cpu_pct_norm"] else 0
        gpu_mean = summary.get("gpu_sm_pct", {}).get("mean", 0) if summary.get("gpu_sm_pct") else 0
        if cpu_mean >= 80 and gpu_mean < 25:
            summary["bottleneck"] = "cpu"
        elif gpu_mean >= 80 and cpu_mean < 50:
            summary["bottleneck"] = "gpu"
        elif cpu_mean >= 60 and gpu_mean >= 60:
            summary["bottleneck"] = "balanced"
        else:
            summary["bottleneck"] = "neither_saturated"
        return summary
