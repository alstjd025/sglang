#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import urlopen


METRIC_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+'
    r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|NaN|[+-]?Inf)$'
)
LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_labels(raw: str) -> Dict[str, str]:
    if not raw:
        return {}
    labels: Dict[str, str] = {}
    for match in LABEL_RE.finditer(raw):
        key = match.group(1)
        value = bytes(match.group(2), "utf-8").decode("unicode_escape")
        labels[key] = value
    return labels


def parse_prometheus_metrics(text: str) -> Iterable[Tuple[str, Dict[str, str], float]]:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = METRIC_LINE_RE.match(line)
        if not m:
            continue
        name = m.group(1)
        labels = parse_labels(m.group(2) or "")
        value_str = m.group(3)
        try:
            value = float(value_str)
        except ValueError:
            continue
        yield name, labels, value


def fetch_text(url: str, timeout_s: float = 3.0) -> str:
    with urlopen(url, timeout=timeout_s) as resp:
        return resp.read().decode("utf-8", errors="replace")


def metric_selected(name: str, exact: List[str], prefixes: List[str]) -> bool:
    if name in exact:
        return True
    for base in exact:
        if name.startswith(base + "_"):
            return True
    for prefix in prefixes:
        if name.startswith(prefix):
            return True
    return False


class RuntimeMetricsCollector(threading.Thread):
    def __init__(
        self,
        role_endpoints: Dict[str, str],
        selectors: Dict[str, Dict[str, List[str]]],
        out_files: Dict[str, Path],
        gpu_file: Path,
        system_file: Path,
        interval_s: float,
        stop_event: threading.Event,
        timeout_s: float = 3.0,
    ):
        super().__init__(daemon=True)
        self.role_endpoints = role_endpoints
        self.selectors = selectors
        self.out_files = out_files
        self.gpu_file = gpu_file
        self.system_file = system_file
        self.interval_s = interval_s
        self.stop_event = stop_event
        self.timeout_s = timeout_s
        self._fps = {}
        self._gpu_fp = None
        self._system_fp = None
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None

        self._prev_cpu_total: Optional[int] = None
        self._prev_cpu_idle: Optional[int] = None

        self._state_lock = threading.Lock()
        self._latest_role_state: Dict[str, Dict[str, object]] = {}
        self._latest_gpu_state: Dict[str, object] = {
            "ts": None,
            "unix_ts": 0.0,
            "samples": [],
            "error": None if self._has_nvidia_smi else "nvidia-smi not found",
        }
        self._latest_system_state: Dict[str, object] = {
            "ts": None,
            "unix_ts": 0.0,
            "cpu_util_pct": None,
            "mem_total_mb": None,
            "mem_used_mb": None,
            "mem_available_mb": None,
            "mem_util_pct": None,
            "error": None,
        }

    def _open_files(self):
        for role, path in self.out_files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fps[role] = path.open("a", encoding="utf-8", buffering=1)
        self.gpu_file.parent.mkdir(parents=True, exist_ok=True)
        self._gpu_fp = self.gpu_file.open("a", encoding="utf-8", buffering=1)
        self.system_file.parent.mkdir(parents=True, exist_ok=True)
        self._system_fp = self.system_file.open("a", encoding="utf-8", buffering=1)

    def _close_files(self):
        for fp in self._fps.values():
            try:
                fp.close()
            except Exception:
                pass
        self._fps.clear()
        if self._gpu_fp is not None:
            try:
                self._gpu_fp.close()
            except Exception:
                pass
            self._gpu_fp = None
        if self._system_fp is not None:
            try:
                self._system_fp.close()
            except Exception:
                pass
            self._system_fp = None

    def _scrape_role(self, role: str, endpoint: str, ts_unix: float, ts_iso: str):
        try:
            text = fetch_text(endpoint, timeout_s=self.timeout_s)
            selector = self.selectors.get(role, {"exact": [], "prefixes": []})
            exact = selector.get("exact", [])
            prefixes = selector.get("prefixes", [])
            fp = self._fps[role]

            metrics_by_name: Dict[str, float] = {}
            metric_series: List[Dict[str, object]] = []

            for name, labels, value in parse_prometheus_metrics(text):
                if not metric_selected(name, exact=exact, prefixes=prefixes):
                    continue

                metrics_by_name[name] = metrics_by_name.get(name, 0.0) + value
                metric_series.append({"name": name, "labels": labels, "value": value})

                rec = {
                    "ts": ts_iso,
                    "unix_ts": ts_unix,
                    "role": role,
                    "endpoint": endpoint,
                    "name": name,
                    "labels": labels,
                    "value": value,
                }
                fp.write(json.dumps(rec, ensure_ascii=True) + "\n")

            with self._state_lock:
                self._latest_role_state[role] = {
                    "ts": ts_iso,
                    "unix_ts": ts_unix,
                    "endpoint": endpoint,
                    "metrics": metrics_by_name,
                    "series": metric_series,
                    "error": None,
                }
        except Exception as e:
            fp = self._fps[role]
            err = {
                "ts": ts_iso,
                "unix_ts": ts_unix,
                "role": role,
                "endpoint": endpoint,
                "error": str(e),
            }
            fp.write(json.dumps({"type": "scrape_error", **err}, ensure_ascii=True) + "\n")
            with self._state_lock:
                self._latest_role_state[role] = {
                    "ts": ts_iso,
                    "unix_ts": ts_unix,
                    "endpoint": endpoint,
                    "metrics": {},
                    "series": [],
                    "error": str(e),
                }

    def _scrape_gpu(self, ts_unix: float, ts_iso: str):
        if not self._has_nvidia_smi or self._gpu_fp is None:
            return
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True, timeout=3)
            samples = []
            for line in out.strip().splitlines():
                fields = [f.strip() for f in line.split(",")]
                if len(fields) < 8:
                    continue
                rec = {
                    "ts": ts_iso,
                    "unix_ts": ts_unix,
                    "gpu_index": int(fields[0]),
                    "gpu_uuid": fields[1],
                    "gpu_name": fields[2],
                    "mem_used_mb": float(fields[3]),
                    "mem_total_mb": float(fields[4]),
                    "util_gpu_pct": float(fields[5]),
                    "util_mem_pct": float(fields[6]),
                    "temperature_c": float(fields[7]),
                }
                samples.append(rec)
                self._gpu_fp.write(json.dumps(rec, ensure_ascii=True) + "\n")

            with self._state_lock:
                self._latest_gpu_state = {
                    "ts": ts_iso,
                    "unix_ts": ts_unix,
                    "samples": samples,
                    "error": None,
                }
        except Exception as e:
            self._gpu_fp.write(
                json.dumps(
                    {
                        "type": "gpu_scrape_error",
                        "ts": ts_iso,
                        "unix_ts": ts_unix,
                        "error": str(e),
                    },
                    ensure_ascii=True,
                )
                + "\n"
            )
            with self._state_lock:
                self._latest_gpu_state = {
                    "ts": ts_iso,
                    "unix_ts": ts_unix,
                    "samples": [],
                    "error": str(e),
                }

    def _read_cpu_times(self) -> Tuple[int, int]:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            line = f.readline().strip()
        if not line.startswith("cpu "):
            raise RuntimeError("unexpected /proc/stat format")
        parts = line.split()
        values = [int(x) for x in parts[1:]]
        if len(values) < 4:
            raise RuntimeError("insufficient cpu stats")
        total = sum(values)
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return total, idle

    def _read_meminfo(self) -> Dict[str, int]:
        wanted = {"MemTotal", "MemAvailable"}
        parsed: Dict[str, int] = {}
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for raw in f:
                if not wanted:
                    break
                if ":" not in raw:
                    continue
                key, rest = raw.split(":", 1)
                if key not in wanted:
                    continue
                num = rest.strip().split()[0]
                parsed[key] = int(num)
                wanted.discard(key)
        if "MemTotal" not in parsed or "MemAvailable" not in parsed:
            raise RuntimeError("failed to parse /proc/meminfo")
        return parsed

    def _scrape_system(self, ts_unix: float, ts_iso: str):
        if self._system_fp is None:
            return
        try:
            cpu_total, cpu_idle = self._read_cpu_times()
            cpu_util_pct: Optional[float] = None
            if self._prev_cpu_total is not None and self._prev_cpu_idle is not None:
                delta_total = cpu_total - self._prev_cpu_total
                delta_idle = cpu_idle - self._prev_cpu_idle
                if delta_total > 0:
                    cpu_util_pct = max(0.0, min(100.0, (1.0 - (delta_idle / delta_total)) * 100.0))

            self._prev_cpu_total = cpu_total
            self._prev_cpu_idle = cpu_idle

            mem = self._read_meminfo()
            mem_total_mb = mem["MemTotal"] / 1024.0
            mem_available_mb = mem["MemAvailable"] / 1024.0
            mem_used_mb = max(0.0, mem_total_mb - mem_available_mb)
            mem_util_pct = (mem_used_mb / mem_total_mb * 100.0) if mem_total_mb > 0 else None

            rec = {
                "ts": ts_iso,
                "unix_ts": ts_unix,
                "cpu_util_pct": cpu_util_pct,
                "mem_total_mb": mem_total_mb,
                "mem_used_mb": mem_used_mb,
                "mem_available_mb": mem_available_mb,
                "mem_util_pct": mem_util_pct,
            }
            self._system_fp.write(json.dumps(rec, ensure_ascii=True) + "\n")

            with self._state_lock:
                self._latest_system_state = {
                    **rec,
                    "error": None,
                }
        except Exception as e:
            err_rec = {
                "type": "system_scrape_error",
                "ts": ts_iso,
                "unix_ts": ts_unix,
                "error": str(e),
            }
            self._system_fp.write(json.dumps(err_rec, ensure_ascii=True) + "\n")
            with self._state_lock:
                self._latest_system_state = {
                    "ts": ts_iso,
                    "unix_ts": ts_unix,
                    "cpu_util_pct": None,
                    "mem_total_mb": None,
                    "mem_used_mb": None,
                    "mem_available_mb": None,
                    "mem_util_pct": None,
                    "error": str(e),
                }

    def get_latest_state(self) -> Dict[str, object]:
        with self._state_lock:
            return copy.deepcopy(
                {
                    "roles": self._latest_role_state,
                    "gpu": self._latest_gpu_state,
                    "system": self._latest_system_state,
                }
            )

    def run(self):
        self._open_files()
        try:
            while not self.stop_event.is_set():
                t0 = time.time()
                ts_iso = now_iso()
                for role, endpoint in self.role_endpoints.items():
                    self._scrape_role(role, endpoint, t0, ts_iso)
                self._scrape_gpu(t0, ts_iso)
                self._scrape_system(t0, ts_iso)
                elapsed = time.time() - t0
                sleep_s = max(0.0, self.interval_s - elapsed)
                self.stop_event.wait(sleep_s)
        finally:
            self._close_files()
