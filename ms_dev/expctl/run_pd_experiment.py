#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
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


@dataclass
class ProcessHandle:
    name: str
    proc: subprocess.Popen
    stdout_fp: object
    stderr_fp: object


class ShutdownRequested(Exception):
    pass


class MetricsScraper(threading.Thread):
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


def wait_http_ready(base_url: str, timeout_s: float, stop_event: Optional[threading.Event] = None) -> str:
    candidates = ["/health", "/metrics", "/v1/models", "/model_info"]
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if stop_event is not None and stop_event.is_set():
            raise ShutdownRequested(f"shutdown requested while waiting for {base_url}")

        for path in candidates:
            if stop_event is not None and stop_event.is_set():
                raise ShutdownRequested(f"shutdown requested while waiting for {base_url}")

            url = base_url.rstrip("/") + path
            try:
                with urlopen(url, timeout=2) as resp:
                    code = getattr(resp, "status", 200)
                    if 200 <= code < 500:
                        return path
            except Exception as e:
                last_error = str(e)

        if stop_event is not None:
            if stop_event.wait(0.5):
                raise ShutdownRequested(f"shutdown requested while waiting for {base_url}")
        else:
            time.sleep(0.5)

    raise RuntimeError(f"timeout waiting for {base_url}. last_error={last_error}")


def launch_process(name: str, script_path: Path, log_dir: Path, workdir: Path) -> ProcessHandle:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    stdout_fp = stdout_path.open("a", encoding="utf-8", buffering=1)
    stderr_fp = stderr_path.open("a", encoding="utf-8", buffering=1)

    proc = subprocess.Popen(
        ["bash", str(script_path)],
        cwd=str(workdir),
        stdout=stdout_fp,
        stderr=stderr_fp,
        start_new_session=True,
        env=os.environ.copy(),
    )
    return ProcessHandle(name=name, proc=proc, stdout_fp=stdout_fp, stderr_fp=stderr_fp)


def stop_process(handle: ProcessHandle, timeout_s: float = 20.0) -> None:
    if handle.proc.poll() is not None:
        try:
            handle.stdout_fp.close()
            handle.stderr_fp.close()
        except Exception:
            pass
        return

    try:
        os.killpg(handle.proc.pid, signal.SIGTERM)
    except Exception:
        pass

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if handle.proc.poll() is not None:
            break
        time.sleep(0.2)

    if handle.proc.poll() is None:
        try:
            os.killpg(handle.proc.pid, signal.SIGKILL)
        except Exception:
            pass

    try:
        handle.stdout_fp.close()
        handle.stderr_fp.close()
    except Exception:
        pass


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _extract_pids_from_ss_line(line: str) -> Set[int]:
    pids: Set[int] = set()
    for m in re.finditer(r"pid=(\d+)", line):
        try:
            pids.add(int(m.group(1)))
        except Exception:
            pass
    return pids


def find_listening_pids_by_ports(ports: List[int]) -> Dict[int, Set[int]]:
    port_to_pids: Dict[int, Set[int]] = {p: set() for p in ports}

    try:
        out = subprocess.check_output(["ss", "-ltnpH"], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return port_to_pids

    for raw in out.splitlines():
        line = raw.strip()
        if not line:
            continue
        pids = _extract_pids_from_ss_line(line)
        if not pids:
            continue

        for port in ports:
            if f":{port} " in line or line.endswith(f":{port}"):
                port_to_pids[port].update(pids)

    return port_to_pids


def kill_pid_or_group(pid: int, timeout_s: float = 8.0) -> bool:
    if not _is_pid_alive(pid):
        return True

    my_pid = os.getpid()
    my_pgid = os.getpgid(my_pid)

    try:
        pgid = os.getpgid(pid)
    except Exception:
        pgid = None

    # Safety: never kill the controller's own process group.
    if pgid is not None and pgid == my_pgid:
        return False

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            os.kill(pid, signal.SIGTERM)
    except Exception:
        pass

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _is_pid_alive(pid):
            return True
        time.sleep(0.2)

    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass

    time.sleep(0.2)
    return not _is_pid_alive(pid)


def cleanup_listening_ports(ports: List[int], timeout_s: float = 8.0) -> Dict[str, object]:
    ports = sorted({int(p) for p in ports if int(p) > 0})
    port_to_pids = find_listening_pids_by_ports(ports)

    all_pids: Set[int] = set()
    for p in ports:
        all_pids.update(port_to_pids.get(p, set()))

    killed: List[int] = []
    failed: List[int] = []

    for pid in sorted(all_pids):
        ok = kill_pid_or_group(pid, timeout_s=timeout_s)
        if ok:
            killed.append(pid)
        else:
            failed.append(pid)

    after = find_listening_pids_by_ports(ports)
    survivors: Dict[int, List[int]] = {
        p: sorted(after.get(p, set())) for p in ports if after.get(p, set())
    }

    return {
        "ports": ports,
        "before": {str(p): sorted(port_to_pids.get(p, set())) for p in ports},
        "killed_pids": killed,
        "failed_pids": failed,
        "survivors": survivors,
    }


def fmt_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def fmt_int(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{int(round(value)):,}"


def fmt_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value < 0:
        return f"{value:.2f}"
    if value <= 1.5:
        return f"{value * 100.0:.1f}%"
    return f"{value:.1f}%"


def fmt_age(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.1f}s"


def metric_value(role_state: Dict[str, object], name: str) -> Optional[float]:
    metrics = role_state.get("metrics")
    if not isinstance(metrics, dict):
        return None
    raw = metrics.get(name)
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def metric_value_any(role_state: Dict[str, object], names: List[str]) -> Optional[float]:
    for name in names:
        val = metric_value(role_state, name)
        if val is not None:
            return val
    return None


def series_sum(
    role_state: Dict[str, object],
    name: str,
    label_key: Optional[str] = None,
    label_value: Optional[str] = None,
) -> Optional[float]:
    series = role_state.get("series")
    if not isinstance(series, list):
        return None

    found = False
    acc = 0.0
    for item in series:
        if not isinstance(item, dict):
            continue
        if item.get("name") != name:
            continue
        labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
        if label_key is not None:
            if labels.get(label_key) != label_value:
                continue
        try:
            acc += float(item.get("value"))
            found = True
        except Exception:
            continue
    if not found:
        return None
    return acc


def count_cb_state(role_state: Dict[str, object], state_name: str, state_code: int) -> Optional[float]:
    series = role_state.get("series")
    if not isinstance(series, list):
        return None

    found = False
    acc = 0.0
    for item in series:
        if not isinstance(item, dict) or item.get("name") != "smg_worker_cb_state":
            continue
        labels = item.get("labels") if isinstance(item.get("labels"), dict) else {}
        value = item.get("value")

        if "state" in labels:
            if labels.get("state") == state_name:
                try:
                    acc += float(value)
                    found = True
                except Exception:
                    pass
            continue

        try:
            if int(round(float(value))) == state_code:
                acc += 1.0
                found = True
        except Exception:
            continue

    if not found:
        return None
    return acc


def hist_mean_ms(role_state: Dict[str, object], base_name: str) -> Optional[float]:
    sum_val = metric_value(role_state, f"{base_name}_sum")
    count_val = metric_value(role_state, f"{base_name}_count")
    if sum_val is None or count_val is None or count_val <= 0:
        return None
    return (sum_val / count_val) * 1000.0


def scrape_age_seconds(role_state: Dict[str, object]) -> Optional[float]:
    unix_ts = role_state.get("unix_ts")
    if unix_ts is None:
        return None
    try:
        return max(0.0, time.time() - float(unix_ts))
    except Exception:
        return None


def process_state(role: str, enabled_roles: Dict[str, bool], handles: Dict[str, ProcessHandle]) -> str:
    if not enabled_roles.get(role, False):
        return "DISABLED"
    handle = handles.get(role)
    if handle is None:
        return "NOT_STARTED"
    code = handle.proc.poll()
    if code is None:
        return f"RUNNING(pid={handle.proc.pid})"
    return f"EXIT(code={code})"


def colorize(text: str, style: str, enabled: bool) -> str:
    if not enabled:
        return text
    styles = {
        "ok": "32",
        "warn": "33",
        "bad": "31",
        "info": "36",
        "muted": "90",
        "title": "1;37",
        "accent": "1;96",
    }
    code = styles.get(style)
    if not code:
        return text
    return f"\033[{code}m{text}\033[0m"


def color_state_text(state: str, enabled: bool) -> str:
    if state.startswith("RUNNING"):
        return colorize(state, "ok", enabled)
    if state.startswith("EXIT"):
        return colorize(state, "bad", enabled)
    if state == "DISABLED":
        return colorize(state, "muted", enabled)
    return colorize(state, "warn", enabled)


def color_high_bad(value: Optional[float], text: str, warn: float, bad: float, enabled: bool) -> str:
    if value is None:
        return text
    if value >= bad:
        return colorize(text, "bad", enabled)
    if value >= warn:
        return colorize(text, "warn", enabled)
    return colorize(text, "ok", enabled)


def color_high_good(value: Optional[float], text: str, warn: float, good: float, enabled: bool) -> str:
    if value is None:
        return text
    if value >= good:
        return colorize(text, "ok", enabled)
    if value >= warn:
        return colorize(text, "warn", enabled)
    return colorize(text, "bad", enabled)


def render_status(
    session_name: str,
    session_dir: Path,
    started_unix: float,
    enabled_roles: Dict[str, bool],
    handles: Dict[str, ProcessHandle],
    snapshot: Dict[str, object],
    use_color: bool,
) -> str:
    roles = snapshot.get("roles") if isinstance(snapshot.get("roles"), dict) else {}
    prefill = roles.get("prefill") if isinstance(roles.get("prefill"), dict) else {}
    decode = roles.get("decode") if isinstance(roles.get("decode"), dict) else {}
    router = roles.get("router") if isinstance(roles.get("router"), dict) else {}

    prefill_err = prefill.get("error") if isinstance(prefill.get("error"), str) else None
    decode_err = decode.get("error") if isinstance(decode.get("error"), str) else None
    router_err = router.get("error") if isinstance(router.get("error"), str) else None

    prefill_age = scrape_age_seconds(prefill)
    decode_age = scrape_age_seconds(decode)
    router_age = scrape_age_seconds(router)

    prefill_ttft_ms = hist_mean_ms(prefill, "sglang:time_to_first_token_seconds")
    decode_tbt_ms = hist_mean_ms(decode, "sglang:inter_token_latency_seconds")

    router_ttft_ms = hist_mean_ms(router, "smg_router_ttft_seconds")
    router_tpot_ms = hist_mean_ms(router, "smg_router_tpot_seconds")

    cb_open = count_cb_state(router, "open", 1)
    cb_half_open = count_cb_state(router, "half_open", 2)

    prefill_running = metric_value(prefill, "sglang:num_running_reqs")
    prefill_queue = metric_value(prefill, "sglang:num_queue_reqs")
    prefill_prealloc_q = metric_value(prefill, "sglang:num_prefill_prealloc_queue_reqs")
    prefill_inflight_q = metric_value(prefill, "sglang:num_prefill_inflight_queue_reqs")
    prefill_bootstrap_fail = metric_value(prefill, "sglang:num_bootstrap_failed_reqs_total")
    prefill_token_usage = metric_value(prefill, "sglang:token_usage")
    prefill_cache_hit = metric_value(prefill, "sglang:cache_hit_rate")

    decode_throughput = metric_value(decode, "sglang:gen_throughput")
    decode_running = metric_value(decode, "sglang:num_running_reqs")
    decode_queue = metric_value(decode, "sglang:num_queue_reqs")
    decode_prealloc_q = metric_value(decode, "sglang:num_decode_prealloc_queue_reqs")
    decode_transfer_q = metric_value(decode, "sglang:num_decode_transfer_queue_reqs")
    decode_transfer_fail = metric_value(decode, "sglang:num_transfer_failed_reqs_total")
    decode_token_usage = metric_value(decode, "sglang:token_usage")
    decode_cache_hit = metric_value(decode, "sglang:cache_hit_rate")

    router_http_active = metric_value_any(router, ["smg_http_requests_active", "smg_http_connections_active"])
    router_worker_active = metric_value(router, "smg_worker_requests_active")
    router_worker_conn = metric_value(router, "smg_worker_connections_active")
    router_pool_size = metric_value(router, "smg_worker_pool_size")

    gpu = snapshot.get("gpu") if isinstance(snapshot.get("gpu"), dict) else {}
    gpu_err = gpu.get("error") if isinstance(gpu.get("error"), str) else None
    gpu_samples = gpu.get("samples") if isinstance(gpu.get("samples"), list) else []

    system = snapshot.get("system") if isinstance(snapshot.get("system"), dict) else {}
    system_err = system.get("error") if isinstance(system.get("error"), str) else None
    system_cpu = float(system.get("cpu_util_pct")) if system.get("cpu_util_pct") is not None else None
    system_mem_used_mb = float(system.get("mem_used_mb")) if system.get("mem_used_mb") is not None else None
    system_mem_total_mb = float(system.get("mem_total_mb")) if system.get("mem_total_mb") is not None else None
    system_mem_util_pct = float(system.get("mem_util_pct")) if system.get("mem_util_pct") is not None else None

    gpu_count = len(gpu_samples)
    gpu_summary = ""
    if gpu_count > 0:
        mem_used = sum(float(s.get("mem_used_mb", 0.0)) for s in gpu_samples if isinstance(s, dict))
        mem_total = sum(float(s.get("mem_total_mb", 0.0)) for s in gpu_samples if isinstance(s, dict))
        util_gpu_values = [float(s.get("util_gpu_pct", 0.0)) for s in gpu_samples if isinstance(s, dict)]
        temp_values = [float(s.get("temperature_c", 0.0)) for s in gpu_samples if isinstance(s, dict)]
        avg_util = sum(util_gpu_values) / len(util_gpu_values) if util_gpu_values else 0.0
        max_temp = max(temp_values) if temp_values else 0.0
        mem_ratio = (mem_used / mem_total * 100.0) if mem_total > 0 else 0.0

        gpu_summary = (
            f"GPU: {colorize(str(gpu_count), 'accent', use_color)} cards | "
            f"mem={mem_used/1024.0:.1f}/{mem_total/1024.0:.1f} GiB "
            f"({color_high_bad(mem_ratio, f'{mem_ratio:.1f}%', warn=80.0, bad=90.0, enabled=use_color)}) | "
            f"avg_util={color_high_bad(avg_util, f'{avg_util:.1f}%', warn=70.0, bad=90.0, enabled=use_color)} | "
            f"max_temp={color_high_bad(max_temp, f'{max_temp:.1f}C', warn=80.0, bad=90.0, enabled=use_color)}"
        )
    else:
        gpu_summary = f"GPU: {colorize('no samples', 'warn', use_color)} ({gpu_err or 'nvidia-smi unavailable'})"

    if system_mem_total_mb is not None and system_mem_used_mb is not None:
        cpu_text = fmt_float(system_cpu, 1) + "%" if system_cpu is not None else "-"
        cpu_colored = color_high_bad(system_cpu, cpu_text, warn=70.0, bad=90.0, enabled=use_color)
        mem_text = f"{system_mem_used_mb/1024.0:.1f}/{system_mem_total_mb/1024.0:.1f} GiB"
        mem_ratio_text = (
            color_high_bad(system_mem_util_pct, f"{fmt_float(system_mem_util_pct, 1)}%", warn=80.0, bad=90.0, enabled=use_color)
            if system_mem_util_pct is not None
            else "-"
        )
        system_summary = f"SYS: cpu={cpu_colored} | ram={mem_text} ({mem_ratio_text})"
    else:
        system_summary = f"SYS: {colorize('no samples', 'warn', use_color)} ({system_err or '/proc unavailable'})"

    prefill_state = color_state_text(process_state('prefill', enabled_roles, handles), use_color)
    decode_state = color_state_text(process_state('decode', enabled_roles, handles), use_color)
    router_state = color_state_text(process_state('router', enabled_roles, handles), use_color)

    prefill_err_text = colorize(prefill_err, 'bad', use_color) if prefill_err else colorize('none', 'ok', use_color)
    decode_err_text = colorize(decode_err, 'bad', use_color) if decode_err else colorize('none', 'ok', use_color)
    router_err_text = colorize(router_err, 'bad', use_color) if router_err else colorize('none', 'ok', use_color)

    lines = []
    lines.append(colorize("=" * 118, "muted", use_color))
    lines.append(
        f"{colorize('PD Experiment Monitor', 'title', use_color)} | "
        f"session={colorize(session_name, 'accent', use_color)} | "
        f"uptime={fmt_duration(time.time() - started_unix)} | now={now_iso()}"
    )
    lines.append(f"session_dir={session_dir}")
    lines.append(
        "processes: "
        + " | ".join(
            [
                f"prefill={prefill_state}",
                f"decode={decode_state}",
                f"router={router_state}",
            ]
        )
    )
    lines.append(colorize("-" * 118, "muted", use_color))

    lines.append(
        f"{colorize('PREFILL', 'accent', use_color)} "
        f"(age={fmt_age(prefill_age)}, err={prefill_err_text}): "
        f"running={color_high_bad(prefill_running, fmt_int(prefill_running), warn=16, bad=64, enabled=use_color)}, "
        f"queue={color_high_bad(prefill_queue, fmt_int(prefill_queue), warn=1, bad=10, enabled=use_color)}, "
        f"prealloc_q={color_high_bad(prefill_prealloc_q, fmt_int(prefill_prealloc_q), warn=1, bad=10, enabled=use_color)}, "
        f"inflight_q={color_high_bad(prefill_inflight_q, fmt_int(prefill_inflight_q), warn=1, bad=10, enabled=use_color)}, "
        f"bootstrap_fail={color_high_bad(prefill_bootstrap_fail, fmt_int(prefill_bootstrap_fail), warn=1, bad=5, enabled=use_color)}"
    )
    lines.append(
        "         "
        f"token_usage={color_high_bad(prefill_token_usage if prefill_token_usage is None else prefill_token_usage*100.0, fmt_pct(prefill_token_usage), warn=70.0, bad=90.0, enabled=use_color)}, "
        f"cache_hit={color_high_good(prefill_cache_hit, fmt_pct(prefill_cache_hit), warn=0.50, good=0.80, enabled=use_color)}, "
        f"TTFT_mean={color_high_bad(prefill_ttft_ms, f'{fmt_float(prefill_ttft_ms, 2)}ms', warn=800.0, bad=2000.0, enabled=use_color)}"
    )

    lines.append(
        f"{colorize('DECODE', 'accent', use_color)}  "
        f"(age={fmt_age(decode_age)}, err={decode_err_text}): "
        f"throughput={color_high_good(decode_throughput, f'{fmt_float(decode_throughput, 2)} tok/s', warn=1.0, good=100.0, enabled=use_color)}, "
        f"running={color_high_bad(decode_running, fmt_int(decode_running), warn=16, bad=64, enabled=use_color)}, "
        f"queue={color_high_bad(decode_queue, fmt_int(decode_queue), warn=1, bad=10, enabled=use_color)}, "
        f"prealloc_q={color_high_bad(decode_prealloc_q, fmt_int(decode_prealloc_q), warn=1, bad=10, enabled=use_color)}, "
        f"transfer_q={color_high_bad(decode_transfer_q, fmt_int(decode_transfer_q), warn=1, bad=10, enabled=use_color)}"
    )
    lines.append(
        "         "
        f"transfer_fail={color_high_bad(decode_transfer_fail, fmt_int(decode_transfer_fail), warn=1, bad=5, enabled=use_color)}, "
        f"token_usage={color_high_bad(decode_token_usage if decode_token_usage is None else decode_token_usage*100.0, fmt_pct(decode_token_usage), warn=70.0, bad=90.0, enabled=use_color)}, "
        f"cache_hit={color_high_good(decode_cache_hit, fmt_pct(decode_cache_hit), warn=0.50, good=0.80, enabled=use_color)}, "
        f"TBT_mean={color_high_bad(decode_tbt_ms, f'{fmt_float(decode_tbt_ms, 2)}ms', warn=60.0, bad=120.0, enabled=use_color)}"
    )

    lines.append(
        f"{colorize('ROUTER', 'accent', use_color)}  "
        f"(age={fmt_age(router_age)}, err={router_err_text}): "
        f"http_active={color_high_bad(router_http_active, fmt_int(router_http_active), warn=32, bad=128, enabled=use_color)}, "
        f"worker_active={color_high_bad(router_worker_active, fmt_int(router_worker_active), warn=32, bad=128, enabled=use_color)}, "
        f"worker_conn={color_high_bad(router_worker_conn, fmt_int(router_worker_conn), warn=32, bad=128, enabled=use_color)}, "
        f"pool_size={color_high_good(router_pool_size, fmt_int(router_pool_size), warn=1, good=2, enabled=use_color)}"
    )
    lines.append(
        "         "
        f"cb_open={color_high_bad(cb_open, fmt_int(cb_open), warn=1, bad=2, enabled=use_color)}, "
        f"cb_half_open={color_high_bad(cb_half_open, fmt_int(cb_half_open), warn=1, bad=3, enabled=use_color)}, "
        f"router_TTFT_mean={color_high_bad(router_ttft_ms, f'{fmt_float(router_ttft_ms, 2)}ms', warn=800.0, bad=2000.0, enabled=use_color)}, "
        f"router_TPOT_mean={color_high_bad(router_tpot_ms, f'{fmt_float(router_tpot_ms, 2)}ms', warn=60.0, bad=120.0, enabled=use_color)}"
    )

    lines.append(gpu_summary)
    lines.append(system_summary)
    lines.append(colorize("=" * 118, "muted", use_color))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run PD router/prefill/decode and scrape selected metrics into a session folder."
    )
    p.add_argument("--session-root", type=str, default="/workspace/sglang/ms_dev/runtime/sessions")
    p.add_argument("--session-name", type=str, default="")
    p.add_argument("--scrape-interval", type=float, default=0.2)
    p.add_argument("--startup-timeout", type=float, default=180.0)
    p.add_argument("--metrics-timeout", type=float, default=3.0)
    p.add_argument("--status-interval", type=float, default=0.2)
    p.add_argument("--quiet-status", action="store_true")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--force-color", action="store_true")
    p.add_argument("--metrics-host", type=str, default="127.0.0.1")
    p.add_argument("--router-host", type=str, default="127.0.0.1")
    p.add_argument("--prefill-port", type=int, default=30002)
    p.add_argument("--decode-port", type=int, default=30001)
    p.add_argument("--router-port", type=int, default=30000)
    p.add_argument("--router-metrics-port", type=int, default=29000)
    p.add_argument("--no-router", action="store_true")
    p.add_argument("--no-prefill", action="store_true")
    p.add_argument("--no-decode", action="store_true")
    p.add_argument("--no-auto-kill-ports", action="store_true")
    p.add_argument("--port-kill-timeout", type=float, default=8.0)
    p.add_argument("--disagg-bootstrap-port", type=int, default=int(os.environ.get("SGLANG_DISAGGREGATION_BOOTSTRAP_PORT", "8998")))
    p.add_argument("--cleanup-extra-ports", type=str, default="")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    ms_dev_dir = Path("/workspace/sglang/ms_dev")
    script_prefill = ms_dev_dir / "sglang_start_server_pd_prefill.sh"
    script_decode = ms_dev_dir / "sglang_start_server_pd_decode.sh"
    script_router = ms_dev_dir / "sglang_start_router_pd.sh"

    started_at = datetime.now()
    started_unix = time.time()
    use_color = (not args.no_color) and (args.force_color or (sys.stdout.isatty() and (os.environ.get("NO_COLOR") is None)))
    session_name = args.session_name or started_at.strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.session_root) / session_name
    process_log_dir = session_dir / "process_logs"
    metrics_dir = session_dir / "metrics"
    meta_dir = session_dir / "meta"

    process_log_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    role_endpoints = {
        "prefill": f"http://{args.metrics_host}:{args.prefill_port}/metrics",
        "decode": f"http://{args.metrics_host}:{args.decode_port}/metrics",
        "router": f"http://{args.metrics_host}:{args.router_metrics_port}/metrics",
    }

    cleanup_ports = [
        args.prefill_port,
        args.decode_port,
        args.router_port,
        args.router_metrics_port,
        args.disagg_bootstrap_port,
    ]
    if args.cleanup_extra_ports.strip():
        for item in args.cleanup_extra_ports.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                cleanup_ports.append(int(item))
            except ValueError:
                print(f"[run_pd_experiment] ignore invalid cleanup port: {item}", flush=True)

    sglang_exact = [
        "sglang:gen_throughput",
        "sglang:num_running_reqs",
        "sglang:num_used_tokens",
        "sglang:token_usage",
        "sglang:full_token_usage",
        "sglang:pending_prealloc_token_usage",
        "sglang:max_total_num_tokens",
        "sglang:cache_hit_rate",
        "sglang:num_retracted_reqs",
        "sglang:num_retracted_requests_total",
        "sglang:num_queue_reqs",
        "sglang:num_prefill_prealloc_queue_reqs",
        "sglang:num_prefill_inflight_queue_reqs",
        "sglang:num_decode_prealloc_queue_reqs",
        "sglang:num_decode_transfer_queue_reqs",
        "sglang:num_transfer_failed_reqs_total",
        "sglang:num_bootstrap_failed_reqs_total",
        "sglang:kv_transfer_speed_gb_s",
        "sglang:kv_transfer_latency_ms",
        "sglang:kv_transfer_bootstrap_ms",
        "sglang:kv_transfer_alloc_ms",
        "sglang:kv_transfer_total_mb",
        "sglang:time_to_first_token_seconds",
        "sglang:inter_token_latency_seconds",
        "sglang:e2e_request_latency_seconds",
        "sglang:queue_time_seconds",
        "sglang:hicache_host_used_tokens",
        "sglang:hicache_host_total_tokens",
    ]

    selectors = {
        "prefill": {"exact": sglang_exact, "prefixes": []},
        "decode": {"exact": sglang_exact, "prefixes": []},
        "router": {
            "exact": [],
            "prefixes": [
                "smg_http_",
                "smg_router_",
                "smg_worker_",
                "smg_worker_cb_",
                "smg_worker_retries_",
                "smg_discovery_",
            ],
        },
    }

    out_files = {
        "prefill": metrics_dir / "prefill_metrics.jsonl",
        "decode": metrics_dir / "decode_metrics.jsonl",
        "router": metrics_dir / "router_metrics.jsonl",
    }

    enabled_roles = {
        "prefill": not args.no_prefill,
        "decode": not args.no_decode,
        "router": not args.no_router,
    }

    role_cmds = {
        "prefill": script_prefill,
        "decode": script_decode,
        "router": script_router,
    }

    handles: Dict[str, ProcessHandle] = {}
    stop_event = threading.Event()
    scraper: Optional[MetricsScraper] = None
    installed_handlers: Dict[int, object] = {}

    def request_shutdown(reason: str) -> None:
        if not stop_event.is_set():
            print(f"[run_pd_experiment] shutdown requested ({reason}); cleaning up...", flush=True)
        stop_event.set()

    def signal_handler(signum, _frame):
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        request_shutdown(signame)

    for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        sig = getattr(signal, sig_name, None)
        if sig is None:
            continue
        installed_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, signal_handler)

    run_meta = {
        "session_name": session_name,
        "session_dir": str(session_dir),
        "started_at": started_at.isoformat(),
        "scrape_interval": args.scrape_interval,
        "metrics_timeout": args.metrics_timeout,
        "status_interval": args.status_interval,
        "quiet_status": args.quiet_status,
        "no_color": args.no_color,
        "force_color": args.force_color,
        "use_color": use_color,
        "no_auto_kill_ports": args.no_auto_kill_ports,
        "port_kill_timeout": args.port_kill_timeout,
        "disagg_bootstrap_port": args.disagg_bootstrap_port,
        "cleanup_ports": sorted(set(cleanup_ports)),
        "enabled_roles": enabled_roles,
        "scripts": {k: str(v) for k, v in role_cmds.items()},
        "role_endpoints": role_endpoints,
        "network": {
            "metrics_host": args.metrics_host,
            "router_host": args.router_host,
            "prefill_port": args.prefill_port,
            "decode_port": args.decode_port,
            "router_port": args.router_port,
            "router_metrics_port": args.router_metrics_port,
        },
    }
    (meta_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")

    if not args.no_auto_kill_ports:
        print(
            f"[run_pd_experiment] pre-cleanup listening ports: {sorted(set(cleanup_ports))}",
            flush=True,
        )
        cleanup_result = cleanup_listening_ports(cleanup_ports, timeout_s=args.port_kill_timeout)
        (meta_dir / "precleanup_ports.json").write_text(
            json.dumps({"ts": now_iso(), **cleanup_result}, indent=2),
            encoding="utf-8",
        )
        if cleanup_result["killed_pids"]:
            print(
                f"[run_pd_experiment] pre-cleanup killed pids: {cleanup_result['killed_pids']}",
                flush=True,
            )
        if cleanup_result["survivors"]:
            print(
                f"[run_pd_experiment] WARNING: ports still occupied after cleanup: {cleanup_result['survivors']}",
                flush=True,
            )

    startup_order = ["prefill", "decode", "router"]

    print(f"[run_pd_experiment] session={session_name}", flush=True)
    print(f"[run_pd_experiment] session_dir={session_dir}", flush=True)
    print("[run_pd_experiment] launching processes...", flush=True)

    try:
        for role in startup_order:
            if not enabled_roles[role]:
                continue
            handle = launch_process(
                name=role,
                script_path=role_cmds[role],
                log_dir=process_log_dir,
                workdir=ms_dev_dir,
            )
            handles[role] = handle
            print(f"[run_pd_experiment] launched {role} pid={handle.proc.pid}", flush=True)

        readiness = {}
        if enabled_roles["prefill"]:
            print(f"[run_pd_experiment] waiting prefill readiness on {args.metrics_host}:{args.prefill_port}...", flush=True)
            readiness["prefill"] = wait_http_ready(
                f"http://{args.metrics_host}:{args.prefill_port}", args.startup_timeout, stop_event=stop_event
            )
            print(f"[run_pd_experiment] prefill ready via {readiness['prefill']}", flush=True)
        if enabled_roles["decode"]:
            print(f"[run_pd_experiment] waiting decode readiness on {args.metrics_host}:{args.decode_port}...", flush=True)
            readiness["decode"] = wait_http_ready(
                f"http://{args.metrics_host}:{args.decode_port}", args.startup_timeout, stop_event=stop_event
            )
            print(f"[run_pd_experiment] decode ready via {readiness['decode']}", flush=True)
        if enabled_roles["router"]:
            print(f"[run_pd_experiment] waiting router readiness on {args.router_host}:{args.router_port}...", flush=True)
            readiness["router"] = wait_http_ready(
                f"http://{args.router_host}:{args.router_port}", args.startup_timeout, stop_event=stop_event
            )
            print(f"[run_pd_experiment] router ready via {readiness['router']}", flush=True)

        (meta_dir / "readiness.json").write_text(
            json.dumps({"ts": now_iso(), "readiness": readiness}, indent=2),
            encoding="utf-8",
        )

        active_role_endpoints = {
            role: endpoint for role, endpoint in role_endpoints.items() if enabled_roles[role]
        }
        active_selectors = {role: selectors[role] for role in active_role_endpoints}
        active_out_files = {role: out_files[role] for role in active_role_endpoints}

        scraper = MetricsScraper(
            role_endpoints=active_role_endpoints,
            selectors=active_selectors,
            out_files=active_out_files,
            gpu_file=metrics_dir / "gpu_metrics.jsonl",
            system_file=metrics_dir / "system_metrics.jsonl",
            interval_s=args.scrape_interval,
            stop_event=stop_event,
            timeout_s=args.metrics_timeout,
        )
        scraper.start()

        print("[run_pd_experiment] metrics scraping started; running. press Ctrl+C to stop.", flush=True)

        next_status_ts = 0.0
        while True:
            if stop_event.is_set():
                raise ShutdownRequested("external termination signal")

            dead = [role for role, h in handles.items() if h.proc.poll() is not None]
            if dead:
                raise RuntimeError(f"one or more processes exited unexpectedly: {dead}")

            if not args.quiet_status and scraper is not None and time.time() >= next_status_ts:
                snap = scraper.get_latest_state()
                block = render_status(
                    session_name=session_name,
                    session_dir=session_dir,
                    started_unix=started_unix,
                    enabled_roles=enabled_roles,
                    handles=handles,
                    snapshot=snap,
                    use_color=use_color,
                )
                if sys.stdout.isatty():
                    print("\033[2J\033[H" + block, flush=True)
                else:
                    print(block, flush=True)
                next_status_ts = time.time() + max(0.5, args.status_interval)

            time.sleep(0.5)

    except (KeyboardInterrupt, ShutdownRequested):
        print("[run_pd_experiment] interrupted by user; shutting down...", flush=True)
        return_code = 0
    except Exception as e:
        err = {"ts": now_iso(), "error": str(e)}
        (meta_dir / "run_error.json").write_text(json.dumps(err, indent=2), encoding="utf-8")
        print(f"[run_pd_experiment] ERROR: {e}", file=sys.stderr, flush=True)
        return_code = 1
    else:
        return_code = 0
    finally:
        stop_event.set()
        if scraper is not None:
            scraper.join(timeout=5.0)
        for role in ["router", "decode", "prefill"]:
            if role in handles:
                stop_process(handles[role])

        for sig, prev_handler in installed_handlers.items():
            try:
                signal.signal(sig, prev_handler)
            except Exception:
                pass

        end_meta = {
            "ts": now_iso(),
            "ended_at": datetime.now().isoformat(),
            "return_code": return_code,
        }
        (meta_dir / "run_end.json").write_text(json.dumps(end_meta, indent=2), encoding="utf-8")

    return return_code


if __name__ == "__main__":
    sys.exit(main())
