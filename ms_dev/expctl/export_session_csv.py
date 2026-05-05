#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


LOCAL_TZ = datetime.now().astimezone().tzinfo
SESSION_TIME_FMT = "%Y%m%d_%H%M%S"
HOUR_FILE_FMT = "%Y-%m-%d_%H"
ALT_HOUR_FILE_FMT = "%Y%m%d_%H"
REQUEST_METRICS_PREFIX = "sglang-request-metrics-"

SERVER_ARG_NAMES = [
    "model_path",
    "served_model_name",
    "port",
    "tp_size",
    "pp_size",
    "mem_fraction_static",
    "schedule_policy",
    "chunked_prefill_size",
    "max_prefill_tokens",
    "kv_cache_dtype",
    "disable_radix_cache",
    "radix_eviction_policy",
    "enable_hierarchical_cache",
    "hicache_ratio",
    "hicache_size",
    "enable_lmcache",
    "attention_backend",
    "dtype",
]

PREFILL_RE = re.compile(
    r"#new-token:\s*(?P<new>\d+),\s+#cached-token:\s*(?P<cached>\d+),\s+"
    r"token usage:\s*(?P<usage>[0-9.]+),\s+#running-req:\s*(?P<running>\d+),\s+"
    r"#queue-req:\s*(?P<queue>\d+).*input throughput \(token/s\):\s*(?P<throughput>[0-9.]+)"
)
DECODE_RE = re.compile(
    r"#running-req:\s*(?P<running>\d+),\s+#token:\s*(?P<token>\d+),\s+"
    r"token usage:\s*(?P<usage>[0-9.]+),.*gen throughput \(token/s\):\s*(?P<throughput>[0-9.]+),\s+"
    r"#queue-req:\s*(?P<queue>\d+)"
)
REQ_TIME_RE = re.compile(
    r"Req Time Stats\(rid=(?P<rid>[^,]+), input len=(?P<input_len>\d+), "
    r"output len=(?P<output_len>\d+), type=(?P<request_type>[^)]+)\): "
    r"queue_duration=(?P<queue_ms>[0-9.]+)ms, "
    r"forward_duration=(?P<forward_ms>[0-9.]+)ms, "
    r"start_time=(?P<start_time>[0-9.]+)"
)
KV_ALLOC_RE = re.compile(
    r"KV Cache is allocated\. #tokens:\s*(?P<tokens>\d+), K size:\s*(?P<k>[0-9.]+)\s+GB, "
    r"V size:\s*(?P<v>[0-9.]+)\s+GB"
)
LOG_TS_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

HISTOGRAM_BASES_SECONDS = [
    "sglang:time_to_first_token_seconds",
    "sglang:inter_token_latency_seconds",
    "sglang:e2e_request_latency_seconds",
    "sglang:queue_time_seconds",
]


@dataclass
class SessionWindow:
    session_name: str
    session_dir: Path
    started_at: datetime
    ended_at: Optional[datetime]

    @property
    def end_for_fallback(self) -> datetime:
        return self.ended_at or (self.started_at + timedelta(hours=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one or more SGLang session folders into session-local CSV files. "
            "The exporter prefers raw files stored under the session directory and "
            "falls back to the legacy global runtime layout when needed."
        )
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        default=Path("/home/nxclab/sglang/ms_dev/runtime/sessions"),
        help="Root directory containing per-session folders.",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        help="Explicit session directory to export.",
    )
    parser.add_argument(
        "--sessions",
        nargs="*",
        help="Explicit session names to export from --session-root.",
    )
    parser.add_argument(
        "--from-session",
        type=str,
        help="Inclusive lower-bound session name when exporting a range.",
    )
    parser.add_argument(
        "--to-session",
        type=str,
        help="Inclusive upper-bound session name when exporting a range.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Export only the latest session from --session-root.",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="exports/csv",
        help="Session-local subdirectory used for CSV outputs.",
    )
    return parser.parse_args()


def dt_to_iso(dt: datetime) -> str:
    return dt.astimezone(LOCAL_TZ).isoformat(timespec="milliseconds")


def parse_iso_datetime(value: str) -> datetime:
    text = value.strip()
    if text.endswith("Z"):
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    else:
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=LOCAL_TZ)
    return dt.astimezone(LOCAL_TZ)


def parse_log_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=LOCAL_TZ)


def parse_session_name(name: str) -> datetime:
    return datetime.strptime(name, SESSION_TIME_FMT).replace(tzinfo=LOCAL_TZ)


def json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def iter_sessions(
    session_root: Path,
    session_dir: Optional[Path],
    selected_names: Optional[list[str]],
    from_name: Optional[str],
    to_name: Optional[str],
    latest: bool,
) -> list[Path]:
    if session_dir is not None:
        return [session_dir]

    if selected_names:
        return [session_root / name for name in selected_names]

    session_dirs = []
    for path in sorted(session_root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if from_name and name < from_name:
            continue
        if to_name and name > to_name:
            continue
        session_dirs.append(path)

    if latest and session_dirs:
        return [session_dirs[-1]]
    return session_dirs


def build_session_window(session_dir: Path) -> SessionWindow:
    run_meta = json_load(session_dir / "meta" / "run_meta.json")
    run_end_path = session_dir / "meta" / "run_end.json"
    ended_at = None
    if run_end_path.exists():
        run_end = json_load(run_end_path)
        ended_raw = run_end.get("ended_at")
        if isinstance(ended_raw, str) and ended_raw:
            ended_at = parse_iso_datetime(ended_raw)

    started_at = parse_iso_datetime(str(run_meta["started_at"]))
    return SessionWindow(
        session_name=str(run_meta["session_name"]),
        session_dir=session_dir,
        started_at=started_at,
        ended_at=ended_at,
    )


def iter_jsonl_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def iter_hour_tokens(started_at: datetime, ended_at: datetime) -> Iterator[str]:
    current = started_at.replace(minute=0, second=0, microsecond=0)
    stop = ended_at.replace(minute=0, second=0, microsecond=0)
    while current <= stop:
        yield current.strftime(HOUR_FILE_FMT)
        yield current.strftime(ALT_HOUR_FILE_FMT)
        current += timedelta(hours=1)


def find_hour_files(root: Path, prefix: str, started_at: datetime, ended_at: datetime) -> list[Path]:
    if not root.exists():
        return []
    tokens = set(iter_hour_tokens(started_at, ended_at))
    paths = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        if prefix and not path.name.startswith(prefix):
            continue
        if any(token in path.name for token in tokens):
            paths.append(path)
    return paths


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def percentile(values: list[float], frac: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * frac
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summarize_numeric(prefix: str, values: list[float]) -> dict[str, Optional[float]]:
    return {
        f"{prefix}_mean": (sum(values) / len(values)) if values else None,
        f"{prefix}_p50": percentile(values, 0.50),
        f"{prefix}_p90": percentile(values, 0.90),
        f"{prefix}_p99": percentile(values, 0.99),
        f"{prefix}_max": max(values) if values else None,
    }


def request_in_window(timestamp: str, started_at: datetime, ended_at: Optional[datetime]) -> bool:
    ts = parse_iso_datetime(timestamp)
    if ts < started_at:
        return False
    if ended_at is not None and ts > ended_at:
        return False
    return True


def runtime_root_for_session(session: SessionWindow) -> Path:
    return session.session_dir.parent.parent


def role_from_relative_path(path: Path, root: Path, default_role: str) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return default_role
    if len(rel.parts) >= 2:
        return rel.parts[0]
    return default_role


def session_request_metric_paths(session: SessionWindow) -> list[tuple[Path, str]]:
    local_root = session.session_dir / "raw" / "request_metrics"
    local_files = iter_jsonl_files(local_root)
    if local_files:
        return [
            (path, role_from_relative_path(path, local_root, "server"))
            for path in local_files
        ]

    legacy_root = runtime_root_for_session(session) / "request_metrics"
    return [(path, "server") for path in find_hour_files(
        legacy_root,
        REQUEST_METRICS_PREFIX,
        session.started_at,
        session.end_for_fallback,
    )]


def session_request_log_paths(session: SessionWindow) -> list[tuple[Path, str]]:
    local_root = session.session_dir / "raw" / "request_logs"
    local_files = iter_jsonl_files(local_root)
    if local_files:
        return [
            (path, role_from_relative_path(path, local_root, "server"))
            for path in local_files
        ]

    legacy_root = runtime_root_for_session(session) / "request_logs"
    return [(path, "server") for path in find_hour_files(
        legacy_root,
        "",
        session.started_at,
        session.end_for_fallback,
    )]


def sanitize_metric_name(name: str) -> str:
    clean = re.sub(r"[^0-9A-Za-z_]+", "_", name.replace(":", "_"))
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean


def extract_request_metrics_row(
    raw: dict[str, Any],
    source_role: str,
    source_file: Path,
) -> Optional[dict[str, Any]]:
    request_finished_ts = safe_float(raw.get("request_finished_ts"))
    if request_finished_ts is None:
        return None
    request_received_ts = safe_float(raw.get("request_received_ts"))
    response_sent_ts = safe_float(raw.get("response_sent_to_client_ts"))
    timestamp = datetime.fromtimestamp(request_finished_ts, tz=LOCAL_TZ)
    finish_reason = raw.get("finish_reason")
    finish_reason_type = None
    if isinstance(finish_reason, dict):
        finish_reason_type = finish_reason.get("type")

    prompt_tokens = safe_int(raw.get("prompt_tokens"))
    cached_tokens = safe_int(raw.get("cached_tokens"))
    cache_hit_ratio = None
    if prompt_tokens and prompt_tokens > 0 and cached_tokens is not None:
        cache_hit_ratio = cached_tokens / prompt_tokens

    return {
        "timestamp": dt_to_iso(timestamp),
        "timestamp_unix": request_finished_ts,
        "rid": raw.get("id"),
        "finish_reason_type": finish_reason_type,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": safe_int(raw.get("completion_tokens")),
        "cached_tokens": cached_tokens,
        "cache_hit_ratio": cache_hit_ratio,
        "total_retractions": safe_int(raw.get("total_retractions")),
        "queue_time": safe_float(raw.get("queue_time")),
        "prefill_waiting_latency": safe_float(raw.get("prefill_waiting_latency")),
        "prefill_launch_latency": safe_float(raw.get("prefill_launch_latency")),
        "e2e_latency": safe_float(raw.get("e2e_latency")),
        "decode_throughput": safe_float(raw.get("decode_throughput")),
        "request_received_ts_epoch": request_received_ts,
        "response_sent_to_client_ts_epoch": response_sent_ts,
        "request_finished_ts_epoch": request_finished_ts,
        "inference_time": safe_float(raw.get("inference_time")),
        "source_role": source_role,
        "source_file": str(source_file),
    }


def extract_request_finished_event(
    raw: dict[str, Any],
    source_role: str,
    source_file: Path,
) -> Optional[dict[str, Any]]:
    if raw.get("event") != "request.finished":
        return None
    meta_info = raw.get("out", {}).get("meta_info")
    if not isinstance(meta_info, dict):
        return None

    timestamp_raw = raw.get("timestamp")
    if not isinstance(timestamp_raw, str):
        return None
    timestamp = parse_iso_datetime(timestamp_raw)

    finish_reason = meta_info.get("finish_reason")
    finish_reason_type = None
    if isinstance(finish_reason, dict):
        finish_reason_type = finish_reason.get("type")

    prompt_tokens = safe_int(meta_info.get("prompt_tokens"))
    cached_tokens = safe_int(meta_info.get("cached_tokens"))
    cache_hit_ratio = None
    if prompt_tokens and prompt_tokens > 0 and cached_tokens is not None:
        cache_hit_ratio = cached_tokens / prompt_tokens

    return {
        "timestamp": dt_to_iso(timestamp),
        "timestamp_unix": timestamp.timestamp(),
        "rid": meta_info.get("id") or raw.get("rid"),
        "finish_reason_type": finish_reason_type,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": safe_int(meta_info.get("completion_tokens")),
        "cached_tokens": cached_tokens,
        "cache_hit_ratio": cache_hit_ratio,
        "total_retractions": safe_int(meta_info.get("total_retractions")),
        "queue_time": safe_float(meta_info.get("queue_time")),
        "prefill_waiting_latency": safe_float(meta_info.get("prefill_waiting_latency")),
        "prefill_launch_latency": safe_float(meta_info.get("prefill_launch_latency")),
        "e2e_latency": safe_float(meta_info.get("e2e_latency")),
        "decode_throughput": safe_float(meta_info.get("decode_throughput")),
        "request_received_ts_epoch": safe_float(meta_info.get("request_received_ts")),
        "response_sent_to_client_ts_epoch": safe_float(meta_info.get("response_sent_to_client_ts")),
        "request_finished_ts_epoch": safe_float(meta_info.get("request_finished_ts")),
        "inference_time": safe_float(meta_info.get("inference_time")),
        "source_role": source_role,
        "source_file": str(source_file),
    }


def collect_request_details(session: SessionWindow) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, source_role in session_request_metric_paths(session):
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extracted = extract_request_metrics_row(payload, source_role=source_role, source_file=path)
                if extracted is None:
                    continue
                if not request_in_window(extracted["timestamp"], session.started_at, session.ended_at):
                    continue
                rows.append(extracted)

    if not rows:
        for path, source_role in session_request_log_paths(session):
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    extracted = extract_request_finished_event(payload, source_role=source_role, source_file=path)
                    if extracted is None:
                        continue
                    if not request_in_window(extracted["timestamp"], session.started_at, session.ended_at):
                        continue
                    rows.append(extracted)

    deduped: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for row in rows:
        key = (row.get("rid"), row.get("request_finished_ts_epoch"), row.get("source_role"))
        current = deduped.get(key)
        if current is None or len(row) > len(current):
            deduped[key] = row
    ordered = sorted(deduped.values(), key=lambda row: (row["timestamp"], row.get("rid") or ""))
    return ordered


def extract_request_event_row(
    raw: dict[str, Any],
    source_role: str,
    source_file: Path,
) -> Optional[dict[str, Any]]:
    timestamp_raw = raw.get("timestamp")
    if not isinstance(timestamp_raw, str):
        return None

    event = raw.get("event")
    if not isinstance(event, str) or not event:
        return None

    timestamp = parse_iso_datetime(timestamp_raw)
    row: dict[str, Any] = {
        "timestamp": dt_to_iso(timestamp),
        "timestamp_unix": timestamp.timestamp(),
        "event": event,
        "rid": raw.get("rid"),
        "source_role": source_role,
        "source_file": str(source_file),
    }

    if event == "request.received":
        obj = raw.get("obj") if isinstance(raw.get("obj"), dict) else {}
        sampling_params = obj.get("sampling_params") if isinstance(obj.get("sampling_params"), dict) else {}
        row.update(
            {
                "stream": obj.get("stream"),
                "log_metrics": obj.get("log_metrics"),
                "sampling_temperature": safe_float(sampling_params.get("temperature")),
                "sampling_max_new_tokens": safe_int(sampling_params.get("max_new_tokens")),
                "sampling_top_p": safe_float(sampling_params.get("top_p")),
            }
        )
        return row

    if event == "request.finished":
        finished = extract_request_finished_event(raw, source_role=source_role, source_file=source_file)
        if finished is None:
            return row
        row.update(finished)
        row["event"] = event
        return row

    if event == "scheduler.status":
        running = raw.get("running_rids") if isinstance(raw.get("running_rids"), list) else []
        queued = raw.get("queued_rids") if isinstance(raw.get("queued_rids"), list) else []
        row.update(
            {
                "running_rids_count": len(running),
                "queued_rids_count": len(queued),
            }
        )
        return row

    return row


def collect_request_events(session: SessionWindow) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, source_role in session_request_log_paths(session):
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                extracted = extract_request_event_row(payload, source_role=source_role, source_file=path)
                if extracted is None:
                    continue
                if not request_in_window(extracted["timestamp"], session.started_at, session.ended_at):
                    continue
                rows.append(extracted)

    deduped: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("event"),
            row.get("rid"),
            row.get("timestamp"),
            row.get("source_role"),
        )
        deduped[key] = row
    return sorted(deduped.values(), key=lambda row: (row["timestamp"], row.get("event") or "", row.get("rid") or ""))


def extract_server_arg(line: str, name: str) -> Optional[str]:
    match = re.search(rf"\b{name}=('(?:[^'\\]|\\.)*'|[^,)]*)", line)
    if not match:
        return None
    raw = match.group(1).strip()
    if raw.startswith("'") and raw.endswith("'"):
        return raw[1:-1]
    return raw


def parse_process_logs(
    session: SessionWindow,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    process_log_dir = session.session_dir / "process_logs"
    summary: dict[str, Any] = {
        "process_prefill_batch_rows": 0,
        "process_decode_batch_rows": 0,
        "process_req_time_stats_rows": 0,
        "process_max_queue_reqs": None,
        "process_max_running_reqs": None,
        "process_max_token_usage": None,
        "process_max_prefill_new_tokens": None,
        "process_max_prefill_cached_tokens": None,
    }
    batch_rows: list[dict[str, Any]] = []
    req_time_rows: list[dict[str, Any]] = []
    if not process_log_dir.exists():
        return summary, batch_rows, req_time_rows

    for path in sorted(process_log_dir.glob("*.stderr.log")):
        role = path.name.replace(".stderr.log", "")
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if f"{role}_arg_model_path" not in summary and "server_args=ServerArgs(" in line:
                    for name in SERVER_ARG_NAMES:
                        value = extract_server_arg(line, name)
                        if value is not None:
                            summary[f"{role}_arg_{name}"] = value

                ts_match = LOG_TS_RE.search(line)
                ts = parse_log_timestamp(ts_match.group("ts")) if ts_match else None

                if summary.get(f"{role}_kv_cache_dtype") is None and "Using KV cache dtype:" in line:
                    summary[f"{role}_kv_cache_dtype"] = line.rsplit(":", 1)[-1].strip()

                if summary.get(f"{role}_ready_at") is None and "The server is fired up and ready to roll!" in line and ts is not None:
                    summary[f"{role}_ready_at"] = dt_to_iso(ts)

                alloc_match = KV_ALLOC_RE.search(line)
                if alloc_match and summary.get(f"{role}_kv_cache_num_tokens") is None:
                    summary[f"{role}_kv_cache_num_tokens"] = int(alloc_match.group("tokens"))
                    summary[f"{role}_kv_cache_k_gb"] = float(alloc_match.group("k"))
                    summary[f"{role}_kv_cache_v_gb"] = float(alloc_match.group("v"))

                prefill_match = PREFILL_RE.search(line)
                if prefill_match and ts is not None:
                    row = {
                        "timestamp": dt_to_iso(ts),
                        "timestamp_unix": ts.timestamp(),
                        "source_role": role,
                        "batch_phase": "prefill",
                        "new_tokens": int(prefill_match.group("new")),
                        "cached_tokens": int(prefill_match.group("cached")),
                        "running_reqs": int(prefill_match.group("running")),
                        "queue_reqs": int(prefill_match.group("queue")),
                        "token_usage": float(prefill_match.group("usage")),
                        "throughput_tokens_per_sec": float(prefill_match.group("throughput")),
                    }
                    batch_rows.append(row)
                    summary["process_prefill_batch_rows"] += 1
                    summary["process_max_prefill_new_tokens"] = max(
                        row["new_tokens"],
                        summary["process_max_prefill_new_tokens"] or 0,
                    )
                    summary["process_max_prefill_cached_tokens"] = max(
                        row["cached_tokens"],
                        summary["process_max_prefill_cached_tokens"] or 0,
                    )
                    summary["process_max_queue_reqs"] = max(
                        row["queue_reqs"],
                        summary["process_max_queue_reqs"] or 0,
                    )
                    summary["process_max_running_reqs"] = max(
                        row["running_reqs"],
                        summary["process_max_running_reqs"] or 0,
                    )
                    summary["process_max_token_usage"] = max(
                        row["token_usage"],
                        summary["process_max_token_usage"] or 0.0,
                    )
                    continue

                decode_match = DECODE_RE.search(line)
                if decode_match and ts is not None:
                    row = {
                        "timestamp": dt_to_iso(ts),
                        "timestamp_unix": ts.timestamp(),
                        "source_role": role,
                        "batch_phase": "decode",
                        "total_tokens": int(decode_match.group("token")),
                        "running_reqs": int(decode_match.group("running")),
                        "queue_reqs": int(decode_match.group("queue")),
                        "token_usage": float(decode_match.group("usage")),
                        "throughput_tokens_per_sec": float(decode_match.group("throughput")),
                    }
                    batch_rows.append(row)
                    summary["process_decode_batch_rows"] += 1
                    summary["process_max_queue_reqs"] = max(
                        row["queue_reqs"],
                        summary["process_max_queue_reqs"] or 0,
                    )
                    summary["process_max_running_reqs"] = max(
                        row["running_reqs"],
                        summary["process_max_running_reqs"] or 0,
                    )
                    summary["process_max_token_usage"] = max(
                        row["token_usage"],
                        summary["process_max_token_usage"] or 0.0,
                    )
                    continue

                req_time_match = REQ_TIME_RE.search(line)
                if req_time_match and ts is not None:
                    req_time_rows.append(
                        {
                            "timestamp": dt_to_iso(ts),
                            "timestamp_unix": ts.timestamp(),
                            "source_role": role,
                            "rid": req_time_match.group("rid"),
                            "input_len": int(req_time_match.group("input_len")),
                            "output_len": int(req_time_match.group("output_len")),
                            "request_type": req_time_match.group("request_type"),
                            "queue_duration_ms": float(req_time_match.group("queue_ms")),
                            "forward_duration_ms": float(req_time_match.group("forward_ms")),
                            "start_time": float(req_time_match.group("start_time")),
                        }
                    )
                    summary["process_req_time_stats_rows"] += 1

    return summary, batch_rows, req_time_rows


def flatten_labels(labels: dict[str, Any]) -> dict[str, Any]:
    return {f"label_{key}": value for key, value in labels.items()}


def collect_runtime_metric_exports(
    session: SessionWindow,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    metrics_dir = session.session_dir / "metrics"
    series_by_role: dict[str, list[dict[str, Any]]] = {}
    wide_by_role: dict[str, list[dict[str, Any]]] = {}
    error_rows: list[dict[str, Any]] = []
    if not metrics_dir.exists():
        return series_by_role, wide_by_role, error_rows

    for path in sorted(metrics_dir.glob("*_metrics.jsonl")):
        if path.name in {"gpu_metrics.jsonl", "system_metrics.jsonl"}:
            continue
        role = path.name.replace("_metrics.jsonl", "")
        series_rows: list[dict[str, Any]] = []
        wide_rows_map: dict[tuple[str, float, str], dict[str, Any]] = {}

        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_ts = payload.get("ts")
                if not isinstance(raw_ts, str):
                    continue
                ts = parse_iso_datetime(raw_ts)
                ts_iso = dt_to_iso(ts)
                unix_ts = safe_float(payload.get("unix_ts")) or ts.timestamp()

                if payload.get("type") == "scrape_error":
                    error_rows.append(
                        {
                            "timestamp": ts_iso,
                            "timestamp_unix": unix_ts,
                            "role": role,
                            "endpoint": payload.get("endpoint"),
                            "error": payload.get("error"),
                            "source_file": str(path),
                        }
                    )
                    continue

                name = payload.get("name")
                labels = payload.get("labels") if isinstance(payload.get("labels"), dict) else {}
                value = safe_float(payload.get("value"))
                if not isinstance(name, str) or value is None:
                    continue

                row = {
                    "timestamp": ts_iso,
                    "timestamp_unix": unix_ts,
                    "role": role,
                    "metric_name": name,
                    "metric_column": sanitize_metric_name(name),
                    "value": value,
                    "labels_json": json.dumps(labels, ensure_ascii=True, sort_keys=True),
                    "source_file": str(path),
                    **flatten_labels(labels),
                }
                series_rows.append(row)

                wide_key = (ts_iso, unix_ts, role)
                wide_row = wide_rows_map.setdefault(
                    wide_key,
                    {
                        "timestamp": ts_iso,
                        "timestamp_unix": unix_ts,
                        "role": role,
                    },
                )
                if "le" not in labels:
                    metric_column = sanitize_metric_name(name)
                    wide_row[metric_column] = (safe_float(wide_row.get(metric_column)) or 0.0) + value

        wide_rows = sorted(wide_rows_map.values(), key=lambda row: row["timestamp"])
        for wide_row in wide_rows:
            for base in HISTOGRAM_BASES_SECONDS:
                sum_col = sanitize_metric_name(f"{base}_sum")
                count_col = sanitize_metric_name(f"{base}_count")
                mean_col = sanitize_metric_name(f"{base}_mean_seconds")
                mean_ms_col = sanitize_metric_name(f"{base}_mean_ms")
                count_val = safe_float(wide_row.get(count_col))
                sum_val = safe_float(wide_row.get(sum_col))
                if count_val is None or sum_val is None or count_val <= 0:
                    continue
                wide_row[mean_col] = sum_val / count_val
                wide_row[mean_ms_col] = (sum_val / count_val) * 1000.0

        series_by_role[role] = sorted(series_rows, key=lambda row: (row["timestamp"], row["metric_name"]))
        wide_by_role[role] = wide_rows

    return series_by_role, wide_by_role, error_rows


def collect_simple_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_ts = payload.get("ts")
            if isinstance(raw_ts, str):
                ts = parse_iso_datetime(raw_ts)
                payload["timestamp"] = dt_to_iso(ts)
                payload["timestamp_unix"] = safe_float(payload.get("unix_ts")) or ts.timestamp()
                del payload["ts"]
                payload.pop("unix_ts", None)
            rows.append(payload)
    return rows


def summarize_session(
    session: SessionWindow,
    request_rows: list[dict[str, Any]],
    request_event_rows: list[dict[str, Any]],
    process_summary: dict[str, Any],
    metric_roles: list[str],
) -> dict[str, Any]:
    prompt_tokens = [float(row["prompt_tokens"]) for row in request_rows if row.get("prompt_tokens") is not None]
    completion_tokens = [float(row["completion_tokens"]) for row in request_rows if row.get("completion_tokens") is not None]
    cached_tokens = [float(row["cached_tokens"]) for row in request_rows if row.get("cached_tokens") is not None]
    e2e_latency = [float(row["e2e_latency"]) for row in request_rows if row.get("e2e_latency") is not None]
    queue_time = [float(row["queue_time"]) for row in request_rows if row.get("queue_time") is not None]
    decode_throughput = [float(row["decode_throughput"]) for row in request_rows if row.get("decode_throughput") is not None]
    cache_hit_ratio = [float(row["cache_hit_ratio"]) for row in request_rows if row.get("cache_hit_ratio") is not None]

    total_prompt_tokens = int(sum(prompt_tokens)) if prompt_tokens else 0
    total_cached_tokens = int(sum(cached_tokens)) if cached_tokens else 0
    weighted_cache_hit_ratio = (total_cached_tokens / total_prompt_tokens) if total_prompt_tokens > 0 else None

    run_meta = json_load(session.session_dir / "meta" / "run_meta.json")
    run_end_path = session.session_dir / "meta" / "run_end.json"
    run_error_path = session.session_dir / "meta" / "run_error.json"
    run_end = json_load(run_end_path) if run_end_path.exists() else {}

    row: dict[str, Any] = {
        "session_name": session.session_name,
        "session_dir": str(session.session_dir),
        "mode": run_meta.get("mode"),
        "timezone": run_meta.get("timezone"),
        "started_at": dt_to_iso(session.started_at),
        "started_at_unix": run_meta.get("started_at_unix"),
        "ended_at": dt_to_iso(session.ended_at) if session.ended_at is not None else None,
        "ended_at_unix": run_end.get("ended_at_unix"),
        "session_duration_sec": (
            (session.ended_at - session.started_at).total_seconds()
            if session.ended_at is not None
            else None
        ),
        "request_count": len(request_rows),
        "request_event_count": len(request_event_rows),
        "cached_request_count": sum(1 for row in request_rows if (row.get("cached_tokens") or 0) > 0),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": int(sum(completion_tokens)) if completion_tokens else 0,
        "total_cached_tokens": total_cached_tokens,
        "weighted_cache_hit_ratio": weighted_cache_hit_ratio,
        "first_request_at": request_rows[0]["timestamp"] if request_rows else None,
        "last_request_at": request_rows[-1]["timestamp"] if request_rows else None,
        "runtime_metric_roles": ",".join(metric_roles),
        "run_end_return_code": run_end.get("return_code"),
        "has_run_error": run_error_path.exists(),
    }
    row.update(summarize_numeric("prompt_tokens", prompt_tokens))
    row.update(summarize_numeric("completion_tokens", completion_tokens))
    row.update(summarize_numeric("cached_tokens", cached_tokens))
    row.update(summarize_numeric("e2e_latency", e2e_latency))
    row.update(summarize_numeric("queue_time", queue_time))
    row.update(summarize_numeric("decode_throughput", decode_throughput))
    row.update(summarize_numeric("cache_hit_ratio", cache_hit_ratio))
    row.update(process_summary)
    return row


def ordered_fieldnames(rows: list[dict[str, Any]], preferred: Optional[list[str]] = None) -> list[str]:
    preferred = preferred or []
    seen = set(preferred)
    fieldnames = list(preferred)
    extra = set()
    for row in rows:
        extra.update(row.keys())
    for name in sorted(extra):
        if name not in seen:
            fieldnames.append(name)
    return fieldnames


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ordered_fieldnames(rows, preferred)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def export_session(session_dir: Path, output_subdir: str) -> list[Path]:
    session = build_session_window(session_dir)
    output_dir = session.session_dir / output_subdir
    request_rows = collect_request_details(session)
    request_event_rows = collect_request_events(session)
    process_summary, process_batch_rows, req_time_rows = parse_process_logs(session)
    runtime_series_by_role, runtime_wide_by_role, runtime_error_rows = collect_runtime_metric_exports(session)
    gpu_rows = collect_simple_metrics(session.session_dir / "metrics" / "gpu_metrics.jsonl")
    system_rows = collect_simple_metrics(session.session_dir / "metrics" / "system_metrics.jsonl")
    summary_row = summarize_session(
        session=session,
        request_rows=request_rows,
        request_event_rows=request_event_rows,
        process_summary=process_summary,
        metric_roles=sorted(runtime_wide_by_role.keys()),
    )

    created: list[Path] = []
    outputs: list[tuple[Path, list[dict[str, Any]], Optional[list[str]]]] = [
        (
            output_dir / "session_summary.csv",
            [summary_row],
            [
                "session_name",
                "mode",
                "timezone",
                "started_at",
                "ended_at",
                "session_duration_sec",
                "request_count",
                "cached_request_count",
                "total_prompt_tokens",
                "total_completion_tokens",
                "total_cached_tokens",
                "weighted_cache_hit_ratio",
                "e2e_latency_mean",
                "e2e_latency_p50",
                "e2e_latency_p90",
                "e2e_latency_p99",
                "decode_throughput_mean",
                "cache_hit_ratio_mean",
                "cache_hit_ratio_p50",
                "cache_hit_ratio_p90",
                "cache_hit_ratio_p99",
                "runtime_metric_roles",
                "session_dir",
            ],
        ),
        (
            output_dir / "request_details.csv",
            request_rows,
            [
                "timestamp",
                "timestamp_unix",
                "rid",
                "finish_reason_type",
                "prompt_tokens",
                "cached_tokens",
                "cache_hit_ratio",
                "completion_tokens",
                "e2e_latency",
                "queue_time",
                "decode_throughput",
                "total_retractions",
                "request_received_ts_epoch",
                "response_sent_to_client_ts_epoch",
                "request_finished_ts_epoch",
                "source_role",
            ],
        ),
        (
            output_dir / "request_events.csv",
            request_event_rows,
            [
                "timestamp",
                "timestamp_unix",
                "event",
                "rid",
                "source_role",
                "finish_reason_type",
                "prompt_tokens",
                "cached_tokens",
                "completion_tokens",
                "e2e_latency",
                "queue_time",
                "decode_throughput",
                "running_rids_count",
                "queued_rids_count",
            ],
        ),
        (
            output_dir / "process_batch_stats.csv",
            process_batch_rows,
            [
                "timestamp",
                "timestamp_unix",
                "source_role",
                "batch_phase",
                "running_reqs",
                "queue_reqs",
                "token_usage",
                "throughput_tokens_per_sec",
                "new_tokens",
                "cached_tokens",
                "total_tokens",
            ],
        ),
        (
            output_dir / "request_time_stats.csv",
            req_time_rows,
            [
                "timestamp",
                "timestamp_unix",
                "source_role",
                "rid",
                "input_len",
                "output_len",
                "request_type",
                "queue_duration_ms",
                "forward_duration_ms",
                "start_time",
            ],
        ),
        (
            output_dir / "gpu_metrics.csv",
            gpu_rows,
            [
                "timestamp",
                "timestamp_unix",
                "gpu_index",
                "gpu_name",
                "mem_used_mb",
                "mem_total_mb",
                "util_gpu_pct",
                "util_mem_pct",
                "temperature_c",
            ],
        ),
        (
            output_dir / "system_metrics.csv",
            system_rows,
            [
                "timestamp",
                "timestamp_unix",
                "cpu_util_pct",
                "mem_total_mb",
                "mem_used_mb",
                "mem_available_mb",
                "mem_util_pct",
            ],
        ),
        (
            output_dir / "runtime_metric_errors.csv",
            runtime_error_rows,
            ["timestamp", "timestamp_unix", "role", "endpoint", "error"],
        ),
    ]

    for path, rows, preferred in outputs:
        if not rows:
            continue
        write_csv(path, rows, preferred=preferred)
        created.append(path)

    for role, rows in runtime_series_by_role.items():
        if rows:
            path = output_dir / f"runtime_metric_series_{role}.csv"
            write_csv(
                path,
                rows,
                preferred=[
                    "timestamp",
                    "timestamp_unix",
                    "role",
                    "metric_name",
                    "metric_column",
                    "value",
                    "labels_json",
                ],
            )
            created.append(path)

    for role, rows in runtime_wide_by_role.items():
        if rows:
            path = output_dir / f"runtime_metrics_{role}.csv"
            write_csv(
                path,
                rows,
                preferred=["timestamp", "timestamp_unix", "role"],
            )
            created.append(path)

    return created


def main() -> None:
    args = parse_args()
    session_dirs = iter_sessions(
        session_root=args.session_root,
        session_dir=args.session_dir,
        selected_names=args.sessions,
        from_name=args.from_session,
        to_name=args.to_session,
        latest=args.latest,
    )
    if not session_dirs:
        raise SystemExit("No session directories matched the selection.")

    all_created: list[Path] = []
    for session_dir in session_dirs:
        created = export_session(session_dir=session_dir, output_subdir=args.output_subdir)
        all_created.extend(created)
        print(f"Exported {session_dir.name} -> {session_dir / args.output_subdir}")

    print(f"Generated {len(all_created)} CSV files")
    for path in all_created:
        print(path)


if __name__ == "__main__":
    main()
