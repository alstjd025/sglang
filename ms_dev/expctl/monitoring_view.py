#!/usr/bin/env python3
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Set


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")

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


def fmt_kv_used_max(used_tokens: Optional[float], max_tokens: Optional[float]) -> str:
    return f"{fmt_int(used_tokens)}/{fmt_int(max_tokens)} tok"


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


def _read_file_head(path: Path, max_bytes: int = 262144) -> Optional[str]:
    try:
        with path.open("rb") as f:
            return f.read(max_bytes).decode("utf-8", errors="replace")
    except Exception:
        return None


def detect_launch_flag(process_log_dir: Path, role: str, flag: str) -> Optional[bool]:
    text = _read_file_head(process_log_dir / f"{role}.stdout.log")
    if text is None:
        return None
    return flag in text


def detect_runtime_feature_flags(
    process_log_dir: Path,
    enabled_roles: Dict[str, bool],
) -> Dict[str, Optional[bool]]:
    states: Dict[str, Optional[bool]] = {
        "prefill_hicache": None,
        "decode_hicache_l2": None,
        "decode_offload": None,
        "server_hicache": None,
        "server_l3": None,
    }

    if enabled_roles.get("prefill", False):
        states["prefill_hicache"] = detect_launch_flag(
            process_log_dir, "prefill", "--enable-hierarchical-cache"
        )

    if enabled_roles.get("decode", False):
        states["decode_hicache_l2"] = detect_launch_flag(
            process_log_dir, "decode", "--enable-hierarchical-cache"
        )
        states["decode_offload"] = detect_launch_flag(
            process_log_dir,
            "decode",
            "--disaggregation-decode-enable-offload-kvcache",
        )

    if enabled_roles.get("server", False):
        states["server_hicache"] = detect_launch_flag(
            process_log_dir, "server", "--enable-hierarchical-cache"
        )
        states["server_l3"] = detect_launch_flag(
            process_log_dir, "server", "--hicache-storage-backend"
        )

    return states


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


def feature_state_text(value: Optional[bool], role_enabled: bool, use_color: bool) -> str:
    if not role_enabled:
        return colorize("N/A", "muted", use_color)
    if value is None:
        return colorize("UNKNOWN", "warn", use_color)
    if value:
        return colorize("ON", "ok", use_color)
    return colorize("OFF", "muted", use_color)


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


def metric_cell(label: str, value: str, label_width: int = 13) -> str:
    return f"{label:<{label_width}} {value}"


def metric_row(items: List[tuple], label_width: int = 13) -> str:
    return " | ".join(metric_cell(k, v, label_width=label_width) for k, v in items)


def render_gpu_lines(gpu_samples: List[dict], gpu_err: Optional[str], use_color: bool) -> List[str]:
    if not gpu_samples:
        return [f"GPU: {colorize('no samples', 'warn', use_color)} ({gpu_err or 'nvidia-smi unavailable'})"]

    valid_samples = [s for s in gpu_samples if isinstance(s, dict)]
    valid_samples.sort(key=lambda s: int(s.get("gpu_index", 9999)))

    lines: List[str] = []
    lines.append(f"GPU: {colorize(str(len(valid_samples)), 'accent', use_color)} cards (per-GPU)")

    for sample in valid_samples:
        gpu_index = int(sample.get("gpu_index", -1))
        mem_used_mb = float(sample.get("mem_used_mb", 0.0))
        mem_total_mb = float(sample.get("mem_total_mb", 0.0))
        util_gpu_pct = float(sample.get("util_gpu_pct", 0.0))
        temp_c = float(sample.get("temperature_c", 0.0))

        mem_ratio = (mem_used_mb / mem_total_mb * 100.0) if mem_total_mb > 0 else None
        mem_text = (
            f"{mem_used_mb / 1024.0:.1f}/{mem_total_mb / 1024.0:.1f} GiB "
            f"({color_high_bad(mem_ratio, f'{fmt_float(mem_ratio, 1)}%', warn=80.0, bad=90.0, enabled=use_color)})"
        )

        util_text = color_high_bad(
            util_gpu_pct,
            f"{fmt_float(util_gpu_pct, 1)}%",
            warn=70.0,
            bad=90.0,
            enabled=use_color,
        )
        temp_text = color_high_bad(
            temp_c,
            f"{fmt_float(temp_c, 1)}C",
            warn=80.0,
            bad=90.0,
            enabled=use_color,
        )

        lines.append(
            "     "
            + metric_row(
                [
                    (f"GPU{gpu_index}", ""),
                    ("mem", mem_text),
                    ("util", util_text),
                    ("temp", temp_text),
                ],
                label_width=6,
            )
        )

    return lines


def render_status(
    session_name: str,
    session_dir: Path,
    started_unix: float,
    enabled_roles: Dict[str, bool],
    handles: Dict[str, ProcessHandle],
    snapshot: Dict[str, object],
    feature_states: Dict[str, Optional[bool]],
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
    prefill_kv_used = metric_value(prefill, "sglang:num_used_tokens")
    prefill_kv_max = metric_value(prefill, "sglang:max_total_num_tokens")
    prefill_cache_hit = metric_value(prefill, "sglang:cache_hit_rate")

    decode_throughput = metric_value(decode, "sglang:gen_throughput")
    decode_running = metric_value(decode, "sglang:num_running_reqs")
    decode_queue = metric_value(decode, "sglang:num_queue_reqs")
    decode_prealloc_q = metric_value(decode, "sglang:num_decode_prealloc_queue_reqs")
    decode_transfer_q = metric_value(decode, "sglang:num_decode_transfer_queue_reqs")
    decode_transfer_fail = metric_value(decode, "sglang:num_transfer_failed_reqs_total")
    decode_token_usage = metric_value(decode, "sglang:token_usage")
    decode_kv_used = metric_value(decode, "sglang:num_used_tokens")
    decode_kv_max = metric_value(decode, "sglang:max_total_num_tokens")
    decode_cache_hit = metric_value(decode, "sglang:cache_hit_rate")

    prefill_kv_ratio = (
        (prefill_kv_used / prefill_kv_max)
        if prefill_kv_used is not None and prefill_kv_max is not None and prefill_kv_max > 0
        else None
    )
    decode_kv_ratio = (
        (decode_kv_used / decode_kv_max)
        if decode_kv_used is not None and decode_kv_max is not None and decode_kv_max > 0
        else None
    )

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

    gpu_lines = render_gpu_lines(gpu_samples, gpu_err, use_color)

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

    prefill_hicache_text = feature_state_text(
        feature_states.get("prefill_hicache"),
        enabled_roles.get("prefill", False),
        use_color,
    )
    decode_hicache_l2_text = feature_state_text(
        feature_states.get("decode_hicache_l2"),
        enabled_roles.get("decode", False),
        use_color,
    )
    decode_offload_text = feature_state_text(
        feature_states.get("decode_offload"),
        enabled_roles.get("decode", False),
        use_color,
    )

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
    lines.append(
        "features: "
        + " | ".join(
            [
                f"prefill_hicache={prefill_hicache_text}",
                f"decode_hicache_l2={decode_hicache_l2_text}",
                f"decode_offload={decode_offload_text}",
            ]
        )
    )
    lines.append(colorize("-" * 118, "muted", use_color))

    lines.append(
        f"{colorize('PREFILL', 'accent', use_color)} "
        f"(age={fmt_age(prefill_age)}, err={prefill_err_text})"
    )
    lines.append(
        "         "
        + metric_row(
            [
                ("running", color_high_bad(prefill_running, fmt_int(prefill_running), warn=16, bad=64, enabled=use_color)),
                ("queue", color_high_bad(prefill_queue, fmt_int(prefill_queue), warn=1, bad=10, enabled=use_color)),
                ("prealloc_q", color_high_bad(prefill_prealloc_q, fmt_int(prefill_prealloc_q), warn=1, bad=10, enabled=use_color)),
                ("inflight_q", color_high_bad(prefill_inflight_q, fmt_int(prefill_inflight_q), warn=1, bad=10, enabled=use_color)),
                ("bootstrap_fail", color_high_bad(prefill_bootstrap_fail, fmt_int(prefill_bootstrap_fail), warn=1, bad=5, enabled=use_color)),
            ]
        )
    )
    lines.append(
        "         "
        + metric_row(
            [
                ("kv_used/max", color_high_bad(prefill_kv_ratio if prefill_kv_ratio is None else prefill_kv_ratio*100.0, fmt_kv_used_max(prefill_kv_used, prefill_kv_max), warn=70.0, bad=90.0, enabled=use_color)),
                ("token_usage", color_high_bad(prefill_token_usage if prefill_token_usage is None else prefill_token_usage*100.0, fmt_pct(prefill_token_usage), warn=70.0, bad=90.0, enabled=use_color)),
                ("cache_hit", color_high_good(prefill_cache_hit, fmt_pct(prefill_cache_hit), warn=0.50, good=0.80, enabled=use_color)),
                ("TTFT_mean", color_high_bad(prefill_ttft_ms, f'{fmt_float(prefill_ttft_ms, 2)}ms', warn=800.0, bad=2000.0, enabled=use_color)),
            ]
        )
    )

    lines.append(
        f"{colorize('DECODE', 'accent', use_color)} "
        f"(age={fmt_age(decode_age)}, err={decode_err_text})"
    )
    lines.append(
        "         "
        + metric_row(
            [
                ("throughput", color_high_good(decode_throughput, f'{fmt_float(decode_throughput, 2)} tok/s', warn=1.0, good=100.0, enabled=use_color)),
                ("running", color_high_bad(decode_running, fmt_int(decode_running), warn=16, bad=64, enabled=use_color)),
                ("queue", color_high_bad(decode_queue, fmt_int(decode_queue), warn=1, bad=10, enabled=use_color)),
                ("prealloc_q", color_high_bad(decode_prealloc_q, fmt_int(decode_prealloc_q), warn=1, bad=10, enabled=use_color)),
                ("transfer_q", color_high_bad(decode_transfer_q, fmt_int(decode_transfer_q), warn=1, bad=10, enabled=use_color)),
            ]
        )
    )
    lines.append(
        "         "
        + metric_row(
            [
                ("transfer_fail", color_high_bad(decode_transfer_fail, fmt_int(decode_transfer_fail), warn=1, bad=5, enabled=use_color)),
                ("kv_used/max", color_high_bad(decode_kv_ratio if decode_kv_ratio is None else decode_kv_ratio*100.0, fmt_kv_used_max(decode_kv_used, decode_kv_max), warn=70.0, bad=90.0, enabled=use_color)),
                ("token_usage", color_high_bad(decode_token_usage if decode_token_usage is None else decode_token_usage*100.0, fmt_pct(decode_token_usage), warn=70.0, bad=90.0, enabled=use_color)),
                ("cache_hit", color_high_good(decode_cache_hit, fmt_pct(decode_cache_hit), warn=0.50, good=0.80, enabled=use_color)),
                ("TBT_mean", color_high_bad(decode_tbt_ms, f'{fmt_float(decode_tbt_ms, 2)}ms', warn=60.0, bad=120.0, enabled=use_color)),
            ]
        )
    )

    lines.append(
        f"{colorize('ROUTER', 'accent', use_color)} "
        f"(age={fmt_age(router_age)}, err={router_err_text})"
    )
    lines.append(
        "         "
        + metric_row(
            [
                ("http_active", color_high_bad(router_http_active, fmt_int(router_http_active), warn=32, bad=128, enabled=use_color)),
                ("worker_active", color_high_bad(router_worker_active, fmt_int(router_worker_active), warn=32, bad=128, enabled=use_color)),
                ("worker_conn", color_high_bad(router_worker_conn, fmt_int(router_worker_conn), warn=32, bad=128, enabled=use_color)),
                ("pool_size", color_high_good(router_pool_size, fmt_int(router_pool_size), warn=1, good=2, enabled=use_color)),
            ]
        )
    )
    lines.append(
        "         "
        + metric_row(
            [
                ("cb_open", color_high_bad(cb_open, fmt_int(cb_open), warn=1, bad=2, enabled=use_color)),
                ("cb_half_open", color_high_bad(cb_half_open, fmt_int(cb_half_open), warn=1, bad=3, enabled=use_color)),
                ("router_TTFT", color_high_bad(router_ttft_ms, f'{fmt_float(router_ttft_ms, 2)}ms', warn=800.0, bad=2000.0, enabled=use_color)),
                ("router_TPOT", color_high_bad(router_tpot_ms, f'{fmt_float(router_tpot_ms, 2)}ms', warn=60.0, bad=120.0, enabled=use_color)),
            ]
        )
    )

    lines.extend(gpu_lines)
    lines.append(system_summary)
    lines.append(colorize("=" * 118, "muted", use_color))
    return "\n".join(lines)


def render_status_single(
    session_name: str,
    session_dir: Path,
    started_unix: float,
    enabled_roles: Dict[str, bool],
    handles: Dict[str, ProcessHandle],
    snapshot: Dict[str, object],
    feature_states: Dict[str, Optional[bool]],
    use_color: bool,
) -> str:
    roles = snapshot.get("roles") if isinstance(snapshot.get("roles"), dict) else {}
    server = roles.get("server") if isinstance(roles.get("server"), dict) else {}

    server_err = server.get("error") if isinstance(server.get("error"), str) else None
    server_age = scrape_age_seconds(server)

    server_ttft_ms = hist_mean_ms(server, "sglang:time_to_first_token_seconds")
    server_tbt_ms = hist_mean_ms(server, "sglang:inter_token_latency_seconds")

    server_throughput = metric_value(server, "sglang:gen_throughput")
    server_running = metric_value(server, "sglang:num_running_reqs")
    server_queue = metric_value(server, "sglang:num_queue_reqs")
    server_prealloc_q = metric_value_any(
        server,
        [
            "sglang:num_prefill_prealloc_queue_reqs",
            "sglang:num_decode_prealloc_queue_reqs",
        ],
    )
    server_inflight_q = metric_value_any(
        server,
        [
            "sglang:num_prefill_inflight_queue_reqs",
            "sglang:num_decode_transfer_queue_reqs",
        ],
    )
    server_retracted = metric_value_any(
        server,
        [
            "sglang:num_retracted_reqs",
            "sglang:num_retracted_requests_total",
        ],
    )

    server_token_usage = metric_value(server, "sglang:token_usage")
    server_kv_used = metric_value(server, "sglang:num_used_tokens")
    server_kv_max = metric_value(server, "sglang:max_total_num_tokens")
    server_cache_hit = metric_value(server, "sglang:cache_hit_rate")

    server_kv_ratio = (
        (server_kv_used / server_kv_max)
        if server_kv_used is not None and server_kv_max is not None and server_kv_max > 0
        else None
    )

    gpu = snapshot.get("gpu") if isinstance(snapshot.get("gpu"), dict) else {}
    gpu_err = gpu.get("error") if isinstance(gpu.get("error"), str) else None
    gpu_samples = gpu.get("samples") if isinstance(gpu.get("samples"), list) else []

    system = snapshot.get("system") if isinstance(snapshot.get("system"), dict) else {}
    system_err = system.get("error") if isinstance(system.get("error"), str) else None
    system_cpu = float(system.get("cpu_util_pct")) if system.get("cpu_util_pct") is not None else None
    system_mem_used_mb = float(system.get("mem_used_mb")) if system.get("mem_used_mb") is not None else None
    system_mem_total_mb = float(system.get("mem_total_mb")) if system.get("mem_total_mb") is not None else None
    system_mem_util_pct = float(system.get("mem_util_pct")) if system.get("mem_util_pct") is not None else None

    gpu_lines = render_gpu_lines(gpu_samples, gpu_err, use_color)

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

    server_state = color_state_text(process_state('server', enabled_roles, handles), use_color)
    server_err_text = colorize(server_err, 'bad', use_color) if server_err else colorize('none', 'ok', use_color)

    server_hicache_text = feature_state_text(
        feature_states.get("server_hicache"),
        enabled_roles.get("server", False),
        use_color,
    )
    server_l3_text = feature_state_text(
        feature_states.get("server_l3"),
        enabled_roles.get("server", False),
        use_color,
    )

    lines = []
    lines.append(colorize("=" * 118, "muted", use_color))
    lines.append(
        f"{colorize('SGLang Experiment Monitor', 'title', use_color)} | "
        f"session={colorize(session_name, 'accent', use_color)} | "
        f"uptime={fmt_duration(time.time() - started_unix)} | now={now_iso()}"
    )
    lines.append(f"session_dir={session_dir}")
    lines.append(f"processes: server={server_state}")
    lines.append(
        "features: "
        + " | ".join(
            [
                f"server_hicache={server_hicache_text}",
                f"server_l3_storage={server_l3_text}",
            ]
        )
    )
    lines.append(colorize("-" * 118, "muted", use_color))

    lines.append(
        f"{colorize('SERVER', 'accent', use_color)} "
        f"(age={fmt_age(server_age)}, err={server_err_text})"
    )
    lines.append(
        "        "
        + metric_row(
            [
                ("throughput", color_high_good(server_throughput, f'{fmt_float(server_throughput, 2)} tok/s', warn=1.0, good=100.0, enabled=use_color)),
                ("running", color_high_bad(server_running, fmt_int(server_running), warn=16, bad=64, enabled=use_color)),
                ("queue", color_high_bad(server_queue, fmt_int(server_queue), warn=1, bad=10, enabled=use_color)),
                ("prealloc_q", color_high_bad(server_prealloc_q, fmt_int(server_prealloc_q), warn=1, bad=10, enabled=use_color)),
                ("inflight_q", color_high_bad(server_inflight_q, fmt_int(server_inflight_q), warn=1, bad=10, enabled=use_color)),
            ]
        )
    )
    lines.append(
        "        "
        + metric_row(
            [
                ("retracted", color_high_bad(server_retracted, fmt_int(server_retracted), warn=1, bad=10, enabled=use_color)),
                ("kv_used/max", color_high_bad(server_kv_ratio if server_kv_ratio is None else server_kv_ratio*100.0, fmt_kv_used_max(server_kv_used, server_kv_max), warn=70.0, bad=90.0, enabled=use_color)),
                ("token_usage", color_high_bad(server_token_usage if server_token_usage is None else server_token_usage*100.0, fmt_pct(server_token_usage), warn=70.0, bad=90.0, enabled=use_color)),
                ("cache_hit", color_high_good(server_cache_hit, fmt_pct(server_cache_hit), warn=0.50, good=0.80, enabled=use_color)),
                ("TTFT_mean", color_high_bad(server_ttft_ms, f'{fmt_float(server_ttft_ms, 2)}ms', warn=800.0, bad=2000.0, enabled=use_color)),
                ("TBT_mean", color_high_bad(server_tbt_ms, f'{fmt_float(server_tbt_ms, 2)}ms', warn=60.0, bad=120.0, enabled=use_color)),
            ],
            label_width=11,
        )
    )

    lines.extend(gpu_lines)
    lines.append(system_summary)
    lines.append(colorize("=" * 118, "muted", use_color))
    return "\n".join(lines)


