#!/usr/bin/env python3
import argparse
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
from typing import Dict, List, Optional, Set
from urllib.request import urlopen
try:
    from .monitoring_view import (
        detect_runtime_feature_flags,
        render_status,
        render_status_single,
    )
except ImportError:
    from monitoring_view import (
        detect_runtime_feature_flags,
        render_status,
        render_status_single,
    )

try:
    from .monitoring_metrics import RuntimeMetricsCollector
except ImportError:
    from monitoring_metrics import RuntimeMetricsCollector



def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class ProcessHandle:
    name: str
    proc: subprocess.Popen
    stdout_fp: object
    stderr_fp: object


class ShutdownRequested(Exception):
    pass




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


def launch_process(
    name: str,
    script_path: Path,
    log_dir: Path,
    workdir: Path,
    env_overrides: Optional[Dict[str, str]] = None,
) -> ProcessHandle:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    stdout_fp = stdout_path.open("a", encoding="utf-8", buffering=1)
    stderr_fp = stderr_path.open("a", encoding="utf-8", buffering=1)

    proc_env = os.environ.copy()
    if env_overrides:
        proc_env.update({k: str(v) for k, v in env_overrides.items()})

    proc = subprocess.Popen(
        ["bash", str(script_path)],
        cwd=str(workdir),
        stdout=stdout_fp,
        stderr=stderr_fp,
        start_new_session=True,
        env=proc_env,
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




def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run SGLang experiment processes and scrape selected metrics into a session folder."
    )
    p.add_argument("--mode", type=str, choices=["pd", "single"], default="pd")
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

    p.add_argument("--single-port", type=int, default=int(os.environ.get("SGLANG_PORT", "30000")))

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
    mode = args.mode

    ms_dev_dir = Path("/workspace/sglang/ms_dev")
    script_prefill = ms_dev_dir / "sglang_start_server_pd_prefill.sh"
    script_decode = ms_dev_dir / "sglang_start_server_pd_decode.sh"
    script_router = ms_dev_dir / "sglang_start_router_pd.sh"
    script_single = ms_dev_dir / "sglang_start_server_request_observability.sh"

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

    if mode == "pd":
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
        startup_order = ["prefill", "decode", "router"]

        readiness_targets = {
            "prefill": f"http://{args.metrics_host}:{args.prefill_port}",
            "decode": f"http://{args.metrics_host}:{args.decode_port}",
            "router": f"http://{args.router_host}:{args.router_port}",
        }

        launch_env_overrides: Dict[str, Dict[str, str]] = {}
    else:
        role_endpoints = {
            "server": f"http://{args.metrics_host}:{args.single_port}/metrics",
        }
        cleanup_ports = [args.single_port]

        selectors = {
            "server": {"exact": sglang_exact, "prefixes": []},
        }

        out_files = {
            "server": metrics_dir / "server_metrics.jsonl",
        }

        enabled_roles = {
            "server": True,
        }

        role_cmds = {
            "server": script_single,
        }
        startup_order = ["server"]

        readiness_targets = {
            "server": f"http://{args.metrics_host}:{args.single_port}",
        }

        launch_env_overrides = {
            "server": {
                "SGLANG_PORT": str(args.single_port),
            }
        }

    if args.cleanup_extra_ports.strip():
        for item in args.cleanup_extra_ports.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                cleanup_ports.append(int(item))
            except ValueError:
                print(f"[run_pd_experiment] ignore invalid cleanup port: {item}", flush=True)

    handles: Dict[str, ProcessHandle] = {}
    stop_event = threading.Event()
    metrics_collector: Optional[RuntimeMetricsCollector] = None
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

    network_meta = {
        "metrics_host": args.metrics_host,
        "router_host": args.router_host,
        "single_port": args.single_port,
        "prefill_port": args.prefill_port,
        "decode_port": args.decode_port,
        "router_port": args.router_port,
        "router_metrics_port": args.router_metrics_port,
    }

    run_meta = {
        "mode": mode,
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
        "network": network_meta,
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

    print(f"[run_pd_experiment] mode={mode}", flush=True)
    print(f"[run_pd_experiment] session={session_name}", flush=True)
    print(f"[run_pd_experiment] session_dir={session_dir}", flush=True)
    print("[run_pd_experiment] launching processes...", flush=True)

    try:
        for role in startup_order:
            if not enabled_roles.get(role, False):
                continue
            handle = launch_process(
                name=role,
                script_path=role_cmds[role],
                log_dir=process_log_dir,
                workdir=ms_dev_dir,
                env_overrides=launch_env_overrides.get(role),
            )
            handles[role] = handle
            print(f"[run_pd_experiment] launched {role} pid={handle.proc.pid}", flush=True)

        readiness = {}
        for role in startup_order:
            if not enabled_roles.get(role, False):
                continue
            target = readiness_targets.get(role)
            if target is None:
                continue
            print(f"[run_pd_experiment] waiting {role} readiness on {target}...", flush=True)
            readiness[role] = wait_http_ready(target, args.startup_timeout, stop_event=stop_event)
            print(f"[run_pd_experiment] {role} ready via {readiness[role]}", flush=True)

        (meta_dir / "readiness.json").write_text(
            json.dumps({"ts": now_iso(), "readiness": readiness}, indent=2),
            encoding="utf-8",
        )

        feature_states = detect_runtime_feature_flags(process_log_dir, enabled_roles)

        active_role_endpoints = {
            role: endpoint for role, endpoint in role_endpoints.items() if enabled_roles.get(role, False)
        }
        active_selectors = {role: selectors[role] for role in active_role_endpoints}
        active_out_files = {role: out_files[role] for role in active_role_endpoints}

        metrics_collector = RuntimeMetricsCollector(
            role_endpoints=active_role_endpoints,
            selectors=active_selectors,
            out_files=active_out_files,
            gpu_file=metrics_dir / "gpu_metrics.jsonl",
            system_file=metrics_dir / "system_metrics.jsonl",
            interval_s=args.scrape_interval,
            stop_event=stop_event,
            timeout_s=args.metrics_timeout,
        )
        metrics_collector.start()

        print("[run_pd_experiment] metrics collection started; running. press Ctrl+C to stop.", flush=True)

        next_status_ts = 0.0
        while True:
            if stop_event.is_set():
                raise ShutdownRequested("external termination signal")

            dead = [role for role, h in handles.items() if h.proc.poll() is not None]
            if dead:
                raise RuntimeError(f"one or more processes exited unexpectedly: {dead}")

            if not args.quiet_status and metrics_collector is not None and time.time() >= next_status_ts:
                snap = metrics_collector.get_latest_state()
                if mode == "single":
                    block = render_status_single(
                        session_name=session_name,
                        session_dir=session_dir,
                        started_unix=started_unix,
                        enabled_roles=enabled_roles,
                        handles=handles,
                        snapshot=snap,
                        feature_states=feature_states,
                        use_color=use_color,
                    )
                else:
                    block = render_status(
                        session_name=session_name,
                        session_dir=session_dir,
                        started_unix=started_unix,
                        enabled_roles=enabled_roles,
                        handles=handles,
                        snapshot=snap,
                        feature_states=feature_states,
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
        if metrics_collector is not None:
            metrics_collector.join(timeout=5.0)

        for role in reversed(startup_order):
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
