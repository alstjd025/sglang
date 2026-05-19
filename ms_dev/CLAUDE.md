# `ms_dev/` — Local Experiment Tooling

Host-local dev scripts for installing SGLang into this machine's `.venv` and running
controlled serving experiments. **Not upstream**; treat as scratch tooling that may
diverge from main.

## Layout

```
ms_dev/
├── env.sh                       # dispatcher: SGLANG_ENV_PROFILE in {single, pd, all}
├── env.common.sh                # shared env vars (paths, model, ports, logging, caches)
├── env.single.sh                # single-server profile (TP, HiCache server-side)
├── env.pd.sh                    # PD profile (prefill/decode roles, ports, HiCache prefill)
├── lib_server.sh                # parse_obs_flags + append_obs_args + launch_server helpers
├── sglang_install_dev.sh        # one-shot: venv, system deps, rust, editable installs
├── start_server_no_pd.sh        # single-server launcher (--mode single)
├── start_server_pd_prefill.sh   # PD prefill role launcher
├── start_server_pd_decode.sh    # PD decode role launcher
├── start_router_pd.sh           # PD router (sglang_router) launcher
├── stop_servers.sh              # graceful stop + port cleanup
├── expctl/                      # Python orchestrator/monitor — see ms_dev/expctl/CLAUDE.md
├── experiments/                 # named, committed experiment wrappers (admission_*.sh)
├── env.local.sh                 # gitignored — personal/host-local overrides, auto-sourced by env.sh
├── admission_control_description.md  # top-level user-facing admission-control guide
└── runtime/                     # gitignored: sessions, caches, logs, cost_models
```

## Env profiles

`env.sh` is the entry point. Each launcher pre-sets `SGLANG_ENV_PROFILE` before sourcing:

- `single` → `env.common.sh` + `env.single.sh`
- `pd` → `env.common.sh` + `env.pd.sh`
- `all` → both (used by `stop_servers.sh` so it knows every port)

Vars are layered: profile-specific files only set what the profile needs; `env.common.sh`
holds anything shared.

### Notable shared env vars (`env.common.sh`)

- `SGLANG_REPO_ROOT`, `SGLANG_VENV_DIR`, `PYTHON_BIN`, `PIP_BIN`
- `SGLANG_MODEL_PATH` — defaults to `meta-llama/Llama-3.3-70B-Instruct`
- `SGLANG_HOST`, `SGLANG_PORT` (single = 31000)
- `SGLANG_ATTENTION_BACKEND` (default `triton`), `SGLANG_SAMPLING_BACKEND`
- `SGLANG_MEM_FRACTION_STATIC` (default `0.85`)
- `SGLANG_SERVE_EXTRA_ARGS` — free-form passthrough to `sglang.launch_server`
- `TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` → redirected under `runtime/cache/`
  because `/tmp` is mounted noexec on this host.
- `SGLANG_NUMA_BIND_V2=0` — must stay 0 on this host (numa-bind v2 writes to /tmp).

### Notable PD env vars (`env.pd.sh`)

- Ports: `SGLANG_PD_PREFILL_PORT=31002`, `SGLANG_PD_DECODE_PORT=31001`,
  `SGLANG_PD_ROUTER_PORT=31000`, `SGLANG_ROUTER_PROMETHEUS_PORT=39000`
- GPUs: `SGLANG_PD_PREFILL_BASE_GPU_ID=0`, `SGLANG_PD_DECODE_BASE_GPU_ID=1`
- `SGLANG_PD_DECODE_OFFLOAD_ENABLE=1` (default; decode L2 HiCache conflicts with PD radix
  cache in current SGLang, so use disaggregation offload path instead)
- HiCache flags: `SGLANG_PD_PREFILL_HICACHE_*`, `SGLANG_PD_DECODE_HICACHE_*`

## `lib_server.sh` patterns

Two helpers used by every launcher:

- `parse_obs_flags "$@"` — reads `--metrics`, `--request-logs`, `--trace`, `--obs=...`
  presets, sets `OBS_METRICS|OBS_REQUEST_LOGS|OBS_TRACE`, leaves unrecognized in
  `OBS_REMAINING_ARGS` (passthrough).
- `append_obs_args ARRAY log_dir metrics_dir crash_dir` — appends sglang CLI args
  conditionally based on the OBS flags. Also exports
  `SGLANG_LOG_SCHEDULER_STATUS_TARGET=stdout` (required when `--enable-metrics`).

A new admission helper follows the same pattern (see "Admission control" below).

## Admission control (Mooncake-style SLO admission)

Predictive SLO-based admission control rejects requests whose predicted TTFT/TBT would
violate configured SLOs. Module:
[python/sglang/srt/managers/admission_control/CLAUDE.md](../python/sglang/srt/managers/admission_control/CLAUDE.md).

### Env vars (added in `env.common.sh`)

| Env var | CLI flag | Default | Notes |
|---|---|---|---|
| `SGLANG_ADMISSION_TTFT_SLO_MS` | `--admission-ttft-slo-ms` | unset (off) | Stage 1 predicted TTFT SLO |
| `SGLANG_ADMISSION_TBT_SLO_MS` | `--admission-tbt-slo-ms` | unset (off) | Stage 2/3 predicted TBT SLO |
| `SGLANG_ADMISSION_PREFILL_COST_MODEL` | `--admission-prefill-cost-model-path` | unset | JSON path |
| `SGLANG_ADMISSION_TBT_COST_MODEL` | `--admission-tbt-cost-model-path` | unset | JSON path |
| `SGLANG_ADMISSION_TBT_EWMA_ALPHA` | `--admission-tbt-ewma-alpha` | `0.1` | Stage 3 smoothing |
| `SGLANG_ADMISSION_TBT_REACTIVE_RATIO` | `--admission-tbt-reactive-ratio` | `0.9` | Stage 3 trip threshold |
| `SGLANG_ADMISSION_DRY_RUN` | `--admission-dry-run` | `0` | Log decisions but always admit |
| `SGLANG_ADMISSION_DECISION_LOG` | `--admission-decision-log` | unset | JSONL path for offline replay (rank-0 only) |

If both SLO env vars are unset/empty, `lib_server.sh::append_admission_args` adds nothing
and admission control is fully off (server behaves as before).

### `lib_server.sh::append_admission_args` (added)

```bash
admission_args=()
append_admission_args admission_args
# ...
launch_server ... "${admission_args[@]}" ...
```

Used by `start_server_no_pd.sh` (Phase A). Phase A only supports
`disaggregation-mode=NULL`; in PD launchers the helper is wired but logs WARN and skips
if `SGLANG_ADMISSION_TTFT_SLO_MS` is set, since admission control is bypassed in PD mode.

### Cost models

Live under `runtime/cost_models/` (gitignored). Generate with
`tools/admission_control/fit_cost_model.py` — see
[tools/admission_control/CLAUDE.md](../tools/admission_control/CLAUDE.md).

### Per-session auto-routing (run_experiment.py)

When `run_experiment.py` detects any `SGLANG_ADMISSION_*_SLO_*` env var at session
start, it:
1. Auto-routes `SGLANG_ADMISSION_DECISION_LOG` to
   `<session_dir>/admission_decisions.jsonl` (unless user pinned it elsewhere).
2. Captures every `SGLANG_ADMISSION_*` env var into
   `<session_dir>/meta/run_meta.json` under the `admission_config` block,
   so post-hoc analysis can recover the active SLOs without grepping
   `process_logs/server.stderr.log`.

Pattern matches request_logs / metrics / crash_dump per-session routing.

### Where to put SLO settings

Three patterns, pick by use case:

| Pattern | What | When |
|---|---|---|
| Shell `export` before run | one-off | quick debug |
| `ms_dev/experiments/<name>.sh` (committed) | named, reproducible | comparing experiments — ★ |
| `ms_dev/env.local.sh` (gitignored, auto-sourced by `env.sh`) | host-local default | personal defaults |

**Don't edit `env.common.sh` / `env.single.sh` / `env.pd.sh` for experiments** —
those are upstream-tracked defaults. The pre-baked wrappers under
`ms_dev/experiments/` cover the common cases (`admission_ratio_only`,
`admission_ratio_with_safety`, `admission_absolute_only`, `admission_dryrun`).

### Live monitoring

`expctl/monitoring_view.py::detect_runtime_feature_flags()` detects `--admission-*` args
on the running process and renders `admission=ON|OFF|DRY_RUN` plus `ttft_slo`/`tbt_slo`
on the live status panel — see [expctl/CLAUDE.md](expctl/CLAUDE.md).

## Project Halo (request-level admission control + tracking)

Request-level admission control + per-request tracking on the single-instance
scheduler. Module:
[python/sglang/srt/managers/halo/CLAUDE.md](../python/sglang/srt/managers/halo/CLAUDE.md).

> Refactored from job-level to request-level on 2026-05-19. Job-level
> design docs are archived in `halo_dev/legacy_document/`; the job-level
> codebase is at git tag `halo-job-level-final`.

### CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--halo-enabled` | off | Master on/off. Off ⇒ zero-cost (no-op hooks). |
| `--halo-admission-policy` | `off` | `off` / `mooncake` / `vss` / `reactive`. |
| `--halo-slo-mode` | `ratio` | TTFT/TBT SLO interpretation: `ratio` (vs solo) or `absolute` (ms). |
| `--halo-admission-kv-cap-ratio` | `0.0` | Stage B′ KV-cache hard cap (0 = off). |
| `--halo-admission-violation-threshold` | `0.2` | vss-policy reject threshold. |
| `--halo-admission-dry-run` | off | Log decisions but always admit. |
| `--halo-admission-decision-log` | unset (auto-routed) | per-decision JSONL. |
| `--halo-{prefill,tbt,step}-cost-model-path` | unset | cost model JSONs. |
| `--halo-tick-interval-ms` | `100` | request-tracker tick interval. |

Per-request SLOs travel in the request body as `halo_ttft_slo` /
`halo_tbt_slo` / `halo_e2e_slo`; `halo_bypass` marks server-internal traffic.

### Env vars (`env.common.sh` → `lib_server.sh::append_halo_args` → `--halo-*`)

| Env var | CLI flag |
|---|---|
| `SGLANG_HALO_ENABLED` | `--halo-enabled` |
| `SGLANG_HALO_TICK_INTERVAL_MS` | `--halo-tick-interval-ms` |
| `SGLANG_HALO_ADMISSION_POLICY` | `--halo-admission-policy` |
| `SGLANG_HALO_SLO_MODE` | `--halo-slo-mode` |
| `SGLANG_HALO_ADMISSION_VIOLATION_THRESHOLD` | `--halo-admission-violation-threshold` |
| `SGLANG_HALO_TBT_REACTIVE_RATIO` | `--halo-tbt-reactive-ratio` |
| `SGLANG_HALO_ADMISSION_DRY_RUN` | `--halo-admission-dry-run` |
| `SGLANG_HALO_ADMISSION_DECISION_LOG` | `--halo-admission-decision-log` |
| `SGLANG_HALO_ADMISSION_KV_CAP_RATIO` | `--halo-admission-kv-cap-ratio` |
| `SGLANG_HALO_{PREFILL,TBT,STEP}_COST_MODEL` | `--halo-{prefill,tbt,step}-cost-model-path` |
| `SGLANG_HALO_COST_MODEL_SAMPLE_{LOG,EVERY}` | `--halo-cost-model-sample-{log,every}` |

`ms_dev/experiments/halo_base.sh` turns Halo on with the step cost model;
set `SGLANG_HALO_ADMISSION_POLICY` to enable a policy. `halo_off.sh` clears
all `SGLANG_HALO_*`.

### Per-session auto-routing (run_experiment.py)

Auto-routes `--halo-admission-decision-log` to
`<session_dir>/admission_decisions.jsonl` (unless pinned) and records the
halo flag snapshot into `meta/run_meta.json`.

## Quick start

```bash
# Install (one-shot)
bash ms_dev/sglang_install_dev.sh

# Single server
bash ms_dev/start_server_no_pd.sh

# PD (3 processes — open 3 terminals or use expctl)
bash ms_dev/start_server_pd_prefill.sh
bash ms_dev/start_server_pd_decode.sh
bash ms_dev/start_router_pd.sh

# Orchestrated (recommended) — runs everything + monitor + saves session
python3 ms_dev/expctl/server_run_experiment.py --mode single
python3 ms_dev/expctl/server_run_experiment.py --mode pd

# Stop everything
bash ms_dev/stop_servers.sh
```

## Host-specific notes

- `sudo` is disabled on this host. `sglang_install_dev.sh` uses `gcsudo`
  (`/engrid/ensh/gpubin/ctn_gcsudo`) via the `${GCSUDO}` variable.
- `/tmp` is mounted noexec → forces `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR`
  redirection AND `SGLANG_NUMA_BIND_V2=0`.
- `runtime/` is gitignored (commit `b1078f0ac`). Never check in session/cache contents.

## Common workflows

### Run an admission-control experiment in dry-run mode

```bash
export SGLANG_ADMISSION_TTFT_SLO_MS=30000
export SGLANG_ADMISSION_TBT_SLO_MS=200
export SGLANG_ADMISSION_PREFILL_COST_MODEL=ms_dev/runtime/cost_models/prefill_llama3-70b.json
export SGLANG_ADMISSION_TBT_COST_MODEL=ms_dev/runtime/cost_models/tbt_llama3-70b.json
export SGLANG_ADMISSION_DRY_RUN=1

python3 ms_dev/expctl/server_run_experiment.py --mode single
# status panel shows: admission=DRY_RUN ttft=30s tbt=200ms
```

After session, replay the decisions to find a good operating point:

```bash
python3 tools/admission_control/replay_admission.py \
    --session-dir ms_dev/runtime/sessions/<TIMESTAMP> \
    --ttft-slo 20000 30000 60000 --tbt-slo 100 150 200
```

Then unset `SGLANG_ADMISSION_DRY_RUN` (or set to `0`) and re-run for real enforcement.
