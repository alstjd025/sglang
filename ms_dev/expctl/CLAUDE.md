# `ms_dev/expctl/` — Experiment Orchestrator

Python-based session runner that launches SGLang processes, scrapes Prometheus +
GPU/system metrics, renders a live status panel, and saves a structured session
folder for offline analysis. User-facing operator docs are in [README.md](README.md).
**This file is for Claude.**

## Module responsibilities

| File | Role |
|---|---|
| `run_experiment.py` | Entry point + orchestrator. Launches/stops processes, readiness probes, session metadata, status loop. |
| `monitoring_metrics.py` | `RuntimeMetricsCollector` (threaded). Periodic Prometheus scrape, GPU/system sampling, JSONL persistence, latest-snapshot accessor. |
| `monitoring_view.py` | Pure renderer. Takes a snapshot dict and produces the status panel string. PD/Single variants. Also hosts `detect_runtime_feature_flags()` which inspects launch args from process logs. |
| `export_session_csv.py` | Post-run: reads `metrics/*.jsonl` + `raw/request_*` + `process_logs/` and emits CSVs under `exports/csv/`. |
| `plot_session_metrics.py` | Post-run: reads CSVs and produces `exports/plots/*.png`. |

Hard rule: collectors don't render, renderers don't collect, run_experiment.py glues
them together. Don't merge concerns.

## Session layout

Created under `SGLANG_RUNTIME_DIR/sessions/YYYYMMDD_HHMMSS/`:

```
sessions/<ts>/
├── meta/         run_meta.json, readiness.json, run_end.json, precleanup_ports.json
├── metrics/      gpu_metrics.jsonl, system_metrics.jsonl, server|prefill|decode|router_metrics.jsonl
├── process_logs/ server.* | prefill.* | decode.* | router.*
├── raw/          request_logs/ request_metrics/ crash_dump/ router_logs/
└── exports/      csv/ plots/   (created by export/plot scripts)
```

`runtime/` is gitignored.

## Modes

`--mode pd` (default): launches prefill + decode + router.
`--mode single`: launches one SGLang server only.

PD ports come from `env.pd.sh`; single port from `SGLANG_PORT` (`env.common.sh`).

## How feature flags reach the panel

`monitoring_view.detect_runtime_feature_flags(process_log_dir, enabled_roles)` reads
the head of each role's process log (which contains the `[start_server_*] launching:`
quoted command line) and grep-extracts CLI flags via:

- `detect_launch_flag(path, role, flag)` → bool ("did this flag appear")
- `detect_launch_arg(path, role, arg_name)` → str | None ("what value followed
  `--arg-name`")

Existing detected flags: `--enable-hierarchical-cache`,
`--disaggregation-decode-enable-offload-kvcache`, `--hicache-storage-backend`,
`--model-path`.

### Adding admission control to the panel

`detect_runtime_feature_flags()` gains:

```python
ttft = detect_launch_arg(process_log_dir, role, "--admission-ttft-slo-ms")
tbt  = detect_launch_arg(process_log_dir, role, "--admission-tbt-slo-ms")
states[f"{role}_admission"] = (ttft is not None) or (tbt is not None)
states[f"{role}_admission_dry_run"] = detect_launch_flag(
    process_log_dir, role, "--admission-dry-run"
)
states[f"{role}_admission_ttft_slo_ms"] = ttft  # str|None — display value
states[f"{role}_admission_tbt_slo_ms"]  = tbt
```

Renderer (`render_status_single` line ~689 area, plus `render_status` for PD) shows
one cell:

- `admission=OFF` (muted) when off
- `admission=ON ttft=30s tbt=200ms` (ok) when active
- `admission=DRY_RUN ...` (warn) when dry-run
- `admission=N/A` (muted) when role disabled

Admission control is single-instance only (Phase A) → only the `server` (single mode)
or `prefill` role (PD mode, future) carries meaningful state.

## How metrics get scraped

`RuntimeMetricsCollector` (line 74 of `monitoring_metrics.py`) is a `threading.Thread`:

- Polls each role's Prometheus endpoint via `fetch_text(url, timeout)` →
  `parse_prometheus_metrics(text)` (handles labels + values).
- Selects metric names via `metric_selected(name, exact, prefixes)` against allowlists.
- Appends one record per scrape to `metrics/<role>_metrics.jsonl`.
- Exposes `get_snapshot()` for `monitoring_view.render_status*()`.

### Adding admission metrics

In the metric allowlist (`monitoring_metrics.py` constants near top), add the prefix
`sglang:admission_` to capture all four admission counters/histograms/gauges. No code
change beyond the allowlist — they flow through the same scrape/persist/snapshot path.

Renderers can pull them via existing helpers:

```python
admit_total  = metric_value(role, "sglang:admission_decisions_total")  # with labels
ttft_p95_ms  = hist_quantile(role, "sglang:admission_predicted_ttft_ms", q=0.95)
tbt_ewma     = metric_value(role, "sglang:admission_tbt_ewma_ms")
```

For PD mode admission is currently bypassed (Phase A) so panel cell stays muted.

## Adding a new metric to status panel + CSV + plot

1. Add metric name/prefix to `monitoring_metrics.py` allowlist.
2. Render in `monitoring_view.py` via existing `metric_value/series_sum/hist_*` helpers.
3. Extract in `export_session_csv.py` to a new column or new CSV.
4. (Optional) Add a default plot in `plot_session_metrics.py`.

Admission control follows this exact flow.

## CSV / plot conventions

Default CSVs (`exports/csv/`):
`session_summary.csv`, `request_details.csv`, `request_events.csv`,
`process_batch_stats.csv`, `request_time_stats.csv`, `runtime_metrics_<role>.csv`,
`runtime_metric_series_<role>.csv`, `gpu_metrics.csv`, `system_metrics.csv`.

Default plots: request overview, runtime overview, GPU overview, system overview,
process batch stats.

### Admission additions

- `exports/csv/admission_decisions.csv` — derived from
  `sglang:admission_decisions_total` time series + (when dry-run) per-decision JSONL
  log. Columns: `ts, decision, reason, predicted_ttft_ms, predicted_tbt_ms,
  ewma_tbt_ms, queue_predicted_ms, rid`.
- `exports/plots/admission_overview.png` — reject-rate timeseries by reason +
  predicted-vs-actual TTFT/TBT scatter.

## Quick reference

```bash
# Live monitor + saved session
python3 ms_dev/expctl/run_experiment.py --mode single

# After run
python3 ms_dev/expctl/export_session_csv.py --session-dir <sess>
python3 ms_dev/expctl/plot_session_metrics.py --session-dir <sess>
```

See [README.md](README.md) for the full operator-facing CLI reference and example
invocations.
