# Halo Module — Job-level Slowdown Tracking (Phase 1)

HALO Phase 1: a job-level observability layer for SGLang. Tracks each job's actual
slowdown ratio (`actual_elapsed / solo_run_elapsed`) versus its SLO. Phase 1
**does not** make admission or scheduling decisions — it only observes.

> Full project rationale + Phase plan: `ms_dev/halo_dev/CLAUDE.md`.
> Phase 2/3 (job-level admission/scheduling) will reuse this module's Job/Registry/
> Tracker primitives without re-defining them.

## Why job-level?

Existing `admission_control/` operates at request granularity (each request's
predicted TTFT/TBT vs SLO). Halo operates at *job* granularity — a job is the
client-supplied grouping of multiple requests that share an SLO target
(e.g., a single agent loop, a LangGraph DAG, a multi-turn conversation).

The SLO is defined as **end-to-end slowdown** of the job versus its solo-run
baseline (e.g., SLO=5 means "the job's total wall time may be at most 5× what
it would take running alone"). Per-request SLOs cannot capture this because a
fast request can mask a slow one in the same job, or vice versa.

## Phase 1 scope

| In scope | Out of scope |
|---|---|
| Job dataclass + JobState enum | Job-level admission gate (Phase 2) |
| JobRegistry (rid ↔ job lookup) | Job-level scheduling priority (Phase 2) |
| 100ms periodic SlowdownTracker.sweep | DAG-aware orchestration (Phase 3) |
| Reuse cost models from `admission_control/` | New cost-model fitting |
| Off-by-default flag | Replacing `admission_control/` |

## Architecture

```
managers/halo/
├── __init__.py             # public API: HaloController, Job, JobRegistry
├── job.py                  # Job dataclass + JobState enum
├── job_registry.py         # JobRegistry: rid↔job, lifecycle
├── slowdown_tracker.py     # sweep() — compute per-request ratios, aggregate to job
└── controller.py           # HaloController: on/off, hook into scheduler
```

## Class summary

| Class | Responsibility |
|---|---|
| `JobState` | Enum: `QUEUED / RUNNING / COMPLETE / REJECTED` (REJECTED reserved for Phase 2) |
| `Job` | task_struct-like dataclass: id, slo, slowdown_max/mean, counters, rids set, history |
| `JobRegistry` | `register/record_admission/record_completion/active_jobs/snapshot` |
| `RequestExecutionInfo` | Plain dataclass the scheduler builds per active req before sweep |
| `SlowdownTracker` | Pure compute. `compute_request_slowdown()` + `sweep()` |
| `HaloConfig` | Snapshot of CLI flags / env-derived parameters |
| `HaloController` | Glue. Owned by scheduler. `register_request / on_request_finished / tick / snapshot` |

## Key invariants

- **Single-threaded access**: all methods called from the scheduler process's main loop.
  No internal locking. Tests must mirror this.
- **rid→job mapping is the source of truth** for ownership. A request's
  `halo_job_id` field is the seed at admission time; afterwards the registry owns it.
- **Initial `slowdown_max = slowdown_mean = job.slo`** (spec choice — measurement-less
  worst-case fallback; tracker overwrites on first sweep).
- **Aggregation**: `slowdown_max` = max over per-request slowdowns; `slowdown_mean`
  = arithmetic mean. Both stored on Job; bounded `slowdown_history` deque
  (`maxlen=64`) stores `(max, mean)` tuples for debugging.

## Slowdown computation

For each active request `r` belonging to job `j`:

```
actual_elapsed_ms   = now - r.first_admitted_ts
solo_prefill_ms     = prefill_cost.estimate_ms(prompt_len, prefix_len_at_admission)
solo_tbt_ms         = tbt_cost.estimate_ms(batch_size=1, per_req_kv=r.kv_len_now)
solo_elapsed_ms     = solo_prefill_ms + solo_tbt_ms * decoded_tokens_so_far
request_slowdown    = actual_elapsed_ms / max(solo_elapsed_ms, eps)
```

Then for each job: `slowdown_max = max(...)` and `slowdown_mean = mean(...)`.

**Cost model brittleness inherited**: the TBT cost model currently has a narrow
prediction range (~55 ms regardless of batch composition — see
`ms_dev/runtime/cost_models/README.md` item 3). That means `solo_tbt_ms`
under-discriminates context, which propagates to slowdown estimates. Phase 1 ships
this as a known limitation; Phase 2 work should refine the cost model first.

## CLI flags (added in `server_args.py`)

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--halo-enabled` | `SGLANG_HALO_ENABLED` | `False` | Master on/off. Off ⇒ zero-cost (no-op hooks) |
| `--halo-default-slo` | `SGLANG_HALO_DEFAULT_SLO` | `5.0` | Used when request omits `halo_slo` |
| `--halo-tick-interval-ms` | `SGLANG_HALO_TICK_INTERVAL_MS` | `100` | Min interval between sweeps |
| `--halo-aggregator` | `SGLANG_HALO_AGGREGATOR` | `max+mean` | Reserved — Phase 1 always tracks both |
| `--halo-job-log` | `SGLANG_HALO_JOB_LOG` | `unset` (auto-routed by `run_experiment.py` to `<session>/halo_jobs.jsonl`) | Per-sweep snapshot JSONL. Rank-0 only |
| `--halo-prefill-cost-model-path` | `SGLANG_HALO_PREFILL_COST_MODEL` | `unset` | Reuses `admission_control` cost model when shared |
| `--halo-tbt-cost-model-path` | `SGLANG_HALO_TBT_COST_MODEL` | `unset` | Same |

### Behavior rules

- `--halo-enabled` off ⇒ controller is `None`; every hook is a one-line null check.
- `--halo-enabled` on but cost model paths unset ⇒ WARN, slowdown sweep skipped
  (jobs are still registered & counted; only `slowdown_*` fields stay at SLO).
- Both cost model paths set + missing/malformed ⇒ WARN, slowdown sweep skipped.
- Request lacks `halo_job_id` while controller enabled ⇒ HTTP 400 reject
  (per spec — strict mode for experiments).
- Request has `halo_job_id` but no `halo_slo` ⇒ default SLO applied.

## External touchpoints (one-line `# HALO:` comments at each site)

| File | What |
|---|---|
| `srt/managers/io_struct.py` | Add `halo_job_id: Optional[str]`, `halo_slo: Optional[float]` to `GenerateReqInput`, `TokenizedGenerateReqInput` |
| `srt/entrypoints/openai/protocol.py` | Same two fields on `ChatCompletionRequest`, `CompletionRequest` |
| `srt/entrypoints/openai/serving_chat.py`, `serving_completions.py` | Forward fields into `GenerateReqInput` |
| `srt/managers/schedule_batch.py::Req` | Store `halo_job_id`, `halo_slo`, `halo_first_admitted_ts` |
| `srt/managers/scheduler.py::__init__` | `init_halo()`: build `HaloController` if `server_args.halo_enabled` |
| `srt/managers/scheduler.py::_add_request_to_queue` | Call `halo_controller.register_request(req)` — returns 400 reject if job_id missing |
| `srt/managers/scheduler.py` finish path | `halo_controller.on_request_finished(req)` |
| `srt/observability/scheduler_metrics_mixin.py` | 100ms wall-clock gate → `halo_controller.tick(build_infos_callable)` |
| `srt/managers/scheduler.py::get_internal_state` | Append `halo_state` block |
| `srt/server_args.py` | 6 new CLI flags + dataclass fields |
| `srt/entrypoints/http_server.py` | 400 path for HALO_NO_JOB_ID reject (passthrough from scheduler) |

## TP-rank-0 dedup

JSONL job log writes and any Prometheus counters initialized only when
`getattr(self, "attn_tp_rank", 0) == 0`. Other ranks construct
`HaloController` but with the log/metrics path nulled so they no-op.

## /server_info snippet

```json
"halo_state": {
  "enabled": true,
  "tick_interval_ms": 100,
  "default_slo": 5.0,
  "active_jobs": 12,
  "jobs": [
    {
      "job_id": "agent-42-step3",
      "state": "running",
      "slo": 5.0,
      "slowdown_max": 2.8,
      "slowdown_mean": 1.4,
      "total_request_number": 4,
      "remaining_request_number": 2,
      "slo_violation_count": 0
    }
  ]
}
```

## Edge cases handled

- `prompt_len == 0` or `decoded_len == 0` very early → `solo_elapsed_ms` could be near 0;
  guard with `max(solo_elapsed_ms, MIN_SOLO_MS=1.0)`.
- Job with all requests finished but still in registry → state transitions to COMPLETE,
  excluded from `active_jobs()`, retained for one further tick for observability, then
  GC'd.
- TP-rank > 0 attempting to write JSONL → constructor receives `decision_log=None`.
- Cost models absent → tracker skips slowdown updates; jobs still tracked by counters.
- Request finishes before any sweep ran → on_request_finished still decrements job counters
  correctly; no slowdown ever recorded for that rid (history empty).

## Files

- `__init__.py` — public re-exports
- `job.py` — `Job`, `JobState`, helper `_now_monotonic()`
- `job_registry.py` — `JobRegistry`
- `slowdown_tracker.py` — `SlowdownTracker`, `RequestExecutionInfo`
- `controller.py` — `HaloController`, `HaloConfig`, helper loaders

## Testing

`test/registered/halo/test_halo_phase1.py` — see that file. Unit tests cover:
Job lifecycle, registry rid↔job mapping, sweep slowdown math, aggregator,
controller on/off, missing-cost-model degradation.
