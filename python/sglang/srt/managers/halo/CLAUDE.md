# Halo Module — Job-level subsystem for SGLang

Project Halo introduces three new responsibilities into SGLang, all scoped to
*jobs* (client-supplied groupings of multiple LLM requests that share a single
end-to-end SLO).

> - **Project plan + decision log**: `ms_dev/halo_dev/CLAUDE.md`
> - **Client-facing API reference** (endpoint schemas, request fields,
>   curl/LangChain recipes): `ms_dev/halo_dev/halo_api_reference.md`
>
> This file is the *internal* design doc — touchpoints, classes,
> invariants. API surface details are summarized here but the
> authoritative source for client integrators is `halo_api_reference.md`.

## Halo's three roles

```
┌──────────────────────────────────────────────────────────────────────┐
│  Project Halo                                                        │
│                                                                      │
│  Role 1. Job-level slowdown tracking                  [Phase 1: ON]  │
│          - SlowdownTracker.sweep() every ~100ms                      │
│          - per-job slowdown_max / slowdown_mean                      │
│                                                                      │
│  Role 2. Job-level admission gate                     [Phase 1: ON]  │
│          - JobRegistry.admit_to_job() — strict validation:           │
│            request must carry halo_job_id AND be pre-registered      │
│          - Phase 2 will add predictive lookahead admission           │
│            that reads Role 1's job state                             │
│                                                                      │
│  Role 3. Job-level scheduling policy                  [Phase 2]      │
│          - fairness over slowdown ratios                             │
│          - reuses Role 1's Job + Registry, no new state              │
└──────────────────────────────────────────────────────────────────────┘
```

The Halo job-level admission gate (Role 2) is **distinct from** SGLang's
existing `admission_control/` module, which is a *per-request, predictive*
gate (Mooncake-style TTFT/TBT SLO check). The two gates compose in series
inside `scheduler._add_request_to_queue`:

```
client → tokenizer → scheduler._add_request_to_queue:
                       ├── _abort_on_queued_limit          (capacity gate)
                       ├── _abort_on_predicted_slo_violation
                       │     ↳ admission_control: per-request predictive
                       ├── _halo_register_or_abort
                       │     ↳ Halo Role 2: job-level admission gate
                       └── waiting_queue.append(req)
```

This file documents the Halo module that owns all three roles. Where roles
overlap with existing SGLang subsystems (`admission_control/`,
scheduling policy), the boundary is called out explicitly below.

## Why job-level?

SGLang's existing `admission_control/` operates at request granularity (each
request's predicted TTFT/TBT vs SLO). Halo operates at *job* granularity — a
job is the client-supplied grouping of multiple requests that share an SLO
target (e.g., a single agent loop, a LangGraph DAG, a multi-turn conversation).

The SLO is defined as **end-to-end slowdown** of the job versus its solo-run
baseline (e.g., SLO=5 means "the job's total wall time may be at most 5× what
it would take running alone"). Per-request SLOs cannot capture this because a
fast request can mask a slow one in the same job, or vice versa.

## Phase 1 scope

| In scope | Out of scope |
|---|---|
| Job dataclass + JobState enum | Job-level admission gate decisions (Phase 2) |
| JobRegistry (rid ↔ job lookup + pre-registration) | Job-level scheduling priority (Phase 2) |
| 100ms periodic SlowdownTracker.sweep | DAG-aware orchestration logic (Phase 3) |
| Reuse cost models from `admission_control/` | New cost-model fitting |
| Off-by-default flag | Replacing `admission_control/` |
| **Option A: `POST /halo/programs`** sets up future-state (call count, DAG) on the Job for Phase 2 to consume | Use of that info — Phase 1 stores it only |
| **Option B: request body fields** `halo_job_id`, `halo_slo` as transport | n/a |

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

| Class | Responsibility | Role |
|---|---|---|
| `JobState` | Enum: `QUEUED / RUNNING / COMPLETE / REJECTED` | shared |
| `Job` | task_struct-like dataclass: id, slo, slowdown_max/mean, counters, rids set, history, Option A pre-registration fields | Role 1 state |
| `JobRegistry` | `register_program / admit_to_job / record_completion / active_jobs / gc_*` | Role 1 + Role 2 |
| `JobAdmissionResult` | Return type of `admit_to_job` — Halo job-level admission decision (distinct from `admission_control.AdmissionDecision`) | Role 2 |
| `RequestExecutionInfo` | Plain dataclass the scheduler builds per active req before sweep | Role 1 |
| `SlowdownTracker` | Pure compute. `compute_request_slowdown()` + `sweep()` | Role 1 |
| `HaloConfig` | Snapshot of CLI flags / env-derived parameters | shared |
| `HaloController` | Glue. Owned by scheduler. `register_program / register_request / on_request_finished / tick / snapshot` | Roles 1 + 2 |

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

## CLI flags + endpoint surface

There are 8 CLI flags (`--halo-*`) and one new endpoint (`POST /halo/programs`)
plus extra body fields (`halo_job_id`, `halo_slo`, `halo_bypass`) on
`/v1/chat/completions` and `/generate`. The **full schemas, status codes,
body shapes, and client recipes** are in
[`ms_dev/halo_dev/halo_api_reference.md`](../../../../../ms_dev/halo_dev/halo_api_reference.md).

Internal-only knobs worth remembering here:
- `--halo-enabled` is the master on/off — when off, controller is None and every
  hook is a one-line null check (zero cost).
- `--halo-program-idle-timeout-seconds` (default 300) drives
  `JobRegistry.gc_idle_programs` from inside `tick()`. `0` disables it.
- Q10 (SLO conflict): pre-registered wins, request value logged as WARN.
- Q11 (re-register): HTTP 409, the existing Job's `to_dict()` is included
  in the response body.
- Q12 (no lazy-create): requests for unregistered job_ids → HTTP 400 with
  reason `HALO_PROGRAM_NOT_REGISTERED`.

### Behavior rules

- `--halo-enabled` off ⇒ controller is `None`; every hook is a one-line null check.
- `--halo-enabled` on but cost model paths unset ⇒ WARN, slowdown sweep skipped
  (jobs are still registered & counted; only `slowdown_*` fields stay at SLO).
- Both cost model paths set + missing/malformed ⇒ WARN, slowdown sweep skipped.
- Request lacks `halo_job_id` while controller enabled ⇒ HTTP 400 reject
  with reason `HALO_NO_JOB_ID` (Q7 — strict mode for experiments).
- Request has `halo_job_id` but program **not pre-registered** ⇒ HTTP 400
  reject with reason `HALO_PROGRAM_NOT_REGISTERED` (Q12). Client must call
  `POST /halo/programs` before the first LLM call.
- Request has `halo_job_id` + program pre-registered but no `halo_slo` ⇒
  pre-registered SLO is used (anyway wins per Q10).

## External touchpoints (one-line `# HALO:` comments at each site)

| File | What |
|---|---|
| `srt/managers/io_struct.py` | (B) `halo_job_id`, `halo_slo` on `GenerateReqInput` + `TokenizedGenerateReqInput`. (A) `HaloRegisterProgramReqInput` / `Output` dataclasses |
| `srt/entrypoints/openai/protocol.py` | (B) Same two fields on `ChatCompletionRequest`, `CompletionRequest` |
| `srt/entrypoints/openai/serving_chat.py`, `serving_completions.py` | (B) Forward fields into `GenerateReqInput` |
| `srt/managers/schedule_batch.py::Req` | (B) Store `halo_job_id`, `halo_slo`, `halo_first_admitted_ts` |
| `srt/managers/tokenizer_control_mixin.py` | (A) New `_COMMUNICATOR_SPECS` entry `register_halo_program`; async `register_halo_program(obj)` helper |
| `srt/managers/scheduler.py::__init__` | `init_halo()`: build `HaloController` if `server_args.halo_enabled` |
| `srt/managers/scheduler.py::init_request_dispatcher` | (A) Register `HaloRegisterProgramReqInput → self.register_halo_program` |
| `srt/managers/scheduler.py::_add_request_to_queue` | (B) Call `halo_controller.register_request(req)` — returns 400 reject if job_id missing or program not pre-registered |
| `srt/managers/scheduler.py` finish path | `halo_controller.on_request_finished(req)` |
| `srt/managers/scheduler.py::register_halo_program` | (A) Scheduler-side handler delegating to `HaloController.register_program` |
| `srt/observability/scheduler_metrics_mixin.py` | 100ms wall-clock gate → `halo_controller.tick(build_infos_callable)` |
| `srt/managers/scheduler.py::get_internal_state` | Append `halo_state` block |
| `srt/server_args.py` | 7 new CLI flags + dataclass fields (`--halo-program-idle-timeout-seconds` is the 7th) |
| `srt/entrypoints/http_server.py` | (A) `POST /halo/programs` route. (B) HTTP 400 path for HALO_NO_JOB_ID / HALO_PROGRAM_NOT_REGISTERED |

## TP-rank-0 dedup

JSONL job log writes and any Prometheus counters initialized only when
`getattr(self, "attn_tp_rank", 0) == 0`. Other ranks construct
`HaloController` but with the log/metrics path nulled so they no-op.

## Prometheus metrics

`managers/halo/metrics.py::HaloMetrics`. Installed by `scheduler.init_halo`
under the same TP-dedup + `--enable-metrics` gating as
`admission_control/metrics.py` so a TP=N deployment doesn't multiply
counters. All metrics live behind the `sglang:` prefix:

```
# counters
sglang:halo_programs_registered_total
sglang:halo_programs_rejected_total{reason}
sglang:halo_requests_admitted_total
sglang:halo_requests_rejected_total{reason}
sglang:halo_slo_violations_total
# gauges (refreshed every sweep, ~100 ms)
sglang:halo_active_jobs
sglang:halo_total_known_jobs
sglang:halo_mean_slowdown_max
sglang:halo_mean_slowdown_mean
sglang:halo_max_slowdown_max
```

Consumed by `ms_dev/expctl/monitoring_view.py` to render two rows on the
live status panel (`halo_active / known / registered / admitted /
rejected` and `mean_smax / mean_smean / worst_smax / slo_violations`).

## /server_info snippet

```json
"halo_state": {
  "enabled": true,
  "tick_interval_ms": 100,
  "default_slo": 5.0,
  "aggregator": "max+mean",
  "program_idle_timeout_seconds": 300.0,
  "cost_models_loaded": {"prefill": true, "tbt": true},
  "active_jobs": 12,
  "total_known_jobs": 18,
  "jobs": [
    {
      "job_id": "agent-42",
      "state": "running",
      "slo": 5.0,
      "slowdown_max": 2.8,
      "slowdown_mean": 1.4,
      "total_request_number": 4,
      "remaining_request_number": 2,
      "slo_violation_count": 0,
      "from_program": true,
      "total_calls_expected": 12,
      "stage_sequence": ["UNDERSTAND", "LOCATE", "..."],
      "expected_input_lens": null,
      "expected_output_lens": null,
      "dag": null
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
