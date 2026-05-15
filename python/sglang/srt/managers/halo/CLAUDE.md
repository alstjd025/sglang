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
│  Role 1. Job-level slowdown tracking                  [ON]          │
│          - SlowdownTracker.sweep() every ~100ms                      │
│          - per-job virtual_job_slowdown (VJS) via compute_job_vjs    │
│                                                                      │
│  Role 2. Job-level admission gate                     [ON]          │
│          - strict validation (halo_job_id + pre-registration)        │
│          - Phase 2 predictive admission: job- and request-scoped     │
│            gates that read each job's VJS (admission_decision.py)    │
│                                                                      │
│  Role 3. Job-level scheduling policy                  [future]      │
│          - fairness over virtual job slowdown                        │
│          - reuses Job + Registry, no new state                       │
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
├── slowdown_tracker.py     # compute_job_vjs() stage-merge VJS + 100ms sweep()
├── controller.py           # HaloController: on/off, hook into scheduler
├── cost_model_sampler.py   # per-step JSONL sampler for fitting the Halo Step Cost Model
│                           # (rank-0, background flusher thread, no-op when path unset).
│                           # See ms_dev/halo_dev/prediction_model.md.
└── admission_decision.py   # ★ Phase 2 predictive admission. AdmissionPredictor ABC +
                            # JobSlowdownAdmissionPredictor (mode "job", per-job stretch
                            # M-2 — the design) + RequestSlowdownAdmissionPredictor
                            # (mode "request", per-request baseline) + Lookahead
                            # (mode "level2", DEPRECATED 2026-05-15) + decide_admission.
                            # See ms_dev/halo_dev/admission_design.md.
```

## Class summary

| Class | Responsibility | Role |
|---|---|---|
| `JobState` | Enum: `QUEUED / RUNNING / COMPLETE / REJECTED` | shared |
| `Job` | task_struct-like dataclass: id, slo, virtual_job_slowdown, completed_call_spans, counters, rids set, history, Option A pre-registration fields | Role 1 state |
| `JobRegistry` | `register_program / admit_to_job / record_completion / active_jobs / gc_*` | Role 1 + Role 2 |
| `JobAdmissionResult` | Return type of `admit_to_job` — Halo job-level admission decision (distinct from `admission_control.AdmissionDecision`) | Role 2 |
| `RequestExecutionInfo` | Plain dataclass the scheduler builds per active req before sweep | Role 1 |
| `JobCallSpan` | Frozen per-call time-span + token features; the unit of compute_job_vjs | Role 1 |
| `SlowdownTracker` | Pure compute. `compute_job_vjs()` (stage-merge VJS) + `sweep()` + cost-model primitives | Role 1 |
| `HaloConfig` | Snapshot of CLI flags / env-derived parameters | shared |
| `HaloController` | Glue. Owned by scheduler. `register_program / register_request / on_request_finished / tick / snapshot` | Roles 1 + 2 |

## Key invariants

- **Single-threaded access**: all methods called from the scheduler process's main loop.
  No internal locking. Tests must mirror this.
- **rid→job mapping is the source of truth** for ownership. A request's
  `halo_job_id` field is the seed at admission time; afterwards the registry owns it.
- **Initial `Job.virtual_job_slowdown = job.slo`** (spec choice — measurement-less
  worst-case fallback; the sweep overwrites it once the job has an in-flight call).
- **One canonical slowdown**: each job carries a single `virtual_job_slowdown`
  (VJS). `slowdown_history` is a bounded (`maxlen=64`) deque of past VJS floats.

## Virtual job slowdown (VJS)

VJS is the job's **lifetime critical-path slowdown** —
`(job critical-path actual ms) / (job critical-path solo ms)` accumulated over
every completed + in-flight call of the job. `SlowdownTracker.compute_job_vjs`
merges the job's call spans into *stages* by wall-clock interval overlap; per
stage `actual` is the interval-union span and `solo` is the critical-path (max)
over the stage's calls with decode steps charged at the batch-of-K step time.
Tool-delay gaps are excluded; concurrent calls merge by critical path.

Both the periodic sweep (`SlowdownTracker.sweep` → `Job.record_vjs`) and the
Phase-2 job-scoped admission gate go through the *same* `compute_job_vjs` — one
source of truth. The **full definition, the stage-merge algorithm, a worked
example, and the cost-model formula** are in
[`ms_dev/halo_dev/prediction_model.md`](../../../../../ms_dev/halo_dev/prediction_model.md) §3.

The solo baseline comes from the **Halo Step Cost Model** (path A,
`--halo-step-cost-model-path`) — a 6/7-coefficient per-step regression that
replaced the old `per_req_kv = total_kv/bs` reparam (the TBT cliff) with a
batch-summed `Σrⱼ` term. A legacy two-model pair (path B) is used only when no
step model is loaded. Cost-model design: `prediction_model.md` §2.

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
- Request with `halo_job_done: true` on its finish ⇒ owning job is
  marked COMPLETE; `gc_completed` drops it after retain_seconds.
- Job with no admit/finish activity for `--halo-quiescent-timeout-seconds`
  (default 300) and zero in-flight requests ⇒ safety-net force-completed
  (WARN logged). Catches clients that crashed or forgot the signal.

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

## JSONL job log (`halo_jobs.jsonl`)

Per-sweep snapshot is throttled to `--halo-job-log-interval-seconds` (default
10s) to keep the file small. Three event kinds:

| Event | When emitted | Throttled by interval? |
|---|---|---|
| `register_program` | client successful `POST /halo/programs` | no — emitted immediately |
| (sweep) `{"ts", "active_jobs": [...]}` | every tick after interval window | yes — every 10s |
| `job_complete` (reason=`halo_job_done`) | client sends `halo_job_done=true` on the last call's body | no — emitted from finish hook |
| `job_complete` (reason=`quiescent_timeout`) | safety-net fallback: a job had no admit/finish for `--halo-quiescent-timeout-seconds` | no — emitted from tick after registry GC |

The `job_complete` row carries the Job.to_dict() at termination time, so
post-hoc analysis sees the final state even when the COMPLETE→GC pair
happens between two sweep snapshots.

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
sglang:halo_mean_vjs
sglang:halo_max_vjs
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
      "virtual_job_slowdown": 2.8,
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
