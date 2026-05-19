# Halo Module — request-level admission control + tracking

**Project Halo** is the umbrella name for a request-level admission-control +
tracking feature added to SGLang's single-instance scheduler. "Halo" is just
the project name — not a role. Off by default (`--halo-enabled`); when off the
controller is `None` and every scheduler hook is a one-line null check.

> **History.** Halo began as a *job-level* subsystem (job-scoped slowdown
> tracking + job admission, `POST /halo/programs`, virtual job slowdown).
> It was refactored to **request-level** on 2026-05-19. The complete
> job-level codebase is preserved at git tag `halo-job-level-final`; its
> design docs are in `ms_dev/halo_dev/legacy_document/`.

## Two components

```
managers/halo/
├── controller.py            HaloController — scheduler-owned top-level glue
├── metrics.py               HaloMetrics — request-level Prometheus metrics
├── cost_model_sampler.py    per-step JSONL sampler for fitting the step cost model
├── request_tracker/         ── the single per-request state store ──
│   ├── record.py            RequestRecord, RequestState, ServerStateSnapshot
│   └── tracker.py           RequestTracker
└── admission_control/       ── the policy-pluggable admission gate ──
    ├── gate.py              HaloAdmissionGate (KV cap + policy dispatch)
    ├── policy.py            AdmissionPolicy ABC, NewRequestInput, PolicyDecision
    ├── policy_mooncake.py   MooncakePolicy  — predictive TTFT/TBT gate
    ├── policy_vss.py        VssPolicy       — memoryless virtual-server-slowdown gate
    ├── policy_reactive.py   (Phase 3 — not yet implemented)
    ├── vss_predictor.py     VSS batch-slowdown kernel (used by VssPolicy)
    ├── cost_model.py        PrefillCostModel, TBTCostModel, HaloStepCostModel
    ├── tbt_tracker.py       TBTEwmaTracker (mooncake Stage-3 reactive EWMA)
    └── decision_log.py      DecisionLogger
```

### request_tracker/ — the single state store

`RequestTracker` holds one `RequestRecord` per in-flight (or recently
finished) request and is the **sole writer** of request state. The admission
gate, the policies, and metrics are read-only consumers. Single-threaded:
every method runs on the scheduler process main loop — no locking.

- `RequestRecord` — canonical per-request state: the three SLOs, prompt/prefix
  lengths, timestamps, decoded count, KV span, derived latency views
  (`ttft_ms`, `tbt_mean_ms`, `e2e_ms`), and terminal `solo_e2e_ms` /
  `e2e_slowdown`. `RequestState` ∈ {QUEUED, RUNNING, FINISHED, REJECTED}.
- `RequestTracker` event handlers (the only mutators): `on_admitted`,
  `on_step`, `on_finished`, `on_rejected`. Readers: `get`, `snapshot`,
  `active_count`. Solo cost-model primitives: `prefill_solo_ms`,
  `decode_step_ms`.
- `ServerStateSnapshot` — the read-only view the gate/policy consult: the
  queued + running `RequestRecord` lists and the current KV-usage ratio.

### admission_control/ — the gate + policies

`HaloAdmissionGate` runs the policy-independent **KV-cache hard cap**
(Stage B′ — reject when KV usage ≥ `--halo-admission-kv-cap-ratio`) then
delegates to exactly one `AdmissionPolicy`:

| `--halo-admission-policy` | Policy | Signal |
|---|---|---|
| `off` | (none) | KV cap only — the gate admits otherwise |
| `mooncake` | `MooncakePolicy` | predictive TTFT/TBT vs the request's `halo_ttft_slo` / `halo_tbt_slo` |
| `vss` | `VssPolicy` | memoryless virtual-server-slowdown vs `halo_e2e_slo` |
| `reactive` | (Phase 3) | measured queueing-inclusive gate — not yet implemented |

Per design decision D5, all three request SLOs are always active; each policy
uses whichever ones its signal speaks to. `--halo-slo-mode` ∈ {ratio, absolute}
sets whether `halo_ttft_slo` / `halo_tbt_slo` are slowdown ratios or ms caps;
`halo_e2e_slo` is always a slowdown ratio.

## HaloController

Scheduler-owned. Public surface (the only methods the scheduler calls):

| Method | Role |
|---|---|
| `register_request(rid, *, ttft_slo, tbt_slo, e2e_slo, prompt_len, prefix_len, kv_usage_ratio)` | Admission hook — run the gate, then `tracker.on_admitted` or raise `HaloRejectError`. |
| `on_request_step(rid, *, decoded_tokens, kv_len)` | Progress update (currently driven via `tick`). |
| `on_request_finished(rid, *, decoded_tokens, kv_len)` | Finalize the record + emit terminal metrics. |
| `on_tbt_sample(per_step_ms)` | Feed the mooncake Stage-3 reactive EWMA. |
| `tick(running_infos)` | Periodic: feed running-batch progress to the tracker + refresh gauges. |
| `snapshot()` | `/halo/status` + `/server_info`. |

`build_halo_controller_from_server_args` is the factory (returns `None` when
`--halo-enabled` is off). The job-level era's `register_program` /
`JobRegistry` are gone.

## Scheduler touchpoints (one-line `# HALO:` comments at each site)

| File / site | What |
|---|---|
| `scheduler.py::init_halo` | Build `HaloController` if `--halo-enabled`; attach `HaloMetrics` on `attn_tp_rank==0` + `--enable-metrics`. |
| `scheduler.py::_add_request_to_queue` | `_halo_admission_gate(req)` — the single gate; True ⇒ rejected (HTTP 400). |
| `scheduler.py::_halo_admission_gate` | Builds prompt/prefix/kv_usage, calls `register_request`, sends `AbortReq` on `HaloRejectError`. |
| `scheduler.py` finish path | `_halo_on_request_finished(req)` → `controller.on_request_finished`. |
| `scheduler.py::_halo_maybe_tick` | Per-loop wall-clock-gated `controller.tick(_halo_running_infos())`. |
| `scheduler.py::get_internal_state` | Appends `halo_state` = `controller.snapshot()`. |
| `io_struct.py`, `schedule_batch.py::Req`, OpenAI `protocol.py` / `serving_*` | Carry `halo_ttft_slo` / `halo_tbt_slo` / `halo_e2e_slo` / `halo_bypass`. |
| `http_server.py` | `GET /halo/status` probe; `halo_bypass=True` on warmup/health traffic. |
| `server_args.py` | `--halo-*` flags. |

`halo_bypass=True` marks server-internal traffic (warmup, health, loopback) so
the gate skips it. There is **no** request endpoint for job pre-registration
and **no** `halo_job_id` — admission is fully per-request.

## CLI flags (`--halo-*`)

`--halo-enabled` (master on/off), `--halo-admission-policy`, `--halo-slo-mode`,
`--halo-admission-kv-cap-ratio`, `--halo-admission-violation-threshold` (vss),
`--halo-tbt-reactive-ratio` (mooncake), `--halo-admission-dry-run`,
`--halo-admission-decision-log`, `--halo-tick-interval-ms`,
`--halo-{prefill,tbt,step}-cost-model-path`,
`--halo-cost-model-sample-log` / `--halo-cost-model-sample-every`.

## Prometheus metrics

`managers/halo/metrics.py::HaloMetrics`, rank-0 + `--enable-metrics` gated:

```
# counters
sglang:halo_requests_admitted_total
sglang:halo_requests_rejected_total{reason}
# histograms — one observation per finished request
sglang:halo_request_ttft_seconds
sglang:halo_request_tbt_seconds
sglang:halo_request_e2e_seconds
sglang:halo_request_e2e_slowdown
# gauge
sglang:halo_active_requests
```

Per-request quantities are histograms (percentiles for free); only the
current active-request count is a gauge.

## Known limitations

- **Tick-driven decode tracking.** `decoded_tokens` / `kv_len` of running
  requests are refreshed by the ~100 ms `tick`, not per forward step — so the
  TBT-mean (`(last_token_ts − first_token_ts) / (decoded − 1)`) carries ±tick
  noise. `first_token_ts` is *not* affected: it is stamped from the scheduler's
  accurate prefill-finished timestamp (`SchedulerReqTimeStats.prefill_finished_time`),
  so TTFT is exact even for a request that finishes within one tick.
- **e2e-slowdown solo baseline is an approximation.** `RequestTracker.on_finished`
  estimates solo decode with the mean-KV span rather than integrating per-step
  cost. The solo-run predictor is slated for a dedicated rework.
- **`vss_predictor.py`** still contains the (now-unused) job-scoped predictor
  and the deprecated lookahead predictor — dead code pending a trim.

## Testing

`test/registered/halo/test_halo_phase1.py` — see that file and
`test/registered/halo/CLAUDE.md`.
