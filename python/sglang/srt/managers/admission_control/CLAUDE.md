# Admission Control Module

Mooncake-style predictive SLO-based admission control for SGLang's single-instance scheduler.
Reject incoming requests **before** queueing if their predicted TTFT or TBT would violate
the configured SLOs. Returns HTTP 429 (matches Mooncake §4.1).

This module is the canonical location for admission control logic. External touchpoints
(scheduler, server_args, metrics mixin) carry one-line comments pointing here.

## Goals

- **Predictive, not capacity-based**: `max_queued_requests` rejects when the queue is full;
  this rejects when the *predicted user experience* would violate SLO, even with empty queue.
- **Hybrid policy** combining three stages — first violation wins (cheapest check first):
  1. **Stage 1 (TTFT predicted)** — Mooncake Algorithm 1 line 18, simplified to single instance
  2. **Stage 2 (TBT predicted)** — current-batch approximation (paper hand-waves this)
  3. **Stage 3 (TBT reactive EWMA)** — safety net for cost-model drift
- **Off by default**: if `--admission-ttft-slo-ms` and `--admission-tbt-slo-ms` are both unset,
  admission control is disabled and the original `_add_request_to_queue` path is unchanged.
- **Single-instance, NULL disaggregation only** for Phase A. Disaggregation modes warn once
  and skip admission. Multi-instance Conductor is Phase B.

## Architecture

```
managers/admission_control/
├── __init__.py          # public API: AdmissionController, AdmissionDecision
├── cost_model.py        # PrefillCostModel (n,p → ms),  TBTCostModel (bs,kv → ms)
├── tbt_tracker.py       # TBTEwmaTracker  (per-step latency → smoothed estimate)
└── controller.py        # AdmissionController.decide() — hybrid policy orchestration
```

### Class summary

| Class | Responsibility | State |
|---|---|---|
| `PrefillCostModel` | `T_prefill ≈ α·d² + β·d + γ` where `d = n − p` | (α,β,γ) loaded from JSON |
| `TBTCostModel` | `TBT ≈ a + b·batch_size + c·total_kv_tokens` | (a,b,c) loaded from JSON |
| `TBTEwmaTracker` | EWMA of measured per-decode-step latency | rolling float, warm-up flag |
| `AdmissionDecision` | dataclass: `admit`, `reason`, `predicted`, `slo` | immutable result |
| `AdmissionController` | combines models + tracker, runs decision flow | references to scheduler state |

## Decision flow (controller.py)

```python
def decide(req, scheduler) -> AdmissionDecision:
    if disabled or wrong disaggregation mode:
        return ADMIT(reason="disabled")

    # ── Stage 1: TTFT predicted ──────────────────────────────────────────
    n = len(req.input_ids)
    p = scheduler.tree_cache.match_prefix_len(req.input_ids)   # 0 on failure
    t_prefill = prefill_cost.estimate_ms(n, p)
    t_queue   = sum(r._predicted_prefill_ms for r in scheduler.waiting_queue)
    pred_ttft = t_queue + t_prefill
    if ttft_slo_ms and pred_ttft > ttft_slo_ms:
        return REJECT("TTFT_PREDICTED", pred_ttft, ttft_slo_ms)

    # ── Stage 2: TBT predicted (current batch approximation) ────────────
    if tbt_slo_ms:
        bs = len(scheduler.running_batch.reqs) + 1
        kv = current_batch_total_kv(scheduler) + n
        pred_tbt = tbt_cost.estimate_ms(bs, kv)
        if pred_tbt > tbt_slo_ms:
            return REJECT("TBT_PREDICTED", pred_tbt, tbt_slo_ms)

        # ── Stage 3: TBT reactive EWMA safety net ────────────────────────
        recent_tbt = tbt_tracker.get()  # 0.0 during warm-up → check skipped
        if tbt_tracker.is_warm() and recent_tbt > tbt_slo_ms * reactive_ratio:
            return REJECT("TBT_REACTIVE", recent_tbt, tbt_slo_ms)

    req._predicted_prefill_ms = t_prefill            # for next request's t_queue
    return ADMIT(pred_ttft=pred_ttft, pred_tbt=pred_tbt)
```

`AdmissionDecision` carries everything needed for log/metrics/dry-run handling.

## External touchpoints

All other modifications outside this directory. Each site has a one-line comment
"`# admission control — see managers/admission_control/CLAUDE.md`".

| File | Where | What |
|---|---|---|
| `managers/scheduler.py` | `__init__` (~line 700–800) | Instantiate `AdmissionController` if `ttft_slo_ms` or `tbt_slo_ms` set; else `None` |
| `managers/scheduler.py` | `_add_request_to_queue` line 2185 area | Call `_abort_on_predicted_slo_violation(req)` after `_abort_on_queued_limit` |
| `managers/scheduler.py` | new method `_abort_on_predicted_slo_violation` | Mirrors `_abort_on_queued_limit` pattern (line 2230); sends 429 `AbortReq` and returns True on reject |
| `server_args.py` | dataclass + arg parser (~line 362, 4498) | Add 7 fields/flags: see "CLI flags" below |
| `observability/scheduler_metrics_mixin.py` | line 517 (`step_time_dict` block) | Also `tbt_tracker.update(per_step_ms)` if controller exists |
| `observability/...metrics.py` (Prometheus) | metric registry | Add 4 metrics: see "Metrics" below |

## CLI flags / env vars

All flags are namespaced `--admission-*`. CLI default = `None` / `False` → admission disabled.

| CLI flag | env var (ms_dev) | Default | Meaning |
|---|---|---|---|
| `--admission-ttft-slo-ms <float>` | `SGLANG_ADMISSION_TTFT_SLO_MS` | unset | Stage 1 SLO. Unset → Stage 1 disabled |
| `--admission-tbt-slo-ms <float>` | `SGLANG_ADMISSION_TBT_SLO_MS` | unset | Stage 2+3 SLO. Unset → Stages 2,3 disabled |
| `--admission-prefill-cost-model-path <path>` | `SGLANG_ADMISSION_PREFILL_COST_MODEL` | unset | JSON `{"alpha","beta","gamma"}` for `PrefillCostModel` |
| `--admission-tbt-cost-model-path <path>` | `SGLANG_ADMISSION_TBT_COST_MODEL` | unset | JSON `{"a","b","c"}` for `TBTCostModel` |
| `--admission-tbt-ewma-alpha <float>` | `SGLANG_ADMISSION_TBT_EWMA_ALPHA` | `0.1` | EWMA smoothing factor |
| `--admission-tbt-reactive-ratio <float>` | `SGLANG_ADMISSION_TBT_REACTIVE_RATIO` | `0.9` | Stage 3 trips at `tbt_slo * ratio` |
| `--admission-dry-run` | `SGLANG_ADMISSION_DRY_RUN` | `False` | Log decisions but always admit (★ tuning mode) |
| `--admission-decision-log <path>` | `SGLANG_ADMISSION_DECISION_LOG` | unset | Append every decision as a JSONL row. Consumed by `tools/admission_control/replay_admission.py`. Only the rank-0 scheduler writes (TP-dedup). |

### Validation rules

- If TTFT_SLO is set but `prefill-cost-model-path` is not → controller logs WARN at startup,
  Stage 1 disabled (lenient mode — server still runs).
- If TBT_SLO is set but `tbt-cost-model-path` is not → Stage 2 disabled, Stage 3 still works
  (reactive EWMA needs no model).
- If model JSON file missing/malformed → WARN, that stage disabled.
- If `disaggregation_mode != NULL` → WARN once, controller disabled entirely (Phase A scope).

## Cost model JSON schema

Both `*-cost-model-path` flags consume small JSON files produced by
`tools/admission_control/fit_cost_model.py`. See [tools/admission_control/CLAUDE.md](../../../../../tools/admission_control/CLAUDE.md).

```json
// PrefillCostModel
{ "alpha": 1.2e-5, "beta": 0.045, "gamma": 12.3,
  "fit_metadata": { "model": "...", "samples": 120, "fit_at": "..." } }

// TBTCostModel
{ "a": 35.0, "b": 1.5, "c": 0.0008,
  "fit_metadata": { ... } }
```

`fit_metadata` is informational only (logged at load time). Coefficients are the only
required fields.

## Metrics (Prometheus)

Registered under existing `sglang:` prefix:

```
sglang:admission_decisions_total{decision="admit"|"reject_dryrun"|"reject", reason="..."}  counter
sglang:admission_predicted_ttft_ms                                                          histogram
sglang:admission_predicted_tbt_ms                                                           histogram
sglang:admission_tbt_ewma_ms                                                                gauge
sglang:admission_queue_predicted_ms                                                         gauge
```

Reasons: `TTFT_PREDICTED`, `TBT_PREDICTED`, `TBT_REACTIVE`.

## Internal state endpoint

Extends `get_internal_state` (`scheduler.py:3389` block) with:

```json
"admission_state": {
  "enabled": true,
  "dry_run": false,
  "ttft_slo_ms": 30000,
  "tbt_slo_ms": 200,
  "tbt_ewma_ms": 145.3,
  "tbt_ewma_warm": true,
  "queue_predicted_total_ms": 8200,
  "decisions_recent": [ ... last 32 ... ]
}
```

## Edge cases handled

- `tree_cache.match_prefix` raises → `prefix_len = 0`, conservative (rejects more).
- EWMA cold start (< warm-up step count, default 100) → Stage 3 skipped.
- Empty prompt (`n = 0`) → `pred_ttft = t_queue`, normal.
- Spec decoding → measured TBT already reflects spec speed-up; cost model fit must be done
  with the same spec config that will run in production.
- Priority scheduling → priority-based eviction (`_abort_on_queued_limit`) runs first;
  admission gate never blocks a higher-priority request that would have evicted a queued one.

## Files quick reference

- `cost_model.py` — `PrefillCostModel`, `TBTCostModel` + `from_json()` loaders
- `tbt_tracker.py` — `TBTEwmaTracker(alpha, warm_up_steps=100)`
- `controller.py` — `AdmissionController.decide(req, scheduler) -> AdmissionDecision`
- `__init__.py` — re-exports `AdmissionController`, `AdmissionDecision` (only public API)

## Testing

`test/registered/admission/test_admission_control.py` — see
[test/registered/admission/CLAUDE.md](../../../../../test/registered/admission/CLAUDE.md).

## Tooling

`tools/admission_control/` — fit + replay scripts. See
[tools/admission_control/CLAUDE.md](../../../../../tools/admission_control/CLAUDE.md).

## ms_dev integration

User's experiment tooling under `ms_dev/` translates env vars to CLI flags and shows
admission state on the live status panel. See [ms_dev/CLAUDE.md](../../../../../ms_dev/CLAUDE.md)
and [ms_dev/expctl/CLAUDE.md](../../../../../ms_dev/expctl/CLAUDE.md).
