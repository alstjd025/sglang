# Halo Admission Control — Design Doc

Code-based design doc for the **admission-control** component of Project Halo,
the request-level admission + tracking subsystem in SGLang's single-instance
scheduler (refactored from job-level to request-level on 2026-05-19). The other
Halo component is `request_tracker`; this doc covers only admission control.

Source tree: `python/sglang/srt/managers/halo/admission_control/` plus the
controller glue in `managers/halo/controller.py`.

## 1. Purpose

Decide **admit / reject for every arriving request before it is queued**. A
rejected request never enters the scheduler queue; the scheduler converts the
rejection into an HTTP 400 abort. This replaces the job-level era's two serial
gates (the standalone Mooncake `_abort_on_predicted_slo_violation` plus the Halo
`_halo_register_or_abort`) with **one gate running exactly one selected policy**.

Off by default (`--halo-enabled`). When off, `HaloController` is never built and
the scheduler hook is a one-line null check.

## 2. The gate — `HaloAdmissionGate` (`gate.py`)

`HaloAdmissionGate.decide(new_req, state) -> AdmissionResult`. Constructed with
an optional `AdmissionPolicy` and a `kv_cap_ratio`. Exact `decide()` flow:

1. **Stage B′ — KV-cache hard cap** (policy-independent). If
   `0.0 < kv_cap_ratio <= 1.0` and `state.kv_usage_ratio >= kv_cap_ratio`,
   return `AdmissionResult(admit=False, reason="HALO_KV_CAP")`.
   `kv_cap_ratio == 0` disables the cap entirely.
2. **Selected policy.** If `self.policy is None` (`--halo-admission-policy off`),
   return `AdmissionResult(admit=True, reason="ADMIT")`. Otherwise call
   `policy.decide(new_req, state)` and wrap its `PolicyDecision` into an
   `AdmissionResult` — `reason` is `"ADMIT"` on admit, else the policy's reason.

`AdmissionResult` is `(admit: bool, reason: str, policy_decision: Optional[PolicyDecision])`.
Stable reason constants live in `gate.py`: `REASON_ADMIT = "ADMIT"`,
`REASON_KV_CAP = "HALO_KV_CAP"`.

## 3. The policy abstraction (`policy.py`)

```python
class AdmissionPolicy(ABC):
    name: str = "abstract"
    @abstractmethod
    def decide(self, new_req: NewRequestInput,
               state: ServerStateSnapshot) -> PolicyDecision: ...
```

| Type | Fields |
|---|---|
| `NewRequestInput` (frozen) | `rid`, `prompt_len`, `prefix_len` (radix match length at admission), `ttft_slo`, `tbt_slo`, `e2e_slo` (all SLOs `Optional[float]`) |
| `PolicyDecision` (frozen) | `admit`, `reason` (`"ADMIT"` when admitted), `predicted_ttft_ms`, `predicted_tbt_ms`, `predicted_e2e_slowdown` (best-effort, `None` when the signal can't produce it), `detail: dict` |

`state` (`ServerStateSnapshot`) is a read-only view from `request_tracker`:
the queued + running `RequestRecord` lists and `kv_usage_ratio`.

**Design note D5:** all three SLOs (`ttft` / `tbt` / `e2e`) are always active;
each policy uses whichever ones its signal speaks to.

**Policy selection** — `HaloController._build_policy` maps
`--halo-admission-policy` (lower-cased) to a policy:

| Flag value | Policy built | Notes |
|---|---|---|
| `off` | `None` | gate runs only the KV cap |
| `mooncake` | `MooncakePolicy` | always built (cost models may be `None`) |
| `vss` | `VssPolicy` | requires the Halo Step Cost Model; if absent → `None` (disabled) |
| `reactive` | `None` | Phase 3 — logs a warning, admission disabled |
| anything else | `None` | logs "unknown admission_policy" |

The cost-model selection: `try_load_halo_step_cost_model` is tried first; only
if it returns `None` are the legacy `try_load_{prefill,tbt}_cost_model` loaded.
All loaders are lenient — a missing/malformed JSON logs a WARN and yields `None`
rather than raising.

## 4. Policies

### 4.1 `MooncakePolicy` (`policy_mooncake.py`, `name = "mooncake"`)

Predictive per-request TTFT/TBT gate — the validation baseline for the
request-level refactor. Constructed with `prefill_cost`, `tbt_cost`, `slo_mode`,
a shared `tbt_tracker`, and `tbt_reactive_ratio` (default 0.9). `decide()`:

**Predict TTFT** (only if `prefill_cost` is set) — queue backlog + this
request's solo prefill:

```
solo_ttft = prefill_cost.estimate_ms(prompt_len, prefix_len)
t_queue   = Σ  prefill_cost.estimate_ms(r.prompt_len, r.prefix_len)  for r in state.queued
pred_ttft = t_queue + solo_ttft
```

`PrefillCostModel.estimate_ms`: `T ≈ α·d² + β·d + γ + δ·prefix_len`, with
`d = max(0, prompt_len − prefix_len)`.

**Predict TBT** (only if `tbt_cost` is set) — cost model on the running batch
plus this request:

```
solo_tbt  = tbt_cost.estimate_ms(1, prompt_len)
bs        = len(state.running) + 1
total_kv  = Σ r.kv_len  for r in state.running   + prompt_len
pred_tbt  = tbt_cost.estimate_ms(bs, total_kv // bs)
```

`TBTCostModel.estimate_ms`: `T ≈ a + b·batch_size + c·per_req_kv`.

**Three-stage check** (first violation wins):

| Stage | Condition | Reason |
|---|---|---|
| 1 — TTFT | `ttft_slo` set, `pred_ttft` set, `_violates(pred_ttft, solo_ttft, ttft_slo)` | `MOONCAKE_TTFT` |
| 2 — TBT predicted | `tbt_slo` set, `pred_tbt` set, `_violates(pred_tbt, solo_tbt, tbt_slo)` | `MOONCAKE_TBT` |
| 3 — TBT reactive | `slo_mode == "absolute"` **and** `tbt_slo` set **and** `tbt_tracker.is_warm()` **and** `ewma > tbt_slo * tbt_reactive_ratio` | `MOONCAKE_TBT_REACTIVE` |

`_violates(predicted, solo, slo)` depends on `--halo-slo-mode`:
- `absolute`: `predicted > slo` (SLOs are millisecond caps).
- `ratio`: `predicted / solo > slo` (slowdown bound vs solo baseline). Skipped
  (returns `False`) when `solo` is `None` or `≤ _RATIO_SOLO_FLOOR_MS = 1.0`ms —
  below that the ratio is numerical noise.

Stage 3 is **absolute-mode only** and uses `TBTEwmaTracker` (`tbt_tracker.py`):
an EWMA over measured per-decode-step latency (`alpha = 0.1`,
`warm_up_steps = 100`). Until `warm_up_steps` updates have arrived `is_warm()`
is `False` and Stage 3 is skipped (no false reject from one early sample).
Negative latencies are ignored so they cannot poison the EWMA. It is fed by
`HaloController.on_tbt_sample` from the scheduler's per-step metrics hook.

`detail` always carries `solo_ttft_ms`, `pred_ttft_ms`, `solo_tbt_ms`,
`pred_tbt_ms` (plus `tbt_ewma_ms` when Stage 3 runs).

### 4.2 `VssPolicy` (`policy_vss.py`, `name = "vss"`)

Memoryless **virtual-server-slowdown** gate, carried over from the job-level
era's request-scoped VSS arm. Requires a `HaloStepCostModel`. Constructed with
`cost_model`, `violation_threshold` (default 0.2), and `chunked_prefill_size`.
`decide()`:

1. Map every running `RequestRecord` to an `ActiveCallInfo`
   (`is_prefill`, `prompt_len`, `prefix_len`, `decoded_tokens`,
   `kv_len_now = r.kv_len or prompt_len + decoded_tokens`).
2. Score with `RequestSlowdownAdmissionPredictor.predict(batch, arrival,
   chunked_prefill_size)` → `{rid: predicted_VSS}`.
3. Build `slos = {rid: e2e_slo}` for running requests that have an `e2e_slo`,
   and apply `decide_admission(predicted, slos, violation_threshold)`.

**The kernel (`vss_predictor.py`).** VSS for in-flight request *i* is a per-step
**contention ratio**:

```
predicted_VSS_i = S_phase(i)⁺ / solo_step_i
```

`S_phase(i)⁺` is the augmented step time of *the current batch plus the new
arrival*; `solo_step_i` is request *i*'s current solo step time. Memoryless — a
pure function of current batch composition and arrival shape, no history.

`_augmented_step_times` returns `(S_extend⁺, S_decode⁺)`: the arrival is added
to **both** halves (it prefills now, then joins decode). `S_extend⁺` is
`cost_model.estimate_step_ms` over the current prefill calls + the arrival;
`S_decode⁺` is over current decode KV lengths + the arrival's input length.

`HaloStepCostModel.estimate_step_ms` form (`halo_step_v1`):
`T ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ + θ_d1·Σrⱼ + θ_d2·bs_d + θ_c`
(a split form `halo_step_split_v1` uses separate prefill/decode intercepts).

**결함 A — chunk-bounded EXTEND step.** `_bounded_extend_step` caps the EXTEND
batch to **one realistic chunked-prefill step**: with chunked prefill ON a
forward pass processes at most `chunked_prefill_size` *new tokens total*, so
each `(n, r)` entry contributes only its share of the remaining budget and
entries past the budget are dropped from the modelled step. Without this cap the
`Σnᵢ²` term blows up on the un-chunked waiting backlog. `chunked_prefill_size`
`None`/`≤ 0` (chunked prefill disabled) ⇒ no cap. The DECODE step is **not**
bounded — a decode step genuinely touches every running request. The per-request
solo prefill step is also chunk-bounded for a consistent denominator. Solo step
times are floored at `_MIN_SOLO_STEP_MS = 1.0`ms to guard the division.

**`decide_admission`** applies the violation-ratio threshold:
`violations = #{ rid : predicted > slos.get(rid, predicted) }` (a request with
no SLO is compared against itself → never a violation),
`violation_ratio = violations / total`, and `admit = violation_ratio <= threshold`.
An empty batch admits trivially. `detail` carries `violation_ratio`,
`violation_count`, `scored_units`.

> `vss_predictor.py` still contains the now-unused job-scoped predictor and a
> deprecated lookahead predictor — dead code pending a trim.

### 4.3 `reactive` — Phase-3 placeholder

`--halo-admission-policy reactive` is **not implemented**. `_build_policy` logs
a warning and returns `None`, so the gate runs only the KV cap. It is intended
as a measured, queueing-inclusive reactive gate; the `vss_predictor` kernel is
slated for reuse as its interference term.

## 5. Reject reason strings

Returned in `AdmissionResult.reason` and used verbatim as Prometheus `reason`
label values and decision-log keys.

| Reason | Raised by | Meaning |
|---|---|---|
| `ADMIT` | gate | admitted (not a rejection) |
| `HALO_KV_CAP` | gate, Stage B′ | KV usage ≥ `kv_cap_ratio` |
| `MOONCAKE_TTFT` | `MooncakePolicy` Stage 1 | predicted TTFT breaches `ttft_slo` |
| `MOONCAKE_TBT` | `MooncakePolicy` Stage 2 | predicted TBT breaches `tbt_slo` |
| `MOONCAKE_TBT_REACTIVE` | `MooncakePolicy` Stage 3 | smoothed TBT EWMA breaches `tbt_slo · ratio` |
| `HALO_VSS_PREDICTED` | `VssPolicy` | VSS violation ratio over threshold |
| `HALO_DISABLED` | controller | constant for the disabled path |

Where they surface:
- **Prometheus** — `HaloController.register_request` calls
  `metrics.record_request_rejected(result.reason)`, feeding the
  `sglang:halo_requests_rejected_total{reason}` counter (rank-0 + `--enable-metrics`).
- **Decision log** — `_log_decision` writes one JSONL row per decision
  (`ts_ns`, `rid`, `policy`, `dry_run`, `decision`, `reason`,
  `predicted_ttft_ms`, `predicted_tbt_ms`, `detail`) to a ring buffer of 32
  (surfaced via `snapshot()` → `/halo/status`) and, when
  `--halo-admission-decision-log` is set, to an append-only file (rank-0 only).

## 6. Controller wiring (`controller.py`)

`HaloController` builds the cost models, the `RequestTracker`, the shared
`TBTEwmaTracker`, and the `HaloAdmissionGate(policy, kv_cap_ratio)`.
`build_halo_controller_from_server_args` is the factory — returns `None` when
`--halo-enabled` is off; otherwise snapshots the `--halo-*` flags into
`HaloConfig`.

`register_request(rid, *, ttft_slo, tbt_slo, e2e_slo, prompt_len, prefix_len,
kv_usage_ratio)` is the admission hook:

1. Build `NewRequestInput` (`prompt_len`/`prefix_len` floored at 0).
2. `snapshot = tracker.snapshot(kv_usage_ratio=...)`.
3. `result = gate.decide(new_req, snapshot)`.
4. `_log_decision(rid, result)` — always logged.
5. If `not result.admit` **and not** `admission_dry_run`:
   `metrics.record_request_rejected(reason)`, `tracker.on_rejected(rid, reason)`,
   then **raise `HaloRejectError(reason, rid)`** — the scheduler converts it to
   an HTTP-400 abort carrying `reason`.
6. Otherwise (admitted, **or** a dry-run would-reject):
   `tracker.on_admitted(...)` records the request — including
   `predicted_ttft_ms` / `predicted_tbt_ms` from the `PolicyDecision` — and
   `metrics.record_request_admitted()` fires.

**Dry-run** (`--halo-admission-dry-run`): a would-reject is still logged but the
request is admitted and tracked anyway — for safe shadow evaluation.

---

*Authoritative source is the code itself plus the module `CLAUDE.md` files
(`managers/halo/CLAUDE.md`, `managers/halo/admission_control/CLAUDE.md`). This
doc tracks them and may lag behind code changes.*
