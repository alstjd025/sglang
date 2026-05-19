# Halo Admission Control

The admission-control component of Project Halo (see `managers/halo/CLAUDE.md`
for the umbrella). A single gate decides admit/reject for every arriving
request *before* it is queued; on reject the scheduler sends HTTP 400.

> **History.** This was once a standalone job-level-era module
> (`managers/admission_control/`, a Mooncake-style per-request TTFT/TBT gate).
> On 2026-05-19 it moved under `managers/halo/` and became *policy-pluggable*:
> "mooncake" is now one selectable policy, not the whole module.

## The gate

`HaloAdmissionGate.decide(new_req, state) -> AdmissionResult`:

1. **Stage B′ — KV-cache hard cap** (policy-independent). Reject with
   `HALO_KV_CAP` when `state.kv_usage_ratio ≥ kv_cap_ratio`
   (`--halo-admission-kv-cap-ratio`; 0 disables, 0 < r ≤ 1 enables).
2. **The selected policy.** `policy is None` (`--halo-admission-policy off`)
   ⇒ admit; otherwise `policy.decide(...)`.

`new_req` is a `NewRequestInput` (rid, prompt/prefix len, the three SLOs);
`state` is a `ServerStateSnapshot` from the request tracker.

## Policy abstraction (`policy.py`)

```python
class AdmissionPolicy(ABC):
    name: str
    def decide(self, new_req: NewRequestInput,
               state: ServerStateSnapshot) -> PolicyDecision: ...
```

`PolicyDecision` carries `admit`, a policy-specific `reason`, best-effort
`predicted_*` fields (for predicted-vs-actual evaluation + the decision log),
and a free-form `detail` dict. Per design decision D5, all three request SLOs
are always active and each policy uses whichever ones its signal speaks to.

## Policies

| Class | `name` | Signal / SLO used |
|---|---|---|
| `MooncakePolicy` (`policy_mooncake.py`) | `mooncake` | Predicts the arrival's TTFT (queue backlog + solo prefill) and TBT (cost model on the running batch). Rejects against `halo_ttft_slo` / `halo_tbt_slo`. 3-stage: TTFT, TBT-predicted, TBT-reactive-EWMA (Stage 3 absolute-`slo_mode` only). |
| `VssPolicy` (`policy_vss.py`) | `vss` | Memoryless virtual-server-slowdown: scores each running request's predicted VSS if the arrival were admitted; rejects when the violation ratio exceeds `--halo-admission-violation-threshold`. VSS is a contention ratio → scored against `halo_e2e_slo`. Needs the Halo Step Cost Model. |
| (Phase 3) `policy_reactive.py` | `reactive` | Measured, queueing-inclusive reactive gate — not yet implemented. |

`--halo-slo-mode` ∈ {ratio, absolute} sets whether `halo_ttft_slo` /
`halo_tbt_slo` are slowdown ratios (vs the request's solo-run baseline) or
absolute millisecond caps. `halo_e2e_slo` is always a slowdown ratio.

## Cost models (`cost_model.py`)

| Class | Formula |
|---|---|
| `PrefillCostModel` | `T_prefill ≈ α·d² + β·d + γ`, `d = prompt_len − prefix_len` |
| `TBTCostModel` | `TBT ≈ a + b·batch_size + c·per_req_kv` |
| `HaloStepCostModel` | Per-step regression; `estimate_step_ms` / `estimate_solo_prefill_total_ms` / `estimate_solo_tbt_ms` |

Loaded via `try_load_{prefill,tbt,halo_step}_cost_model`. The step model wins
over the legacy prefill/tbt pair when both are configured. `TBTEwmaTracker`
(`tbt_tracker.py`) smooths measured per-decode-step latency for the mooncake
Stage-3 reactive check.

`vss_predictor.py` holds the VSS batch-slowdown kernel (`_augmented_step_times`,
chunk-bounded EXTEND step, `RequestSlowdownAdmissionPredictor`, `decide_admission`)
that `VssPolicy` wraps. It still contains the now-unused job-scoped predictor
and the deprecated lookahead predictor — dead code pending a trim.

## Reject reasons

`HALO_KV_CAP` (gate), `MOONCAKE_TTFT` / `MOONCAKE_TBT` / `MOONCAKE_TBT_REACTIVE`
(mooncake policy), `HALO_VSS_PREDICTED` (vss policy). Used as Prometheus
`reason` labels and decision-log keys.

## Testing

`test/registered/admission/test_admission_control.py` exercises the legacy
`AdmissionController` (still present for reference). The request-level gate +
policies are covered by `test/registered/halo/test_halo_phase1.py`.
