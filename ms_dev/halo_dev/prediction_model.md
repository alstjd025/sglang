# Halo prediction model — Step Cost Model + Virtual Job Slowdown

This document specifies, code-1:1, the two prediction components of Project
Halo:

1. **Halo Step Cost Model** — predicts the wall-clock time of one SGLang
   forward step from its batch composition.
2. **Virtual Job Slowdown (VJS)** — the per-job slowdown metric, built on
   top of the cost model.

Both are used by job slowdown tracking (the periodic sweep) and by Phase 2
admission control. The admission decision logic itself is in
[`admission_design.md`](admission_design.md); this document is the
*measurement / prediction* layer underneath it.

Authoritative source files:
- `python/sglang/srt/managers/admission_control/cost_model.py` — `HaloStepCostModel`
- `python/sglang/srt/managers/halo/slowdown_tracker.py` — `SlowdownTracker`, `compute_job_vjs`
- `python/sglang/srt/managers/halo/job.py` — `Job`, `JobCallSpan`

---

## 1. Why a step cost model

Halo needs a *solo-run baseline*: "how long would this work take if the job
ran alone on the server?" Slowdown = actual / solo.

The earlier Mooncake-style admission used two separate models — a prefill
TTFT model and a TBT model with a `per_req_kv = total_kv / batch_size`
reparameterization. That TBT model had a **cliff**: its output sat in a
narrow ~9–14 ms band regardless of batch composition, because dividing the
batch KV by the batch size cancels the very load signal it should track.
Measured TBT under load was ~5× higher than the model predicted.

The Halo Step Cost Model fixes this by regressing **one whole forward step**
on the *summed* batch composition (`Σrⱼ`, not `Σrⱼ / bs`), so the model
output scales with real batch load. The form is adapted from MuxWise
(ASPLOS '26, arXiv 2504.14489); §7 records what was borrowed and what was
re-designed.

---

## 2. The cost model — `HaloStepCostModel`

### 2.1 Per-step formula

A forward step processes some prefill (EXTEND) work and/or some decode work.
The step time is regressed as:

```
T_step ≈ θ_p1·Σ nᵢ²              prefill: new-token self-attention
       + θ_p2·Σ (nᵢ·rᵢ)          prefill: new-token × cached-prefix cross-attention
       + θ_p3·Σ nᵢ               prefill: per-new-token MLP / FFN
       + θ_d1·Σ rⱼ               decode: batch KV-sum (attention memory bandwidth)
       + θ_d2·bs_d               decode: per-request decode overhead
       + intercept               fixed per-step cost
```

where, over the requests in the step:
- `nᵢ` = new (uncached) prefill tokens of prefill request *i* = `prompt_len − prefix_len`
- `rᵢ` = cached prefix length of prefill request *i*
- `rⱼ` = current KV span of decode request *j*
- `bs_d` = number of decode requests in the step

### 2.2 Two forms — split (default) and unified

The host runs **chunked prefill ON, `--enable-mixed-chunk` OFF**, so a step is
either EXTEND-only or DECODE-only — never mixed. That makes the prefill and
decode intercepts separable, so the **split form** (`halo_step_split_v1`,
7 coefficients) is the default — it fits prefill steps and decode steps
independently, each with its own intercept:

- prefill-only step intercept: `θ_c_p`
- decode-only step intercept:  `θ_c_d`

The **unified form** (`halo_step_v1`, 6 coefficients) keeps a single shared
intercept `θ_c`; use it only when MIXED steps occur (`--enable-mixed-chunk`
on). `HaloStepCostModel.is_split` selects the behavior; `_prefill_const()` /
`_decode_const()` return `θ_c_p` / `θ_c_d` in split mode and fall back to
`θ_c` in unified mode.

### 2.3 Code interface — three methods

`cost_model.py::HaloStepCostModel` exposes exactly three public estimators.

**(a) `estimate_step_ms(prefill_infos, decode_infos) -> float`** — the full
batched estimate.
- `prefill_infos`: iterable of `(n, r)` tuples (new tokens, cached prefix).
- `decode_infos`: iterable of `r` (decode KV spans).
- Computes `Σnᵢ²`, `Σ(nᵢrᵢ)`, `Σnᵢ`, `Σrⱼ`, `bs_d`, applies the formula.
- Split mode: adds `θ_c_p` if the step has any prefill, `θ_c_d` if it has any
  decode, both if mixed (over-estimates by one intercept — acceptable since
  split mode assumes mixed steps don't occur).

**(b) `estimate_solo_prefill_total_ms(n, r) -> float`** — one request's
prefill, alone. Equivalent to `estimate_step_ms([(n, r)], [])`:
```
θ_p1·n² + θ_p2·n·r + θ_p3·n + θ_c_p
```

**(c) `estimate_solo_tbt_ms(r) -> float`** — one decode step for a solo
decoder at KV span `r`. Equivalent to `estimate_step_ms([], [r])`:
```
θ_d1·r + θ_d2·1 + θ_c_d
```

### 2.4 JSON schema

`HaloStepCostModel.from_json(path)` loads a file like:

```json
{
  "form": "halo_step_split_v1",
  "theta_p1": 1.0e-7, "theta_p2": 4.0e-7, "theta_p3": 0.013,
  "theta_d1": 1.5e-5, "theta_d2": 0.18,
  "theta_c_p": 70.0, "theta_c_d": 22.0,
  "model": "Llama-3.3-70B-Instruct", "hw": "B200x2 TP=2"
}
```

Unified form uses `"form": "halo_step_v1"` and a single `"theta_c"` instead
of the `theta_c_p` / `theta_c_d` pair. Malformed / missing files raise
`CostModelLoadError`.

Cost-model JSON files are fit per-(GPU type × TP size) — see §6. They live
under `ms_dev/runtime/cost_models/` (gitignored).

---

## 3. Virtual Job Slowdown (VJS)

### 3.1 Definition

A job is a client-supplied group of LLM calls (an agent chain) sharing one
SLO. The SLO is a slowdown bound: the job's end-to-end LLM time (tool-call
gaps excluded) may be at most `τ ×` its solo-run time.

**Virtual job slowdown** is the running estimate of that ratio:

```
VJS = (job critical-path actual ms) / (job critical-path solo ms)
```

accumulated over the job's whole life so far — every **completed** call plus
every **in-flight** call. It is recomputed every sweep (~100 ms) and stored
on `Job.virtual_job_slowdown` (initial value = the job's SLO, the
measurement-less worst-case fallback).

Three properties the definition must satisfy, and how:
- **tool-delay gaps excluded** — only time inside LLM calls is counted.
- **concurrent calls merged by critical path** — if a job issues calls in
  parallel, the overlapping span counts once (the longest call decides it),
  not as the sum.
- **the job's own self-batching is not slowdown** — if a job issues K calls
  concurrently, those K calls batch together even when the job runs alone;
  that cost belongs in *both* actual and solo, so it cancels.

### 3.2 The stage-merge model

`SlowdownTracker.compute_job_vjs(spans, fallback_slo)` implements it. A
`JobCallSpan` (`job.py`) is one call's record:

```
JobCallSpan(admitted_ts, end_ts, prompt_len, prefix_len, decoded_tokens, kv_len)
```

`admitted_ts` / `end_ts` are monotonic seconds; for an in-flight call
`end_ts` is "now", for a completed call it is the finish time.

**Step 1 — merge spans into stages.** Sort spans by `admitted_ts`, sweep,
and merge any span that starts at or before the running stage's current max
end time. Each resulting **stage** is one connected component of the
interval-overlap graph — equivalently, one contiguous busy span of the job.
Sequential calls (with or without a gap between them) land in separate
stages; concurrent calls land in the same stage.

**Step 2 — per-stage actual.**
```
stage_actual = max(end_ts) − min(admitted_ts)        # over the stage's spans
```
This is the interval-union length of the stage (contiguous, so just the
span). Summing over stages excludes the gaps between stages — that is the
tool-delay exclusion.

**Step 3 — per-stage solo.** The stage's calls run as one batch (the job
alone, self-batching preserved). The DECODE step time for that batch:
```
decode_step = decode_step_ms([kv_len of the stage's decode-phase calls])
```
A call is decode-phase iff `decoded_tokens > 0`. Then each call's solo time:
```
call_solo = prefill_solo_ms(prompt_len, prefix_len) + decoded_tokens · decode_step
stage_solo = max over the stage's calls of call_solo        # critical path
```
Charging decode steps at the batch-of-K `decode_step` (not batch-of-1) is
what keeps a job's self-batching out of the slowdown: a job running alone
has actual decode steps at batch-K too, so VJS ≈ 1.

**Step 4 — job total.**
```
job_actual_ms = Σ stage_actual · 1000
job_solo_ms   = Σ stage_solo
VJS = job_actual_ms / max(job_solo_ms, MIN_SOLO_MS=1.0)
```
Returns `fallback_slo` when there are no spans, no cost model, `job_solo ≤ 0`,
or the ratio is non-finite.

For a single-call stage, `decode_step_ms([kv])` reduces to
`estimate_solo_tbt_ms(kv)` and `call_solo` to the plain per-request solo —
the general formula degenerates to the obvious one-request case.

### 3.3 The two solo primitives

`compute_job_vjs` builds `call_solo` from two `SlowdownTracker` methods that
encapsulate the path-A (step model) / path-B (legacy) branch:

**`prefill_solo_ms(prompt_len, prefix_len)`** — one call's solo prefill.
```
n_new = max(0, prompt_len − prefix_len)        # radix-cache hit excluded
path A: step_cost.estimate_solo_prefill_total_ms(n_new, prefix_len)
path B: prefill_cost.estimate_ms(prompt_len, prefix_len)   # legacy; computes n−p itself
none:   0.0
```
The `n_new = prompt_len − prefix_len` term is important: the cached prefix
(`prefix_len` tokens, a radix-cache hit) is **not** re-prefilled, so it must
not be counted as new-token prefill compute.

**`decode_step_ms(kv_lens)`** — one DECODE step for a batch with the given
per-request KV spans.
```
empty list: 0.0
path A: step_cost.estimate_step_ms([], kv_lens)
path B: tbt_cost.estimate_ms(batch_size=K, per_req_kv=Σkv // K)
```

### 3.4 Worked example

Job with four calls; calls 2 and 3 ran concurrently:

| call | interval `[admit, end]` (s) |
|---|---|
| 1 | `[0, 10]` |
| 2 | `[12, 25]` |
| 3 | `[14, 22]` — overlaps call 2 |
| 4 | `[40, 50]` (50 = now, in-flight) |

Stages (overlap-connected components): `{1}`, `{2, 3}`, `{4}`.

```
stage {1}:     actual = 10−0  = 10 s,   solo = call_solo₁
stage {2,3}:   actual = 25−12 = 13 s,   solo = max(call_solo₂, call_solo₃)
               where call_soloⱼ uses decode_step_ms([kv₂, kv₃])  (batch of 2)
stage {4}:     actual = 50−40 = 10 s,   solo = call_solo₄

job_actual = (10 + 13 + 10) · 1000 = 33000 ms
job_solo   = call_solo₁ + max(call_solo₂, call_solo₃) + call_solo₄
VJS        = 33000 / job_solo
```

The gaps `[10,12]` and `[25,40]` (tool delays) are not counted. Calls 2 and 3
contribute one critical-path stage, with their decode steps charged at the
batch-of-2 step time.

### 3.5 Lifecycle — where the spans come from

- **Admission** sets `Req.halo_first_admitted_ts`.
- **Per tick**, `scheduler._halo_build_request_execution_infos` snapshots
  every in-flight Halo request into a `RequestExecutionInfo`
  (`rid, job_id, prompt_len, prefix_len_at_admission, decoded_tokens_so_far,
  kv_len_now, elapsed_ms, admitted_ts`).
- **`SlowdownTracker.sweep`** groups those by job, turns each into an
  in-flight `JobCallSpan` via `span_from_info`, prepends the job's
  `completed_call_spans`, calls `compute_job_vjs`, and stores the result via
  `Job.record_vjs`.
- **On finish**, `scheduler._halo_on_request_finished` builds the finished
  request's final `RequestExecutionInfo`; `HaloController.on_request_finished`
  freezes it into the owning job's `completed_call_spans`
  (`Job.record_completed_call`) so the call keeps contributing after the
  request object is gone.

A job with completed calls but zero in-flight calls (between rounds, during
a tool delay) is not re-swept — its VJS is unchanged anyway, since no work
is happening.

### 3.6 One source of truth — sweep and admission

Both the periodic sweep and the Phase 2 job-scoped admission gate compute a
job's VJS through the **same `compute_job_vjs`**:
- the sweep records it onto `Job.virtual_job_slowdown` (monitoring, JSONL);
- `HaloController._build_active_jobs_input` calls it to fill
  `JobLookaheadInput.current_vjs`, which the job-scoped predictor multiplies
  by a per-phase stretch (see `admission_design.md` §4).

There is no second VJS formula. The request-scoped admission baseline
(`admission_design.md` §4.2) deliberately does *not* use `compute_job_vjs` —
it scores each request on its own batch-of-1 solo (`_solo_ms_for_request`),
which is the point of that baseline.

---

## 4. Known limitations

1. **KV-cache pressure / eviction cliff is not modeled.** `Σrⱼ` is a proxy
   for decode attention memory bandwidth, but the model has no term for the
   sharp non-linearity when `token_usage_pct` nears the pool limit
   (eviction, preemption). When that happens, measured step times jump and
   VJS over-shoots. Mitigation belongs to a later scheduling phase.
2. **Flat-KV decode approximation.** A call's decode solo uses its
   *current* `kv_len` for all of its `decoded_tokens` steps, rather than
   integrating over the KV span as it grows. Slight under-estimate of solo
   for long generations.
3. **Snapshot KV in concurrent stages.** A concurrent stage charges every
   call's decode steps at the batch-of-K step time computed from the
   *current* KV set; as shorter calls finish the batch shrinks, so this
   slightly over-estimates `stage_solo`. Bounded and small.
4. **Cost-model accuracy bounds VJS accuracy.** VJS is only as good as the
   fitted coefficients. First-fit validation (split form, λ=0.3 workload)
   showed the cliff resolved and tail/p90 close to ground truth, with
   mean/p50 mildly conservative — see this file's git history for the
   detailed first-fit numbers.

---

## 5. CLI flags / env vars

| CLI flag | env var | Effect |
|---|---|---|
| `--halo-step-cost-model-path` | `SGLANG_HALO_STEP_COST_MODEL` | Load a `HaloStepCostModel`. When set, supersedes the legacy pair. |
| `--halo-prefill-cost-model-path` | `SGLANG_HALO_PREFILL_COST_MODEL` | Legacy path-B prefill model (fallback). |
| `--halo-tbt-cost-model-path` | `SGLANG_HALO_TBT_COST_MODEL` | Legacy path-B TBT model (fallback). |
| `--halo-cost-model-sample-log` | `SGLANG_HALO_COST_MODEL_SAMPLE_LOG` | Per-step JSONL sampler output, for fitting. |

`ms_dev/experiments/halo_base.sh` auto-selects the step-model JSON for the
host's GPU tag (`b200x4`, `b200x2`, …). Path B is used only when no step
model is loaded; if all three are set the step model wins and the legacy
pair is ignored (logged INFO).

---

## 6. Fitting a cost model

The model is fit per hardware (GPU type × TP size). Workflow:

1. **Collect** per-step samples: launch the server with
   `SGLANG_HALO_COST_MODEL_SAMPLE_LOG=auto` (forces `--disable-overlap-schedule`
   so step-latency measurement is valid) and drive any mixed-composition
   workload. Samples land in `<session>/halo_cost_samples.jsonl`.
2. **Fit** with `tools/halo/fit_halo_cost_model.py` — OLS over the 6/7
   coefficients. Default form is `split`. Produces the JSON of §2.4.
3. **Use**: point `SGLANG_HALO_STEP_COST_MODEL` at the JSON (or let
   `halo_base.sh` pick it by GPU tag).

Sanity checks on a fresh fit: R² > 0.9, all θ finite, `θ_d1 > 0` (the
KV-sum term must be positive — that was the headline cliff fix). Full
workflow: `tools/halo/README.md`.

---

## 7. Relation to MuxWise

**Borrowed**: the idea of regressing a whole forward step on summed batch
composition, and the split prefill/decode term structure with separate
intercepts.

**Re-designed for this host**: the `Σrⱼ` decode term replacing Mooncake's
`per_req_kv = total_kv / bs` reparameterization (the cliff fix); the
`Σ(nᵢrᵢ)` prefill cross-term (cache-hit-heavy agent workloads make it
significant); split as the default form because `--enable-mixed-chunk` is
off here; and the VJS stage-merge layer (§3), which is Halo-specific —
MuxWise has no job-level slowdown notion.

DistServe's (OSDI '24) simulator was considered but not ported: it assumes
disaggregated prefill/decode, which this single-instance NULL-disaggregation
setup does not use.
