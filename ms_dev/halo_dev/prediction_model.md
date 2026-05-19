# Project Halo — Cost / Prediction Models

Internal doc for Halo's latency cost models: what they predict, their exact
formulas, JSON schema, the per-step sampler, and the offline fit workflow.

> Framing: Project Halo is a **request-level** admission-control + tracking
> subsystem in SGLang's single-instance scheduler (`managers/halo/`,
> off by default behind `--halo-enabled`). See `managers/halo/CLAUDE.md`.

Code is in `python/sglang/srt/managers/halo/admission_control/cost_model.py`
unless stated otherwise.

## 1. Purpose

The cost models estimate **forward-step latency** (and its prefill / decode
specializations) in milliseconds, from cheap batch-composition features. They
are pure regressions — no GPU calls — so admission policies and the request
tracker can predict latency on the scheduler hot path.

Consumers:

- **Admission policies** — `MooncakePolicy` (predicted TTFT/TBT vs
  `halo_ttft_slo` / `halo_tbt_slo`) and `VssPolicy` (virtual-server-slowdown
  via `vss_predictor.py`, scored against `halo_e2e_slo`). `VssPolicy` requires
  the Halo Step Cost Model.
- **Solo-run baselines** — `RequestTracker` (`request_tracker/tracker.py`)
  uses the model to compute each request's *solo* (single-request, no
  contention) prefill and decode time. The solo baseline is the denominator
  of the e2e-slowdown ratio.

Three models coexist in `cost_model.py`. All load coefficients from JSON; a
malformed/missing file raises the typed `CostModelLoadError`, which the
`try_load_*` loaders catch and downgrade to "stage disabled" (return `None`)
rather than crash.

## 2. The models and their exact formulas

### PrefillCostModel (legacy, per-request)

`estimate_ms(prompt_len, prefix_len)` with `d = max(0, prompt_len - prefix_len)`:

```
T_prefill_ms ≈ α·d² + β·d + γ + δ·prefix_len
```

| Coef | Meaning |
|---|---|
| `alpha` | quadratic prefill compute (attention-in-prompt-length) |
| `beta`  | linear prefill compute |
| `gamma` | fixed per-request overhead |
| `delta` | cost of loading the matched radix-cache prefix (~7 µs/token on B200×4 Llama-3.3-70B). Optional in JSON, defaults to `0.0` (back-compat) |

### TBTCostModel (legacy, per-step approximation)

`estimate_ms(batch_size, per_req_kv)` — caller computes
`per_req_kv = total_kv // batch_size`:

```
T_tbt_ms ≈ a + b·batch_size + c·per_req_kv
```

Linear: TBT grows with batch size (compute overhead) and per-request KV span
(attention memory bandwidth). Higher-order terms are absorbed by re-fitting.

### HaloStepCostModel (current, per-step)

Estimates one forward-step's latency from full batch composition. The caller
decides what the `Σ` sums contain — a batch of 1 for solo baselines, the real
or hypothetical batch for admission lookahead. Two **forms**, selected at
JSON-load time from the `form` field:

**`halo_step_v1` (unified, 6 coefficients)** — `estimate_step_ms(...)`:

```
T_step_ms ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ
          + θ_d1·Σrⱼ + θ_d2·bs_d + θ_c
```

**`halo_step_split_v1` (split, 7 coefficients)** — prefill and decode steps
each get their own intercept; same 5 slopes:

```
T_prefill_step ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ + θ_c_p
T_decode_step  ≈ θ_d1·Σrⱼ  + θ_d2·bs_d + θ_c_d
```

| Coef | Term | Meaning |
|---|---|---|
| `theta_p1` | `Σnᵢ²`    | self-attention on new (uncached) tokens |
| `theta_p2` | `Σ(nᵢrᵢ)` | cross-attention: new tokens × cached prefix |
| `theta_p3` | `Σnᵢ`     | per-token MLP/FFN |
| `theta_d1` | `Σrⱼ`     | decode attention memory BW (batch KV sum) |
| `theta_d2` | `bs_d`    | per-request decode overhead |
| `theta_c`  | const     | unified-mode shared intercept |
| `theta_c_p` / `theta_c_d` | const | split-mode prefill / decode intercepts |

`nᵢ` = new tokens of prefill request *i*; `rᵢ` = its cached prefix length;
`rⱼ` = KV span of decode request *j*; `bs_d` = decode batch size.

In `estimate_step_ms`, split mode picks the intercept matching the step:
prefill-only → `θ_c_p`, decode-only → `θ_c_d`, mixed (rare when
`--enable-mixed-chunk` is off) → both intercepts summed (slight over-estimate,
acceptable). The split form mirrors MuxWise's original separated form and is
justified when MIXED steps never occur.

Solo shortcuts (batch of 1, for the Halo R1 / e2e-slowdown denominator):

| Method | Equivalent to | Form |
|---|---|---|
| `estimate_solo_prefill_total_ms(n, r)` | `estimate_step_ms([(n,r)], [])` — `θ_p1·n² + θ_p2·n·r + θ_p3·n + θ_c[_p]` |
| `estimate_solo_tbt_ms(r)` | `estimate_step_ms([], [r])` — `θ_d1·r + θ_d2·1 + θ_c[_d]` |

### Which supersedes which

`HaloStepCostModel` **supersedes** the legacy `PrefillCostModel` /
`TBTCostModel` pair. When the step model is configured it is used; the legacy
pair is the fallback. `RequestTracker` selects in this order:

1. `step_cost` set → Halo Step Cost Model path (preferred)
2. else `prefill_cost` / `tbt_cost` set → legacy two-model path
3. else → solo estimates return `0.0`

Setting all three logs an INFO line and the legacy pair is ignored.

## 3. JSON schema + loader flags

Each model is one JSON object. `fit_metadata` is free-form and ignored by the
loaders. Loaded via `try_load_{prefill,tbt,halo_step}_cost_model` from
`server_args` paths:

| Flag | Loads |
|---|---|
| `--halo-prefill-cost-model-path` | `PrefillCostModel` |
| `--halo-tbt-cost-model-path` | `TBTCostModel` |
| `--halo-step-cost-model-path` | `HaloStepCostModel` (auto-detects form) |

**PrefillCostModel:**
```json
{ "alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0, "fit_metadata": {} }
```
`delta` optional (defaults `0.0`). `alpha`/`beta`/`gamma` required, numeric.

**TBTCostModel:**
```json
{ "a": 0.0, "b": 0.0, "c": 0.0, "fit_metadata": {} }
```

**HaloStepCostModel — `halo_step_v1`:**
```json
{
  "form": "halo_step_v1",
  "theta_p1": 0.0, "theta_p2": 0.0, "theta_p3": 0.0,
  "theta_d1": 0.0, "theta_d2": 0.0, "theta_c": 0.0,
  "fit_metadata": {}
}
```

**HaloStepCostModel — `halo_step_split_v1`:**
```json
{
  "form": "halo_step_split_v1",
  "theta_p1": 0.0, "theta_p2": 0.0, "theta_p3": 0.0, "theta_c_p": 0.0,
  "theta_d1": 0.0, "theta_d2": 0.0, "theta_c_d": 0.0,
  "fit_metadata": {}
}
```
`form` must be one of the two identifiers or load fails. On split load,
`theta_c` is mirrored from `theta_c_d` so form-agnostic callers see a value.

## 4. The per-step cost-model sampler

`managers/halo/cost_model_sampler.py::HaloCostModelSampler` collects the fit
data. The scheduler hot path calls `observe_step(batch, step_time_ms)` once
per forward step; the sampler extracts the 5-tuple of `Σ` features plus the
measured step time, pushes onto a bounded in-memory deque, and a daemon thread
flushes JSONL every `flush_interval_ms` — no I/O on the scheduler thread.

- Enabled **only** when `--halo-cost-model-sample-log <path>` is set;
  otherwise `scheduler.halo_cost_sampler` stays `None` and the hook
  short-circuits. `--halo-cost-model-sample-every N` subsamples every Nth step.
- TP-dedup: only `attn_tp_rank == 0` instantiates a sampler
  (`build_halo_cost_sampler_from_server_args`).
- Skips IDLE / DRAFT_EXTEND / SPLIT_PREFILL etc.; keeps EXTEND / MIXED /
  DECODE. Within EXTEND/MIXED, a req with `extend_input_len > 1` counts as
  prefill, `== 1` as decode.
- Queue-full drops the **oldest** sample and logs a periodic WARN.

Each JSONL row:
```json
{"step_idx":1,"ts_ns":0,"sum_n_sq":0.0,"sum_nr":0.0,"sum_n":0,"sum_r":0,
 "bs_d":0,"step_time_ms":0.0,"num_prefill_reqs":0,"num_decode_reqs":0,
 "max_kv":0,"forward_mode":"DECODE"}
```

**`--disable-overlap-schedule` is required.** Overlap-schedule mode pipelines
CPU and GPU work, so the measured `step_time_ms` no longer reflects a single
forward pass and would invalidate the fit. The experiment orchestrator
(`expctl/server_run_experiment.py`) appends this flag automatically when the
sampler is enabled.

## 5. The fitting workflow (`tools/halo/`)

`tools/halo/fit_halo_cost_model.py` does an offline OLS fit
(`numpy.linalg.lstsq`) of the sampler JSONL into a `HaloStepCostModel` JSON.
See `tools/halo/README.md` for the full runbook.

End-to-end: **(1)** launch a server with `--halo-cost-model-sample-log` (+
overlap off) → **(2)** drive any workload producing mixed batch compositions →
**(3)** run `fit_halo_cost_model.py` → **(4)** relaunch with
`--halo-step-cost-model-path` pointing at the fitted JSON.

```bash
python tools/halo/fit_halo_cost_model.py \
    --samples <session>/halo_cost_samples.jsonl \
    --output  halo_step_split_llama3-70b_b200x4.json \
    --model "Llama-3.3-70B-Instruct" --hw "B200x4 TP=4"
```

- `--form` ∈ `{split, unified}`, **default `split`** (since 2026-05-14): fits
  EXTEND rows and DECODE rows independently (own intercepts), MIXED rows
  dropped with a WARN. `--form unified` does the joint 6-coefficient fit and
  is preferred when MIXED steps appear (`--enable-mixed-chunk` on).
- Filters: `--filter-forward-mode`, `--min/--max-step-time-ms`,
  `--min-samples` (default 200, refuse to fit below it).
- Reports `n_samples`, `RMSE_ms`, `R²`, and the θ coefficients; sanity targets
  are `R² > 0.9`, all θ finite, `θ_d1 > 0`.

## 6. Known limitations

- **Coefficients are not portable.** Each model is fit per
  **(GPU type × TP size)** — and effectively per (model, kernel config). A
  JSON fit on B200×4 TP=4 is invalid on other hardware; re-collect and re-fit
  when hardware or parallelism changes.
- **Solo-e2e baseline is an approximation.** `RequestTracker.on_finished`
  computes `solo_e2e_ms = solo_prefill + solo_decode`, where `solo_decode`
  uses the **mean-KV span** (`prompt_len + decoded_tokens/2`) for a single
  representative decode step rather than integrating per-step cost as KV
  grows. The e2e-slowdown ratio is `e2e_ms / max(solo_e2e_ms, MIN_SOLO_MS)`
  (floor `MIN_SOLO_MS = 1.0` guards the division). The solo predictor is
  slated for a dedicated rework.

---

The source code is authoritative. If this doc and `cost_model.py` /
`cost_model_sampler.py` / `tracker.py` / `fit_halo_cost_model.py` disagree,
trust the code.
