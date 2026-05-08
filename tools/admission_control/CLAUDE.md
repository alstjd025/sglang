# Admission Control Tooling

Standalone scripts for fitting cost models and analyzing/tuning admission decisions
without touching production traffic. Companion to the main module at
[python/sglang/srt/managers/admission_control/CLAUDE.md](../../python/sglang/srt/managers/admission_control/CLAUDE.md).

## Files

| File | Purpose |
|---|---|
| `fit_cost_model.py` | Drive a running SGLang server through a sweep of (n,p) and/or (batch_size, kv) and emit a JSON cost-model file |
| `replay_admission.py` | Take a dry-run decision log and report what would be admitted/rejected at any candidate SLO |

Both are pure clients — no SGLang internals. Run anywhere the server is reachable.

---

## fit_cost_model.py

Produces JSON consumable by `--admission-prefill-cost-model-path` and `--admission-tbt-cost-model-path`.

### Targets

- `--target prefill` — fits `T_prefill ≈ α·d² + β·d + γ` (d = n − p). Output:
  `{"alpha", "beta", "gamma", "fit_metadata": {...}}`.
- `--target tbt` — fits `TBT ≈ a + b·batch_size + c·per_req_kv` (per_req_kv = total_kv // bs).
  Output: `{"a", "b", "c", "fit_metadata": {...}}`.

### Method (prefill)

For each (n, p) cell:

1. (Optional) `POST /flush_cache` to start with an empty radix cache.
2. Generate `p` random token ids, send as a warm-up request with
   `max_new_tokens=1`. SGLang inserts the prefix into the radix cache.
3. Build the measure prompt = warm-up ids + `n - p` fresh random ids.
4. POST to `/generate` with `max_new_tokens=1`, `stream=true`. Time the wall
   clock from request send to first SSE chunk = TTFT ≈ T_prefill when the
   server is idle.
5. Validate: response's `meta_info.cached_tokens` should match `p` (modulo
   page-size rounding). The script logs a WARN if they diverge.
6. Repeat `--repeats` times per cell.

Then `numpy.polyfit(d, t, deg=2)` over all `(d=n-p, t=ttft_ms)` samples
fits α, β, γ. Pure-Python fallback is included so the tool works without
numpy.

The default grid covers d ∈ {0, 128, 256, 512, 1k, 2k, 4k, 8k, 16k} at three
prefix sizes (0, 8k, 32k) — chosen from the production distribution
(d median ~1.5k, p99 ~12k; cache hit ratio median 93-96%).

### Method (TBT)

For each (batch_size, prompt_len) cell:

1. Build `batch_size` distinct random prompts of length `prompt_len`. Distinct
   contents prevent the radix cache from collapsing them into one entry.
2. Open `batch_size` concurrent streaming `/generate` calls in a thread pool,
   each with `max_new_tokens = warmup_tokens + measure_tokens`.
3. Per stream, time the inter-chunk intervals. Discard the first
   `warmup_tokens` (covers initial prefill + decode ramp-up) and keep the
   next `measure_tokens` intervals.
4. Across all streams, take the median of all retained intervals as the
   cell's TBT estimate. `total_kv ≈ batch_size * prompt_len` is the cell's
   KV occupancy proxy at decode start.

Linear fit `TBT = a + b·bs + c·per_req_kv` (where per_req_kv = total_kv // bs)
via `numpy.linalg.lstsq` (or pure-Python normal equations as fallback).

The regressor used to be `total_kv` instead of `per_req_kv`; on production data
that fitted a non-physical negative `b` because per-request KV (the dominant
attention-bandwidth cost per step) is collinear with `bs` when cells have
uniform per-request length. Switching to `per_req_kv` separates batch-size
overhead from per-request memory cost cleanly.

Caveat: TBT depends on more than just (bs, per_req_kv) — quantization,
attention backend, sequence length distribution within the batch all matter —
so re-fit when any of those change.

### Usage

The actual CLI uses **subcommands** `prefill` and `tbt`, not `--target`. Defaults
target the user's production workload (Llama-3.3-70B, agent-style traces with
median prompt 18-30k tokens, 93-96% cache hit ratio, d=n-p median ~1500).

```bash
# Prefill cost model (uses production-tuned default grid)
python tools/admission_control/fit_cost_model.py prefill \
    --server http://localhost:31000 \
    --output ms_dev/runtime/cost_models/prefill_llama3-70b.json

# Override grid:
python tools/admission_control/fit_cost_model.py prefill \
    --server http://localhost:31000 \
    --output prefill.json \
    --n-list 100 500 1000 2000 4000 8000 16000 \
    --p-list 0 1000 4000 \
    --repeats 3 \
    --flush-each

# TBT cost model — runs concurrent batches in-process; no other traffic should
# hit the server during this run. Default grid bs∈{1,4,8,16,24,32} ×
# prompt_len∈{1k,4k,16k,32k,64k}, with cells whose total_kv > 1.2M skipped to
# stay within VRAM and bracket production p99 (≈920k total_kv, ≈61k per_req_kv).
python tools/admission_control/fit_cost_model.py tbt \
    --server http://localhost:31000 \
    --output ms_dev/runtime/cost_models/tbt_llama3-70b.json \
    --warmup-tokens 10 --measure-tokens 30
```

### Output location convention

Per the design, cost models live under `ms_dev/runtime/cost_models/` (gitignored —
they are HW/model specific and regenerated when either changes). The directory is
created by the fit script if missing.

### Re-fit triggers

- Different model
- Different GPU type / TP size
- Different `--mem-fraction-static`, `--chunked-prefill-size`, `--max-prefill-tokens`
- Different attention backend
- After enabling/disabling speculative decoding

---

## replay_admission.py

Takes a JSONL decision log (produced by `--admission-decision-log`) and reports what
the policy would have decided under one or more candidate SLO settings. Lets you tune
SLOs against real traffic before flipping enforcement on.

### Producing the input

The scheduler appends one row per decision when `SGLANG_ADMISSION_DECISION_LOG`
(or `--admission-decision-log`) is set. With `SGLANG_ADMISSION_DRY_RUN=1` every
reject becomes "would-have-rejected" — perfect for sweeping SLOs without
actually losing requests.

The simplest path:

```bash
source ms_dev/experiments/admission_dryrun.sh
python3 ms_dev/expctl/run_experiment.py --mode single
# ...drive your stress workload...
```

`run_experiment.py` auto-routes the decision log to
`<session_dir>/admission_decisions.jsonl` when SLO env vars are set and
`SGLANG_ADMISSION_DECISION_LOG` is unset. Only the attn_tp_rank=0 scheduler
writes (TP-dedup), so a TP=N deployment produces one row per decision, not N.

### Inputs

- `--decision-log <path>` — required JSONL path.
- `--ttft-slo <ms>` (repeatable) — candidate TTFT SLO(s) to evaluate; pass `0` to disable.
- `--tbt-slo <ms>` (repeatable) — candidate TBT SLO(s); pass `0` to disable.
- `--ttft-slo-ratio <r>` (repeatable) — candidate TTFT slowdown ratios vs solo-run; pass `1.0` to disable.
- `--tbt-slo-ratio <r>` (repeatable) — candidate TBT slowdown ratios; pass `1.0` to disable.
- `--reactive-ratio <r>` — Stage 3 trip = `tbt_slo * r` (default 0.9).
- `--csv-out <path>` — write the summary table to CSV.
- `--details-out <path>` — when exactly one SLO combo is given, write a
  per-decision CSV including `original_admit / original_reason / replay_verdict`
  for diff-style inspection.

### Outputs

Stdout summary (Cartesian product of all four SLO axes):

```
TTFT_SLO  TBT_SLO  TTFTrx  TBTrx  total  admit  rTTFT  rTTFTr  rTBTp  rTBTr  rTBTrx  rej_pct
---------------------------------------------------------------------------------------------------------
        0         0    1.50    1.00      25      14       0       11       0       0        0    44.00%
        0         0    2.00    1.00      25      14       0       11       0       0        0    44.00%
        0         0    5.00    1.00      25      17       0        8       0       0        0    32.00%
     2000        100    1.00    1.00      25       2      23        0       0       0        0    92.00%
```

Columns:
- `rTTFT` / `rTTFTr` — rejects from absolute / ratio TTFT stage
- `rTBTp` / `rTBTr` / `rTBTrx` — rejects from TBT predicted / ratio / reactive

`--csv-out` produces the same data with the same columns.

### Caveat — counterfactual, not simulation

Replay re-thresholds the predictions stored at decision time; it does NOT recompute
predictions from a different cost model, and it does NOT simulate the feedback effect
of changing the admit/reject decision (the `queue_predicted_ms` in row N reflects the
actual outcome of rows 0..N-1, not the counterfactual). When you sweep down to a
tighter SLO than what was active during the dry-run, reported reject counts are an
**upper bound** — under real enforcement, rejecting earlier requests would shrink the
queue and admit more later ones. For an accurate operating-point estimate, re-run the
workload with the chosen SLO under real enforcement.

---

## File layout

```
tools/admission_control/
├── CLAUDE.md
├── fit_cost_model.py
└── replay_admission.py
```

No package init needed — both are CLI entrypoints.
