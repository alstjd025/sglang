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
- `--target tbt` — fits `TBT ≈ a + b·batch_size + c·total_kv_tokens`. Output:
  `{"a", "b", "c", "fit_metadata": {...}}`.

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

Linear fit `TBT = a + b·bs + c·kv` via `numpy.linalg.lstsq` (or pure-Python
normal equations as fallback). Caveat: TBT depends on more than just (bs,
kv) — quantization, attention backend, sequence length distribution within
the batch all matter — so re-fit when any of those change.

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
# hit the server during this run.
python tools/admission_control/fit_cost_model.py tbt \
    --server http://localhost:31000 \
    --output ms_dev/runtime/cost_models/tbt_llama3-70b.json \
    --batch-sizes 1 4 8 16 32 \
    --prompt-lens 1024 4096 16384 \
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

Takes a JSONL decision log (from `--admission-dry-run` mode) and reports what *would*
have been admitted/rejected under one or more candidate SLO settings.

Use this to tune SLOs against real traffic before turning admission control on for real.

### Inputs

- `--decision-log <path>` — JSONL file produced by the controller in dry-run mode.
  Each line: `{ts, rid, predicted_ttft, predicted_tbt, queue_predicted, ...}`.
  Path defaults to `<session>/admission_decisions.jsonl` if `--session-dir` is given.
- `--ttft-slo <ms>` (repeatable) — candidate TTFT SLO(s).
- `--tbt-slo <ms>` (repeatable) — candidate TBT SLO(s).

### Outputs

stdout table + optional CSV:

```
TTFT_SLO   TBT_SLO   total   admit   rej_TTFT   rej_TBT_pred   rej_TBT_react   reject_pct
30000      200       12031   11420   411        180            20              5.1%
30000      150       12031   10980   411        585            55              8.7%
20000      200       12031   10885   946        180            20              9.5%
...
```

CSV format identical (`--csv-out path`).

### Usage

```bash
python tools/admission_control/replay_admission.py \
    --session-dir ms_dev/runtime/sessions/20260507_180000 \
    --ttft-slo 20000 30000 60000 \
    --tbt-slo 100 150 200 300 \
    --csv-out reports/slo_sweep.csv
```

The session dir is auto-discovered from `meta/run_meta.json` if a path is given;
otherwise pass `--decision-log` directly.

### Caveat

Replay is a **counterfactual** — it assumes the predictor's outputs at request arrival
were correct. It does not simulate feedback: e.g., a real reject would have changed the
queue state for subsequent requests. Treat results as upper bounds on reject rate.

---

## File layout

```
tools/admission_control/
├── CLAUDE.md
├── fit_cost_model.py
└── replay_admission.py
```

No package init needed — both are CLI entrypoints.
