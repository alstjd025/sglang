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

For each (prompt_len, prefix_ratio) combination:

1. Build a long enough "warm-up" prompt of `prompt_len * prefix_ratio` tokens, send a
   throwaway request to populate the radix cache.
2. Build a "measure" prompt = warm-up tokens + fresh suffix to total `prompt_len`.
3. POST to `/generate` with `max_new_tokens=1`. Use streaming response to extract the
   timestamp of the first generated token; subtract send time = TTFT (≈ T_prefill when
   queue is empty).
4. Repeat ≥3 times per cell, take the median.

Run `numpy.polyfit(d, t, deg=2)` over collected `(d, t)` samples.

### Method (TBT)

For each (target_batch_size, kv_total) combination:

1. Submit `target_batch_size` long-running requests with `max_new_tokens` large enough
   to span the measurement window, so the running batch size stabilizes.
2. Wait for the running batch to reach the target (poll `/get_server_info`).
3. Sample TBT for ≥30 decode steps (extract from request streaming intervals or from
   `step_time_dict` via internal-state endpoint).
4. Median over samples.

Linear fit: `TBT = a + b·bs + c·kv` via `numpy.linalg.lstsq`.

### Usage

```bash
# Prefill cost model
python tools/admission_control/fit_cost_model.py \
    --server http://localhost:31000 \
    --target prefill \
    --prompt-lens 128 512 1024 2048 4096 8192 16384 \
    --prefix-ratios 0.0 0.5 0.9 \
    --repeats 3 \
    --output ms_dev/runtime/cost_models/prefill_llama3-70b.json

# TBT cost model (do this on a separate run; needs no concurrent traffic)
python tools/admission_control/fit_cost_model.py \
    --server http://localhost:31000 \
    --target tbt \
    --batch-sizes 1 4 8 16 32 \
    --kv-totals 1024 4096 16384 65536 \
    --warmup-steps 10 --measure-steps 30 \
    --output ms_dev/runtime/cost_models/tbt_llama3-70b.json
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
