# tools/halo — Halo Step Cost Model: data collection, fit, and use

Offline tooling for the **Halo Step Cost Model** (post-Phase-1 follow-up to
the §18 TBT cost-model cliff).

> Full design + rationale: [`ms_dev/halo_dev/prediction_model.md`](../../ms_dev/halo_dev/prediction_model.md)
> · Decision summary: `ms_dev/halo_dev/CLAUDE.md` §20

## Files

| File | Purpose |
|---|---|
| `fit_halo_cost_model.py` | OLS fit of 6 coefficients from per-step JSONL samples → `halo_step_v1` JSON consumed by `--halo-step-cost-model-path` |

## End-to-end workflow

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────────┐
│ 1. Server with  │ →  │ 2. Drive any │ →  │ 3. Fit OLS  │ →  │ 4. Server with │
│    --halo-cost- │    │    workload  │    │             │    │    --halo-step │
│    model-sample-│    │ (Halo on/off │    │ tools/halo/ │    │    -cost-model │
│    log <path>   │    │  doesn't     │    │ fit_halo_   │    │    -path       │
│    + overlap off│    │  matter)     │    │ cost_model  │    │                │
└─────────────────┘    └──────────────┘    └─────────────┘    └────────────────┘
       (collect)         (load model)      (offline OLS)        (use new model)
```

---

## 1. Collect per-step samples

Set **one** env var (`SGLANG_HALO_COST_MODEL_SAMPLE_LOG`) to enable
collection; everything else is automatic.

### Terminal 1 — Server orchestrator (`expctl/server_run_experiment.py`)

```bash
# "auto" → file is auto-routed to <session>/halo_cost_samples.jsonl.
# Use an absolute path instead if you need to pin the location.
SGLANG_HALO_COST_MODEL_SAMPLE_LOG=auto \
python3 ms_dev/expctl/server_run_experiment.py --mode single --session-name halo_cost_fit
```

What this does automatically:
- creates `ms_dev/runtime/sessions/<ts>_halo_cost_fit/`
- routes the sampler JSONL to `<session>/halo_cost_samples.jsonl`
- appends `--disable-overlap-schedule` to the server command line
  (overlap mode would invalidate the step-latency measurement; the sampler
  refuses to run there)
- forwards `SGLANG_HALO_*` env vars to the server process
- snapshots all `SGLANG_HALO_*` values into
  `<session>/meta/run_meta.json::halo_config`

Other env vars you can set (all optional):

| Env var | Effect |
|---|---|
| `SGLANG_HALO_COST_MODEL_SAMPLE_LOG=<abs path>` | pin the output JSONL outside the session dir |
| `SGLANG_HALO_COST_MODEL_SAMPLE_EVERY=<N>` | subsample: write every Nth step. Default 1. Bump up if hot-path queue pressure shows in WARNs or disk fills too fast. |
| `SGLANG_HALO_ENABLED=1` | also run the Halo controller. Not needed for fit-data collection — the sampler is fully independent. |

You don't need to set anything else — `expctl/server_run_experiment.py` already
launches the single server, scrapes Prometheus, renders the live panel, and
saves the session folder.

### Terminal 2 — Drive a workload

The fit-data collection is workload-agnostic. Any workload that produces
mixed batch compositions is fine; `MIXED` steps come from continuous
batching naturally when concurrency is non-trivial.

Typical command from `Agent_applications/agent_motivation_experiment`
(matches the host's `project-halo-phase1-client` branch):

```bash
cd Agent_applications/agent_motivation_experiment

# Example: parallel tool-delay sweep, λ=0.3, 10 minutes
python run_experiment.py \
    --workload swe_bench_coding_parallel_tool_delay \
    --mode poisson-sweep \
    --baseline-dir results/baseline_20260424-180204 \
    --lambda-list 0.3 \
    --duration-min 10 \
    --halo-enabled --tau 5.0 \
    --session-name slowdown_test_v3
```

`--halo-enabled` on the client side is **optional** for fit-data collection:

- the **sampler is fully independent** of the Halo controller — it records
  per-step features whether or not Halo runs;
- keeping `--halo-enabled` on means the server will still compute slowdown
  ratios using the **legacy** cost model and write `halo_jobs.jsonl`, which
  is useful as a "before" baseline to compare against the refit run later.
  No harm to the fit data either way.

Things to vary if the recovered RMSE/R² look weak:

- **Concurrency**: a fixed λ that's high enough to produce decode batches
  of bs ≥ 4 most of the time but not so high that the queue saturates.
- **Prompt-length mix**: bigger spread of `n_i` gives the prefill terms
  better identifiability.
- **Cache-hit ratio**: agent-style workloads (parallel_tool_delay) have
  ~90 % cache hit, which is what we want to fit for. Avoid pure synthetic
  benchmarks for production-targeted fits.

A 10-minute run at ~5 jobs/min generates several hundred thousand step
samples — far more than needed (a few thousand are enough for OLS).

### Where the file lands

After the session ends, look for:

```
ms_dev/runtime/sessions/<ts>_halo_cost_fit/halo_cost_samples.jsonl
ms_dev/runtime/sessions/<ts>_halo_cost_fit/meta/run_meta.json   # halo_config block
```

## 2. Fit the model

The default form is **split** (since 2026-05-14). It fits prefill (EXTEND step)
and decode (DECODE step) separately with their own intercepts (θ_c_p, θ_c_d),
mirroring MuxWise's original form. Validation showed it produces the best-
balanced tail/max accuracy on this host without the cliff or the deflation
seen in unified — see `ms_dev/halo_dev/prediction_model.md` §17.

```bash
.venv/bin/python tools/halo/fit_halo_cost_model.py \
    --samples $SESSION/halo_cost_samples.jsonl \
    --output  ms_dev/runtime/cost_models/halo_step_split_llama3-70b_b200x4.json \
    --model   "Llama-3.3-70B-Instruct" \
    --hw      "B200x4 TP=4"
```

To opt into the **unified** form (single shared θ_c, joint fit over all step
types — preferred when MIXED steps appear in the sample log, e.g. when
`--enable-mixed-chunk` is on):

```bash
.venv/bin/python tools/halo/fit_halo_cost_model.py \
    --samples $SESSION/halo_cost_samples.jsonl \
    --output  ms_dev/runtime/cost_models/halo_step_llama3-70b_b200x4.json \
    --model   "Llama-3.3-70B-Instruct" \
    --hw      "B200x4 TP=4" \
    --form unified
```

To verify which step types appear:
```bash
jq -r .forward_mode $SESSION/halo_cost_samples.jsonl | sort -u
```
If only `EXTEND` and `DECODE` show up → stay with the default split form.

The script reports:
- **n_samples** (after filters)
- **RMSE_ms** — residual error
- **R²** — fraction of variance explained
- **Coefficients** (θ_p1 … θ_c)

Sanity checks before trusting the JSON:
- R² > 0.9 (preferably > 0.95)
- All θ coefficients finite, no NaNs
- θ_d1 > 0 (KV-summed term should be positive — that was the headline fix)

Useful filters when diagnosing fit subsets:

| Flag | Use |
|---|---|
| `--filter-forward-mode DECODE` | decode-only fit (verify θ_d1/θ_d2/θ_c) |
| `--filter-forward-mode EXTEND,MIXED` | prefill-containing steps (verify θ_p* identifiability) |
| `--min-step-time-ms 1.0` | drop noisy near-zero rows |
| `--max-step-time-ms 1000.0` | drop outliers (e.g., preemption events) |
| `--min-samples 200` | refuse to fit too-small datasets (default) |

## 3. Use the fitted model

The easy path: source `ms_dev/experiments/halo_base.sh` — it sets
`SGLANG_HALO_STEP_COST_MODEL` to the split JSON path by default. This is
the recommended way to launch Halo on this host:

```bash
source ms_dev/experiments/halo_base.sh
python3 ms_dev/expctl/server_run_experiment.py --mode single --session-name <name>
```

If running outside that wrapper, set the env var directly:

```bash
SGLANG_HALO_ENABLED=1 \
SGLANG_HALO_STEP_COST_MODEL=ms_dev/runtime/cost_models/halo_step_split_llama3-70b_b200x4.json \
python3 ms_dev/expctl/server_run_experiment.py --mode single --session-name halo_step_verify
```

When `SGLANG_HALO_STEP_COST_MODEL` is set, the server **supersedes** the
legacy `SGLANG_HALO_PREFILL_COST_MODEL` / `SGLANG_HALO_TBT_COST_MODEL` pair.
Setting all three logs an INFO line and ignores the legacy pair. The server
auto-detects whether the JSON is `halo_step_v1` (unified) or
`halo_step_split_v1` (split) from the `form` field — no extra config needed.
To opt back into legacy, explicitly set `SGLANG_HALO_STEP_COST_MODEL=""`
before sourcing the wrapper.

The validation experiment is just a normal Halo run: drive the same workload
with `--halo-enabled --halo-slo 5` and compare the `halo_jobs.jsonl`
slowdown distribution to the pre-fit one. The §18 cliff symptom is a tight
band of ratios; the fix should produce a spread that varies with load.

---

## Form / coefficients

See `python/sglang/srt/managers/admission_control/cost_model.py` —
`HaloStepCostModel`:

```
T_step_ms ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ
          + θ_d1·Σrⱼ + θ_d2·bs_d + θ_c
```

Two convenience shortcuts for the Halo R1 slowdown denominator (one-request
batch):

- `estimate_solo_prefill_total_ms(n, r)` — 1 prefill request, full pass
- `estimate_solo_tbt_ms(r)` — bs=1 decode against context length `r`

## Tests

All in `stage-a-test-cpu` (no GPU needed):

```bash
.venv/bin/python test/registered/halo/test_halo_step_cost_model.py
.venv/bin/python test/registered/halo/test_halo_cost_model_sampler.py
.venv/bin/python test/registered/halo/test_fit_halo_cost_model.py
```

Coverage:
- `test_halo_step_cost_model.py` — `HaloStepCostModel` arithmetic, JSON
  roundtrip, loader error paths (22 tests).
- `test_halo_cost_model_sampler.py` — feature extraction, subsampling,
  background flush, queue-full drop counter (13 tests).
- `test_fit_halo_cost_model.py` — synthesize with known θ, fit, verify OLS
  recovery (noiseless: exact; with 1 ms noise: RMSE < 2 ms, R² > 0.99)
  (7 tests).
