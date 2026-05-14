# `ms_dev/halo_dev/verify/` — Halo functional verification

Standalone scripts that exercise the Halo subsystem against the four
validation questions raised in the Phase 1 review (see
`ms_dev/halo_dev/CLAUDE.md`):

1. **slowdown accuracy** — does Halo's per-job slowdown track reality
   (vs. cost-model baseline and solo-run measurement)?
2. **multi-call jobs** — does `slowdown_max` / `slowdown_mean` aggregate
   correctly when a job has multiple in-flight requests?
3. **register/admit overhead** — is the hot-path cost of `register_program`
   and `admit_to_job` negligible?
4. **sweep overhead** — does the periodic sweep stay well below the tick
   interval so it can't perturb scheduling / admission?

The scripts are split by what they need:

| Script | Needs | Validates |
|---|---|---|
| `microbench_overhead.py` | Python only (no sglang server) | 3 + 4 |
| `e2e_smoke.sh` | running sglang server (`--halo-enabled`) | 2 + 3 (E2E) |
| (future) `verify_slowdown_vs_actual.py` | finished Agent_applications + sglang session | 1 |

## Run

### Offline microbench (no server)

```bash
.venv/bin/python3 ms_dev/halo_dev/verify/microbench_overhead.py
```

Reports register / admit hot-path mean/p99 latency (in microseconds) and
sweep cost at several `(num_jobs × reqs_per_job)` matrix points. The
"safe margin" rule we use: sweep < tick_interval × 1% (default 100ms
tick → sweep should stay well under 1 ms).

### E2E smoke (live server)

**Recommended — orchestrated runner with live monitor panel:**

```bash
# Terminal 1
source ms_dev/experiments/halo_base.sh
python3 ms_dev/expctl/server_run_experiment.py --mode single
# (metrics + request-logs are default-on for the launcher; --metrics is a
#  launcher OBS flag, not a run_experiment.py flag, so don't pass it here.)

# Terminal 2 (after "Server ready" appears in terminal 1)
bash ms_dev/halo_dev/verify/e2e_smoke.sh
```

While the smoke fires requests, the terminal 1 panel shows Halo state
live: a `halo=ON ...` feature cell and two rows of runtime counters
(`halo_active known registered admitted rejected` and
`mean_smax mean_smean worst_smax slo_violations`).

**Standalone (no monitor panel, plain server):**

```bash
source ms_dev/experiments/halo_base.sh
bash ms_dev/start_server_no_pd.sh
# in another terminal:
bash ms_dev/halo_dev/verify/e2e_smoke.sh
```

The smoke script itself sends a `POST /halo/programs`, fires N
`chat.completions` carrying `halo_job_id`/`halo_slo`, then queries
`/server_info` and prints the `halo_state` block so you can eyeball
`slowdown_max` / `slowdown_mean` and counters.

## Acceptance hints

- microbench: `register_program` and `admit_to_job` < 50 µs mean. Sweep
  with 100 jobs × 10 reqs each (= 1000 RequestExecutionInfo) < 1 ms mean.
  Anything beyond that is worth investigating.
- e2e_smoke: counts in `halo_state.jobs[]` match expected N; status panel
  shows `halo=ON`.
- slowdown_vs_actual (future): mean absolute relative error between
  Halo's `slowdown_max` and the realized job latency / baseline ratio.
  > 30 % typically points at the TBT cost-model cliff
  (`runtime/cost_models/README.md` item 3), not Halo itself.

## Reference numbers (2026-05-12, local run)

Captured on the same host SGLang is being developed against. Numbers
will vary with hardware, but should stay in the same order of magnitude.

### Hot-path overhead (verification #3, CPU only)

| Function | mean | p99 | Notes |
|---|---|---|---|
| `register_program` (fresh job per call) | **99 µs** | 189 µs | Job-once cost; insert + Job dataclass + log |
| `register_request` (admit to existing job) | **0.96 µs** | 1.04 µs | Hot path: every LLM call |
| `on_request_finished` | **0.45 µs** | 0.52 µs | Hot path: every finish |

Per-call Halo overhead ≈ **1.4 µs / LLM request**. LLM calls are tens to
hundreds of ms wall-clock → Halo's added latency is well under 0.001 %.

### Sweep overhead (verification #4)

| Load                          | mean    | % of 100 ms tick |
|-------------------------------|---------|------------------|
| 1 job × 1 req                 | 1.3 µs  | < 0.001 %        |
| 10 jobs × 10 reqs (100 reqs)  | 58 µs   | 0.06 %           |
| **100 jobs × 10 reqs (1000)** | **590 µs**  | **0.6 %**    |
| 1000 jobs × 1 req             | 1.0 ms  | 1.0 %            |
| 100 jobs × 50 reqs (5000)     | 2.8 ms  | 2.8 %            |

Verdict — sweep is single-threaded inside the scheduler event loop and
*can* perturb scheduling in principle. Empirically, up to ~1000 in-flight
requests it stays well within a 1 % budget at the 100 ms tick interval.
Higher concurrency (5000+) crosses the budget and would need either a
larger `--halo-tick-interval-ms` or sweep-side optimization (e.g. per-job
incremental accumulators instead of full re-aggregation).

### What this means for Phase 1 experiments

- Typical Agent_applications load (tens to a few hundred concurrent
  requests) → Halo overhead is in the noise. Safe to leave on.
- Pathological load (multi-thousand concurrent jobs) → tune
  `--halo-tick-interval-ms` upward; the wall-clock gate already only
  fires when due, so coarser tick = less work.

### Note on `register_program`'s 99 µs

It's job-once, so even 1000 jobs/sec only spends 100 ms on Halo
registration total. Not worth optimizing in Phase 1; if Phase 2 needs
it, candidates are: skip the JSONL log on rank-0 in batch-register mode,
or reuse a Job pool.
