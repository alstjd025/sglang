# Project Halo — status & post-migration plan

A handoff doc: what is **done**, and what to **do next** — written so work
can resume cleanly after the GPU-instance migration (B200×4 → B200×2).

- Branch: `project_halo_request_level` (pushed to `origin`).
- Job-level recovery point: commit `fc2d2d877` (tag `halo-job-level-final`,
  local — push with `git push origin halo-job-level-final`).
- Authoritative design docs: `admission_control.md`, `halo_api_reference.md`,
  `prediction_model.md`, `CLAUDE.md` (this directory) + the in-code
  `python/sglang/srt/managers/halo/CLAUDE.md`.

---

## ✅ Done

Project Halo was refactored from **job-level** to **request-level** admission
control + tracking. The request-level system works and the repo is green.

### Phase 0 — branch + archive
- Tagged the job-level codebase `halo-job-level-final`; branched
  `project_halo_request_level`.
- Job-level design docs → `ms_dev/halo_dev/legacy_document/`.

### Phase 1 — request-level structural refactor
- `managers/admission_control/` moved under `managers/halo/admission_control/`.
- New `request_tracker/` — `RequestTracker`, the single per-request state
  store (one `RequestRecord` per request; sole writer).
- New policy layer — `HaloAdmissionGate` (KV-cache hard cap + policy),
  `AdmissionPolicy` ABC, `MooncakePolicy`, `VssPolicy`
  (`policy_reactive.py` is a Phase-3 stub).
- `HaloController` rewritten request-level; `job.py` / `job_registry.py` /
  `slowdown_tracker.py` deleted; scheduler rewired to a single gate.
- Request body fields: `halo_job_id` / `halo_slo` / `halo_job_done` →
  per-request `halo_ttft_slo` / `halo_tbt_slo` / `halo_e2e_slo`.
- `POST /halo/programs` + `halo_job_id` removed — admission is fully
  per-request. CLI flags: `--halo-admission-policy {off,mooncake,vss,reactive}`,
  `--halo-slo-mode {ratio,absolute}`, etc.

### Phase 2 — first-token accuracy
- `RequestTracker.first_token_ts` is stamped from the scheduler's accurate
  `SchedulerReqTimeStats.prefill_finished_time` (not the ±100 ms tick), so
  TTFT is exact even for sub-tick requests. Decode progress stays tick-driven.

### Phase 4 — docs + cleanup
- `vss_predictor.py` trimmed to the request-level kernel (~400 lines of dead
  job-level code removed).
- Code-based docs written (this directory); job-level `CLAUDE.md` archived.
- ms_dev tooling (`env.common.sh`, `lib_server.sh`, `experiments/halo_*.sh`,
  `expctl/server_run_experiment.py`, `expctl/monitoring_view.py`) updated to
  the request-level `--halo-*` flags + metric names.

### Tests
All green: halo phase1 23, predictors 18, step-cost 33, sampler 13, fit 9,
admission 72. `test/registered/halo/`.

---

## ⏭️ To do — after the instance migration

### A. Migration checklist (B200×4 → B200×2)

These are **invalidated by the hardware change** — regenerate, do not copy:

- [ ] **TP size** — `--tp 4` → `--tp 2` (set in the server-launch path).
- [ ] **Cost models refit** — prefill / TBT / Halo step cost model are fit
      per-`(GPU type × TP size)`; the B200×4 JSONs do not transfer. Refit on
      B200×2 (`tools/halo/fit_halo_cost_model.py`). `halo_base.sh` auto-picks
      the filename by HW tag (`b200x2`).
- [ ] **Baseline re-measure** — the `baseline_20260424…` poisson-sweep
      baseline is 4-GPU; not comparable to 2-GPU runs. Run a fresh baseline.
- [ ] **KV-capacity regime** — aggregate KV roughly ⅓ of before → recheck
      `max_running_requests`, `chunked_prefill_size`, `mem_fraction_static`,
      and the meaning of `--halo-admission-kv-cap-ratio` (same ratio, very
      different absolute token count).
- [ ] **λ sweep range** — server capacity ~halved → the meaningful λ regime
      shifts down; re-pick the sweep points.
- [ ] **`--sglang-ssh-host`** — `NXC7` → the new instance hostname (in the
      Agent_applications `run_experiment.py` client + its CLAUDE.md default).
- [ ] Model weights re-download; `.venv` re-create (install script).

Already done (no action needed): the ms_dev env-var wiring is already on the
request-level `--halo-*` flags.

### B. Phase 3 — the reactive admission policy (the next dev work)

The research contribution: a **measured, queueing-inclusive** admission
policy — `policy_reactive.py` (currently a stub). Build it *after* the
migration (it needs refit cost models + experiments to validate).

**Why it is needed** (proven by the VSS_v2 experiment, 2026-05-18):
contention-only signals (VSS) are *structurally queueing-blind*. A per-step
contention ratio saturates (~3.5) well below the SLO while the real
slowdown (measured VJS ≈ 25) is dominated by **queueing** — a request
sitting in the waiting queue making zero progress. A step-snapshot cannot
see queueing; it is an emergent property of arrival-vs-service over time.

**Design direction** (from the design discussion; candidate, not final):
- **Decompose** slowdown ≈ `(queueing_delay + interference) / solo`, with
  KV-cache occupancy as a *state variable* driving the queueing dynamics
  (preemption / retraction → re-queueing). The three components live on
  different time scales (interference: ms; KV: seconds; queueing: tens of s).
- The proven working queueing signal already exists in the codebase: the
  mooncake `t_queue` term — `Σ predicted_prefill_ms over waiting_queue`,
  the waiting-queue backlog. (Verified from the `admission_ratio_tp_tau5`
  decision logs: 100 % of mooncake's rejects came from the TTFT/`t_queue`
  stage; the TBT/contention stage never fired.)
- **Measure queueing as absolute delay time**, not a ratio — a queued
  request's wait is always a finite, observed number of seconds (no 0/0,
  no ∞). Candidate signals: Little's-law predicted wait `Q / drain_rate`
  (predictive), or measured waiting-queue residence.
- **Smooth + damp**: EWMA (low-pass, noise) + a deadband / hysteresis
  (two thresholds — anti-oscillation, the open-loop instability concern).
- **Two-step build**: a reactive feedback gate first; upgrade to an
  MPC-lite forward-simulation feasibility gate only if the reactive gate
  oscillates or reacts too late.
- Reuse: the VSS batch-slowdown kernel in `vss_predictor.py`
  (`_augmented_step_times`, chunk-bounded EXTEND) for the *interference*
  term; the request tracker's measured TTFT/TBT/e2e for evaluation.
- The reference baseline to beat = `MooncakePolicy` (the mooncake gate).

Selected by `--halo-admission-policy reactive`. `_build_policy` in
`controller.py` already has the `reactive` branch (currently logs "not
implemented" and disables).

### C. Deferred tidy-ups (no functional impact)

- **D19 — single gate.** The standalone `admission_control` Mooncake gate
  (`scheduler._abort_on_predicted_slo_violation`, `--admission-*` flags) is
  still present but dormant by default. Mooncake also runs as a Halo policy
  now; unifying the two call sites is pending.
- **Known limitations** (see `python/sglang/srt/managers/halo/CLAUDE.md`):
  tick-granular decode tracking (TBT-mean ±tick noise); the solo-e2e
  baseline is a mean-KV-span approximation — the **solo-run predictor is
  slated for a dedicated rework** (relevant to the `halo_e2e_slo` SLO,
  whose admission-time prediction needs an output-length estimate).

---

## Where to look

| Want | File |
|---|---|
| Module internals, scheduler touchpoints | `python/sglang/srt/managers/halo/CLAUDE.md` |
| Gate + policies design | `ms_dev/halo_dev/admission_control.md` |
| Client API, flags | `ms_dev/halo_dev/halo_api_reference.md` |
| Cost models, fitting | `ms_dev/halo_dev/prediction_model.md` |
| Project overview | `ms_dev/halo_dev/CLAUDE.md` |
| Job-level history | `ms_dev/halo_dev/legacy_document/` + tag `halo-job-level-final` |
