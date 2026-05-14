# `ms_dev/experiments/` — Reproducible Experiment Wrappers

Each `.sh` here is a **named experiment configuration**. Source it (or `bash` it
in a wrapper) before launching `run_experiment.py` so the SGLANG_* env knobs
are set in a way that's easy to re-run, diff, and version-control.

## Why

`ms_dev/env.common.sh` is the upstream-tracked default. Editing it for an
experiment creates dirty git diffs and makes runs hard to compare. Instead:

- **One-off ad-hoc**: `export FOO=bar` in your shell, run.
- **Reproducible experiment**: write a small wrapper here, commit/share it.
- **Personal host-local default**: write `ms_dev/env.local.sh` (gitignored,
  auto-sourced by `env.sh`).

## Pattern

```bash
#!/usr/bin/env bash
# Source this file before running ms_dev/expctl/server_run_experiment.py.
# Usage:   source ms_dev/experiments/<name>.sh && python3 ms_dev/expctl/server_run_experiment.py --mode single

# ... export SGLANG_* env vars here ...
```

The launcher scripts (`start_server_no_pd.sh` etc.) inherit the env from
the calling shell, so anything `export`ed here flows all the way to
`sglang.launch_server` via `lib_server.sh`.

## Existing experiments

| File | Purpose |
|---|---|
| `admission_ratio_only.sh` | Admission control with ratio-based SLOs only (TTFT 2x, TBT 3x). No absolute caps. Stage 3 EWMA disabled by design. |
| `admission_ratio_with_safety.sh` | Recommended for real runs: ratio SLOs + loose absolute caps as catastrophic-prevention safety net (Stage 3 EWMA active). |
| `admission_absolute_only.sh` | Admission control with absolute ms SLOs only (TTFT 5s, TBT 200ms). All 3 stages active. |
| `admission_ratio_ttft_only.sh` | Hybrid: TTFT ratio + absolute TBT SLO with reactive EWMA. Use when TBT cost model is brittle. |
| `admission_dryrun.sh` | Same as `_with_safety.sh` but with dry-run on — for SLO tuning against real traffic. |
| `admission_off.sh` | Unsets every `SGLANG_ADMISSION_*` env var so the next run launches sglang with no admission control at all. Useful as a clean baseline or after sourcing a wrapper you want to undo. |
| `halo_base.sh` | **Project Halo Phase 1** — turns on job-level slowdown tracking (observation only, no admission/scheduling decisions). Each request MUST carry `halo_job_id` (HTTP 400 otherwise). Per-sweep snapshot logged to `<session>/halo_jobs.jsonl`. See `ms_dev/halo_dev/CLAUDE.md`. |
| `halo_off.sh` | Unsets every `SGLANG_HALO_*` env var so the next run launches sglang with no Halo tracking. |

After running an experiment session, the resulting session folder contains
`meta/run_meta.json` with `admission_config` and (when active) `halo_config`
blocks recording which knobs were active. Diff `run_meta.json` across runs
to compare configurations.
