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
# Source this file before running ms_dev/expctl/run_experiment.py.
# Usage:   source ms_dev/experiments/<name>.sh && python3 ms_dev/expctl/run_experiment.py --mode single

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
| `admission_dryrun.sh` | Same as `_with_safety.sh` but with dry-run on — for SLO tuning against real traffic. |

After running an experiment session, the resulting session folder contains
`meta/run_meta.json` with an `admission_config` block recording which knobs
were active. Diff `run_meta.json` across runs to compare configurations.
