# Legacy documents — Halo job-level design

These documents describe the **job-level** Halo subsystem (job-scoped slowdown
tracking + job-level admission control). They were archived here on
**2026-05-19** when the project pivoted to **request-level** admission control
and tracking.

They are kept for reference only and are **not maintained**. The request-level
design supersedes them.

## What moved here

| File | Was |
|---|---|
| `admission_design.md` | Phase 2 job-level admission design (VSS, KV cap, VJS) |
| `halo_api_reference.md` | Job-level client API (`POST /halo/programs`, `halo_job_id`) |
| `prediction_model.md` | Job-lifetime VJS + Halo step cost model |

## Recovery point

The complete job-level codebase + docs at their final state are preserved at
git tag **`halo-job-level-final`** (commit `fc2d2d877`). Check that tag out to
restore the job-level system in full.

## Current docs

Request-level docs are written fresh, code-based, under `ms_dev/halo_dev/`
(`admission_control.md`, `halo_api_reference.md`, `prediction_model.md`).
