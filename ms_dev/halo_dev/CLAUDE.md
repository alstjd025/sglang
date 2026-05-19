# Project Halo — development overview

Project Halo is a **request-level admission-control + tracking** subsystem
added to SGLang's single-instance scheduler. "Halo" is the umbrella project
name, not a role. Off by default (`--halo-enabled`); when off, every
scheduler hook is a one-line null check.

> **History.** Halo began *job-level* (job-scoped slowdown tracking + job
> admission, `POST /halo/programs`, virtual job slowdown). It was refactored
> to **request-level** on 2026-05-19. The complete job-level codebase is
> preserved at git tag `halo-job-level-final`; its design docs +
> decision log are in `legacy_document/`.

## Components

```
managers/halo/
├── request_tracker/    — RequestTracker: the single per-request state store
└── admission_control/  — the policy-pluggable admission gate
                          (policies: off / mooncake / vss / reactive)
```

A request arrives → the scheduler runs the **admission gate** (KV-cache
hard cap + the selected policy) → admitted requests are tracked by the
**request tracker** (per-request TTFT / TBT / e2e / slowdown). The
`HaloController` (owned by the scheduler) wires the two together.

## Documentation map

| Doc | Scope |
|---|---|
| [`admission_control.md`](admission_control.md) | Admission gate + the mooncake / vss / reactive policies |
| [`halo_api_reference.md`](halo_api_reference.md) | Client-facing API — request body SLO fields, `--halo-*` flags, `GET /halo/status` |
| [`prediction_model.md`](prediction_model.md) | Cost models (prefill / TBT / step) + the fit workflow |
| `python/sglang/srt/managers/halo/CLAUDE.md` | Module internals — controller surface, scheduler touchpoints, metrics, limitations |
| `python/sglang/srt/managers/halo/admission_control/CLAUDE.md` | Admission-control module internals |

The **code is authoritative**; these docs are written from it.

## Phase status

| Phase | State |
|---|---|
| Phase 1 — request-level structural refactor (request_tracker, policy gate, controller, scheduler rewire, io_struct/flags) | ✅ done |
| Phase 2 — accurate first-token timestamp (from `SchedulerReqTimeStats.prefill_finished_time`) | ✅ done |
| Phase 3 — new **reactive** admission policy (measured, queueing-inclusive: queueing-delay decomposition + EWMA + deadband) | ⏳ pending — after the GPU-instance migration |

## Open items

- **Phase 3 reactive policy** — the request-level research contribution.
  A measured, queueing-inclusive gate; `policy_reactive.py` is a stub today.
  Deferred until after the B200×4 → B200×2 instance migration (cost models
  must be refit per `(GPU type × TP size)` — see `prediction_model.md`).
- **D19 — single gate.** The standalone `admission_control` Mooncake gate
  (`scheduler._abort_on_predicted_slo_violation`, the `--admission-*` flags)
  is still present but dormant by default. Mooncake now also exists as a
  Halo policy (`--halo-admission-policy mooncake`); unifying the two is a
  pending tidy-up — no functional impact.
- **Known limitations** — tick-granular decode tracking, the approximate
  solo-e2e baseline: see the module `CLAUDE.md` "Known limitations".

## Experiment tooling

`ms_dev/experiments/halo_base.sh` turns Halo on; set
`SGLANG_HALO_ADMISSION_POLICY` to enable a policy. Flags / env vars: see
`halo_api_reference.md` and `ms_dev/CLAUDE.md`.
