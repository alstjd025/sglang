# Halo Phase 1 Tests

Pure-Python unit tests for the `managers/halo/` module. No GPU, no model load —
runs in the `stage-a-test-cpu` suite. Covers:

| Area | Coverage |
|---|---|
| `Job` dataclass | Lifecycle (admitted → completed), initial VJS = SLO, record_vjs |
| `JobRegistry` | rid↔job mapping, lazy creation, completion bookkeeping, gc_completed |
| `SlowdownTracker` | compute_job_vjs stage-merge VJS, cost-model primitives, sweep, no-cost-model fallback |
| `HaloController` | enable/disable, register_request strict mode (raises on missing job_id), tick wall-clock gate, snapshot |
| `build_halo_controller_from_server_args` | factory returns None when off |

What is **not** covered here (Phase 2 / integration):
- Live HTTP request → scheduler → Halo flow (requires running server)
- Multi-TP rank dedup (single-process test)
- Cost-model fitting (admission_control's existing test exercises that)

Run:

```bash
python3 test/registered/halo/test_halo_phase1.py
```
