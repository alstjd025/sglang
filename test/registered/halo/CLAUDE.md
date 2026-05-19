# Halo Tests

Pure-Python unit tests for the request-level `managers/halo/` module. No GPU,
no model load — runs in the `stage-a-test-cpu` suite.

`test_halo_phase1.py` covers:

| Area | Coverage |
|---|---|
| `RequestTracker` / `RequestRecord` | `on_admitted` / `on_step` / `on_finished` / `on_rejected`, first-token stamping, TBT mean, e2e + slowdown at finish, `snapshot` partitioning, solo primitives |
| `HaloAdmissionGate` | KV-cache hard cap (reject / disabled), None-policy admit |
| `MooncakePolicy` | admits without cost models, TTFT-ratio reject, absolute `slo_mode` |
| `VssPolicy` | admits empty batch, decide runs with a populated running batch |
| `HaloController` | off-by-default factory, register→tracker record, KV-cap reject raises `HaloRejectError`, dry-run, finish, tick |

Cost models are duck-typed fakes (`_FakePrefill` / `_FakeTbt` / `_FakeStep`)
so the tests stay model-file-free.

Other halo test files in this directory (`test_halo_step_cost_model.py`,
`test_halo_cost_model_sampler.py`, `test_fit_halo_cost_model.py`) cover the
cost-model machinery and are unaffected by the request-level refactor.

What is **not** covered here (integration):
- Live HTTP request → scheduler → Halo flow (requires a running server)
- Multi-TP rank dedup (single-process test)

Run:

```bash
python3 test/registered/halo/test_halo_phase1.py
```
