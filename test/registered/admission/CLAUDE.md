# Admission Control Tests

CI-registered tests for the admission control module
(`python/sglang/srt/managers/admission_control/`).

## Files

| File | What it tests |
|---|---|
| `test_admission_control.py` | Unit tests for `PrefillCostModel`, `TBTCostModel`, `TBTEwmaTracker`, `AdmissionController.decide()` decision matrix; integration test that boots a real single server with admission flags and verifies HTTP 429 on synthetic SLO-violating traffic. |

## CI registration

Per [test/README.md](../../README.md), every file under `test/registered/` must call a
`register_*_ci` function at module level with **literal** `est_time` and `suite`. AST-parsed,
so no f-strings or variables.

Use `stage-b-test-1-gpu-small` — the integration portion needs a GPU but only briefly,
and the unit portion is CPU-only:

```python
from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")
```

## Running locally

```bash
# Single file
python3 test/registered/admission/test_admission_control.py

# Single test method
python3 test/registered/admission/test_admission_control.py \
    TestAdmissionController.test_ttft_predicted_reject

# Whole suite (matches CI)
python3 test/run_suite.py --hw cuda --suite stage-b-test-1-gpu-small
```

CI appends `-f` (failfast). Test files MUST end with the canonical
`if __name__ == "__main__": unittest.main()` and must not mutate `sys.argv` — see
[test/README.md](../../README.md).

## Test matrix

Unit tests (no GPU, no server):

- **PrefillCostModel**: load/save round-trip, estimate matches `α·d² + β·d + γ` exactly,
  graceful fallback on missing/malformed JSON.
- **TBTCostModel**: same shape for `a + b·bs + c·per_req_kv`.
- **TBTEwmaTracker**: warm-up flag turns true after N updates; `get()` returns smoothed
  value; cold-start returns 0 and `is_warm()` is False.
- **AdmissionController.decide()** decision matrix:
  - All disabled → ADMIT
  - TTFT only configured, queue empty, prediction under SLO → ADMIT (records `_predicted_prefill_ms`)
  - TTFT only, queue causes prediction over SLO → REJECT(reason=TTFT_PREDICTED)
  - TBT only, predicted over SLO → REJECT(reason=TBT_PREDICTED)
  - TBT only, predicted under, EWMA over `slo*ratio` → REJECT(reason=TBT_REACTIVE)
  - Cold EWMA: no false TBT_REACTIVE rejects.
  - Dry-run flag set: same decision returned but `admit=True` always; metric/log path
    distinguishes via `dry_run_would_reject` flag.
  - `disaggregation_mode != NULL` → ADMIT with reason="disabled" + warns once.

Integration test (GPU, single server):

- Boot server with `--admission-ttft-slo-ms 1`, fixed cost model, dry-run OFF.
- POST a single non-trivial generate request → expect HTTP 429 with body containing
  reason `TTFT_PREDICTED`.
- Boot with `--admission-dry-run` and same SLO → expect HTTP 200 (admitted).
- Verify `/get_server_info` exposes `admission_state` block.

## Adding new tests

Use SGLang's `CustomTestCase` pattern (see `write-sglang-test` skill). Mock cost models
via small in-line JSON to keep unit tests pure. For integration, use the smallest viable
model (`Qwen/Qwen2.5-0.5B` or similar) — admission logic is model-agnostic.
