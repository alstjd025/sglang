# Halo API Reference (client-facing)

Everything Halo adds to SGLang's HTTP / OpenAI-compatible API and CLI.
Targeted at **client integrators** (Agent_applications, custom load
drivers, ad-hoc curl scripts).

For internal design / phase plan / decision log, see:
- `ms_dev/halo_dev/CLAUDE.md` — project-wide design + decision history
- `python/sglang/srt/managers/halo/CLAUDE.md` — module internals

> All identifiers below appear in code as `halo_*` so they don't collide
> with admission_control's `admission_*` namespace.

## Activation (server-side)

Halo is off by default. To turn on, launch sglang with `--halo-enabled`
plus at least the SLO-related cost model paths so slowdown math has
something to chew on. The pre-baked wrapper `ms_dev/experiments/halo_base.sh`
sets sensible defaults:

```bash
source ms_dev/experiments/halo_base.sh
python3 ms_dev/expctl/server_run_experiment.py --mode single
```

### CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--halo-enabled` | `False` | Master on/off. Off → controller is None, all hooks are null-checks |
| `--halo-default-slo` | `5.0` | Default slowdown SLO used when a request omits `halo_slo` |
| `--halo-tick-interval-ms` | `100` | Min wall-clock interval between sweeps |
| `--halo-aggregator` | `max+mean` | Reserved; Phase 1 always tracks both |
| `--halo-job-log` | auto-routed | Per-sweep JSONL snapshot path. `run_experiment.py` auto-sets it to `<session>/halo_jobs.jsonl` |
| `--halo-prefill-cost-model-path` | unset | Same JSON schema as admission_control's prefill cost model |
| `--halo-tbt-cost-model-path` | unset | Same for TBT |
| `--halo-program-idle-timeout-seconds` | `300` | Pre-registered programs that never receive a request get GC'd after this many seconds. `0` disables idle GC |
| `--halo-job-log-interval-seconds` | `10` | How often (seconds) `halo_jobs.jsonl` gets a full active-jobs snapshot. Slowdown sweep + Prometheus gauges still tick at `--halo-tick-interval-ms`; only the verbose JSONL log is throttled. `0` reverts to per-sweep logging. Note: `event=register_program` and `event=job_complete` rows are emitted immediately regardless of this interval |
| `--halo-quiescent-timeout-seconds` | `300` | Safety net for jobs the client never closes with `halo_job_done=true`. A job with zero in-flight requests and no admit/finish activity for this many seconds is force-flipped to COMPLETE (then GC'd after retain_seconds). Tune up for workloads with legitimately long mid-chain waits (human-in-the-loop, external API). `0` disables |

### Env-var equivalents (translated to flags by `lib_server.sh::append_halo_args`)

`SGLANG_HALO_ENABLED`, `SGLANG_HALO_DEFAULT_SLO`,
`SGLANG_HALO_TICK_INTERVAL_MS`, `SGLANG_HALO_AGGREGATOR`,
`SGLANG_HALO_JOB_LOG`, `SGLANG_HALO_PREFILL_COST_MODEL`,
`SGLANG_HALO_TBT_COST_MODEL`. Set them in `env.local.sh` or a wrapper
under `ms_dev/experiments/`.

## Endpoints

### `GET /halo/status` — server-side Halo configuration probe

Lightweight, scheduler-free. Lets a client check whether the server has
`--halo-enabled` on, plus the relevant defaults, before issuing LLM
traffic. A client that wants Halo on should call this at startup and
abort if `enabled` is false (avoids `HALO_NO_JOB_ID` rejecting every
request later).

**No request body.** Always returns 200:

```json
{
  "enabled": true,
  "default_slo": 5.0,
  "tick_interval_ms": 100.0,
  "program_idle_timeout_seconds": 300.0,
  "cost_models": {
    "prefill_path": "/path/to/prefill.json",
    "tbt_path": "/path/to/tbt.json"
  }
}
```

**Curl:**

```bash
curl http://127.0.0.1:31000/halo/status
```

### `POST /halo/programs` — pre-register a job (Option A)

Required before the job's first LLM call when Halo is enabled. Carries
optional future-state info (chain length, expected stages, DAG) that
Phase 2 admission/scheduling will use.

**Request body** (application/json, ≤ 16 KiB total):

```json
{
  "job_id": "agent-42",                          // required, str (non-empty)
  "slo": 5.0,                                    // required, float > 0
  "total_calls": 12,                             // optional, int
  "stage_sequence": ["UNDERSTAND", "LOCATE", "PLAN", "..."],  // optional
  "expected_input_lens": [200, 350, 410, "..."], // optional, list[int]
  "expected_output_lens": [80, 120, 90, "..."],  // optional, list[int]
  "dag": { "type": "linear" }                    // optional, JSON-able
}
```

**Responses:**

| Status | When | Body shape |
|---|---|---|
| `200` | Fresh registration | `{"registered": true, "job_id": "...", "active_jobs": N}` |
| `409` | Same `job_id` already registered | `{"registered": false, "reason": "JOB_ID_ALREADY_REGISTERED", "existing": {...}}` |
| `400 BAD_JSON` | Body isn't JSON | `{"registered": false, "reason": "BAD_JSON", "detail": "..."}` |
| `400 BAD_FIELDS` | Unknown / wrongly-typed fields | `{"registered": false, "reason": "BAD_FIELDS", "detail": "..."}` |
| `400 MISSING_FIELDS` | `job_id` empty or `slo` ≤ 0 | `{"registered": false, "reason": "MISSING_FIELDS", "detail": "..."}` |
| `400 BODY_TOO_LARGE` | Body > 16 KiB | `{"registered": false, "reason": "BODY_TOO_LARGE", "limit_bytes": 16384}` |
| `400 HALO_DISABLED` | Server has `--halo-enabled` off | `{"registered": false, "reason": "HALO_DISABLED"}` |

**Curl example:**

```bash
curl -X POST http://127.0.0.1:31000/halo/programs \
    -H 'Content-Type: application/json' \
    -d '{"job_id":"agent-42","slo":5.0,"total_calls":4,
         "stage_sequence":["UNDERSTAND","LOCATE","PLAN","VERIFY"]}'
```

### `POST /v1/chat/completions` and `POST /generate` — extra body fields

These existing endpoints gain three optional fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `halo_job_id` | `str` | unset | Required when Halo is on. Identifies the job this call belongs to. Must match a previously registered program (per Q12 — no lazy create) |
| `halo_slo` | `float` | unset → server default | Per-job slowdown SLO. If the program was pre-registered (Q10), this is **ignored** and the pre-registered SLO is used (server logs WARN on mismatch) |
| `halo_bypass` | `bool` | `false` | Skip Halo entirely for this request. Reserved for server-internal traffic (warmup, self-loopback). Real clients should never set this |
| `halo_job_done` | `bool` | `false` | "This is the last LLM call of the job." On this request's finish, the server flips the owning Job's state to COMPLETE → `gc_completed` drops it after retain_seconds. Required for clean termination — without it the job stays RUNNING until the quiescent safety net trips (default 5 min). |

### Strict-mode reject paths (when `--halo-enabled` is on)

| Status | reason field | When |
|---|---|---|
| `400` | `HALO_NO_JOB_ID` | Request body lacks `halo_job_id` (Q7) |
| `400` | `HALO_PROGRAM_NOT_REGISTERED` | `halo_job_id` was never seen by `POST /halo/programs` (Q12) |

Server returns these via the standard SGLang abort path, so client
libraries see them as HTTP 400 with a JSON body containing the reason.

### `GET /server_info` — runtime state snapshot

Existing endpoint. When Halo is on, gains a `halo_state` block:

```json
"halo_state": {
  "enabled": true,
  "default_slo": 5.0,
  "tick_interval_ms": 100.0,
  "aggregator": "max+mean",
  "program_idle_timeout_seconds": 300.0,
  "cost_models_loaded": {"prefill": true, "tbt": true},
  "active_jobs": 1,
  "total_known_jobs": 1,
  "jobs": [
    {
      "job_id": "agent-42",
      "state": "running",
      "slo": 5.0,
      "virtual_job_slowdown": 2.8,
      "total_request_number": 4,
      "remaining_request_number": 2,
      "slo_violation_count": 0,
      "from_program": true,
      "total_calls_expected": 4,
      "stage_sequence": ["UNDERSTAND", "LOCATE", "PLAN", "VERIFY"],
      "expected_input_lens": null,
      "expected_output_lens": null,
      "dag": null
    }
  ]
}
```

`jobs[]` is the top 32 most-slowed active jobs (sorted by
`virtual_job_slowdown` desc); `total_known_jobs` includes
completed-but-not-yet-GC'd ones.

## Client integration recipes

### Plain curl (one job, N sequential calls)

The last call carries `halo_job_done: true` so the server closes the
job cleanly. Without it the job sits RUNNING until the quiescent
safety net trips.

```bash
BASE=http://127.0.0.1:31000
curl -X POST $BASE/halo/programs -H 'Content-Type: application/json' \
    -d '{"job_id":"demo-1","slo":5.0,"total_calls":3}'
for i in 1 2 3; do
  DONE=false
  [ "$i" -eq 3 ] && DONE=true
  curl -X POST $BASE/v1/chat/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"call $i\"}],
           \"max_tokens\":32,\"halo_job_id\":\"demo-1\",\"halo_slo\":5.0,
           \"halo_job_done\":$DONE}"
done
```

### LangChain `ChatOpenAI`

`extra_body` forwards arbitrary fields straight into the OpenAI
request body, so pass `halo_job_id` and `halo_slo` there:

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    base_url="http://127.0.0.1:31000/v1",
    api_key="dummy",
    model=MODEL,
    extra_body={"halo_job_id": "demo-1", "halo_slo": 5.0},
)
# before the first .invoke / .stream, register the program once:
import httpx
httpx.post("http://127.0.0.1:31000/halo/programs",
           json={"job_id": "demo-1", "slo": 5.0, "total_calls": 3})
```

### Agent_applications (planned, Q15 separate PR)

The runner has a per-job `ChainState.job_id` already. The wiring is:

1. At the top of `swe_bench_coding/agent.py::run_job`, POST `/halo/programs`
   with `job_id=state["job_id"]`, `slo=tau`, `total_calls=chain_length`,
   `stage_sequence=state["stage_sequence"]`.
2. In `make_llm`, pass `extra_body={"halo_job_id": ..., "halo_slo": ...}`
   into `ChatOpenAI(...)`.
3. `_detect_admission_rejection` already handles HTTP 400 — Halo's 400s
   ride the same path; just include `HALO_*` reasons in the rejection
   message capture if needed.

## Cost models

Halo reuses the JSON cost models produced by
`tools/admission_control/fit_cost_model.py`. Same files, same schema.
Pass them via `--halo-prefill-cost-model-path` and
`--halo-tbt-cost-model-path` (or the env vars). Phase 1 doesn't add
new fitting code — slowdown accuracy is bounded by these models'
accuracy (see `ms_dev/runtime/cost_models/README.md`).

## Operational notes

### TP-dedup
Counter metrics + JSONL log writes are gated to `attn_tp_rank=0` so a
TP=N deployment doesn't inflate values by tp_size. Per-rank reject log
lines (`[halo] REJECT rid=... reason=...`) still appear on every rank
— same behavior as admission_control.

### Idle GC
Programs registered but never given an LLM request get GC'd after
`--halo-program-idle-timeout-seconds` (default 300 s). A WARN log
fires when this happens — useful for catching wired clients that
register but forget to call.

### Single-instance only (Phase 1)
Halo is single-server / NULL disaggregation only. In PD modes the
controller is a no-op (the flag is still passed but `_halo_register_or_abort`
returns early). The `halo_bypass` field passes through harmlessly.

### Phase 2 outlook
The pre-registered metadata (`total_calls`, `stage_sequence`,
`expected_*_lens`, `dag`) is stored but not consumed in Phase 1.
Phase 2 admission/scheduling will read it to make lookahead decisions
("this job will issue K more calls — admit or evict?"). Clients that
already wire Option A today will need no API changes when Phase 2 ships.
