# Project Halo — Client-Facing API Reference

Project Halo is a **request-level** admission-control + tracking subsystem in
the SGLang single-instance scheduler. Clients interact with it entirely through
**per-request SLO fields in the request body** plus a set of **server CLI
flags** that operators set at launch.

- Admission is **fully per-request**. There is **no job pre-registration**: the
  old `POST /halo/programs` endpoint and the `halo_job_id` field were removed
  in the 2026-05-19 request-level refactor. Do not send a job id — there is no
  longer any.
- Halo is **off by default**. When the server is launched without
  `--halo-enabled`, the per-request fields below are accepted and ignored, and
  there is zero scheduler overhead.
- When enabled, every request is evaluated independently by an admission gate
  (a policy-independent KV-cache hard cap plus one pluggable policy). A
  rejected request fails admission with HTTP 400.

## 1. Request body fields

These optional fields may be attached to any generation request. All are
optional; omit them to leave the corresponding SLO unset.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `halo_ttft_slo` | float | `null` | Time-to-first-token SLO. Interpreted as **absolute milliseconds** or as a **slowdown ratio vs the solo-run baseline**, depending on the server's `--halo-slo-mode`. |
| `halo_tbt_slo` | float | `null` | Time-between-tokens SLO. Same units as `halo_ttft_slo` — controlled by `--halo-slo-mode`. |
| `halo_e2e_slo` | float | `null` | End-to-end SLO. **Always a slowdown ratio** (actual e2e latency / solo-run e2e latency), regardless of `--halo-slo-mode`. |
| `halo_bypass` | bool | `false` | Marks server-internal traffic (warmup, health, self-loopback) so the admission gate skips it. **Real clients should leave this `false`** — it is not a security boundary, just a "not real user traffic" marker. |

Notes:

- All three SLOs are always active when Halo is enabled. Each admission policy
  uses whichever SLOs its signal speaks to (see the policy table below).
- `--halo-slo-mode` affects only `halo_ttft_slo` and `halo_tbt_slo`.
  `halo_e2e_slo` is always a ratio.

### Endpoints that accept these fields

| Endpoint | Source struct |
|---|---|
| `POST /generate` | `GenerateReqInput` (`io_struct.py`) |
| `POST /v1/chat/completions` | `ChatCompletionRequest` (`openai/protocol.py`) |
| `POST /v1/completions` | `CompletionRequest` (`openai/protocol.py`) |

## 2. Server CLI flags (`--halo-*`)

Operators set these at server launch. They are the single source of truth for
Halo behavior; clients only need to know `--halo-slo-mode` to interpret their
SLO values correctly (and can read it back via `GET /halo/status`).

| Flag | Type / choices | Default | Meaning |
|---|---|---|---|
| `--halo-enabled` | flag (store_true) | `false` | Master switch. Enables request-level admission control + tracking. Off by default; when off, zero impact. |
| `--halo-tick-interval-ms` | float | `100.0` | Min wall-clock interval (ms) between Halo request-tracker ticks. |
| `--halo-admission-policy` | `off` \| `mooncake` \| `vss` \| `reactive` | `off` | Admission policy (see below). `off` = no policy (KV cap still applies). |
| `--halo-slo-mode` | `ratio` \| `absolute` | `ratio` | Interpretation of `halo_ttft_slo` / `halo_tbt_slo`: `ratio` = slowdown bound vs solo-run baseline; `absolute` = millisecond cap. (`halo_e2e_slo` is always a ratio.) |
| `--halo-admission-violation-threshold` | float | `0.2` | `vss` policy: reject when more than this fraction [0,1] of in-flight requests are predicted to breach their SLO. |
| `--halo-tbt-reactive-ratio` | float | `0.9` | `mooncake` policy Stage-3: the reactive TBT EWMA trips at `tbt_slo * this ratio` (absolute slo_mode only). |
| `--halo-admission-dry-run` | flag (store_true) | `false` | Compute + log the admission decision but **always admit**. |
| `--halo-admission-decision-log` | str (path) | `null` | JSONL path for per-decision admission logs (rank-0 only). |
| `--halo-admission-kv-cap-ratio` | float | `0.0` | KV-cache hard cap. Reject when KV-cache pool usage ratio is at/above this value. Active iff `0 < ratio <= 1`; `0.0` disables it. Independent of `--halo-admission-policy`. |
| `--halo-prefill-cost-model-path` | str (path) | `null` | Prefill cost model JSON (legacy two-model pair). Used by `mooncake` + solo baselines when no step cost model is set. |
| `--halo-tbt-cost-model-path` | str (path) | `null` | TBT cost model JSON (legacy two-model pair). |
| `--halo-step-cost-model-path` | str (path) | `null` | Halo Step Cost Model JSON (`form='halo_step_v1'`). Supersedes the legacy prefill/tbt pair. Required by the `vss` policy. |
| `--halo-cost-model-sample-log` | str (path) | `null` | Per-forward-step JSONL output for fitting the Step Cost Model (rank `attn_tp_rank==0` only). Unset = sampler disabled. |
| `--halo-cost-model-sample-every` | int | `1` | Subsample rate for `--halo-cost-model-sample-log`: write every Nth step. |

### Admission policies (`--halo-admission-policy`)

| Value | Policy | Signal |
|---|---|---|
| `off` | (none) | KV cap only — the gate otherwise admits. |
| `mooncake` | MooncakePolicy | Predictive TTFT/TBT vs the request's `halo_ttft_slo` / `halo_tbt_slo`. |
| `vss` | VssPolicy | Memoryless virtual-server-slowdown vs `halo_e2e_slo`. |
| `reactive` | (Phase 3) | Measured queueing-inclusive gate — not yet implemented. |

## 3. `GET /halo/status`

A lightweight, always-200 server-status probe. It does not touch the
scheduler — it just reflects `server_args`. Clients can call it at startup to
discover whether Halo is enabled and how to interpret their SLO values before
issuing traffic.

Response body:

```json
{
  "enabled": false,
  "admission_policy": "off",
  "slo_mode": "ratio",
  "kv_cap_ratio": 0.0,
  "tick_interval_ms": 100.0
}
```

| Field | Type | Source |
|---|---|---|
| `enabled` | bool | `--halo-enabled` |
| `admission_policy` | string | `--halo-admission-policy` |
| `slo_mode` | string | `--halo-slo-mode` |
| `kv_cap_ratio` | float | `--halo-admission-kv-cap-ratio` |
| `tick_interval_ms` | float | `--halo-tick-interval-ms` |

## 4. Client recipe

### Check Halo status at startup

```bash
curl -s http://localhost:30000/halo/status
# -> {"enabled":true,"admission_policy":"vss","slo_mode":"ratio",
#     "kv_cap_ratio":0.9,"tick_interval_ms":100.0}
```

Use `slo_mode` from this response to decide whether your `halo_ttft_slo` /
`halo_tbt_slo` values should be milliseconds (`absolute`) or slowdown ratios
(`ratio`). If `enabled` is `false`, the SLO fields are accepted but ignored.

### Issue a request carrying SLO fields

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "my-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "halo_ttft_slo": 2.0,
    "halo_tbt_slo": 2.0,
    "halo_e2e_slo": 3.0
  }'
```

With `slo_mode = ratio`, the example above asks for TTFT and TBT no worse than
2x the solo-run baseline and end-to-end latency no worse than 3x. With
`slo_mode = absolute`, `halo_ttft_slo` / `halo_tbt_slo` would instead be
millisecond caps (`halo_e2e_slo` stays a ratio either way).

If the admission gate rejects the request, the call fails with HTTP 400. The
same fields work identically on `/generate` and `/v1/completions`.

## 5. No job pre-registration

Halo admission is fully per-request. There is no job-registration endpoint and
no `halo_job_id` field — the job-level `POST /halo/programs` API was removed in
the 2026-05-19 request-level refactor. Every request is admitted or rejected on
its own SLOs and the current server state; nothing needs to be registered
ahead of time.

---

*This reference is derived from the SGLang source as of the 2026-05-19
request-level refactor (`io_struct.py`, `openai/protocol.py`, `http_server.py`,
`server_args.py`, `managers/halo/CLAUDE.md`). The code is authoritative — if it
diverges from this document, trust the code.*
