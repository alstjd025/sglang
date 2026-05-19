# Halo Phase 2 — Admission Control Design

> Project Halo Phase 2 의 핵심 deliverable. Phase 1 (job slowdown tracking)
> 위에 *job-level admission decision* 을 구현. `ms_dev/halo_dev/CLAUDE.md`
> §21 에서 가리키는 상세 문서.
>
> **2026-05-16 갱신** — admission 신호를 *lifetime VJS* 에서 *memoryless VSS*
> 로 교체 + KV-cache hard cap (Stage B′) 추가. 본 문서는 그 최신 상태를
> 코드와 1:1 로 기술한다.

## 1. 목적

> SLO 를 맞출 수 있는 capacity 로 serving system 을 최대한 유지하고, 들어온 job 들에 대해서는 SLO attainment rate 를 maximize.

이 문장이 Halo 전체의 motivation. 본 admission control 의 *고전 시스템* 적 의미:

> "이 새 job 을 받으면 *기존 진행 중 job 들의 SLA 보장이 깨질지* 미리 평가해서, 깨질 것 같으면 거부."

핵심은 **새 job 의 SLA 보장이 아니라 기존 작업들의 SLA 보호**.

설계의 핵심 arm 은 *job-scoped* 게이트 (mode `job`). 이와 비교하기 위한
*request-scoped* baseline (mode `request`) 도 같은 framework 안에 구현 —
"request 단위 admission 이 job 단위보다 나쁘다" 를 실험으로 보이는 용도 (§4.2).

## 2. LLM serving admission control 의 특수성

| # | 특수성 | 의미 |
|---|---|---|
| 1 | Multi-call job | 한 job 이 여러 LLM call. SLO 는 *job 전체* end-to-end slowdown 으로 정의 |
| 2 | Continuous batching | request 단위가 아니라 *batch step* 단위로 시간이 흐름 |
| 3 | Prefill ≠ Decode | 두 phase 비용 성질 다름 — split cost model 로 해결 (Phase 1 follow-up) |
| 4 | KV memory budget | 동시 in-flight 의 KV 가 메모리 점유. pool 이 차면 queueing / preemption cliff |

기존 work 와의 비교:
- **Mooncake**: per-request TTFT/TBT 예측, multi-call 누적 안 봄
- **DistServe simulator**: M/D/1 steady-state, burst 약함
- **MuxWise**: scheduling 위주, admission 부차적
- **우리 (Halo Phase 2)**: *job 단위* admission + *서버 순간 혼잡도(VSS)* 기반 결정 — novel

## 3. 결정 요약 (2026-05-14 → 2026-05-15 → **2026-05-16 VSS refactor**)

| ID | 내용 |
|---|---|
| Stage A | 두 scope. **`job`** (per-job memoryless VSS) = 설계. **`request`** (per-request, no aggregation) = "request-scoped 가 더 나쁘다" baseline. `level2` deprecated → `job` 자동 폴백 + WARN. `level0` 은 `job` 의 옛 alias. |
| Stage B | **Per-job concurrency hard cap** — declared 시 enforce, 미선언 시 무제한 (D4) |
| **Stage B′** | **KV-cache hard cap (신규 2026-05-16)** — KV pool usage 가 ratio 이상이면 새 job 의 첫 request reject. admission_mode 와 독립. cost model 이 못 보는 eviction cliff 를 hard gate 로 막음 |
| 용어 | **측정값** `virtual job slowdown` (VJS) — job lifetime cumulative, SlowdownTracker 가 sweep 마다 갱신, SLO *측정* 전용. **예측 신호** `virtual server slowdown` (VSS) — admission 이 쓰는 *순간* 혼잡도, memoryless. **둘은 분리** (§4.0). |
| 결정 단위 | mode `job`: **job 단위** — 첫 LLM request 만 Stage A, 후속 자동 admit (chain 보호). mode `request`: **request 단위** — 매 request Stage A. |
| 사용 정보 | **현재 batch composition 만** — active call 들의 현재 prompt/prefix/KV + 새 request 의 prompt/prefix. **VJS·elapsed history·DAG·remaining lengths 전부 안 씀.** |
| VSS 정의 | 새 도착을 더한 *augmented batch step time* ÷ 각 unit 의 *현재 solo step time*. memoryless (§4.0). |
| SLO 침해 비율 threshold | **0.2 (20%)** default (D3) |
| Cap 초과 시 동작 | **reject** (P2) — pending queue 는 추후 |
| Dry-run mode | **포함** (P3) — Stage A 와 Stage B′ 둘 다 dry-run 존중 |
| 한계 인지 | VSS 는 batch step time 만 봄 — KV pool *cliff* 는 Stage B′ 가 별도 hard gate 로 처리 (§5.2, §12) |

### 3.1 왜 VJS → VSS 로 바꿨나 (2026-05-16)

1차 라운드 (2026-05-15) 의 mode `job` 은 `predicted = current_vjs × stretch`
였다 — `current_vjs` 가 job 의 *lifetime cumulative* slowdown. 사용자 환경
실험 (λ 0.075–0.5, 80 min) 에서 병적 거동 관찰:

- 저부하에서도 높은 rejection, throughput 초기 peak → 붕괴 → 느린 부분 회복.
- 진단: lifetime-cumulative VJS 는 신호 안에 **적분기(integrator)** 가 들어있어
  *과거를 기억* 하고 부하 변화에 *지연(lag)* 된다. job 이 끝나도 남은 job 의
  VJS 는 즉시 안 떨어진다 → admission 이 닫힌 채로 머문다 (복원력 없는 열린 루프).

**결론**: lifetime-cumulative VJS 는 SLO 를 *측정* 하기엔 옳지만 admission 을
*제어* 하기엔 틀렸다. → admission 신호를 **memoryless VSS** 로 분리 (§4.0).
VJS 는 SlowdownTracker / 리포팅 / 향후 R3 scheduling 용으로 그대로 유지.

## 4. Stage A — Predictive admission (memoryless VSS)

### 4.0 VSS — virtual server slowdown

admission 이 쓰는 신호. 한 active unit i 가 새 도착을 admit 했을 때 겪을
**예측 slowdown**:

```
predicted_VSS_i = S_phase(i)⁺ / solo_step_i
```

- **분자** `S_phase(i)⁺` — *현재 batch + 새 도착* 을 cost model 에 넣은 step
  time. 서버의 post-admission 혼잡도. 모든 unit 이 공유.
- **분모** `solo_step_i` — unit i 가 *자기 call(들)만* 단독 실행할 때의 step
  time. unit 의 *현재* KV/prompt 로만 계산 — 누적 이력 없음.

memoryless: 현재 batch composition + 새 도착의 첫 call shape 만의 함수.
job 이 끝나 batch 가 줄면 다음 step 에 `S⁺` 가 즉시 떨어지고 모든
`predicted_VSS_i` 가 즉시 하락 → cumulative-VJS 게이트에 없던 빠른
self-regulating 피드백.

### 4.1 mode `job` — Per-job VSS

`JobSlowdownAdmissionPredictor`. 결정을 *job 단위* 로 — 한 job 의 첫 request 만
Stage A 를 거치고, 후속 request 는 자동 admit (chain 보호).

**사용자 환경에서 step 의 의미**:
- chunked prefill ON + `--enable-mixed-chunk` OFF.
- 한 step (`run_batch()` 1 호출) 은 *EXTEND-only* 또는 *DECODE-only*. MIXED 없음.
- prefill 과 decode 가 시간적으로 *분리* 되어 번갈아 step 으로 실행.

**알고리즘** (`admission_decision.py::JobSlowdownAdmissionPredictor.predict`,
`_augmented_step_times`):

```
# 0. active call snapshot 에서 kv_len_now == 0 (waiting queue / retracted —
#    어떤 forward step 에도 없음) request 는 제외 (결함 A, 2026-05-18).

# 1. 현재 batch composition 을 prefill / decode 로 분리
current_prefill = [(n_i, r_i)] for prefill-phase active calls   # n_i = prompt-prefix, r_i = prefix
current_decode  = [kv_j]       for decode-phase active calls    # kv_j = kv_len_now

n_new = max(0, first_call_input_len - first_call_prefix_len)
r_new = first_call_prefix_len

# 2. Augmented step times — 새 도착을 더한 batch (VSS 의 공유 분자).
#    EXTEND batch 는 _bounded_extend_step 으로 한 chunked-prefill step
#    (≤ chunked_prefill_size 토큰) 으로 cap — 결함 A: un-chunked Σnᵢ² 폭발 차단.
extend_batch = _bounded_extend_step(current_prefill + [(n_new, r_new)],
                                    chunked_prefill_size)
S_extend⁺ = cost_model.estimate_step_ms(extend_batch, [])
S_decode⁺ = cost_model.estimate_step_ms([], current_decode + [first_call_input_len])

# 3. 각 active job 별 predicted_VSS — 자기 phase 의 augmented step ÷ 자기 solo step
for each active job i (active_calls 비어있으면 skip):
    if i 의 active call 중 prefill 이 하나라도 있으면:           # prefill 우선
        solo_i = cost_model.estimate_step_ms(
                     _bounded_extend_step(i 의 prefill (n,r), chunked_prefill_size), [])
        predicted_VSS_i = S_extend⁺ / max(solo_i, 1.0)
    else:
        solo_i = cost_model.estimate_step_ms([], i 의 decode-call KV 들)
        predicted_VSS_i = S_decode⁺ / max(solo_i, 1.0)
```

- 새 도착은 EXTEND batch 에 `(n_new, r_new)` 로, DECODE batch 에
  `first_call_input_len` (prefill 후 KV) 로 들어간다. decode 투영은 DAG
  lookahead 가 아니라 "이 request 는 어차피 decode 한다" 는 필연 — 빼면
  decode-heavy 서버에서 게이트가 새 job 의 주 효과를 못 본다.
- `current_vjs` / elapsed / DAG / remaining / expected_output **안 씀**.
- 비용: cost_model 호출 = 2 (augmented) + active job 수 (solo).
- **결함 A 수정 (2026-05-18) 의 귀결**: EXTEND step 이 ≤ `chunked_prefill_size`
  로 bound 되므로 prefill-phase job 의 VSS 는 한 자릿수(공유 chunked step ÷
  자기 chunked step) 로 내려온다. 새 prefill 부하는 step 을 *무겁게* 만들지
  않고 *step 수* 를 늘릴 뿐인데(chunked prefill 의 성질) 그 효과(queueing)는
  VSS 가 못 본다. 따라서 수정 후 admission 판정은 사실상 *decode VSS* 가
  주도한다 — prefill-phase unit 은 위반에 거의 기여하지 않음. §12 참고.

**사용자 예시 검증** (A prefill chunk=4096, B/C decode KV=3000/7000, 새 req
prompt=8000/prefix=7500 → n_new=500):

| Job | 현재 단계 | predicted_VSS 식 |
|---|---|---|
| — | — | `S_extend⁺ = cost([(4096,0),(500,7500)], [])`, `S_decode⁺ = cost([], [3000,7000,8000])` |
| A | prefill | `S_extend⁺ / cost([(4096,0)], [])` |
| B | decode | `S_decode⁺ / cost([], [3000])` |
| C | decode | `S_decode⁺ / cost([], [7000])` |

### 4.2 mode `request` — Per-request VSS (baseline)

`RequestSlowdownAdmissionPredictor`. **비교용 baseline** — "request-scoped
admission 이 job-scoped 보다 나쁘다" 를 실험으로 보이기 위한 의도적으로 naive
한 arm.

VSS 식·cost model 은 §4.1 과 **완전히 동일**. 차이는 단 두 가지:

1. **Job aggregation 없음** — 모든 in-flight call 하나하나를 *독립 단위* 로
   score. 예측 dict key 가 job_id 가 아니라 rid. 한 call 의 solo 는 그 call
   자기 KV/prompt 하나로 계산.
2. **매 request 마다 결정** — controller 가 첫 request 뿐 아니라 *모든*
   request 에 Stage A. 따라서 mid-chain request 도 reject 될 수 있음.

```
S_extend⁺, S_decode⁺ = _augmented_step_times(...)   # §4.1 과 동일

for each active call c (모든 job 의 active_calls flatten):
    if c.is_prefill:
        solo_c = cost_model.estimate_step_ms([(n_c, r_c)], [])
        predicted_VSS_c = S_extend⁺ / max(solo_c, 1.0)
    else:
        solo_c = cost_model.estimate_step_ms([], [c.kv_len_now])
        predicted_VSS_c = S_decode⁺ / max(solo_c, 1.0)
```

mode `job` 과 `request` 의 차이는 **scoring 단위뿐** — VSS 산식·cost model 은
동일하다. active call 이 job 당 정확히 1 개면 두 모드의 예측치는 같다
(`test_matches_job_predictor_for_one_call_per_job` 이 이를 검증).

**왜 더 나쁠 것으로 예상되나** (실험 가설): chain 보호가 없어 mid-chain reject
발생 → 앞 call 들에 쓴 연산 낭비. 또 한 job 의 여러 call 이 각각 violation 으로
세어져 violation_ratio 가 job-scoped 보다 noisy.

### 4.3 level2 — SLO-driven lookahead (DEPRECATED 2026-05-15)

> ⚠️ `admission_mode=level2` 가 들어와도 controller 가 WARN 후 mode `job` 으로
> 자동 폴백.

본 모드는 *DAG / remaining lengths / expected_output_lens* 같은 declared 정보를
깊게 활용했음. 2026-05-15 design decision (현재 상태만 사용) 이후로 사용 안 함.
코드는 `admission_decision.py::LookaheadAdmissionPredictor` 에 남아 있으나
admission path 에서는 호출 안 됨.

### 4.4 결정 함수 — `decide_admission`

```
predicted_slowdowns: Dict[unit_id, predicted_VSS]   # unit = job 또는 request
slos:                Dict[unit_id, SLO]             # 같은 key 체계

violations      = count of u where predicted_slowdowns[u] > slos[u]
violation_ratio = violations / max(1, len(predictions))

if violation_ratio > threshold (default 0.2):
    return REJECT(REASON_HALO_ADMISSION_PREDICTED, ...)
else:
    return ADMIT(...)
```

`decide_admission` 은 unit 종류에 무관 — key 가 job_id (mode `job`) 든 rid
(mode `request`) 든 동일하게 동작. active unit 이 0 개면 항상 admit.

## 5. Stage B / B′ — Capacity hard caps

Stage A 가 *slowdown 예측* 게이트라면, Stage B / B′ 는 *capacity 제약* 게이트.
둘은 hard cap — 예측이 아니라 명시적 한계.

### 5.1 Stage B — Per-job concurrency hard cap

**철학**: Application 이 *나는 동시에 X 개까지 던집니다* 라고 declare 하면
server 가 그 promise 를 enforce.

```python
# HaloController.register_request, Stage A 통과 후:
cap = job.declared_max_concurrency       # None → 무제한 (D4)
if cap is not None and job.in_flight_count >= cap:
    raise HaloRejectError(REASON_HALO_CONCURRENCY_CAP, rid)
```

- declared field 는 `register_program` body 의 optional field.
- 결과: HTTP 400 `HALO_CONCURRENCY_CAP`.

### 5.2 Stage B′ — KV-cache hard cap (신규 2026-05-16)

**문제**: VSS 는 *batch 에 들어간* call 의 step time 만 본다. KV pool 이 차면
새 request 가 batch 에 *못 들어가* queueing 되거나, 실행 중 request 가
preemption / recompute 당한다 — cost model `estimate_step_ms` 는 매끄러운
다항식이라 이 **eviction cliff** 를 표현할 항이 없다. 게다가 preemption 이
일어나면 쫓겨난 request 가 batch 를 떠나 step time 이 *오히려 떨어져* VSS 가
거꾸로 신호를 준다. → 연속 신호(VSS)에 섞지 않고 **별도 hard gate** 로 처리.

**알고리즘** (`HaloController.register_request`):

```python
# 새 job 의 첫 request 에 대해서만:
if is_first_request and kv_cap_enabled and kv_usage_ratio is not None:
    if kv_usage_ratio >= admission_kv_cap_ratio:
        # dry-run 이면 log 만, 아니면 reject
        raise HaloRejectError(REASON_HALO_KV_CAP, rid)
```

- `kv_usage_ratio` — scheduler 가 `get_pool_stats().get_kv_token_stats()[1]`
  (KV pool usage, 0..1) 를 매 admission 시점에 전달.
- `admission_kv_cap_ratio` — `--halo-admission-kv-cap-ratio`. **0 < ratio < 1
  일 때만 활성** (default 0.0 = 비활성). `kv_cap_enabled` 가 이 조건.
- **admission_mode 와 독립** — `admission_mode=off` 여도 ratio 만 켜면 작동.
  → VSS-only / KV-cap-only / 둘다 의 3-way ablation 이 가능.
- **새 job 의 첫 request 에만** 적용. 후속 request 는 우회 — Phase 2 결정
  ("한 번 admit 된 job 의 후속 request 는 무조건 admit") 과 일관하고, pending
  queue 가 아직 없어 후속 request 를 잡아둘 수단이 없다 (후속을 거절하면
  실행 중 job 이 깨진다).
- dry-run 존중 — `mode=kv_cap` row 를 decision log 에 쓰고 admit.
- 결과: HTTP 400 `HALO_KV_CAP`.

**한계**: 신규 job 의 output length 를 모르므로 (현재-상태-만 결정) lifetime
peak KV 는 못 본다. 현재 usage 즉시값 + 신규 첫 request 만으로 판단 — decode
가 길어지며 뒤늦게 cliff 에 닿는 경우는 못 막는다. 또 *이미 batch 에서 도는*
request 의 decode-time KV 성장도 못 막는다 (SGLang 자체 retraction 이
last-resort). pending queue / per-job concurrency budget / drop 은 추후 작업.

## 6. 코드 구조

```
python/sglang/srt/managers/halo/admission_decision.py
├── ActiveCallInfo                  (frozen — 한 in-flight call 의 현재 shape)
├── JobLookaheadInput               (frozen — per-active-job snapshot: slo + active_calls)
├── NewJobInput                     (frozen — 새 도착의 첫 call shape)
├── AdmissionDecisionResult         (frozen — admit/reject + per-unit 예측 dict)
├── AdmissionPredictor              (ABC)
├── _bounded_extend_step(...)       (EXTEND batch 를 한 chunked step 으로 cap — 결함 A)
├── _augmented_step_times(...)      (S_extend⁺, S_decode⁺ — 공유 VSS 분자)
├── JobSlowdownAdmissionPredictor       (mode "job")
├── RequestSlowdownAdmissionPredictor   (mode "request")
├── LookaheadAdmissionPredictor         (mode "level2", DEPRECATED)
└── decide_admission(predictions, slos, threshold) → AdmissionDecisionResult

python/sglang/srt/managers/halo/controller.py
├── HaloConfig.admission_kv_cap_ratio   (Stage B′ 설정)
├── HaloController.kv_cap_enabled       (0 < ratio < 1)
├── register_request(..., kv_usage_ratio)   Q7 → Q12 → Stage B′ → Stage A → Stage B → admit
├── _build_active_jobs_input(...)       RequestExecutionInfo → JobLookaheadInput
├── _stage_a_decide(...)                predict + decide_admission
├── _log_admission_decision(...)        Stage A JSONL row
└── _log_kv_cap_decision(...)           Stage B′ JSONL row (mode="kv_cap")
```

기존 파일 수정: `job.py` (declared_max_concurrency / in_flight_count),
`scheduler.py` (`_halo_register_or_abort` 가 kv_usage_ratio 계산·전달),
`server_args.py` (CLI 플래그), `metrics.py` (reject reason 라벨).

## 7. JSON / decision log 스키마

### 7.1 admission_decisions.jsonl — Stage A row (mode job/request)

```json
{
  "ts_ns": 1717800000000000,
  "rid": "rid-abc-001", "job_id": "agent-42",
  "mode": "job", "is_first_call": true, "dry_run": false,
  "decision": "reject", "reason": "HALO_ADMISSION_PREDICTED",
  "violation_ratio": 0.31, "violation_count": 5, "active_units_total": 16,
  "threshold": 0.2,
  "predicted_slowdowns": {"job-1": 4.2, "job-2": 5.8, ...},
  "horizon_sec": null
}
```
`predicted_slowdowns` 값 = predicted VSS. key 는 mode 가 `job` 이면 job_id,
`request` 이면 rid.

### 7.2 admission_decisions.jsonl — Stage B′ KV-cap row

```json
{
  "ts_ns": 1717800000000000,
  "rid": "rid-abc-001", "job_id": "agent-42",
  "mode": "kv_cap", "is_first_call": true, "dry_run": false,
  "decision": "reject", "reason": "HALO_KV_CAP",
  "kv_usage_ratio": 0.95, "kv_cap_ratio": 0.90
}
```

## 8. CLI 플래그 / env vars

| CLI 플래그 | env var | Default | 설명 |
|---|---|---|---|
| `--halo-admission-mode {off,job,request}` | `SGLANG_HALO_ADMISSION_MODE` | `off` | Stage A scope. `level0`/`level2` 는 deprecated alias |
| `--halo-admission-violation-threshold <f>` | `SGLANG_HALO_ADMISSION_VIOLATION_THRESHOLD` | `0.2` | active unit 중 이 비율 초과가 SLO 침해 예상 시 reject (D3) |
| `--halo-admission-kv-cap-ratio <f>` | `SGLANG_HALO_ADMISSION_KV_CAP_RATIO` | `0.0` (비활성) | **Stage B′**. KV pool usage 가 이 값 이상이면 새 job reject. 0<ratio<1 일 때만 활성. admission_mode 와 독립 |
| `--halo-admission-dry-run` | `SGLANG_HALO_ADMISSION_DRY_RUN` | `0` | 로그만 + 항상 admit. Stage A + B′ 둘 다 적용 |
| `--halo-admission-decision-log <path>` | `SGLANG_HALO_ADMISSION_DECISION_LOG` | unset | JSONL 출력. 비어 있으면 expctl 가 `<session>/admission_decisions.jsonl` 로 auto-route (admission_mode≠off **또는** kv-cap 활성 시) |
| `--halo-admission-lookahead-horizon-sec <f>` | `SGLANG_HALO_ADMISSION_LOOKAHEAD_HORIZON_SEC` | `0` | level2 (deprecated) 의 horizon |

### Ablation 조합

| admission_mode | kv_cap_ratio | 효과 |
|---|---|---|
| `off` | `0` | admission 없음 (Phase 1 strict-mode 만) |
| `job` | `0` | VSS-only |
| `off` | `0.90` | KV-cap-only |
| `job` | `0.90` | VSS + KV cap (full) |

## 9. Workflow — 요청 처리 흐름

```
client
  ├── POST /halo/programs {job_id, slo, ..., declared_max_concurrency}
  │      └── server: register_program ─ Phase 1
  │
  └── POST /v1/chat/completions  (body: halo_job_id=, halo_slo=)
         └── scheduler._halo_register_or_abort:
             ├── kv_usage_ratio = get_pool_stats().get_kv_token_stats()[1]  (kv_cap_enabled 시)
             │
             └── HaloController.register_request:
                 ├── 1. Q7  — halo_job_id missing → REJECT HALO_NO_JOB_ID
                 ├── 2. Q12 — not registered      → REJECT HALO_PROGRAM_NOT_REGISTERED
                 │
                 │   is_first_request = (job.total_request_number == 0)
                 │
                 ├── 3. Stage B′ — KV cap  (★ 2026-05-16)
                 │      if is_first_request and kv_cap_enabled
                 │         and kv_usage_ratio >= admission_kv_cap_ratio:
                 │             log kv_cap row
                 │             if not dry_run: → REJECT HALO_KV_CAP
                 │
                 ├── 4. Stage A — Predictive VSS  (★)
                 │      run_stage_a = mode != off
                 │                    and (mode == "request" or is_first_request)
                 │      if run_stage_a:
                 │          predictions = predictor.predict(active_jobs, new_job)   # {unit: predicted_VSS}
                 │          decision    = decide_admission(predictions, slos, threshold)
                 │          log Stage A row
                 │          if reject and not dry_run: → REJECT HALO_ADMISSION_PREDICTED
                 │      (mode "job": 후속 request 는 Stage A skip)
                 │
                 └── 5. Stage B — Concurrency cap
                        if job.declared_max_concurrency and in_flight_count >= cap:
                            → REJECT HALO_CONCURRENCY_CAP
```

## 10. 단계별 PR 순서

| # | 내용 | 상태 |
|---|---|---|
| PR1–PR4 | AdmissionPredictor ABC + job/level2 predictor + controller 통합 + Stage B | ✅ |
| PR-refactor (2026-05-15) | job-level reject, snapshot-only per-job stretch, level2 deprecate | ✅ |
| PR-request-baseline (2026-05-15) | mode 개명 (`level0`→`job`, 신규 `request`), request-scoped predictor | ✅ |
| **PR-vss-kvcap (2026-05-16)** | **admission 신호를 lifetime VJS → memoryless VSS 로 교체 (`predicted_VSS = S⁺/solo`, `current_vjs` 제거). KV-cache hard cap (Stage B′) 추가. mode `job`+`request` 둘 다 VSS 통일.** | ✅ |
| **PR-결함A-fix (2026-05-18)** | **결함 A 수정 — EXTEND batch 를 `_bounded_extend_step` 으로 한 chunked-prefill step (≤ `chunked_prefill_size`) 으로 cap (un-chunked Σnᵢ² 폭발 차단) + `kv_len_now==0` waiting/retracted request 를 VSS batch 에서 제외. `chunked_prefill_size` 를 scheduler→register_request→predict 로 배선.** | ✅ |
| PR5 | 검증 실험 (mode={off,job} × kv_cap={off,on} ablation) + 문서 | ⏳ |

## 11. 단위 테스트 전략

### 11.1 `test_halo_admission_predictors.py` (31 tests)
- + `test_chunked_prefill_size_caps_extend_vss` — 결함 A: 거대 prefill 도착이
  cap 으로 폭발 안 함.
- `decide_admission`: violation_ratio, threshold 경계, missing SLO, frozen dataclass
- `JobSlowdownAdmissionPredictor`: 빈 active → 빈 dict, 단일 decode/prefill job 의
  `S⁺/solo`, mixed (사용자 예시), 큰 도착 → 큰 VSS, **memoryless (elapsed 무관)**,
  active_calls 없는 job skip, declared 무시
- `RequestSlowdownAdmissionPredictor`: rid key (no aggregation), per-call phase별
  solo, **job 당 call 1 개면 job-predictor 와 동일** (VSS 통일 검증)

### 11.2 `test_halo_phase1.py` (78 tests)
- + `test_waiting_queue_requests_excluded_from_vss` — 결함 A: `kv_len_now==0`
  request 가 VSS batch 에서 제외됨.
- `TestPhase2AdmissionIntegration`: off skip, 빈 active admit, 위반 시 reject
  (mock predictor), dry-run, decision log, level2 폴백, 후속 request skip,
  request mode 매 request 검사
- `TestPhase2KvCap` (★ 신규 8 tests): default 비활성, pool full reject,
  pool below admit, ratio=None skip, **후속 request 우회**,
  **admission_mode 와 독립**, dry-run, decision log `mode=kv_cap` row
- `TestPhase2StageBConcurrencyCap`: declared cap enforce

### 11.3 통합 / e2e
- 같은 워크로드에 §8 의 4 조합 ablation. SLO attainment / rejection / throughput /
  oscillation 비교.

## 12. 알려진 한계

- **VSS 는 KV pool cliff 를 모름.** cost model step time 은 매끄러운 다항식 →
  eviction/preemption cliff 표현 불가. Stage B′ KV cap 이 별도 hard gate 로 보완.
  단 Stage B′ 도 신규 첫 request 의 *현재* usage 만 봄 — decode-time KV 성장으로
  뒤늦게 닿는 cliff, 실행 중 request 의 성장은 못 막음 (§5.2).
- **현재 시점만.** 미래 phase 변화 (prefill 중 job 이 decode 합류) 안 봄.
  memoryless 의 의도된 trade-off.
- **Cost model 부재 시.** step cost model 안 로드되면 admission_predictor=None →
  Stage A skip. Stage B′ 는 cost model 불필요 — ratio 만 켜면 작동.
- **Mixed-phase job.** 한 job 의 active call 에 prefill/decode 가 섞이면 *prefill
  우선* 으로 `S_extend⁺` 받음 (단순화). 사용자 환경에선 거의 발생 안 함.
- **Pending queue / concurrency budget / drop 미구현.** Stage B′ 가 후속 request
  를 못 잡아 우회시키는 것, fan-out job 의 unfairness 등은 추후 작업.
- **★ VSS 는 queueing 에 구조적으로 눈멂.** VSS = step-ratio (`batch step ÷
  solo step`) 라, *어떤 forward step 에도 안 들어간* request — waiting queue
  대기, KV 부족 preemption(retraction), chunked-prefill 청크 대기 — 는 진전 0
  으로 시간만 흐르는데 VSS 식에 담을 "step" 이 없다. 결함 A 수정으로 `kv==0`
  request 를 *명시적으로 제외* 하면서 이 한계가 분명해졌다: **수정 후 VSS 는
  `running_batch` 안의 contention 만 보고, `waiting_queue` 는 통째로 못 본다.**
  과부하 서버에서 slowdown 의 지배적 성분이 queueing 인데 — 이게 VSS-only
  접근의 근본 한계 (재설계 논의 §14 참고).
- **chunked_req 스냅샷 gap.** chunked-prefill 진행 중인 request (`self.chunked_req`,
  한 번에 ≤1개) 는 `waiting_queue` 에도 `running_batch.reqs` 에도 없어서
  `_halo_build_request_execution_infos` 가 못 잡을 수 있다. 한 번에 하나뿐이라
  영향은 작지만 알려진 gap.

## 13. 미정 (PR5 검증 후 결정)

- Threshold default 0.2, kv_cap_ratio 0.90 이 적절한지
- pending queue / per-job concurrency budget / doomed-job drop 도입 시점
- Phase 3 (R3 scheduling rebalance) 진입 시점

## 14. 검증 결과 + 접근법 비교 (2026-05-16~18, λ sweep)

워크로드 `swe_bench_coding_parallel_tool_delay` (SWE-bench, 프롬프트 14k–75k
토큰, job 당 15–30 call), B200×4 TP=4, tau=5. 네 정책 비교 (`no_admission` /
`mooncake` = `admission_control` ratio 모드 / `halo_v1` = 옛 cumulative-VJS /
`halo_VSS` = 신규 memoryless VSS).

### 14.1 Job goodput (submit window [20,60)분, goodput = SLO 안 완주)

| λ | mooncake | halo_v1 (옛 VJS) | halo_VSS |
|---|---|---|---|
| 0.075 | 64 (40%) | 7 (4%) | — |
| 0.1 | 62 (29%) | 0 (0%) | **71 (33%)** |
| 0.2 | 78 (16%) | 37 (8%) | — |
| 0.3 | 51 (7%) | 21 (3%) | — |
| 0.4 | 48 (5%) | 19 (2%) | 11 (1%) |
| 0.5 | 56 (5%) | 25 (2%) | 0 (0%) |

- **옛 cumulative-VJS Halo 는 전 구간 꼴찌, 확정적으로 망가짐** (§3.1 진단).
- **Mooncake(per-request ratio) 는 일관된 baseline.**
- **VSS 는 bimodal** — λ0.1 에서 Mooncake 를 이기고, λ0.4/0.5 에서 붕괴.

### 14.2 받은 job vs 완주 (submit window [0,80)분)

| λ | policy | received | completed | comp/recv | mid-chain reject |
|---|---|---|---|---|---|
| 0.1 | mooncake | 368 | 157 | 43% | 139 |
| 0.1 | halo_VSS | 292 | 214 | **73%** | 0 |
| 0.5 | mooncake | 1038 | 138 | 13% | 816 |
| 0.5 | halo_VSS | 367 | 103 | **28%** | 0 |

- **job-level commit (후속 mid-chain reject 안 함) 은 데이터로 입증된 진짜
  이점** — VSS 가 받은 job 의 완주율이 Mooncake 의 2–3배. Mooncake 는 job 을
  admit 해놓고 중간 call 을 거절 (λ0.5 에서 816 job) → 앞 연산 낭비.

### 14.3 VSS 의 붕괴 원인 = 결함 A (un-chunked prefill 비용)

- 예측 VSS 분포가 bimodal: p50 ≈ 3–5 (정상, decode), 그러나 p90 50→138→172,
  max 525→1037→1303 (λ 0.1→0.4→0.5). prefill-phase unit 의 VSS 폭발.
- 원인: `_augmented_step_times` 가 prefill 중인 모든 request 의 *전체*
  `n = prompt−prefix` 를 한 step 에 넣고 `Σnᵢ²` 2차항을 계산 — chunked
  prefill 환경에서 실제 한 step 은 ≤ `chunked_prefill_size`(16384) 토큰인데
  un-chunked 로 ~160× 과대추정. λ 오를수록 backlog 커져 악화.
- → **PR-결함A-fix (2026-05-18)** 로 수정 (§4.1, §10). 단 수정 후에도 VSS 는
  queueing 에 눈멂 (§12) — 이건 버그가 아니라 step-ratio 의 본질적 한계.
- 부수 사실: KV pool usage 가 전 런·전 구간 0.96–0.97 (포화), mean VJS 8–10
  (tau=5 의 ~2배) — 워크로드가 capacity 대비 크게 oversubscribed.

### 14.4 재설계 논의 (길1 / 길2 — PR5 후 결정)

세 시도의 본질 (slowdown = contention 항 + queueing 항 분해):

| | 보는 것 | 즉각성 | 결과 |
|---|---|---|---|
| VJS | contention+queueing 둘 다 (실측) | 평생 누적 → 지연 | 망함 |
| VSS | contention 만 (cost-model, 순간) | 즉각 | queueing 눈멂 |
| Mooncake | contention 만 (per-request) | 즉각 | mid-chain reject 낭비 |

- queueing 은 *시간-적분* 현상 — 순간 step 스냅샷에 원리적으로 안 담김.
  VSS 가 그걸 못 보는 게 고부하 실패의 구조적 원인.
- **길1 (현재):** 결함 A 만 고치고 (PR-결함A-fix) KV cap 켜고 λ={0.075,0.1,0.2}
  재실험 → 깨끗한 데이터로 VSS-vs-Mooncake 재판정. VSS-with-bug 도 λ0.1 에선
  이미 Mooncake 와 비등했으므로 시사적.
- **길2 (재설계 후보, 미구현):** admission 신호를 *측정* 기반·*최근 윈도우*·
  *queueing 포함* 으로. 후보 = `server_slowdown = (K·Δt)/Σwork ≤ tau`
  (K = 활성 request, Σwork = 윈도우 내 실제 진행 solo-work; 굶는 request 는
  work 0 기여 → queueing 자동 반영; 예산 불필요, tau 만 필요). per-job 절대
  예산 `(tau−1)·S_job_solo` 은 job 전체 solo 시간이 필요한데 서버가 모름 — 그래서
  budget-free server_slowdown 이 후보. lost-time `L = T − W` 분해, flip-count
  (doomed job 제외) 논의는 진행 로그에 있음.
- 결정 트리거: 길1 재실험 데이터. 고친 VSS 가 meaningful regime(λ 0.075–0.2)
  에서 Mooncake 를 일관되게 이기면 → 그대로. 지면 → 길2.

### 14.5 길1 재실험 결과 — VSS_v2 (2026-05-18, λ=0.1)

세션 `260518_0201_halo_admission_job_VSS_v2_lambda_0p1`,
`admission_decisions.jsonl` (11,757 predicted VSS 값):

- **결함 A 수정 확인** — predicted VSS: p90 50→**3.7**, max 525→**4.0**. 폭발
  인공물 완전 제거.
- **그러나 고친 VSS 는 구조적으로 toothless** — predicted VSS 의 *전체 범위*
  가 [1, 4] 인데 SLO=5 → `violation_ratio` 가 항상 정확히 0 → **reject 0건.**
  threshold 튜닝으로 해결 불가 (신호 범위가 임계 아래).
- 동시에 측정 VJS = 25 (worst 500) — 서버는 ~25× 과부하.
- 원인: VSS = per-step *contention*. decode step cost 는 상수항(θ_c_d)이
  지배 → batch 가 커져도 비율이 ~3.5 에서 saturate. 실제 25× 의 대부분은
  *queueing* (대기 / preemption) 인데 step-ratio 는 이를 구조적으로 못 봄.
- v1 λ0.1 의 "선전" 은 결함 A 버그 덕분 — spurious prefill VSS 가 *유일한*
  reject 원천이었다. 수정 → 0 reject → 사실상 no_admission.

**결론**: VSS-only admission 은 meaningful regime 에서 작동하지 않음 — 데이터로
확정. **길2 (queueing 포함 측정 기반 재설계, §14.4 candidate) 가 다음 단계.**
