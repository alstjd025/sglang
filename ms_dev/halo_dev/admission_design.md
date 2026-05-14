# Halo Phase 2 — Admission Control Design

> Project Halo Phase 2 의 핵심 deliverable. Phase 1 (R1 — slowdown tracking)
> 위에 R2/R3 의 일부를 *job-level admission decision* 으로 구현. `ms_dev/halo_dev/CLAUDE.md`
> §21 에서 가리키는 상세 문서.

## 1. 목적

> SLO 를 맞출 수 있는 capacity 로 serving system 을 최대한 유지하고, 들어온 job 들에 대해서는 SLO attainment rate 를 maximize.

이 문장이 Halo 전체의 motivation. 본 admission control 의 *고전 시스템* 적
의미:

> "이 새 job 을 받으면 *기존 진행 중 job 들의 SLA 보장이 깨질지* 미리 평가해서, 깨질 것 같으면 거부."

핵심은 **새 job 의 SLA 보장이 아니라 기존 작업들의 SLA 보호**.

## 2. LLM serving admission control 의 특수성

| # | 특수성 | 의미 |
|---|---|---|
| 1 | Multi-call job | 한 job 이 여러 LLM call. *누적* slowdown 추적 필요 |
| 2 | Continuous batching | request 단위가 아니라 *batch step* 단위로 elapsed 증가 |
| 3 | Prefill ≠ Decode | 두 phase 비용 성질 다름 — split cost model 로 해결 (Phase 1 follow-up) |
| 4 | KV memory budget | 동시 in-flight 의 KV 가 메모리 점유. 한 job burst 가 다른 job 의 cache 를 evict |

기존 work 와의 비교:
- **Mooncake**: per-request TTFT/TBT 예측, multi-call 누적 안 봄
- **DistServe simulator**: M/D/1 steady-state, burst 약함
- **MuxWise**: scheduling 위주, admission 부차적
- **우리 (Halo Phase 2)**: *job 의 남은 calls 의 cumulative slowdown* 을 예측해 결정 — novel

## 3. 결정 요약 (2026-05-14 → 2026-05-15 refinement 후)

| ID | 내용 |
|---|---|
| Stage A | **Level 0 (per-job snapshot stretch, M-2)** 만 활성. `--halo-admission-mode=level2` 는 *legacy* 라 자동 폴백 + WARN. |
| Stage B | **Per-job concurrency hard cap** — declared 시 enforce, 미선언 시 무제한 (D4) |
| 용어 | R1 의 측정값 = `virtual job slowdown`, admission 예측값 = `predicted virtual job slowdown` |
| 결정 단위 | **Job 단위** — 한 job 의 *첫 LLM request* 만 Stage A 거침. 후속 request 는 자동 admit (chain 끊김 방지) |
| 사용 정보 | **현재 상태만** — active job 들의 *현재* VJS + 새 request 의 prompt/prefix 만. DAG / remaining call lengths / expected output 안 씀 |
| Stretch 정의 | 각 active job 의 *자기 단계 (prefill/decode) 의 step time 변화율* (M-2) |
| SLO 침해 비율 threshold | **0.2 (20%)** default (D3) |
| Cap 초과 시 동작 | **reject** (P2) — queue 도입은 추후 |
| `declared_max_concurrency` | `register_program` body 의 *명시 field* (P4) |
| Dry-run mode | **포함** (P3) |
| 한계 인지 | **KV cache pressure 직접 미반영** (Σrⱼ 가 attention memory BW proxy 일 뿐). *Eviction / preemption cliff* 는 R3 단계 |

## 4. Stage A — Predictive admission

### 4.1 Level 0 — Per-job snapshot stretch (M-2, current default)

**철학** (2026-05-15 결정):
- *현재 시점만* 본다. 미래 phase 변화 (예: 현재 prefill 중인 job 이 decode 합류) 안 봄.
- *각 active job 의 현재 단계* (prefill / decode) 에 따라 *그 단계의 step time 변화율* 을 그 job 의 stretch 로 사용.

**사용자 환경에서 step 의 의미**:
- chunked prefill ON + `--enable-mixed-chunk` OFF.
- 한 step (`run_batch()` 1 호출) 은 *EXTEND-only* 또는 *DECODE-only*. MIXED 안 일어남.
- prefill 과 decode 가 시간적으로 *분리* 되어 번갈아 step 으로 실행.

**알고리즘**:

```
# 1. 현재 batch composition 을 prefill / decode 로 분리
current_prefill = [(n_i, r_i)] for prefill-phase active calls
current_decode  = [kv_j]       for decode-phase active calls

n_new = max(0, first_call_input_len - first_call_prefix_len)
r_new = first_call_prefix_len

# 2. EXTEND step stretch — prefill-phase active job 이 받음
if current_prefill is non-empty:
    base_ext = cost_model([current_prefill], [])
    aug_ext  = cost_model([current_prefill + (n_new, r_new)], [])
    stretch_extend = aug_ext / base_ext
else:
    stretch_extend = 1.0   # 안 쓰임

# 3. DECODE step stretch — decode-phase active job 이 받음
#    새 request 는 KV = prompt_len 으로 decode batch 합류한다고 가정
if current_decode is non-empty:
    base_dec = cost_model([], [current_decode])
    aug_dec  = cost_model([], [current_decode + first_call_input_len])
    stretch_decode = aug_dec / base_dec
else:
    stretch_decode = 1.0

# 4. 각 active job 별 stretch 선택 + predicted_VJS 계산
for each active job i:
    job_is_in_prefill = any(call.is_prefill for call in i.active_calls)
    stretch_i = stretch_extend if job_is_in_prefill else stretch_decode

    if i.current_solo_elapsed_ms > 0:
        current_VJS_i = i.current_actual_elapsed_ms / i.current_solo_elapsed_ms
    else:
        current_VJS_i = i.slo  # R1 initial slowdown_max convention

    predicted_VJS_i = current_VJS_i × stretch_i
```

**사용자 예시 검증** (A prefill 4096 chunk, B/C decode KV=3000/7000, 새 req prompt=8000/prefix=7500):

| Job | 현재 단계 | 받는 stretch | predicted_VJS 표현 |
|---|---|---|---|
| A | prefill | stretch_extend = cost([(4096,0), (500,7500)], []) / cost([(4096,0)], []) | current_VJS_A × stretch_extend |
| B | decode | stretch_decode = cost([], [3000, 7000, 8000]) / cost([], [3000, 7000]) | current_VJS_B × stretch_decode |
| C | decode | (same as B) | current_VJS_C × stretch_decode |

- 비용: cost_model 호출 최대 4 회 (각 stretch 의 baseline + augmented). 빈 batch 면 호출 skip.
- 의도적으로 제외: 미래 phase 합류 (A → decode 후), KV cache pressure, prefill burst 동안의 누적 대기, 새 request 후속 call 들의 부담.

### 4.2 Level 2 — SLO-driven lookahead (DEPRECATED 2026-05-15)

> ⚠️ 2026-05-15 부터 *deprecated*. `admission_mode=level2` 가 들어와도 controller 가
> WARN 후 Level 0 으로 자동 폴백.

본 모드는 *DAG / remaining lengths / expected_output_lens* 같은 declared 정보를 깊게
활용했음. 2026-05-15 design decision (현재 상태만 사용) 이후로 사용 안 함.
코드는 `python/sglang/srt/managers/halo/admission_decision.py::LookaheadAdmissionPredictor`
에 남아 있으나 admission path 에서는 호출 안 됨.

추후 (`expected_output_lens` 가 더 신뢰할 만해지면) 부활 가능성 있음.

### 4.3 결정 함수 — `decide_admission`

```
predicted_virtual_job_slowdowns: Dict[job_id, predicted_VJS]
slos:                            Dict[job_id, SLO]

violations      = count of j where predicted_VJS_j > slos[j]
violation_ratio = violations / max(1, len(predictions))

if violation_ratio > threshold (default 0.2):
    return REJECT(REASON_HALO_ADMISSION_PREDICTED, ...)
else:
    return ADMIT(...)
```

## 5. Stage B — Per-job concurrency hard cap

**철학**: Application 이 *나는 동시에 X 개 까지 던집니다* 라고 declare 하면, server 가 그 *promise* 를 enforce.

```python
class Job:
    declared_max_concurrency: Optional[int] = None  # None → 제한 없음 (D4)
    in_flight_count: int = 0                         # admit / finish 시 갱신

# In HaloController.register_request, after Stage A admit:
if job.declared_max_concurrency is not None:
    if job.in_flight_count >= job.declared_max_concurrency:
        raise HaloRejectError(REASON_HALO_CONCURRENCY_CAP, rid)
    job.in_flight_count += 1

# In HaloController.on_request_finished:
job.in_flight_count -= 1
```

- 첫 라운드 첫 동작: **reject** (queue 가 아님 — 단순)
- Declared field 는 `register_program` body 에 새 optional field 로 추가
- 결과: HTTP 400 `HALO_CONCURRENCY_CAP`. Client 측 `_detect_admission_rejection` 이 자연 처리

## 6. 코드 구조

### 6.1 신규 파일

```
python/sglang/srt/managers/halo/
├── admission_decision.py     ★ 신규
│   ├── JobLookaheadInput          (dataclass — per-active-job snapshot)
│   ├── NewJobInput                (dataclass — register 요청 객체 + declared structure)
│   ├── AdmissionDecisionResult    (dataclass — admit / reject + diagnostic dict)
│   ├── AdmissionPredictor         (ABC — predict(active, new, cost_model) → predictions)
│   ├── SnapshotAdmissionPredictor (Level 0 구현)
│   ├── LookaheadAdmissionPredictor (Level 2 구현, slice-by-slice 시뮬)
│   └── decide_admission(predictions, slos, threshold) → AdmissionDecisionResult
```

### 6.2 기존 파일 수정

| 파일 | 변경 |
|---|---|
| `managers/halo/__init__.py` | 새 export (AdmissionPredictor 등) |
| `managers/halo/job.py` | `declared_max_concurrency`, `in_flight_count` 추가 + `to_dict` 갱신 |
| `managers/halo/job_registry.py` | `register_program(...)` 시그니처에 `declared_max_concurrency` 추가 |
| `managers/halo/controller.py` | `HaloConfig` 에 admission mode/threshold/horizon/dry_run/decision_log 추가. `register_request` 안 strict-mode 후 *Stage A* (mode != off) → *Stage B* (cap) 순서로 호출. 새 reject reasons. on_request_finished 에 in_flight_count 감소 |
| `managers/halo/metrics.py` | admission 카운터 추가 (admitted_total, rejected_total{reason}) |
| `managers/io_struct.py` | `HaloRegisterProgramReqInput` 에 `declared_max_concurrency: Optional[int]` 추가 |
| `entrypoints/http_server.py` | `POST /halo/programs` body parse 시 새 field 인식 |
| `server_args.py` | 5 개 새 CLI 플래그 (admission-mode, threshold, horizon, dry-run, decision-log) |

### 6.3 ms_dev / expctl 통합

| 파일 | 변경 |
|---|---|
| `ms_dev/env.common.sh` | 5 개 새 env vars (SGLANG_HALO_ADMISSION_*) |
| `ms_dev/lib_server.sh::append_halo_args` | env var → CLI 변환 |
| `ms_dev/expctl/server_run_experiment.py` | env var snapshot 확장. decision_log auto-route. meta 기록 |
| `ms_dev/experiments/halo_base.sh` | 기본은 admission off 유지. 사용자가 명시적으로 켜는 패턴 |

## 7. JSON / decision log 스키마

### 7.1 register_program body 확장 (Optional field 추가)
```json
{
  "job_id": "agent-42",
  "slo": 5.0,
  "total_calls": 12,
  "stage_sequence": ["UNDERSTAND", "LOCATE", "PLAN", ...],
  "expected_input_lens":  [1024, 18000, ...],
  "expected_output_lens": [200, 600, ...],
  "dag": {"type": "parallel_rounds", "rounds": [...]},
  "declared_max_concurrency": 8        ← 신규 (Optional)
}
```

### 7.2 admission_decisions.jsonl (rank-0 only, auto-routed)
```json
{
  "ts_ns": 1717800000000000,
  "rid": "rid-abc-001",
  "job_id": "agent-42",
  "mode": "level0",
  "dry_run": false,
  "decision": "reject",
  "reason": "HALO_ADMISSION_PREDICTED",
  "violation_ratio": 0.31,
  "active_jobs_total": 16,
  "violation_count": 5,
  "threshold": 0.2,
  "predicted_virtual_job_slowdowns": {
    "job-1": 4.2, "job-2": 5.8, "job-3": 7.1, ...
  },
  "horizon_sec": null
}
```
(JSONL 키 `predicted_virtual_job_slowdowns` 는 2026-05-15 rename. 이전 이름: `predicted_slowdowns`.)

목적:
- offline replay (`tools/halo/replay_admission.py` — 향후. 다른 threshold/mode 비교)
- 실험 분석 (mode=off/level0/level2 sweep 결과 비교)

## 8. CLI 플래그 / env vars

| CLI 플래그 | env var | Default | 설명 |
|---|---|---|---|
| `--halo-admission-mode {off,level0,level2}` | `SGLANG_HALO_ADMISSION_MODE` | `off` | Stage A 켜기/끄기 + level 선택 |
| `--halo-admission-violation-threshold <f>` | `SGLANG_HALO_ADMISSION_VIOLATION_THRESHOLD` | `0.2` | xx%. 0.2 = active job 중 20% 이상 SLO 초과 예상 시 reject |
| `--halo-admission-lookahead-horizon-sec <f>` | `SGLANG_HALO_ADMISSION_LOOKAHEAD_HORIZON_SEC` | `0` (= SLO-driven) | Level 2 의 horizon. 0 이면 SLO 도달 시점 자동 |
| `--halo-admission-dry-run` | `SGLANG_HALO_ADMISSION_DRY_RUN` | `0` | 로그만 + 항상 admit (튜닝 모드) |
| `--halo-admission-decision-log <path>` | `SGLANG_HALO_ADMISSION_DECISION_LOG` | unset | JSONL 출력. 비어 있으면 expctl 가 `<session>/admission_decisions.jsonl` 로 auto-route |

## 9. Workflow — 요청 처리 흐름

```
client
  ├── POST /halo/programs {job_id, slo, total_calls, ..., declared_max_concurrency}
  │      └── server: register_program ─ Phase 1
  │
  └── POST /v1/chat/completions  (body: halo_job_id=, halo_slo=)
         └── server scheduler.handle_generate_request:
             ├── admission_control (Mooncake-style, optional) — Phase 1 그대로
             │
             └── HaloController.register_request:
                 ├── 1. Strict mode check (Phase 1)
                 │    - halo_job_id missing → REJECT HALO_NO_JOB_ID
                 │    - not in registry      → REJECT HALO_PROGRAM_NOT_REGISTERED
                 │
                 ├── 2. Stage A — Predictive admission (Phase 2 ★)
                 │    is_first_request = (job.total_request_number == 0)   ← 2026-05-15
                 │    if mode != off and is_first_request:
                 │      predictor   = SnapshotAdmissionPredictor   (level2 → auto-fallback)
                 │      predictions = predictor.predict(active_jobs, new_job)
                 │                  = {job_id: predicted_virtual_job_slowdown}
                 │      decision    = decide_admission(predictions, slos, threshold)
                 │      log to admission_decisions.jsonl
                 │      if reject and not dry_run:
                 │          → REJECT HALO_ADMISSION_PREDICTED
                 │    (follow-up requests in the same job skip Stage A entirely —
                 │     job-level reject: never break a chain mid-flight)
                 │
                 └── 3. Stage B — Concurrency cap (Phase 2 ★)
                      if job.declared_max_concurrency is set:
                          if job.in_flight_count >= cap:
                              → REJECT HALO_CONCURRENCY_CAP
                          job.in_flight_count += 1
```

## 10. 단계별 PR 순서

| # | 내용 | 영향 범위 | 상태 |
|---|---|---|---|
| **PR1** | `AdmissionPredictor` ABC + `SnapshotAdmissionPredictor` (Level 0) + 단위 테스트 | 라이브러리만 | ✅ |
| **PR2** | `HaloController` 통합: register_request 안 Stage A 호출, dry-run, decision log, CLI 플래그, env vars, expctl auto-route | controller + scheduler + plumbing | ✅ |
| **PR3** | `LookaheadAdmissionPredictor` (Level 2) + 단위 테스트 | 라이브러리만 | ✅ (그 후 2026-05-15 deprecated) |
| **PR4** | Stage B (concurrency cap) — Job/Registry 확장, register_program field, controller check, 단위 테스트 | Job/Registry + register_program + controller | ✅ |
| **PR-refactor** | **2026-05-15 design refresh** — job-level reject (`is_first_request` 분기), snapshot-only per-job stretch (M-2), declared-driven 코드 deprecate, level2 → level0 자동 폴백, `predicted_slowdowns` → `predicted_virtual_job_slowdowns` rename | admission_decision + controller + tests + docs | ✅ |
| **PR5** | 검증 실험 + 분석 + 문서 갱신 (CLAUDE.md / prediction_model.md / 본 문서 §검증 결과) | 문서만 | ⏳ |

## 11. 단위 테스트 전략

### 11.1 `test_halo_admission_predictors.py`
- `decide_admission`: violation_ratio 계산, threshold 경계, missing SLO 처리, frozen dataclass
- `SnapshotAdmissionPredictor` (M-2 per-job stretch):
  - 빈 active set → 빈 dict
  - 단일 decode-phase job → stretch_decode
  - 단일 prefill-phase job → stretch_extend
  - mixed (사용자 예시: A prefill, B/C decode) → 각자 phase 별 stretch
  - 더 큰 새 도착 → 더 큰 predicted VJS
  - current_solo_ms = 0 인 fresh job → SLO 로 fallback
  - declared 필드 있어도 결과 동일 (declared 안 씀)
- `LookaheadAdmissionPredictor` (DEPRECATED) — 회귀만 잡는 약한 검증

### 11.2 `test_halo_phase1.py`
- `TestPhase2AdmissionIntegration`:
  - admission_mode=off → Stage A skip
  - admission_mode=level0 + 빈 active → admit
  - admission_mode=level0 + 부담 큰 시나리오 → reject
  - dry-run 모드 → 결정 reject 여도 admit
  - decision log JSONL 1 row + 새 키 `predicted_virtual_job_slowdowns`
  - **`test_level2_falls_back_to_level0_with_warn`** — DEPRECATED 폴백
  - **`test_followup_requests_skip_stage_a`** — 같은 job 의 두 번째 request 는 Stage A 건너뜀
- `TestPhase2StageBConcurrencyCap`:
  - declared 없으면 무제한
  - declared 있으면 초과 시 reject
  - finish 후 slot 해제
  - admission_mode=off 와도 독립 작동

### 11.3 통합 / e2e
- 같은 워크로드 (parallel_tool_delay λ=0.5) 에 mode=off / mode=level0 비교
- SLO attainment rate / rejection rate / throughput 측정

## 12. 알려진 한계

- **KV cache pressure 미반영.** Cost model 식은 `Σrⱼ`, `Σ(nᵢrᵢ)` 를 변수로 쓰지만,
  *token_usage_pct 가 limit 에 가까워질 때의 cliff* (eviction, preemption) 은 직접
  모델링 안 됨. R3 (scheduling) 단계에서 별도 처리 예정.
- **현재 시점만.** 미래 phase 변화 (현재 prefill 중인 job 이 decode 합류하는 시점)
  안 봄. 사용자 결정 (2026-05-15).
- **Cost model 부재 시.** Step cost model 안 로드되면 admission_predictor = None →
  Stage A 스킵 + WARN. 모든 request 가 strict-mode + Stage B 만 통과.
- **Mixed-phase job.** 한 job 의 active call 중 prefill / decode 가 섞이면
  *prefill 우선* 으로 stretch_extend 받음 (단순화). 사용자 환경에선 거의 발생 안 함.
- **Cumulative wait effect.** prefill burst 동안 *decode 중인 job 들이 누적 대기* 받는
  영향은 명시적으로 모델링 안 함. R1 의 *과거* 측정값에 이미 누적 반영된 것으로
  처리 (snapshot 의 한계).

## 13. 미정 (PR5 검증 후 결정)

- Threshold default 0.2 가 적절한지
- KV cache pressure cliff 가 cost model 정확도에 미치는 영향 측정
- Phase 3 (R3 scheduling rebalance) 진입 시점

## 14. 검증 결과 (placeholder — PR5 후 채움)

(사용자 실험 후 작성)
- 같은 워크로드 sweep across mode={off, level0}
- SLO attainment rate 비교
- Capacity (admitted job/min) 비교
- Rejection rate 비교
