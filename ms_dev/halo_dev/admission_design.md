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

## 3. 결정 요약 (2026-05-14 사용자 컨펌)

| ID | 내용 |
|---|---|
| Stage A | **Level 0 (snapshot scaling) + Level 2 (SLO-driven lookahead)** 둘 다 구현. CLI `--halo-admission-mode {off, level0, level2}` 로 선택 |
| Stage B | **Per-job concurrency hard cap** — declared 시 enforce, 미선언 시 무제한 (D4) |
| Slowdown 합산 | **max over concurrent requests** in a job (D2) |
| SLO 침해 비율 threshold | **0.2 (20%)** default (D3) — 예측된 active job 중 20% 이상이 SLO 초과 예상되면 reject |
| Cap 초과 시 동작 | **reject** (P2) — queue 도입은 추후 |
| `declared_max_concurrency` | `register_program` body 의 *명시 field* (P4). `dag` 분석으로 추론 X |
| Dry-run mode | **포함** (P3) — Mooncake admission_control 의 `--admission-dry-run` 과 동일 패턴 |
| 가정 수준 | **Option 2** — declared 정보만 사용. server 가 추정 안 함 (opt-in protection) |

## 4. Stage A — Predictive admission

### 4.1 Level 0 — Snapshot scaling

**철학**: "지금의 부하가 그 job 의 남은 시간 동안 유지된다" 라는 *time-invariant* 가정.

```
current_step_ms   = cost_model.estimate_step_ms(prefill=[], decode=active_call_kvs)
augmented_step_ms = cost_model.estimate_step_ms(prefill=[new_job.first_call], decode=active_call_kvs)

for each active job i (with declared remaining_calls):
    remaining_solo_i  = sum over declared remaining stages of (solo_prefill + solo_tbt × expected_output_len)
    remaining_actual_i = remaining_solo_i × (augmented_step_ms / current_step_ms)
        # batch 가 새 job 추가로 stretch 된 비율로 *남은 시간 stretch*

    final_actual_i = current_actual_elapsed_i + remaining_actual_i
    final_solo_i   = current_solo_elapsed_i  + remaining_solo_i
    predicted_slowdown_i = final_actual_i / final_solo_i
```

- 비용: cost_model 호출 2 회 + 산술
- 한계: 다른 job 종료 / 새 job 후속 call 의 batch 효과 못 봄

### 4.2 Level 2 — SLO-driven lookahead

**철학**: 1초 단위 slice 로 batch composition 의 *시간 변화* 를 직접 시뮬.

```
horizon_sec = max over active jobs of:
                (current_solo_elapsed_i + remaining_solo_i) × SLO_i / 1000
              # 가장 빠른 SLO 도달 시점 — 그 이후는 어차피 위반 확정

t = 0; slice = 1.0 sec
sim_batch = (active calls deep-copy) ∪ {new_job.first_call (prefill 시작)}

while t < horizon_sec and any job has remaining work:
    step_ms = cost_model.estimate_step_ms(prefill=sim_batch.prefill_infos, decode=sim_batch.decode_infos)
    steps_in_slice = (slice × 1000) / step_ms

    for call in sim_batch:
        if call.phase == prefill and prefill_remaining > 0:
            advance prefill (chunked) for slice
        elif call.phase == decode:
            call.decoded += steps_in_slice
            call.elapsed_actual += slice × 1000
            if call.decoded >= call.declared_output_len:
                sim_batch.remove(call)
                next_call = job_of_call.advance_to_next(stage_sequence, dag)
                if next_call: sim_batch.add(next_call)

    t += slice

for each active job i:
    predicted_slowdown_i = job_i.total_elapsed_actual / job_i.total_solo
```

- 비용: horizon=30s → 30 slice → cost_model 30 회 호출 ≈ 수십 μs
- 한계: declared `expected_output_lens` 없으면 call 종료 시점 모름 → Level 0 fallback

### 4.3 결정 함수 — `decide_admission`

```
predictions: Dict[job_id, predicted_slowdown_max]
slos:        Dict[job_id, SLO]

violations = [j for j, s in predictions.items() if s > slos[j]]
violation_ratio = len(violations) / max(1, len(predictions))

if violation_ratio > threshold (default 0.2):
    return REJECT(REASON_HALO_ADMISSION_PREDICTED, predictions, violation_ratio)
else:
    return ADMIT(predictions, violation_ratio)
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
| `ms_dev/expctl/run_experiment.py` | env var snapshot 확장. decision_log auto-route. meta 기록 |
| `ms_dev/experiments/halo_observe_only.sh` | 기본은 admission off 유지. 사용자가 명시적으로 켜는 패턴 |

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
  "mode": "level2",
  "dry_run": false,
  "decision": "reject",
  "reason": "HALO_ADMISSION_PREDICTED",
  "violation_ratio": 0.31,
  "active_jobs_total": 16,
  "violation_count": 5,
  "threshold": 0.2,
  "predicted_slowdowns": {
    "job-1": 4.2, "job-2": 5.8, "job-3": 7.1, ...
  },
  "horizon_sec": 23.5
}
```

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
                 │    if mode != off:
                 │      predictor = SnapshotPredictor or LookaheadPredictor
                 │      predictions = predictor.predict(active_jobs, new_job, cost_model)
                 │      decision    = decide_admission(predictions, slos, threshold)
                 │      log to admission_decisions.jsonl
                 │      if reject and not dry_run:
                 │          → REJECT HALO_ADMISSION_PREDICTED
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
| **PR3** | `LookaheadAdmissionPredictor` (Level 2) + 단위 테스트 | 라이브러리만 (PR2 의 통합 자동 인식) | ✅ |
| **PR4** | Stage B (concurrency cap) — Job/Registry 확장, register_program field, controller check, 단위 테스트 | Job/Registry + register_program + controller | ✅ |
| **PR5** | 검증 실험 + 분석 + 문서 갱신 (CLAUDE.md / prediction_model.md / 본 문서 §검증 결과) | 문서만 | ⏳ |

## 11. 단위 테스트 전략

### 11.1 `test_halo_admission_predictors.py` (신규)
- SnapshotPredictor: 합성 active jobs (N=3, 길이/SLO 다양) + 새 job → 예측 slowdown 정확
- LookaheadPredictor: 1초 slice 시뮬이 정확하게 진행, call 종료 시 다음 call 들어감, declared 부족 시 Level 0 fallback
- decide_admission: violation_ratio 계산 정확, threshold 경계 동작

### 11.2 `test_halo_phase1.py` (확장)
- `TestHaloController`: register_request 가 Stage A → Stage B 순서로 호출되는지 (mock predictor)
- dry-run 모드에서 reject 결정해도 admit 되는지
- Stage B: cap 초과 시 reject, finish 시 in_flight 감소
- 새 reject reasons 가 HaloRejectError 에 정상 propagate

### 11.3 통합 / e2e
- 같은 워크로드 (parallel_tool_delay λ=0.3) 에 mode=off / level0 / level2 비교
- SLO attainment rate / rejection rate / throughput 측정

## 12. 미정·주의 사항

- **Cost model fallback**: cost model 없으면 Stage A 작동 불가 → WARN + 모든 admit (lenient)
- **Declared 부족**: 새 job 또는 active job 의 declared 정보 부족 → Level 0 는 default length 가정 (보수적), Level 2 는 Level 0 fallback
- **Decision log 크기**: 매 request 마다 1 라인 → 부하 시 큰 양. background thread 로 flush (sampler 와 같은 패턴) — PR2 에서 검토
- **Concurrency 가 정확하지 않은 케이스**: client 가 declared 보다 더 많이 던지면 cap 초과 → reject. application 단 retry 책임
- **Phase 2 R3 (scheduling rebalance)**: 본 문서 범위 아님. admission 만으로 부족하면 별도 작업

## 13. 검증 결과 (placeholder — PR5 후 채움)

(PR1~PR4 코드 완성 + 사용자 실험 후 작성)
- 같은 워크로드 sweep across mode={off, level0, level2}
- SLO attainment rate 비교
- Capacity (admitted job/min) 비교
- Rejection rate 비교
- Level 0 vs Level 2 의 *추가 정확도* 가 실제로 의미 있는지 ablation

## 14. 결정 미정 (PR5 검증 후)

- Threshold default 0.2 가 적절한지 (검증 결과 보고 결정)
- Decision log size — batched flush 필요한지
- Phase 3 (R3 scheduling) 진입 시점 — admission 만으로 SLO attainment 가 부족한 부하 spectrum 보이면 시작
