SGlang 프레임워크 개선
Project Halo
1. **SLO를 맞출 수 있는 capacity로 serving system을 최대한 유지하고, 들어온 job들에 대해서는 SLO attainment rate을 maximize.**
- 이걸 위해 기존 work들은 이걸 개별 request, token level로 tracking, scheduling, orchestration 했으나, 우리는 job level로 orchestration 해함

1단계 개발 내용.

## System Design

### Interface

- 인터페이스 설계에 따라 뒤쪽이 달라질듯 (논문에서는 implementation 으로 빠질 내용)
- Goal: Provide job-level info, SLO to server (kind of job registration)
    - SLO는 Job의 solo-run 대비 slowdown ratio임. 예를들어, SLO가 5인 job 이 있다면 그 job의 end-to-end latency가 (tool calling 제외) 그 job이 서버에서 solo-run 했을때의 5배를 넘으면 안되는 것.
- Option A (LLM-program의 기본적인 DAG까지는 안다고 가정. 당연히 runtime에 변할수는 있음)
    - Program compile 기반
        - 사용자가 python script (Langgraph 등)에 LLM call(request), program_key, SLO 기록
        - Halo에 해당 program에 LLM call이 몇번인지, dag형태의 call간 관계 파싱 및 SLO 등록.
        - [질문] SGLang frontend language 필요하다면 약간 수정해서 사용 가능할지?
- Option B (LLM-program에 대해 아무것도 모른다고 가정. 그냥 call(request)가 속한 program의 id와 SLO만 알 수 있음)
    - OpenAi-API 등(혹은 지금 Agent_application에서 사용 가능할 Api)을 약간 확장 또는 수정
        - Request 보내는 필드에 SLO와 program key (id) 넣어서 함께 보낼 수 있도록 함.

  [고민] Option A가 가능한지 검토. Option A가 가능하다면, job-level scheduling이 좀 더 정교하게 될 수 있을듯. Option B는 구현이 좀 더 간단하겠으나 충분한 정보가 없을 수 있음.

### Job 자료구조 (상태구조)

- Goal: 시스템에 존재하는 job 의 자료구조 정의 (linux 의 task_struct와 흡사)
    - 구조
        - Job ID (정수)
        - Job state: Running, Queued, Admission, Rejected, Complete
        - SLO (정수)
        - Slowdown_ratio (실수)
        - Total_request_number (정수)
        - Remaining_request_number (정수)
        - [고민] 추가로 더 필요할 자료구조가?
    - 용도
        - Admission control과 scheduling에 사용
            - Admission control goal: SLO를 맞출 수 있는 capacity로 serving system을 최대한 유지 [1단계 개발 이후 진행 예정]
            - Scheduling goal: Job 들이 SLO slowdown ratio 안에서 최대한 fair하게 slowdown 되도록 serve. [1단계 개발 이후 진행 예정]

### D1. Job-level slowdown tracking

- Goal: 시스템에 존재하는 job, admit decision 할 job의 **slowdown ratio** tracking
    - Slowdown ratio: Job 마다 지금까지 진행한 토큰 생성이 solo-run  대비 얼마나 느려졌는지 주기적(실행 주기ex, 100ms)으로 기록
    - 값: 실수 (예를들면, 값이 1.5일때는 solo-run 대비 1.5배 느려진것, 1보다 작으면 더 빨리 실행된것)
    - 실행 주기마다 시스템에 존재하는, 지금 batch 에 들어가서 실행중이 아닌, running queue에 있는, call 들을 해당 call이 속한 job의 metadata에 slowdown ratio 갱신.
- 예시
    - 시스템에 job이 처음 들어온 경우
        - 처음 들어온 job의 slowdown ratio 는 SLO와 동일 (SLO 𝛕가 5라면 기본값은 5)
    - 실행 주기마다 할 일
        - **Execution history 기반의 slowdown tracking**
            - running queue에 있는 request sweep
                - 각 request의 100ms 동안의 inference 기록에서 각 request의 input length, KV cache hit rate, decode length 파악
                - solo-run latency 도출
                    - [solo-run latency 도출 방식 고민] 성능 modeling이나 profile lookup 방식 중 고민
                    - [overhead 고민] 매번 처음부터 끝까지 history를 보면서 계산하면 overhead가 좀 있을수도?
            - ex)  지난 inference 기록에서 각 request의 input length, KV cache hit rate, decode length 파악 → solo-run latency 도출
            - actual inference time / solo-run latency → job’s slowdown ratio
            - [고민] 만약 동일한 job에 속한 call이 여러개 있다면?


개발 요구사항
- 시스템에는 꼭 필요한 내용만 최소한의 수정으로 구현하려고 노력할것.
- 개발 전 SGLang의 소스코드와, 개발 명세를 면밀히 검토하고 어떻게 구현, 개발하면 될지 최대한 상세한 계획을 작성해서 아래에 기입할것
- 내 요구사항이 애매하거나, 추가적인 검토, 결정이 필요한 부분, 애매한 부분은 절대 바로 코딩시작하지말고 나에게 확실한 확인과 의견을 듣고 진행할것. (내가 위 개발 명세에 적은 고민이나 결정이 필요할것으로 예쌍되는 부분에도 너의 의견을 주고, 나와 함께 고민해서 결정한 후에 개발을 시작할것.)
- SGLang의 코딩 컨벤션을 충실히 따르되, 수정, 추가한 부분에 대해 주석으로 반드시 설명을 달것 (HALO 임도 표시).
- 추가되거나 수정되는 기능에서 예상되는 문제나 이슈가 있다면, 그것도 주석으로 설명할것 (예를들면, 성능 overhead, 정확도 문제 등), 그리고 그에 따른 디자인 변화나 수정이 필요한 부분 역시 나에게 확인 받고 진행할것.
- 개발은 최대한 객체지향적으로, policy와 mechanism을 분리할것. 예를들면, scheduling policy와 scheduling mechanism을 분리해서 구현할것. 그래야 나중에 policy를 바꿔도 mechanism은 그대로 사용할 수 있음.
- 개발은 최대한 내용을 쉽게 on/off 할 수 있도록, 각종 파라미터는 쉽게 넣을 수 있도록, 그리고 logging이나 디버깅, 모니터링을 쉽게 할 수 있도록 (가급적이면 기존 admission control이나 scheduling code를 참고해서) 구현할것.
- 수정하거나 추가하는 디렉토리, 및 파일이 있는 위치에 반드시 CLAUDE.md 파일을 만들어서, 그 파일에 개발 계획과 구현 내용, 그리고 개발하면서 생긴 이슈나 고민 등을 상세히 기록할것. 그리고 그 파일은 항상 최신 상태로 유지할것. 그리고 각 디렉토리에 있는 CLAUDE.md나 READNME.md 파일을 잘 읽어볼것.
- 현재 실험이 진행중인데, 이 소스코드를 지금 수정해도 괜찮을지 고민해볼것. 지금 실험 결과에 영향이 있으면 안됨.
- 현재 브랜치를 기준으로, 새로운 개발 브랜치를 만들어 작업할것.


[Claude 개발 계획]
========================================================================

> 작성: 2026-05-12, Claude (개발 진입 전 검토 단계)
> 본 섹션은 개발 명세 검토 + 코드베이스 스캔 결과 + 미결정 사항에 대한 의견을 정리.
> **사용자 컨펌 받기 전에는 어떠한 코드 수정도 진행하지 않음.**

## 0. 작업 환경 (코딩 시작 전 셋업)

- 현재 브랜치: `admission-control-mooncakelike`. 실험이 이 브랜치에서 진행 중이므로
  여기에 더 손대지 말고 **새 브랜치 분기** (제안: `project-halo-phase1`).
- 분기 시점: 현재 HEAD (`28797d9e6 Add admission control shell scripts`) 기준.
- **모든 Halo 코드는 off-by-default**. `--halo-enabled` 또는
  `SGLANG_HALO_ENABLED=1` 일 때만 활성. 비활성 시점에서는 진입점에서 즉시 early-return
  → 기존 실험 / admission_control 동작에 0-impact 보장.
- 기존 `admission_control` 모듈은 **건드리지 않음**. Halo는 job-level 별도 레이어로 구성.

## 1. Phase 1 범위 정리 (명세 vs 실제 작업)

명세상 1단계는 다음 3개:
- **Interface** — Job 등록 메커니즘 (Option A vs B)
- **Job 자료구조** — task_struct 유사 객체
- **D1. Job-level slowdown tracking** — 100ms tick으로 sweep

Phase 1 에서 **하지 않을 것**:
- Job-level admission control (명세 본문에도 "1단계 개발 이후 진행 예정"으로 명시)
- Job-level scheduling policy (동일)
- Option A의 program-compile / DAG 파싱
- Multi-job fairness, deadline aware 등

Phase 1 의 핵심 산출물은 결국 **"관찰만 하는 레이어"**: job을 식별/등록/추적할 수 있게 만들고,
slowdown 데이터를 수집/노출까지만. 결정 로직은 0건.

## 2. 미결정 [고민] 항목 — 내 의견 + 사용자 결정 필요

### Q1. Option A vs Option B — 인터페이스 형태

**내 의견: Phase 1은 Option B 단독으로 가는 것을 강력 추천.**

근거:
1. 명세 원칙 "최소한의 수정으로 구현" 과 가장 부합. SGLang frontend DSL 수정은 그
   자체로 큰 작업이고 Phase 1 deliverable (slowdown tracking) 과 직접 연결 안 됨.
2. Phase 1의 목표는 *job-level 관찰* 인데, 이걸 위해 필요한 정보는 `job_id + SLO`
   뿐. DAG 정보는 슬로우다운 계산에 안 쓰임 → Option A가 주는 정보 우위가 Phase 1
   에서는 의미가 없음.
3. Option B 의 메커니즘 (request에 metadata field 추가) 자체가 추후 Option A 의
   *전송 계층* 이 됨. 즉 B는 A의 부분집합이라 future-compatible.
4. 실험 application은 Agent_applications/ 의 OpenAI 호출 → 헤더/필드 한두개 추가만
   하면 끝. 기존 LangGraph 코드 거의 수정 안 함.

→ 단, **Option A 가능성 자체는 architecture 단에서 막지 않도록** 다음을 보장:
- Job 자료구조에 `dag: Optional[Any]` 자리 비워두기 (None 허용).
- JobRegistry 등록 API 가 "사후 등록" (request로부터 lazy 등록) + "사전 등록"
  (compile 시점 등록) 양쪽 다 받을 수 있게 설계.

### Q2. Solo-run latency 도출 방식

명세: `[성능 modeling이나 profile lookup 방식 중 고민]`

**내 의견: 기존 `admission_control/cost_model.py` 의 `PrefillCostModel` +
`TBTCostModel` 을 그대로 재사용.**

근거:
1. 이미 작성/테스트/피팅된 cost model 이 있음 (`ms_dev/runtime/cost_models/*.json`).
2. Solo-run latency = cost model을 "유휴 상태" 파라미터로 evaluate 한 값:
   - solo TTFT = `prefill_cost.estimate_ms(n, p)` (큐 0, 다른 부하 0 가정)
   - solo TBT  = `tbt_cost.estimate_ms(bs=1, per_req_kv=현재 KV 길이)`
3. admission_control 이 이미 동일 계산을 `REASON_TTFT_RATIO` / `REASON_TBT_RATIO`
   판단에 쓰고 있음. 같은 시스템을 별도 profile lookup 으로 중복 구현하면
   유지보수 부담만 늘어남.
4. profile-lookup 방식은 정확하지만 sparse 한 lookup table + 보간 로직이 필요해
   "최소 수정" 원칙에 안 맞음.

**주의**: 현재 commit `c3b5a075f` 에 TBT cost model 의 cliff 문제가 노트되어 있음.
즉 cost model이 부정확한 시점에 slowdown 계산도 같이 부정확해짐 → **slowdown
값의 신뢰도** 가 cost model 정확도에 종속. 이건 Phase 1 에서는 알려진 한계로
받아들이고 (decision log에 같이 dump), 모델 개선은 별도 트랙으로.

### Q3. 100ms tick 의 hook 위치 / overhead

**내 의견: scheduler_metrics_mixin 의 forward-step block에 wall-clock 체크 추가.**

배경: SGLang scheduler는 event-driven이라 별도 100ms 타이머가 없음.
forward step 단위로 metrics_mixin이 콜백 받음 (variable freq, 보통 수십 ms).

설계:
```python
# scheduler_metrics_mixin.py, step block 안
now = time.monotonic()
if self.halo_tracker is not None and now - self._halo_last_tick >= 0.1:
    self.halo_tracker.sweep(
        self.waiting_queue,
        self.running_batch.reqs,
    )
    self._halo_last_tick = now
```
- 100ms 미만 forward step: skip → forward-step 자체 추가 비용 0 (한 줄 비교).
- 100ms 이상 경과: sweep 1회.

Overhead 추정: sweep = (waiting_queue + running_batch) 길이만큼 cost model
evaluate. 동시 1000 req라도 1ms 미만. Phase 1 에서는 충분.

별도 asyncio thread는 **만들지 않음** — scheduler state는 single-threaded
assumption 하에 동작하므로 lock-free path 유지가 안전.

### Q4. Job 자료구조 - 추가로 필요한 필드?

명세 기본 필드: Job ID / state / SLO / slowdown_ratio / total/remaining req count.

**내 의견: 다음을 추가하길 권장.**

```python
@dataclass
class Job:
    # 명세 정의 필드
    job_id: int
    state: JobState              # Queued/Running/Admission/Rejected/Complete
    slo: float                   # slowdown SLO (예: 5.0)
    slowdown_ratio: float        # 현재 관측 슬로우다운
    total_request_number: int
    remaining_request_number: int

    # 추가 권장 필드 (모두 Phase 1 동작에 필요)
    first_seen_ts: float                   # job 등록 시점 (logging/디버깅)
    last_update_ts: float                  # 마지막 sweep 갱신 시점
    request_ids: set[str]                  # 이 job에 속한 rid (bounded 권장)
    slowdown_history: deque[float]         # 최근 N개 ratio (smoothing/디버깅용)
    slo_violation_count: int               # 현재까지 slowdown_ratio > slo 였던 횟수

    # 향후 Option A를 위한 placeholder (Phase 1에선 None)
    dag: Optional[Any] = None
```

`request_ids` 는 set으로 두되 완료된 req은 즉시 제거 → 메모리 bound.

### Q5. 초기 slowdown_ratio 값

명세: "처음 들어온 job의 slowdown ratio는 SLO와 동일 (SLO 𝛕가 5라면 기본값은 5)"

**의견 충돌 있음** — 사용자 확인 필요.

명세대로 초기값=SLO 로 두면:
- 의미: "아직 측정 안 됨 → 일단 worst-case (SLO 한계치) 로 둠 → fail-safe"
- Phase 2 admission 단에서: "초기값=SLO 인 job은 SLO 빠듯한 후보로 보임"

대안: 초기값 = 1.0
- 의미: "아직 측정 못 했고, 측정 전엔 slowdown 없다고 가정"
- 자연스러운 통계량 (실측 없으면 1.0 이 정상)

**내 추천: 초기값=1.0 (또는 명시적으로 NaN/None — `slowdown_ratio: Optional[float]`).**
- "아직 sweep 안 돈 job" 과 "sweep 돌았는데 1.0" 을 구분할 수 있어 디버깅에 유리.
- fail-safe 효과는 Phase 2 admission 알고리즘 안에서 `if ratio is None: use_slo`
  처럼 별도 처리하면 됨.

→ 사용자 결정 필요: **(a) 명세 그대로 SLO** vs **(b) 1.0** vs **(c) None/NaN**.

### Q6. 같은 job 안에 동시 실행 중인 request 가 여러 개일 때

명세: `[고민] 만약 동일한 job에 속한 call이 여러개 있다면?`

**내 의견: per-request slowdown을 계산하되, job 레벨로는 "max" 집계 (worst-case).**

근거:
- SLO 정의 = "job 전체 latency 가 solo-run의 N배 이내". 여러 call이 병렬로 돌면
  job latency = max(call latencies) (DAG가 linear/parallel 어떻든 가장 느린 call이
  완료돼야 다음으로 넘어감).
- 따라서 *현재 시점* job slowdown 추정치 = 진행 중 call들의 slowdown 중 최댓값.
- 평균을 쓰면 한 call이 SLO 위반인데 다른 call이 잘 돌면 마스킹됨.

추가: `slowdown_history` 에는 max값과 함께 mean도 함께 보관해 디버깅 용도로 노출.

### Q7. 기존 `admission_control` 와의 관계

**Phase 1 에서는 완전히 독립. 서로 모르는 척 동작.**

- admission_control: per-request 예측 기반 SLO 게이트 (이미 동작 중)
- Halo phase 1: per-job 슬로우다운 *관찰* (결정 없음)

둘은 같은 cost model JSON을 *읽기* 단계에서 공유하지만 (Q2 참고), 서로의 결정에
간섭하지 않음. Phase 2 에서 admission_control 을 흡수/대체할지 별도 레이어로
유지할지는 그때 결정.

### Q8. 실험 동시 진행과의 호환성

- 새 브랜치에서만 작업 → 현재 실험 브랜치 영향 0.
- 새 브랜치를 실험 환경에 가져다 쓰더라도, **`--halo-enabled` 미지정 시
  zero-impact**: scheduler에서 `if self.halo_tracker is None: return` 으로 종료.
- env var: `SGLANG_HALO_ENABLED=0` 디폴트.
- 실험 진행 중 코드를 만지지 않을 것: 이건 사용자가 새 브랜치를 *실제로 쓰기
  시작하는* 시점에 결정. 그 전까지는 admission_control 브랜치는 frozen.

## 3. 구현 모듈 구성

```
python/sglang/srt/managers/halo/                ← 신규 패키지
├── __init__.py                                 # public API: HaloTracker, Job, JobRegistry
├── CLAUDE.md                                   # 모듈 내부 문서 (코드와 함께 작성)
├── job.py                                      # Job dataclass + JobState enum
├── job_registry.py                             # JobRegistry: rid→job_id, job_id→Job
├── slowdown_tracker.py                         # 100ms sweep + per-request slowdown 계산
└── controller.py                               # HaloController: 위 모듈 조립 + on/off
```

외부 touch points (한 줄짜리 hook + 주석):

| 파일 | 무엇을 |
|---|---|
| `srt/managers/io_struct.py` | `GenerateReqInput`/`TokenizedGenerateReqInput`에 `halo_job_id: Optional[str]`, `halo_slo: Optional[float]` 추가 |
| `srt/entrypoints/openai/protocol.py` | `ChatCompletionRequest`/`CompletionRequest`에 동일 필드 추가 |
| `srt/entrypoints/openai/serving_chat.py` (+ completions) | OpenAI request → GenerateReqInput 변환 시 위 필드 forward |
| `srt/managers/schedule_batch.py::Req` | `halo_job_id`, `halo_slo` 필드 추가 |
| `srt/managers/scheduler.py::__init__` | `--halo-enabled` 시 `HaloController` 인스턴스화 |
| `srt/managers/scheduler.py::_add_request_to_queue` | admission_control 직후, JobRegistry에 req→job 매핑 등록 |
| `srt/managers/scheduler.py::handle_finish_request` (또는 retire path) | job의 remaining_request_number 감소 + 완료 시 state=Complete |
| `srt/observability/scheduler_metrics_mixin.py` | step block에 100ms tick → `halo_tracker.sweep()` |
| `srt/server_args.py` | `--halo-enabled`, `--halo-tick-interval-ms`, `--halo-aggregator` (max|avg|p95) |
| `srt/entrypoints/http_server.py::get_internal_state` | `halo_state` 블록 노출 (job 목록 + 슬로우다운) |
| `ms_dev/env.common.sh` | `SGLANG_HALO_*` env var → CLI flag 변환 (lib_server.sh::append_halo_args) |
| `ms_dev/experiments/halo_*.sh` | 시나리오별 wrapper (off / observe_only / per_slo) |
| `ms_dev/expctl/monitoring_view.py` | 라이브 패널에 `halo=ON/OFF`, 대표 job 수 / 평균 slowdown 표기 |

## 4. 로깅 / 모니터링 / on-off

기존 admission_control 패턴을 그대로 재사용:

- **Python logger**: `sglang.halo` 네임스페이스, level INFO/DEBUG.
- **JSONL decision log** (선택): `--halo-job-log <path>` 또는 run_experiment.py 가
  `<session>/halo_jobs.jsonl` 자동 라우팅. job 등록 / 상태 변화 / sweep 결과(매 100ms
  스냅샷) 를 append.
- **Prometheus metrics** (rank-0 only, TP-dedup):
  - `sglang:halo_active_jobs` (gauge)
  - `sglang:halo_job_slowdown_ratio{job_id="..."}` — 라벨 카디널리티 위험성 있음 →
    개별 job 라벨 대신 히스토그램으로 권장: `sglang:halo_job_slowdown_histogram`.
  - `sglang:halo_slo_violations_total` (counter)
- **internal-state endpoint**: `/server_info` 응답에 `halo_state` 블록 append.
- **expctl 패널**: `halo=ON jobs=12 worst_slowdown=2.8x violations=1`.

## 5. 테스트 계획

- `test/registered/halo/test_halo_phase1.py` (신규)
  - Job dataclass 라이프사이클
  - JobRegistry 등록/제거 (req→job 매핑 + 완료 시 자동 cleanup)
  - SlowdownTracker.sweep() 시뮬레이션 (가짜 req 리스트 + 가짜 cost model)
  - Aggregator (max/avg) 정확성
  - off-by-default 동작 (controller=None일 때 hook이 zero-cost)
- 통합 테스트: ms_dev/experiments/halo_observe_only.sh 로 짧은 실험 1회 돌려
  `halo_jobs.jsonl` 이 생성되고 slowdown 분포가 합리적인지 확인.

## 6. 구현 순서 (제안)

1. **브랜치 분기** + 빈 패키지 디렉토리 + CLAUDE.md 골격
2. `Job` dataclass + `JobRegistry` + 유닛테스트
3. `SlowdownTracker.sweep()` + 유닛테스트 (cost model은 admission_control 것 재사용)
4. `HaloController` 조립 + on-off 로직
5. scheduler 통합 (init + _add_request_to_queue + finish path)
6. metrics_mixin 100ms tick hook
7. server_args + CLI/env var
8. OpenAI/io_struct 메타데이터 패스스루
9. /server_info halo_state 노출
10. ms_dev wrapper + expctl 모니터링 표시
11. 통합 테스트 + 짧은 실험 + 문서 업데이트

각 단계마다 커밋 단위로 끊고, 사용자 푸시 정책 따라 push는 사용자가 직접.

## 7. 결정사항 (2026-05-12 사용자 컨펌)

| # | 항목 | 결정 |
|---|---|---|
| 1 | 인터페이스 | **Option B 단독** — request에 `halo_job_id`, `halo_slo` 필드 추가. Option A 자리는 `Job.dag: Optional` 로 비워둠 |
| 2 | 초기 `slowdown_ratio` | **SLO 값** (명세대로). 의미: 측정 전엔 worst-case로 둔다 (fail-safe) |
| 3 | 집계 방식 | **max + mean 둘 다 보관** — `slowdown_max`, `slowdown_mean` 두 필드 유지 |
| 4 | 브랜치명 | `project-halo-phase1` |
| 5 | `halo_job_id` 타입 | **`str`** (정수 ID를 쓰면 `"42"` 형식 string으로) |
| 6 | `halo_slo` 누락 | **서버 디폴트 SLO 사용** — `--halo-default-slo` (제안 디폴트: `5.0`) |
| 7 | `halo_job_id` 누락 (Halo enabled 상태) | **HTTP 400 거부** — 엄격 모드. 실험 점검에 유리 |
| 8 | 커밋/푸시 | **Claude 단계별 커밋, push는 사용자가 직접** (admission_control 작업과 동일) |

## 8. 위 결정 반영해 확정된 데이터 모델

```python
class JobState(Enum):
    QUEUED = "queued"          # 처음 등록만 됨, 아직 sweep 안 됨
    RUNNING = "running"        # 1개 이상 request가 진행 중
    COMPLETE = "complete"      # remaining_request_number == 0
    REJECTED = "rejected"      # Phase 2 admission이 거절한 경우 (Phase 1엔 미사용)

@dataclass
class Job:
    job_id: str                          # Option B request field
    slo: float                           # 클라이언트 지정 or 서버 디폴트
    state: JobState = JobState.QUEUED
    slowdown_max: float = field(init=False)   # __post_init__: = self.slo
    slowdown_mean: float = field(init=False)  # __post_init__: = self.slo
    total_request_number: int = 0
    remaining_request_number: int = 0
    first_seen_ts: float = field(default_factory=time.monotonic)
    last_update_ts: float = field(default_factory=time.monotonic)
    request_ids: set[str] = field(default_factory=set)
    slowdown_history: deque[tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=64))   # (max, mean) 쌍
    slo_violation_count: int = 0
    dag: Optional[Any] = None            # Option A 자리 (Phase 1엔 None)
```

## 9. 확정된 코딩 작업 순서

1. ✅ `project-halo-phase1` 브랜치 분기
2. ✅ `python/sglang/srt/managers/halo/__init__.py` + 모듈 CLAUDE.md 골격
3. ✅ `halo/job.py` — `Job` dataclass + `JobState` enum
4. ✅ `halo/job_registry.py` — `JobRegistry` (rid↔job 매핑, scheduler 호출)
5. ✅ `halo/slowdown_tracker.py` — `SlowdownTracker.sweep()` (cost model 재사용)
6. ✅ `halo/controller.py` — `HaloController` 조립 + on/off
7. ✅ `server_args.py` — `--halo-enabled`, `--halo-default-slo`, `--halo-tick-interval-ms`, `--halo-decision-log` 등
8. ✅ `io_struct.py` / `protocol.py` / `serving_chat.py` / `serving_completions.py` / `schedule_batch.py::Req` — 메타데이터 패스스루
9. ✅ `scheduler.py::__init__` — controller 초기화
10. ✅ `scheduler.py::_add_request_to_queue` — JobRegistry 등록 (job_id 없으면 400 reject)
11. ✅ `scheduler.py` finish path — remaining_request_number 감소 + state 갱신
12. ✅ `scheduler_metrics_mixin.py` — 100ms tick → `halo_controller.tick()`
13. ✅ `/server_info` halo_state 노출
14. ✅ `test/registered/halo/test_halo_phase1.py` 작성
15. ✅ `ms_dev/` 통합 (env vars, lib_server.sh, experiments wrapper, expctl monitoring)
16. ✅ 짧은 통합 실험 1회 + 결과 캡처
17. ✅ 최종 문서 갱신

> 각 단계 끝나면 커밋. push 는 사용자가 직접.