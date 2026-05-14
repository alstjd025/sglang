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

  Option A와 B를 일단 모두 구현. 개발 Phase 2에서 job 의 미래 상태 (call 수, DAG구조 등) 이 어느정도 필요할 것으로 예상됨. 
  
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

> **분리된 문서**:
> - **클라이언트/API 관점** (endpoint, request 필드, curl/LangChain 예제, 응답
>   status code 표 등): [halo_api_reference.md](halo_api_reference.md)
> - **모듈 내부 디자인** (touchpoints, 책임 경계, 클래스 표):
>   [`python/sglang/srt/managers/halo/CLAUDE.md`](../../python/sglang/srt/managers/halo/CLAUDE.md)
> - **검증/벤치마크**: [verify/CLAUDE.md](verify/CLAUDE.md)
>
> 이 파일은 **프로젝트 전체 디자인 결정 + 진행 로그**가 남는 곳. API 세부는
> 별도로 분리해서 길이를 통제.

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

**갱신 (2026-05-12, 사용자 명세 재정의)**:
> "Option A와 B를 일단 모두 구현. Phase 2 admission control 시 job 의 미래 상태
> (call 수, DAG 구조 등) 이 필요할 것으로 예상됨."

**최종 결정: A + B 둘 다 구현. B 는 transport 계층, A 는 그 위의 사전 등록 계층.**

두 옵션은 배타적이지 않고 **계층적**:

```
Layer 1 (Option B): 매 LLM request 의 body 에 {halo_job_id, halo_slo} 동봉
                    → transport. job 단위 식별 + per-call SLO 전달.
                    → 이미 Phase 1 에서 구현 완료.

Layer 2 (Option A): POST /halo/programs 한 번 호출하여 미래 정보 동봉
                    {job_id, slo, total_calls, stage_sequence, dag, ...}
                    → control plane. Phase 2 admission/scheduling 결정 기반.
                    → Phase 1 에서는 등록만 받고 *저장*. 활용은 Phase 2.
```

**핵심**: 사용자가 Option A 미사용 시 → 그대로 Option B 동작 (lazy-create).
사용자가 Option A 사용 시 → 동일 transport 위에서 Job 객체가 더 풍부한 정보 보유.

Phase 1 deliverable (slowdown 관찰) 만 두고 보면 A 의 추가 정보는 *현재* 활용 안 됨.
그러나 Phase 2 lookahead admission ("앞으로 K call 더 올 거니까 지금 받으면 SLO 못
맞춤") 에 필요. 명세대로 **Phase 1 에서 A 도 미리 깔아두자**.

> 자세한 디자인은 §11 "Option A 디자인" 참고.

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
- 통합 테스트: ms_dev/experiments/halo_base.sh 로 짧은 실험 1회 돌려
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
| 1 | 인터페이스 | ~~Option B 단독~~ → **A + B 둘 다** (2026-05-12 명세 갱신). B 는 transport, A 는 그 위의 사전 등록 계층. 자세한 디자인은 §11. |
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

## 9. 확정된 코딩 작업 순서 + 진행 상황

> ✅ = 완료, ⏳ = 사용자가 직접 진행할 단계.

1. ✅ `project-halo-phase1` 브랜치 분기
2. ✅ `python/sglang/srt/managers/halo/__init__.py` + 모듈 CLAUDE.md 골격
3. ✅ `halo/job.py` — `Job` dataclass + `JobState` enum
4. ✅ `halo/job_registry.py` — `JobRegistry` (rid↔job 매핑, scheduler 호출)
5. ✅ `halo/slowdown_tracker.py` — `SlowdownTracker.sweep()` (cost model 재사용)
6. ✅ `halo/controller.py` — `HaloController` 조립 + on/off
7. ✅ `server_args.py` — `--halo-enabled`, `--halo-default-slo`, `--halo-tick-interval-ms`, `--halo-job-log`, cost model paths
8. ✅ `io_struct.py` / `protocol.py` / `serving_chat.py` / `serving_completions.py` / `schedule_batch.py::Req` / `tokenizer_manager.py` — 메타데이터 패스스루
9. ✅ `scheduler.py::__init__` → `init_halo()`
10. ✅ `scheduler.py::_add_request_to_queue` — `_halo_register_or_abort` (400 reject)
11. ✅ `scheduler_output_processor_mixin.py` finish path — `_halo_on_request_finished`
12. ✅ `scheduler.py` event loops — 100ms wall-clock-gated `_halo_maybe_tick()`
13. ✅ `/server_info` halo_state 노출
14. ✅ `test/registered/halo/test_halo_phase1.py` — 19 tests, all green (CPU stage-a)
15. ✅ `ms_dev/` 통합 (env vars, lib_server.sh, experiments wrapper, expctl monitoring)
16. ⏳ 짧은 통합 실험 1회 + 결과 캡처 (사용자 실행 필요 — `source ms_dev/experiments/halo_base.sh && python3 ms_dev/expctl/server_run_experiment.py --mode single`)
17. ⏳ Agent_applications 클라이언트 측에 `halo_job_id` / `halo_slo` 필드 추가 (Phase 1 strict mode → 빠지면 400)

## 10. 커밋 히스토리 (전체, 시간 역순)

### `sglang` repo · 브랜치 `project-halo-phase1` (분기: `28797d9e6` from `admission-control-mooncakelike`)

**최신 (2026-05-15) — Phase 2 admission refactor round**:
```
b8ff4a984 refactor(halo): admission Phase 2 — job-level, snapshot-only, per-job stretch  ← §21 갱신
3b33beda9 docs(halo): refresh §10 commit history (29→30) — Phase 2 admission round
5fb6c3d9f add(halo): Phase 2 admission control — Stage A (Level 0+2) + Stage B
eb52d10b4 docs(halo): refresh §10 commit history (26→29) — Step Cost Model round
```

**Step Cost Model round (2026-05-14)**:
```
672280e4b feat(halo): adopt split form as default — observe_only wrapper wires it up
dc39b5b3e add(halo): Step Cost Model — unified + split forms, sampler, fit tool  ← §15/§17
a83b1bade docs(halo): refresh §10 commit history (5→26) + new §19 status snapshot
```

**2026-05-13 (compaction 직전까지)**:
```
2ac0fed5d add(halo): jsonl event=job_complete row on explicit signal + quiescent path
e85c4fd61 docs(halo): §18 follow-up on TBT cost-model cliff
3181ae6e9 docs(halo): halo_job_done + quiescent fallback + decision log
03a334bc6 add(halo): split monitor labels — halo_rej_no_id / halo_rej_not_reg + halo_ prefix
c03797057 add(halo): explicit halo_job_done body field + quiescent safety net  ← A 핵심
f25fd41ea fix(halo): don't mark job COMPLETE mid-chain (rolled back later by c03797057)
6d6074623 fix(halo): throttle JSONL job log to 10s
cbf717a7d add(halo): GET /halo/status — lightweight server-config probe
181cfb187 fix(halo): mark /health internal inference with halo_bypass=True
443d1add4 fix(halo): forward halo_* fields from TokenizedGenerateReqInput into Req()
b1afff118 docs(halo): split client-facing API surface into halo_api_reference.md
d378cdde3 fix(halo): bypass admission gate for server-internal traffic (warmup)
df60ccda2 docs(halo): correct verify guide
67810b236 add(halo): Prometheus metrics + live runtime state in expctl panel
834ea7356 add(halo): Phase 1 verification harness — microbench + e2e_smoke
99cd25bd0 refactor(halo): clarify role boundaries — admit_to_job + JobAdmissionResult
6c4720d73 docs(halo): Option A end-to-end progress + strict-mode guidance
18bf66a8b add(halo): Option A IPC + HTTP — POST /halo/programs end-to-end
802c5bf20 add(halo): Option A core — Job extension, register_program, strict-mode admission
2c3ebee91 docs(halo): A+B 둘 다 구현 방향으로 명세 갱신
```

**초기 (2026-05-12)**:
```
f321ff493 docs(halo): Phase 1 progress + ms_dev/CLAUDE.md Halo section
c72f90286 add(halo): Phase 1 ms_dev tooling — env vars, launcher, experiments, expctl panel
37200978d test(halo): Phase 1 unit tests — Job, Registry, Tracker, Controller, factory
3334bf55e add(halo): Phase 1 scheduler integration + request-metadata passthrough
acdfc411f add(halo): Phase 1 module skeleton — Job, JobRegistry, SlowdownTracker, HaloController
e496ea707 docs(halo): Phase 1 plan + decisions in ms_dev/halo_dev/CLAUDE.md
```

총 **32 commits** (30 + Phase 2 admission refactor 2). Push 는 사용자가 직접 (`git push origin project-halo-phase1`).

### `Agent_applications` repo · 브랜치 `project-halo-phase1-client` (분기: `4d3cc51` from `main`)

```
0d596e8 add(halo-client): send halo_job_done=true on the chain's last call
b5e5e5f fix(halo-client): record HALO_* rejections in metrics.csv
d2058ef add(halo): Project Halo Phase 1 client wiring across all workloads
```

총 **3 commits**. Push 는 `bash safe_push.sh` 또는 `git push -u origin project-halo-phase1-client`.

총 5개 커밋 (+ Halo 진행상황 doc 커밋 1개 = `f321ff493`). push 는 사용자가 직접
(`git push -u origin project-halo-phase1`).

> 각 단계 끝나면 커밋. push 는 사용자가 직접.

========================================================================
# Option A 확장 — 2026-05-12 갱신 후 추가
========================================================================

## 11. Option A 디자인 (사전 등록 계층)

### 11.1 개념

**Option A = Option B 의 *superset*. transport 는 그대로, control plane 만 추가.**

```
┌────────────────────────────────────────────────────────────────────┐
│  Client (LangGraph run_job 진입)                                   │
│                                                                    │
│  ① POST /halo/programs  ← Option A — chain 시작 전 1회             │
│     body: {job_id, slo, total_calls, stages, dag, ...}             │
│                                                                    │
│  ② POST /v1/chat/completions  ← Option B — 매 call                 │
│     body: {model, messages, halo_job_id, halo_slo, ...}            │
└────────────────────────────────────────────────────────────────────┘
                              ↓
┌────────────────────────────────────────────────────────────────────┐
│  SGLang server                                                     │
│                                                                    │
│  ① http_server → TokenizerManager (zmq) → Scheduler                │
│     → HaloController.register_program(job_id, slo, **info)         │
│     → JobRegistry: Job 객체 미리 생성 + 미래 정보 저장             │
│                                                                    │
│  ② tokenizer → scheduler → _add_request_to_queue                   │
│     → _halo_register_or_abort                                      │
│       case A: job_id 가 사전 등록 → 그 Job 의 rid 추가              │
│       case B: job_id 가 미등록 → lazy-create (Option B 동작)        │
└────────────────────────────────────────────────────────────────────┘
```

Phase 1 에서는 Option A 의 미래 정보를 *저장만* 하고, slowdown 계산엔 미사용.
Phase 2 admission/scheduling 알고리즘이 이 정보를 활용.

### 11.2 신규 HTTP endpoint

`POST /halo/programs` — 신규 top-level route (admission_control 에는 대응 endpoint
없음. Halo 만의 control plane).

**전체 schema / status code / 응답 본문 형태**: [halo_api_reference.md](halo_api_reference.md)
의 "`POST /halo/programs`" 섹션 참고. 여기서는 *내부 디자인 결정* 만 기록:

- 외부 routing: `http_server.py @app.post("/halo/programs")`
- IPC: TokenizerManager `register_halo_program` communicator → Scheduler
  `register_halo_program` 핸들러 → `HaloController.register_program`
- body 크기 상한: 16 KiB (Q14, http_server.py 에서 강제)
- status code 매핑: fresh=200 / dup=409 / disabled or malformed=400 (Q11)

### 11.3 신규 IO struct (io_struct.py)

```python
# HALO: Option A — see managers/halo/CLAUDE.md.
@dataclass
class HaloRegisterProgramReqInput:
    job_id: str
    slo: float
    total_calls: Optional[int] = None
    stage_sequence: Optional[List[str]] = None
    expected_input_lens: Optional[List[int]] = None
    expected_output_lens: Optional[List[int]] = None
    dag: Optional[Dict[str, Any]] = None

@dataclass
class HaloRegisterProgramReqOutput:
    registered: bool
    job_id: str
    reason: Optional[str] = None         # set when registered=False
    active_jobs: int = 0
    existing: Optional[Dict[str, Any]] = None
```

### 11.4 Job 구조 확장

```python
@dataclass
class Job:
    # ... 기존 필드 ...

    # HALO: Option A — pre-registered program metadata.
    # All optional; None 이면 Option B (lazy-create) job.
    total_calls_expected: Optional[int] = None
    stage_sequence: Optional[List[str]] = None
    expected_input_lens: Optional[List[int]] = None
    expected_output_lens: Optional[List[int]] = None
    dag: Optional[Dict[str, Any]] = None      # ← 기존 dag: Optional[Any] 자리 강화
    from_program: bool = False                # True 면 사전 등록된 Job
```

`Job.to_dict()` 도 이 필드들을 dump 하도록 확장.

### 11.5 JobRegistry 확장

```python
def register_program(
    self,
    job_id: str,
    slo: float,
    *,
    total_calls: Optional[int] = None,
    stage_sequence: Optional[List[str]] = None,
    expected_input_lens: Optional[List[int]] = None,
    expected_output_lens: Optional[List[int]] = None,
    dag: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Job]:
    """
    Pre-register a job before any LLM request arrives.

    Returns (newly_registered: bool, job: Job).
    - True  → fresh registration (200 OK)
    - False → job_id already exists (409 Conflict by default)
    """
    existing = self._jobs.get(job_id)
    if existing is not None:
        return False, existing
    job = Job(
        job_id=job_id,
        slo=slo,
        total_calls_expected=total_calls,
        stage_sequence=stage_sequence,
        expected_input_lens=expected_input_lens,
        expected_output_lens=expected_output_lens,
        dag=dag,
        from_program=True,
    )
    self._jobs[job_id] = job
    return True, job
```

기존 `record_admission()` 도 수정:
- job_id 가 이미 존재하고 `from_program=True` 이면 → 기존 Job 의 slo 우선. request 의
  `halo_slo` 와 다르면 WARN 로그 (값은 무시).
- 미존재면 → 현재 동작 그대로 (lazy-create, `from_program=False`).

### 11.6 HaloController API

```python
def register_program(
    self,
    job_id: str,
    slo: float,
    **future_info,
) -> HaloRegisterProgramReqOutput:
    """Public entry for Option A pre-registration."""
    if not self.config.enabled:
        return HaloRegisterProgramReqOutput(
            registered=False, job_id=job_id, reason="HALO_DISABLED")
    fresh, job = self.registry.register_program(job_id, slo, **future_info)
    if not fresh:
        return HaloRegisterProgramReqOutput(
            registered=False, job_id=job_id,
            reason="JOB_ID_ALREADY_REGISTERED",
            existing=job.to_dict())
    if self._log is not None:
        self._log.write({"ts": time.monotonic(), "event": "register_program",
                         "job": job.to_dict()})
    return HaloRegisterProgramReqOutput(
        registered=True, job_id=job_id,
        active_jobs=len(self.registry.active_jobs()))
```

### 11.7 IPC 경로 (TokenizerManager ↔ Scheduler)

`flush_cache` / `get_internal_state` 패턴 그대로 mirror:

1. `http_server.py` 에 `@app.post("/halo/programs")` 추가.
2. Handler 는 body → `HaloRegisterProgramReqInput` → `tokenizer_manager.register_halo_program(req)` 호출.
3. TokenizerManager: zmq 로 scheduler 에 forward, await response.
4. Scheduler 의 dispatch table 에 `HaloRegisterProgramReqInput` 핸들러 등록 (event loop 의 `process_input_requests` 라우팅).
5. Scheduler 가 `self.halo_controller.register_program(...)` 호출, 결과를 `HaloRegisterProgramReqOutput` 으로 tokenizer 에 반환.
6. Tokenizer 가 HTTP response 로 변환.

### 11.8 클라이언트 (Agent_applications) 측 통합

코드 예제와 단계별 wiring 가이드 (LangChain `ChatOpenAI` 의 `extra_body`,
`run_job` 진입 시 `register_halo_program` 호출 등) 는
[halo_api_reference.md](halo_api_reference.md) 의 "Client integration recipes"
및 "Agent_applications" 섹션 참고.

> Agent_applications 변경은 SGLang server 변경 commit/merge 후 별도 PR. Phase 1
> 안에서는 server-side 만 완료하고 사용자가 직접 client wiring.

## 12. Option A 구현의 예상 어려움 + 위험

### 12.1 LoC 추정

| 작업 | LoC | 난이도 | 위치 |
|---|---|---|---|
| `Job` 구조 확장 | ~30 | 낮음 | `halo/job.py` |
| `JobRegistry.register_program()` | ~30 | 낮음 | `halo/job_registry.py` |
| `HaloController.register_program()` | ~25 | 낮음 | `halo/controller.py` |
| `record_admission()` 사전 등록 인지 | ~15 | 낮음 | `halo/job_registry.py` |
| 신규 IO struct | ~40 | 중 | `srt/managers/io_struct.py` |
| TokenizerManager handler | ~40 | 중 | `srt/managers/tokenizer_manager.py` |
| Scheduler dispatch + handler | ~40 | 중 | `srt/managers/scheduler.py` |
| HTTP endpoint | ~50 | 중 | `srt/entrypoints/http_server.py` |
| 단위 테스트 (register / fallback / 409) | ~80 | 낮음 | `test/registered/halo/` |
| 통합 smoke test (curl + LLM call) | ~30 | 낮음 | manual or in test |
| **합계** | **~380** | | |

기존 Phase 1 코드 (~830 LoC) 대비 ~45% 증가. **substantial 하지만 모든 항목이 기존 패턴 mirror**.

### 12.2 주요 위험 & 완화

#### W1. HTTP 요청 도착 순서 race
**문제**: `register_program` HTTP 요청과 첫 LLM request 가 거의 동시에 출발하면, 서버에
LLM request 가 먼저 도착할 수도 있음 (둘 다 비동기).

**시나리오**:
1. Client: register_program 보냄 (TCP A)
2. Client: 곧바로 chat.completions 보냄 (TCP B)
3. Server: B 가 먼저 도착 → `record_admission` 이 lazy-create (`from_program=False`)
4. Server: A 가 도착 → `register_program` 이 보니 이미 존재 → 409? 아니면 갱신?

**완화 옵션**:
- (a) 클라이언트 contract: register_program 200 받은 후 LLM 호출 — 가장 안전하지만 매번 RTT 1회 추가
- (b) 서버 측 idempotent upgrade: lazy-create Job 에 `from_program=False` 인 상태면
  나중 도착한 register_program 이 "사전 정보 보강" 으로 처리 (slo 충돌은 W3 정책 따름)
- **권장: (a) + (b) 병행**. Contract 는 (a) 로 권장. 그러나 race 발생 시 (b) 동작도
  지원해서 안전성 확보.

#### W2. 동일 job_id 재등록 정책
- (a) HTTP 409 (현재 권장)
- (b) Idempotent — 두 번째가 새 값으로 덮어쓰기, WARN 로그
- (c) Silently ignore

권장 **(a)** 인데 W1 의 race 와 충돌 가능성 있음 → (b) 도 검토 가치 있음.
사용자 확인 필요 (Q11).

#### W3. SLO 충돌 (사전 등록 vs request body)
- register_program 에서 slo=5.0 등록, 이후 chat.completions 에 halo_slo=3.0 도착.
- 옵션:
  - (a) 사전 등록 우선, request 의 halo_slo 는 WARN 후 무시
  - (b) request 우선 (last-write-wins)
  - (c) 모두 reject (HTTP 400)
- **권장: (a)**. program 이 contract.

#### W4. Multi-instance future-proofing
Phase 2 에서 PD/multi-instance 가 되면, program registry 도 cross-scheduler 동기화
필요. 현재 단일 scheduler 에 등록되므로 PD/DP-attn 등 멀티 rank 시 program 이 한
rank 에만 알려질 수 있음.

**완화**: 현재 `_add_request_to_queue` 는 rank 0 (또는 attn_tp_rank=0) 만 처리하는
가정에 기대고 있는데, Halo 도 동일 가정 → 단일 scheduler 의 controller 만 program
인지. 다른 rank 는 register_program 결과를 신경 안 써도 됨 (어차피 그 rank 가
admission 안 함).

**위험**: scheduler 가 멀티프로세스로 fork 되는 경우 (DP-attn N=4 등). program
등록을 받은 scheduler 만 알고 있고 다른 ranks 는 모름 → request 가 다른 rank 로
가면 lazy-create. 이건 Phase 2 multi-instance Conductor 작업 때 통합.

→ Phase 1 에서는 **NULL disaggregation + single scheduler 만 보장**. 다른 모드면
register_program 이 200 OK 를 주되 ("등록은 받음") 실제로는 무시 / WARN.

#### W5. Program 객체의 빈-사용 GC
- 사용자가 program 등록만 하고 LLM 호출은 하나도 안 보냈을 때, Job 이 계속 메모리에
  남는다.
- 현재 `gc_completed()` 는 state == COMPLETE 인 것만 정리.
- **해결**: program 등록 후 일정 시간 (예: `--halo-program-idle-timeout` 디폴트 300s)
  동안 request 가 0개면 GC. WARN 로그.

#### W6. DAG 필드 크기
- 자유 형식 dict → 너무 크면 안 됨.
- **해결**: server 측 body 파싱 시 16KB hard cap. 초과 시 HTTP 400.

#### W7. strict-mode 와 사전 등록 강제
- 현재: `--halo-enabled` 만 ON → job_id 누락 시 400 (사전 등록 무관).
- 추가 옵션: `--halo-require-program-registration` (디폴트 OFF). ON 이면 사전 등록 안
  된 job_id 도 400. 실험 점검에 유용한 추가 엄격 모드.

### 12.3 검증 포인트 (코딩 시작 전 확인 필요)

1. **`ChatCompletionRequest` Pydantic `extra=` 기본 동작**: forbid 면 langchain
   `model_kwargs` 가 reject 됨. 현재 protocol.py 에 명시적 `extra=` 없으면 기본
   `ignore` → 통과. 사용자 실험 직전에 grep 한 번이면 끝.
2. **TokenizerManager IPC pattern 정확한 위치**: 기존 `flush_cache`, `set_internal_state`
   가 어떻게 구현돼 있는지 보고 그대로 mirror.
3. **http_server.py 의 route 등록 컨벤션**: Halo 가 OpenAI namespace 가 아니므로
   `/halo/programs` 라는 top-level path 가 SGLang 다른 internal endpoint 들과 충돌
   안 하는지.

## 13. 새 결정 사항 (Option A 추가에 따라, 2026-05-12 사용자 컨펌)

| # | 항목 | 결정 |
|---|---|---|
| Q9 | API endpoint 경로 | **`POST /halo/programs`** (top-level neutral path) |
| Q10 | 사전등록 SLO vs request `halo_slo` 충돌 | **사전 등록 우선 + WARN 로그**. request 의 `halo_slo` 는 ignore |
| Q11 | 동일 job_id 재등록 | **HTTP 409 Conflict** (명시적). race 발생 시 client 가 처리 |
| Q12 | 미사전등록 job_id 의 LLM request | **항상 reject (HTTP 400)** — Halo enabled 면 program 사전등록 *필수*. lazy-create fallback 제거 |
| Q13 | Program idle GC | **새 옵션 `--halo-program-idle-timeout` 도입, 디폴트 300s**. register 후 N초간 request 0개면 GC + WARN |
| Q14 | DAG 필드 형태 | **`Dict[str, Any]` (JSON-able) + 16KB 상한**. schema 강제 안 함 |
| Q15 | Agent_applications 클라이언트 통합 시점 | **SGLang server 변경 완료 후 별도 PR**. Phase 1 안에서는 server-side 까지 |
| Q16 | register_program JSONL 로깅 위치 | **기존 `halo_jobs.jsonl` 에 `event` 필드 추가**. event ∈ {register_program, sweep, complete} |

### Q12 의 의미 — Phase 1 기존 동작 변경

Q12 = "항상 reject" 결정에 따라 **Phase 1 의 기존 lazy-create 동작이 변경**:

- 기존 (B 단독): Halo enabled + `halo_job_id` 있음 → 그 자리에서 Job 생성 (lazy-create).
- 신규 (A+B): Halo enabled + `halo_job_id` 있음 + **사전등록 안 됨** → HTTP 400 reject.
- 즉 클라이언트는 `register_program` 200 받은 후에 LLM call. 안 그러면 모든 call 이 400.

**구현 영향**:
- `JobRegistry.record_admission()` 의 "create new Job if not exists" 분기를 **400 reject**
  분기로 변경. 분기 명만 바꾸면 됨 (~10 LoC).
- `HaloController.register_request()` 의 reject 사유에 새 reason `HALO_PROGRAM_NOT_REGISTERED`
  추가.
- scheduler 의 `_halo_register_or_abort` 의 HTTP message 도 갱신.
- 기존 단위 테스트 중 lazy-create 동작을 검증하던 1~2개는 변경 필요 (reject 동작으로).

**운용 영향**:
- 실험 진행 전 Agent_applications 의 client wiring (Q15) 이 *반드시 먼저* 완료돼야 함.
  안 그러면 Halo enabled 한 실험은 모두 400 폭주.
- server-side 작업이 끝난 직후 실험을 못 돌리고, Agent_applications PR 까지 끝나야 비로소 검증 가능.
- 이건 Phase 1 deliverable 의 *통합 검증 시점* 이 클라이언트 PR 뒤로 미뤄진다는 뜻.
  설계상 의도된 거라면 그대로 진행.

위 영향이 의도와 맞는지 코딩 시작 전에 마지막으로 한 번 더 확인 권장 (혹시 "Option A
미사용 시에는 B fallback 그대로 두는" 더 부드러운 운용을 원하시면 Q12 를 (a) 또는 (b)
로 재고).

## 14. 갱신된 작업 순서 (Option A 포함)

Phase 1 은 §9 의 1~15 완료된 상태. **Option A 추가 작업 + Q12 영향 반영**:

1. ✅ §13 사용자 컨펌 완료 (Q9~Q16)
2. ✅ `halo/job.py` — `Job` 구조 확장 (total_calls_expected 외 5필드 + from_program + is_idle helper)
3. ✅ `halo/job_registry.py` — `register_program()` 메서드 추가 + **`record_admission` 의 lazy-create 분기를 `AdmissionResult(admit=False, reason=PROGRAM_NOT_REGISTERED)` 으로 변경 (Q12)** + `gc_idle_programs()` (Q13)
4. ✅ `halo/controller.py` — `register_program()` public API + `HaloRegisterProgramResult` + tick에 idle GC 통합 + reject reason 상수
5. ✅ `io_struct.py` — `HaloRegisterProgramReqInput`/`Output`
6. ✅ `tokenizer_control_mixin.py` — `_COMMUNICATOR_SPECS` 에 추가, `register_halo_program(obj)` async helper
7. ✅ `scheduler.py` — dispatch table 등록 + `register_halo_program` 핸들러 + reject reason 별 message
8. ✅ `http_server.py` — `POST /halo/programs` route (16KB cap, status code 매핑)
9. ✅ `server_args.py` — `--halo-program-idle-timeout-seconds` (디폴트 300s)
10. ✅ 단위 테스트 — 31개 그린 (register fresh/409/SLO conflict/idle GC/strict reject/controller wiring)
11. ⏳ 통합 smoke test (실제 server 띄우고 curl POST /halo/programs → register 200 → LLM call → 200, 미등록 LLM call → 400) — 사용자 실험 시 검증
12. ✅ `ms_dev/halo_dev/CLAUDE.md` + `python/sglang/srt/managers/halo/CLAUDE.md` Option A 반영
13. ⏳ (별도 PR) Agent_applications client wiring: register_halo_program helper + halo_job_id/halo_slo passthrough in ChatOpenAI

각 단계 끝나면 커밋. push 는 사용자.

## 15. Option A 추가 커밋 히스토리

```
18bf66a8b add(halo): Option A IPC + HTTP — POST /halo/programs end-to-end
802c5bf20 add(halo): Option A core — Job extension, register_program, strict-mode admission
2c3ebee91 docs(halo): A+B 둘 다 구현하는 방향으로 명세 갱신, Option A 디자인/위험/결정사항 추가
```

Option A 작업 전체: 3 커밋, ~720 LoC (코드) + ~430 LoC (문서/테스트). Phase 1 전체
(Phase 1 만 = ~830 LoC) 대비 86% 증가 — 사전 추정 (~380 LoC) 대비 두 배. 주된 차이는
SLO conflict / 16KB cap / 단위 테스트 추가 코드.

다음 (필요 시): Agent_applications 측 client wiring (Q15) → 통합 실험.

## 16. Mental model 보정 — Halo 는 SGLang 안의 *서브시스템*

§1, §11 에서 Phase 1 의 Halo 를 "관찰만 하는 레이어" 로 정의했었음. 하지만 사용자
컨펌 + 실제 구현을 마치고 보니 **더 정확한 표현은**: Halo 는 SGLang 안에 도입된
*서브시스템* 이고, 그 안에 **세 가지 새 책임**이 있음.

| Role | Phase 1 상태 | Phase 2 예정 |
|---|---|---|
| **R1. Job-level slowdown tracking** | ✅ 활성 (SlowdownTracker.sweep) | 더 정교한 cost-model / per-stage tracking |
| **R2. Job-level admission gate** | ✅ 활성 (strict validation: job_id 누락 / 미등록 → 400) | predictive lookahead — R1 의 job 상태를 읽어 admit/reject 결정 |
| **R3. Job-level scheduling policy** | ❌ Phase 2 | fair-slowdown scheduling |

### `admission_control/` 과의 책임 분리

| | 동작 단위 | 결정 시점 | 결정 근거 |
|---|---|---|---|
| `managers/admission_control/` | per-request | request 도착 | 예측된 TTFT/TBT vs SLO (Mooncake §4.1) |
| Halo R2 (이 모듈) | per-job | request 도착 (Phase 1) / + 주기 sweep (Phase 2) | job 의 누적 slowdown vs SLO 비율 |

둘은 `scheduler._add_request_to_queue` 안에서 **직렬로** 작동:

```
_abort_on_queued_limit          # 용량 게이트
  ↓
_abort_on_predicted_slo_violation # admission_control: per-request 예측
  ↓
_halo_register_or_abort         # Halo R2: per-job 검증
  ↓
waiting_queue.append(req)
```

### Rename (2026-05-12)

§7 직후 mental model 보정을 반영해 다음 rename 적용:

- `JobRegistry.record_admission` → **`admit_to_job`** (이름 자체가 admission 결정임을 분명히 함)
- `AdmissionResult` → **`JobAdmissionResult`** (admission_control 의 `AdmissionDecision` 과 구분)
- `result.admit` → **`result.admitted`** (과거형 → "결정 났다" 가독성)

CLAUDE.md (이 파일 + `managers/halo/CLAUDE.md`) 에 3-role 도식 + 경계 표를 명시.

### 향후 (Phase 2) 자연스러운 통합

Phase 2 에서 admission decision 알고리즘이 들어올 때:
- Halo R2 의 `admit_to_job` 가 "registered/not-registered" 만 보던 단순한 게이트에서,
  R1 의 job 상태 (`slowdown_max`, `slowdown_mean`, `total_calls_expected` 등) 를 읽어
  *예측 기반* admission 도 함께 결정하도록 확장.
- `admission_control/` 은 그대로 두되 (Mooncake 스타일 per-request 게이트), Halo R2 가
  그 *위에 추가* 되는 형태로 작동.

이 디자인이라면 Phase 2 에서 Halo 모듈 외부 코드 변경 거의 없음 (`scheduler.py` 의 hook
한 줄, server_args 의 새 플래그 정도).

## 17. Job 종료 시그널 — explicit `halo_job_done` (2026-05-12 결정)

### 배경: chain 중간 조기 COMPLETE 버그

Slowdown 실험 (`260512_2128_slowdown_test_lambda_0p3`) 에서 *parallel_tool_delay* 워크로드의 44 reject (`HALO_PROGRAM_NOT_REGISTERED`). 원인:
- `Job.on_request_completed` 가 `remaining_request_number==0` 시 즉시 COMPLETE
- parallel_tool_delay 의 *tool_delay 3-10s* 가 5s retain 윈도우 초과 → `gc_completed` GC
- 다음 round 의 call 도착 → registry 에서 미등록 → 400

### 1차 fix 의 한계 — total_calls_expected 의존

f25fd41ea 에서 `total_calls_expected` 만큼 admit 됐을 때만 COMPLETE 로 가드. 그러나:
- 동적 DAG (conditional branch, runtime-결정 chain) 에서 정확한 count 미리 모름
- 과소 추정: 9번째 call → reject (버그 재발)
- 과대 추정: 영영 COMPLETE 안 됨 → idle GC (300s) 까지 점유

### 최종 결정 (c03797057): explicit signal

| | 동작 |
|---|---|
| **client 책임** | chain 의 마지막 LLM call 의 body 에 `halo_job_done: true` 동봉 |
| **server 동작** | `_halo_on_request_finished` 가 그 flag 받으면 `mark_job_done(rid)` → state=COMPLETE → 다음 sweep 의 `gc_completed` 가 retain_seconds 후 GC |
| **`on_request_completed`** | 자동 COMPLETE 로직 *완전 제거*. 단순히 counter 갱신만 |
| **안전망** | `gc_quiescent_jobs(quiescent_seconds=300)` — `--halo-quiescent-timeout-seconds` 로 조정 가능. 클라이언트 crash / signal 누락 case 처리 |

### Implementation

| 영역 | 변경 |
|---|---|
| `io_struct.GenerateReqInput` + `Tokenized*` + `protocol.ChatCompletion*` | `halo_job_done: bool = False` |
| `serving_chat`/`serving_completions`/`tokenizer_manager` | passthrough |
| `schedule_batch.Req` | `halo_job_done` 필드 |
| `scheduler.handle_generate_request` | Req 생성 시 전달 |
| `scheduler._halo_on_request_finished` | flag forward |
| `HaloController.on_request_finished(rid, halo_job_done=False)` | flag True 시 `mark_job_done` 먼저, `record_completion` 나중 |
| `Job.on_request_completed` | state 전환 로직 *제거* |
| `Job.mark_done()` | state=COMPLETE 강제 |
| `JobRegistry.mark_job_done(rid)` | rid → job 룩업 후 mark_done |
| `JobRegistry.gc_quiescent_jobs(N)` | running/queued 상태인 idle job 자동 COMPLETE (WARN 로그) |
| `HaloController.tick()` | gc_quiescent_jobs → gc_completed → gc_idle_programs 순 |
| `server_args.halo_quiescent_timeout_seconds` (default 300) | + CLI flag + env var passthrough |

### Client wiring (Agent_applications)

| | wiring |
|---|---|
| `make_llm(...)` | `halo_job_done: bool = False` keyword arg → `extra_body["halo_job_done"]` |
| `ChainState` + `create_chain_state` | `halo_done_llm: Optional[Any]` |
| `invoke_with_tracking` | `call_index == state["chain_length"]` 시 `state["halo_done_llm"]` 으로 swap |
| 3 워크로드 `run_job` | `llm` + `halo_done_llm` 두 인스턴스 같이 생성 |

### 명시 종료 시그널의 효과

- chain 길이 정확히 몰라도 OK (사용자 의도)
- gap (tool_delay) 길이 무관 — quiescent 까지는 안전
- dynamic DAG 도 wiring 가능 (마지막 call 시점에 client 가 결정)
- 안전망: quiescent timeout 으로 client crash 케이스 대비

## 18. Open issue — TBT cost-model cliff (follow-up to §2 Q2)

§2 Q2 의 짧은 메모와 `runtime/cost_models/README.md` item 3, `managers/halo/CLAUDE.md`
"Cost model brittleness inherited" 의 통합 follow-up. **A, B 검증이 끝난 뒤 별도
트랙으로 진행**.

### 증상 정량 (260512_2159 세션 측정)

| 항목 | 값 |
|---|---|
| 실측 TBT mean (λ=0.3 부하, parallel_tool_delay) | **50.8 ms** |
| TBT cost-model 예측 @ batch=1, kv=2000 | **11.4 ms** |
| 예측/실측 비율 | **~22 %** — model이 ~5x underestimate |

현재 식 `tbt_ms ≈ a + b·batch_size + c·per_req_kv`, fit 결과
`a=9.25, b=0.066, c=0.00105`. per_req_kv 가 0~5000 사이면 출력이 9~14ms 좁은
구간에 평탄 — **부하 변화에 거의 무반응 (cliff)**. 결과적으로 Halo slowdown
ratio = actual / predicted 가 인플레이션.

### 모델이 못 잡는 변수 (검토 후보, 영향 큰 순)

| 변수 | 왜 영향 큰가 | 가용성 |
|---|---|---|
| **total_batch_kv** (running batch 모든 req KV 합) | Attention 은 batch 전체 KV 메모리에서 동시에 읽음 — 메모리 BW 가 *합* 에 종속 | scheduler 에서 직접 가능 (already aggregated in `_halo_build_request_execution_infos`) |
| **num_running_reqs** | TBT compute = kernel launch + sched + attn 병렬도 — *batch size 의 초선형* | `running_batch.reqs` |
| **max_seqlen** | 일부 attention kernel 은 max KV 에 padding/chunking → max 에 종속 | running batch max |
| **num_prefill_in_progress** (chunked-prefill 토큰) | prefill 이 같은 GPU 의 decode time 을 훔침 — 무시 못 함 | scheduler chunked-prefill state |
| **GPU mem saturation (token_usage_pct)** | 일정 임계 넘으면 swap/evict → 갑작스런 TBT cliff (진짜 cliff) | mem pool |
| **spec-decoding accept rate** | spec 켰을 때 평균 TBT 가 accept rate 에 강하게 종속 | spec stats |

### 데이터 수집 옵션 (3 가지)

| 방식 | 설명 | 장 | 단 |
|---|---|---|---|
| **(α) workload-driven calibration** | baseline + load run 의 실측 TBT 와 scheduler batch composition 시계열을 같이 기록 → offline regression | 본 workload 에 fit. 빠른 적용 | 일반화 약함 |
| **(β) grid 확장 + 다변수** | `tools/admission_control/fit_cost_model.py` 의 controlled grid 에 multi-req 시나리오 추가 | 일반화 좋음 | 학습 시간 길음 (몇 시간) + grid 설계 부담 |
| **(γ) per-step instrumentation** | scheduler 가 step 마다 `(num_running, total_kv, max_kv, prefill_tokens, measured_tbt)` 5-tuple 기록 (rank-0) → run 후 fit_cost_model.py 가 regression | 자동. 운용 부담 적음. 실험 부하 시점 분포에 정확 fit | 새 instrumentation 필요 (40 LoC) |

추천: **(γ)** — 새 instrumentation 인 만큼 phase 2 cost-model refit 의 기반이 되고
admission_control 의 TBT EWMA tracker 와도 자연 통합 가능.

### 현실적 대안 — model-less baseline TBT

baseline run (concurrency=1) 의 측정 TBT 통계를 직접 사용:
```
solo_tbt_ms = baseline_run.tbt_mean_ms_for_this_workload
```
- 장: 즉시 정확 (실측 자체). cliff 없음. ~10 LoC.
- 단: workload-specific. 새 workload 면 새 baseline 필요. 일반 모델 학습 회피.

Phase 1 의 *짧은* 대안. (γ) 가 무거우면 임시 (α) 또는 model-less 로 가도 됨.

### Action items (사용자 컨펌 후)

- [ ] C1: 변수 후보 중 어디까지 추가? (추천: total_batch_kv, num_running_reqs, max_seqlen, prefill_tokens 4개)
- [ ] C2: 모델 형태 — 다중선형만 / 분할선형 / lookup? (추천: 다중선형 먼저 → 잔차 보고)
- [ ] C3: 데이터 수집 — α / β / γ / model-less? (추천: γ)
- [ ] C4: 진행 시점 — Halo A·B 검증 후 (현재 결정: A·B 먼저)

이 자료는 Halo Phase 1 의 *알려진 한계* 로 표시되어 있음:
- `python/sglang/srt/managers/halo/CLAUDE.md` "Cost model brittleness inherited"
- `python/sglang/srt/managers/halo/slowdown_tracker.py` 모듈 docstring
- `ms_dev/runtime/cost_models/README.md` item 3
- `ms_dev/halo_dev/CLAUDE.md` §2 Q2 (의사결정 노트) + 이 §18 (follow-up 정량 + 옵션)

## 19. 현재 상태 (2026-05-14) — 한 페이지 요약

> compaction 이후 새 context 에서 빠르게 catch-up 하기 위한 스냅샷.

### Phase 1 deliverable 진행

| | 의도된 동작 | 검증 상태 |
|---|---|---|
| **Job tracking** (R1) | 매 100ms sweep, per-job `slowdown_max/mean` 갱신 | ✅ 260513_1550 세션에서 mean 2.03x / worst 2.56x 관찰. 부하 반영 정상 |
| **Job-level admission gate** (R2, strict mode) | `halo_job_id` + 사전등록 필수, 둘 다 안되면 400 | ✅ 검증됨 (reject 0건) |
| **Job-level scheduling** (R3) | n/a — Phase 2 작업 | ⏳ |
| **Option A: `POST /halo/programs`** | client 가 chain 시작 시 1회 등록. body 에 chain_length 등 | ✅ |
| **Option B: request body fields** | `halo_job_id`, `halo_slo`, `halo_bypass`, `halo_job_done` | ✅ |
| **Explicit termination** | client 가 마지막 call body 에 `halo_job_done=true` → server mark COMPLETE | ✅ 260513_1550 에서 1 job 정상 종료 확인 |
| **Quiescent safety net** | 300s idle 후 자동 COMPLETE | ✅ 단위 테스트 통과, 정상 운용에선 미트립 |
| **Monitor panel** | `halo_*` prefix + reject reason split | ✅ |
| **Verify harness** | microbench + e2e_smoke + slowdown_vs_actual (Phase 2) | ✅ A/B 부분 |

### 알려진 한계 (Phase 2 향)

- **§18 cost-model cliff**: ✅ Halo Step Cost Model (split form, §17) 으로 **해소**.
  옛 식 worst-case +89 % 인플레이션 → split 검증 (260513_2342 세션) 에서 -35 %
  deflation 으로 균형. p90 은 +4 % 로 거의 정확. 분포 모양이 ground-truth 와
  닮아짐. mean/p50 은 약간 보수적 (+17 % / +34 %) — Halo SLO 결정에는 안전한
  방향 (false-negative 적음).
  Validation 정량 + open question: [`prediction_model.md`](prediction_model.md) §17.
- **Prefill 식 정확도는 sample 부족**: EXTEND step 645개로 fit 됨. V1 refit
  (λ spectrum 확장 + 긴 sweep) 시 함께 개선될 가능성. 본 라운드에선 미진행.
- **Mass-job 분석 도구 없음**: per-request ground-truth 비교를 일회용 inline
  python 으로 수행 (§15·§17 표). 재사용 가능한 스크립트로는 미작성.
- **R3 (scheduling)** 미구현 — Phase 2 작업.
- **Multi-instance (PD/DP)**: Halo 는 단일 scheduler 가정. PD 모드에선 controller
  no-op + WARN. Phase 2 Conductor 단에 통합 필요.

### Push 상태 (2026-05-14)

- sglang `project-halo-phase1`: 26 commits **이미 push 됨** + 본 라운드의 step-cost-model 작업
  분량 누적 (commit 대기 중)
- Agent_applications `project-halo-phase1-client`: 변경 없음 (client 측은 본 작업
  영향 0 — server-side cost-model 만 바뀜)

push 정책상 사용자가 직접 푸시.

### 사용자 결정 미결 항목 (2026-05-14 시점)

| ID | 내용 | 컨텍스트 |
|---|---|---|
| **V1 refit** | λ=[0.1, 0.3, 0.5, 0.7] × 30~60 min 데이터로 step model 재학습할지. Phase 2 R3 lookahead 가기 전 권장. | [`prediction_model.md`](prediction_model.md) §15 V1 |
| **검증 1 분석 스크립트** | 재사용 가능한 `verify_slowdown_vs_actual.py` 작성 시점. 현재는 inline. | `verify/CLAUDE.md` |
| **Phase 2 R3 진입 시점** | 현 split model 그대로 갈지 V1 refit 후 갈지 | §17 끝 |

### 다음 가능한 action 후보 (대화 재개 시)

1. **(끝남)** §20 / §17 — split form 채택, default 로 설정. 1차 라운드 완료.
2. **V1 refit (선택, 권장)** — λ spectrum × 긴 sweep 으로 split model refit.
   prefill 정확도 + tail 정확도 모두 추가 개선 여지. §15 V1.
3. **Phase 2 R3 lookahead** — admission decision 작업 진입. V1 refit 안 거치고
   가도 일단 split model 이 ground-truth 와 균형 잡힌 분포라 진행 가능.
4. **commit + push** (사용자 직접) — 본 라운드의 모든 작업.

---

## 20. Halo Step Cost Model (§18 cliff 해결안)

§18 cliff 를 해결하기 위한 새 cost-model 도입. **결정된 설계·식·변수 의미·
fit 절차·변경 파일·단계별 실행 순서의 상세는 [`prediction_model.md`](prediction_model.md)** 에 정리. 본 절은 한눈에 보는 요약만.

### 식 (한 step 회귀 대상, 6 계수)
```
T_step ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ
       + θ_d1·Σrⱼ + θ_d2·bs_d + θ_c
```
- prefill 항: 새 토큰 self-attention(`Σnᵢ²`) + 새 토큰 × cached prefix cross-attention(`Σ(nᵢ·rᵢ)`) + FFN(`Σnᵢ`)
- decode 항: 배치 KV 합산(`Σrⱼ`) + per-request 오버헤드(`bs_d`)
- 옛 식과의 핵심 차이: decode 항이 `per_req_kv` (나눔) 대신 `Σrⱼ` (합산) — cliff 의 직접 원인 제거

### 핵심 결정사항 (사용자 컨펌, 2026-05-13 → 14)

| ID | 결정 |
|---|---|
| 모델 형태 | `HaloStepCostModel` — 두 form 지원: **unified (`halo_step_v1`, 6 계수)** + **split (`halo_step_split_v1`, 7 계수, MuxWise 원본 형태)** |
| Form 선택 기준 | MIXED step 발생 시 unified, 없으면 split. 본 호스트 (mixed_chunk off) 에선 **split default 채택** (2026-05-14, ground-truth 검증 후 — [`prediction_model.md`](prediction_model.md) §17). `fit_halo_cost_model.py --form` default=`split`. `halo_base.sh` 가 split JSON 을 default export. |
| Solo vs batched | 같은 모델·계수, caller 가 입력 모드 선택. Halo R1 분모는 solo (1 요청 가정) |
| `Σ(nᵢ·rᵢ)` cross-term | 포함 (cache hit ≥ 90% 워크로드라 비중 큼) |
| Contention guard | 없음 (drift 보이면 추후 EWMA 보정 추가) |
| 모듈 분리 | 안 함 — `admission_control/cost_model.py` 안에 `HaloStepCostModel` 추가 (옛 클래스 공존) |
| Naming | "Halo Step Cost Model" (외부 시스템 이름 사용 안 함) |
| admission_control 변경 | **이번 작업 범위 아님** — 옛 식 (Mooncake-like) 그대로 살아 있음. 미래 Phase 2 작업 |
| Legacy 공존 | 새 path 있으면 new, 없으면 legacy 자동 fallback. 양 모드 다 유지 |

### 단계별 실행 순서 (각 단계가 PR 한 개 단위)

1. ✅ `HaloStepCostModel` 클래스 + loader + 단위 테스트
2. ✅ per-step instrumentation (`--halo-cost-model-sample-log`) + 단위 테스트
3. ✅ `tools/halo/fit_halo_cost_model.py` + JSONL fixture 단위 테스트
4. ✅ Halo controller·tracker 로드/호출 분기 + 단위 테스트
5. ✅ 데이터 수집 1차 (260513_2014 세션, λ=0.3, 10 min, 6,982 step)
6. ✅ Fit → `halo_step_llama3-70b_b200x4.json` (R²=0.96, RMSE=23 ms) → 검증 실험 (260513_2147 세션) → ground truth vs 옛/새 식 비교 완료. [`prediction_model.md`](prediction_model.md) §15 에 결과 + 한계 + 개선 경로 기록.
7. ✅ `runtime/cost_models/README.md`, `tools/halo/README.md` (워크플로·명령어), `tools/halo/CLAUDE.md` (짧은 pointer) 등 문서 갱신 (monitor panel 은 minor 로 보류)

**진행 상황**: 단계 1–7 모두 완료. 코드/문서/검증 1차 라운드 종료.
**결과 요약**: §18 cliff 의 *인플레이션 방향* 오차는 해소, *deflation 방향* 의 새 오차가 tail 에 생김. 평균 정확도 ~2x 향상. R3 lookahead 에서 쓰기 전엔 V1 refit (다양한 λ + 긴 sweep) 권장.

### 신규 CLI 플래그

| 플래그 | env var | 효과 |
|---|---|---|
| `--halo-step-cost-model-path` | `SGLANG_HALO_STEP_COST_MODEL` | Halo 가 새 모델 사용. set 시 legacy halo 두 path 무시 |
| `--halo-cost-model-sample-log` | `SGLANG_HALO_COST_MODEL_SAMPLE_LOG` | per-step JSONL 출력 (rank-0 only, 데이터 수집용) |

legacy `--halo-prefill-cost-model-path`, `--halo-tbt-cost-model-path` 는 그대로 살아 있음.

### 참고
- 상세: [`prediction_model.md`](prediction_model.md)
- §18 (cliff 정량·진단)
- `python/sglang/srt/managers/halo/CLAUDE.md` "Cost model brittleness inherited"
- 식 출처(영감): MuxWise (ASPLOS '26 arXiv 2504.14489). DistServe (OSDI '24) 의 simulator 는 disaggregated 가정이라 직접 이식 안 함.

### 참고 — 변경된 파일 목록 (sglang)

```
python/sglang/srt/managers/halo/             ← 신규 패키지
  ├── __init__.py
  ├── CLAUDE.md
  ├── job.py
  ├── job_registry.py
  ├── slowdown_tracker.py
  ├── controller.py
  └── metrics.py
python/sglang/srt/managers/{io_struct, schedule_batch, scheduler, tokenizer_manager, tokenizer_control_mixin}.py
python/sglang/srt/managers/scheduler_output_processor_mixin.py
python/sglang/srt/entrypoints/{http_server, openai/{protocol, serving_chat, serving_completions}}.py
python/sglang/srt/server_args.py
test/registered/halo/{CLAUDE.md, test_halo_phase1.py}          ← 39 tests
ms_dev/{env.common.sh, lib_server.sh, start_server_no_pd.sh, CLAUDE.md}
ms_dev/experiments/{halo_base.sh, halo_off.sh, README.md}
ms_dev/expctl/{run_experiment.py, monitoring_view.py, CLAUDE.md}
ms_dev/halo_dev/{CLAUDE.md, halo_api_reference.md, verify/{CLAUDE.md, microbench_overhead.py, e2e_smoke.sh}}
```

---

## 21. Halo Phase 2 — Admission Control

§3 의 Role 2 (job-level admission gate) 의 *predictive* + *enforcement* 단계.
Phase 1 은 strict-mode 만 (halo_job_id + pre-registered 둘 다 있어야 통과).
Phase 2 는 그 위에 **새 job 받기 결정** 을 *기존 job 들의 예측 slowdown* 으로
판정.

**상세 설계·식·코드 구조·PR 순서**: [`admission_design.md`](admission_design.md).

### 목적 한 줄

> 새 job 을 admit 했을 때 *기존 active job 들의* 예측 slowdown 중 xx% 이상이
> SLO 를 넘을 것 같으면 reject. 새 job 의 SLA 가 아니라 *기존 작업 보호* 가
> 목적.

### 두 Stage

| Stage | 목적 | 입력 | 출력 |
|---|---|---|---|
| **A — Predictive (Level 0 only, M-2)** | 새 job 받으면 *기존 active job 들의 predicted virtual job slowdown 분포*. 결정은 *job 단위* — 첫 request 만 검사, 후속 request 는 자동 admit | active jobs + new job's first request + cost model | admit / reject + per-job 예측치 |
| **B — Concurrency hard cap** | application 이 declare 한 *동시 in-flight* 약속을 enforce | declared_max_concurrency + current in_flight_count | admit / reject |

### Level 0 (현재 default), Level 2 (DEPRECATED)

- **Level 0 — Per-job snapshot stretch (M-2, 2026-05-15 design)**:
  - 각 active job 의 *현재 단계* (prefill / decode) 에 따라 *그 단계의 step time 변화율* 을 stretch 로 사용.
  - `stretch_extend = aug_extend / base_extend`, `stretch_decode = aug_decode / base_decode`
  - `predicted_VJS_i = current_VJS_i × stretch_i`
  - 사용 정보: *현재 batch + 새 request 의 prompt/prefix*. **DAG / remaining / expected_output 안 씀**.
- **Level 2 (DEPRECATED 2026-05-15)**: 옛 1 초 slice lookahead. 코드는 남아있지만 admission path 에서 호출 안 됨 (controller 가 자동 level0 폴백 + WARN).

### 새 components (계획)

```
managers/halo/admission_decision.py   ← 신규
  AdmissionPredictor / SnapshotAdmissionPredictor / LookaheadAdmissionPredictor
  decide_admission(...)
managers/halo/job.py                  declared_max_concurrency, in_flight_count
managers/halo/controller.py            register_request 안 Stage A → Stage B
server_args.py                         5 새 CLI 플래그 (--halo-admission-*)
ms_dev/env.common.sh + lib_server.sh   5 새 env vars (SGLANG_HALO_ADMISSION_*)
test/registered/halo/test_halo_admission_predictors.py  ← 신규
```

### 단계별 PR 순서

| # | 내용 | 영향 |
|---|---|---|
| PR1 | AdmissionPredictor ABC + Level 0 + 단위 테스트 | 라이브러리만 |
| PR2 | Controller 통합 + dry-run + CLI + env vars + decision log | 통합 |
| PR3 | Level 2 lookahead + 단위 테스트 | 라이브러리 |
| PR4 | Stage B (Concurrency cap) | Job/Registry + register_program |
| PR5 | 검증 실험 + 문서 갱신 | 문서 |

### 진행 상태 (2026-05-14)

| PR | 상태 | 산출물 |
|---|---|---|
| PR1 | ✅ | `managers/halo/admission_decision.py` (Snapshot + ABC) + 14 unit tests |
| PR2 | ✅ | `HaloController.register_request` 가 Stage A 호출 + dry-run + decision log + 5 CLI 플래그 + 5 env vars + expctl auto-route + 6 controller tests |
| PR3 | ✅ | `LookaheadAdmissionPredictor` (1초 slice 시뮬) + 7 lookahead tests |
| PR4 | ✅ | `Job.declared_max_concurrency` + `in_flight_count` + `register_program` body field + Stage B reject path + 4 cap tests |
| PR5 | ⏳ | 사용자 환경 검증 실험 (mode=off / level0 / level2 비교) — 코드 ready |

본 라운드 통합 test 수: **201 (43+33+13+9+72 + 4 신규 Phase 2 sets)**.

---

### 참고 — 변경된 파일 목록 (Agent_applications)

```
agent_motivation_experiment/run_experiment.py                  ← --halo-enabled / --halo-slo + readiness probe
agent_motivation_experiment/workloads/{base, halo_helpers}.py
agent_motivation_experiment/workloads/swe_bench_coding/{agent, workload}.py
agent_motivation_experiment/workloads/swe_bench_coding_tool_delay/workload.py
agent_motivation_experiment/workloads/swe_bench_coding_parallel_tool_delay/workload.py
agent_motivation_experiment/{CLAUDE.md, workloads/{AGENTS, CLAUDE}.md}
CLAUDE.md                                                       ← repo root
```