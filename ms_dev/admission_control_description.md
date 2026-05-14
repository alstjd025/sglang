# Admission Control — 사용 설명서

## 0. 한 줄 요약

> **새 요청이 들어왔을 때 SLO를 못 맞출 것 같으면 그 자리에서 거절**해서, 이미 받은 요청들의 latency가 망가지지 않게 보호한다.

> 무조건 받고 큐에 쌓아두면 → 한 명도 SLO 못 맞춤.
> 미리 거절 → 받은 요청들은 SLO 안에 끝, 거절된 요청들은 클라이언트가 backoff/재시도/다른 worker로 fallback.

이 페이지는 SGLang에 추가한 Mooncake-style admission control의 **사용법**과 **내부 동작**을 모두 다룹니다.

코드 위치: `python/sglang/srt/managers/admission_control/`. 자세한 모듈 설계는
`python/sglang/srt/managers/admission_control/CLAUDE.md`.

---

## 1. 빠른 시작 (3분)

### 1-A. Cost model 한 번 만들기 (모델/HW 변경 시만)

```bash
# 1) admission OFF 상태로 서버 띄움
unset SGLANG_ADMISSION_TTFT_SLO_MS SGLANG_ADMISSION_TBT_SLO_MS
unset SGLANG_ADMISSION_TTFT_SLO_RATIO SGLANG_ADMISSION_TBT_SLO_RATIO
bash ms_dev/start_server_no_pd.sh --metrics

# 2) 다른 터미널에서 fit 실행 (~2분)
python tools/admission_control/fit_cost_model.py prefill \
    --server http://localhost:31000 \
    --output ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json

python tools/admission_control/fit_cost_model.py tbt \
    --server http://localhost:31000 \
    --output ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json
```

→ JSON 두 개. 한 번만 만들어두면 모델/HW가 안 바뀌는 한 재사용.

**언제 다시 fit?** 모델 / GPU 종류 / TP 크기 / quantization / attention backend / mem-fraction 중 하나라도 바뀌면.

### 1-B. Dry-run으로 SLO 튠

가장 간단한 방법은 미리 만들어둔 **experiment wrapper script**를 source하는 것:

```bash
source ms_dev/experiments/admission_dryrun.sh
python3 ms_dev/expctl/server_run_experiment.py --mode single
```

이렇게 하면:
- `admission_dryrun.sh`가 SLO env vars 모두 export
- `run_experiment.py`가 SLO 활성화를 감지 → **`admission_decisions.jsonl`을 자동으로 세션 폴더에 라우팅**
- `meta/run_meta.json`에 `admission_config` 블록 자동 캡처

직접 export하려면:

```bash
# 모드 A: 절대값 SLO만
export SGLANG_ADMISSION_TTFT_SLO_MS=2000
export SGLANG_ADMISSION_TBT_SLO_MS=100

# OR 모드 B: 비율 SLO만 (solo-run 대비)
export SGLANG_ADMISSION_TTFT_SLO_RATIO=2.0
export SGLANG_ADMISSION_TBT_SLO_RATIO=3.0

# OR 모드 C: 둘 다 (★ 권장 — Stage 3 EWMA 안전망 활성화)
export SGLANG_ADMISSION_TTFT_SLO_MS=30000        # 안전 cap
export SGLANG_ADMISSION_TTFT_SLO_RATIO=2.0       # 정상 운영 기준
export SGLANG_ADMISSION_TBT_SLO_MS=500
export SGLANG_ADMISSION_TBT_SLO_RATIO=3.0

# 공통
export SGLANG_ADMISSION_PREFILL_COST_MODEL=ms_dev/runtime/cost_models/prefill_llama3-70b_b200x4.json
export SGLANG_ADMISSION_TBT_COST_MODEL=ms_dev/runtime/cost_models/tbt_llama3-70b_b200x4.json
export SGLANG_ADMISSION_DRY_RUN=1                 # ★ 거절 안 함, 결정만 기록

# DECISION_LOG는 명시 안 해도 됨 — run_experiment가 자동으로 세션 폴더에 라우팅
python3 ms_dev/expctl/server_run_experiment.py --mode single
```

### 1-C. SLO 튠 (replay)

```bash
SESS=ms_dev/runtime/sessions/<session_name>
python tools/admission_control/replay_admission.py \
    --decision-log $SESS/admission_decisions.jsonl \
    --ttft-slo 0 \
    --tbt-slo 0 \
    --ttft-slo-ratio 1.5 2.0 3.0 5.0 \
    --tbt-slo-ratio 2.0 3.0 5.0 \
    --csv-out $SESS/admission_replay.csv
```

→ 각 (절대 SLO × 비율 SLO) 조합에서 거절률 확인. 운영 목표에 맞는 값 결정.

세션 폴더에 `admission_replay.csv`로 저장하면 다른 산출물과 함께 보관됨.

### 1-D. 본격 사용 (dry-run 끄기)

```bash
unset SGLANG_ADMISSION_DRY_RUN  # 또는 =0
bash ms_dev/start_server_no_pd.sh --metrics
```

→ 이제 진짜로 HTTP 429 거절.

---

## 2. SLO 설정 — 두 가지 방식

### 방식 A: 절대값 SLO (ms 단위)

| Env var | CLI 플래그 | 의미 |
|---|---|---|
| `SGLANG_ADMISSION_TTFT_SLO_MS=2000` | `--admission-ttft-slo-ms 2000` | TTFT 가 2초 넘기면 거절 |
| `SGLANG_ADMISSION_TBT_SLO_MS=100` | `--admission-tbt-slo-ms 100` | TBT 가 100ms 넘기면 거절 |

**장점**: 사용자 경험을 직접적으로 보장. "응답 첫 토큰까지 2초 안에" 같은 SLA 그대로.
**단점**: 요청 길이가 다양하면 평균 기준으로 정해야 해서, 큰 요청은 늘 거절 / 작은 요청은 거의 안 거절됨.

### 방식 B: 비율 SLO (slowdown ratio)

| Env var | CLI 플래그 | 의미 |
|---|---|---|
| `SGLANG_ADMISSION_TTFT_SLO_RATIO=2.0` | `--admission-ttft-slo-ratio 2.0` | TTFT가 solo-run 대비 2배 넘게 느려지면 거절 |
| `SGLANG_ADMISSION_TBT_SLO_RATIO=3.0` | `--admission-tbt-slo-ratio 3.0` | TBT가 solo-run 대비 3배 넘게 느려지면 거절 |

**Solo-run baseline** = 그 요청을 idle한 서버에서 단독으로 돌렸을 때의 latency.
- TTFT solo = `α·d² + β·d + γ + δ·p` (큐 비어있을 때)
- TBT solo = TBT cost model을 `bs=1` + 요청 자체 KV로 평가

**장점**: 요청 크기에 자동 적응. 50k 프롬프트 (solo 500ms)는 1초까지 OK, 500토큰 프롬프트 (solo 30ms)는 60ms까지만 OK — 둘 다 ratio=2에서.
**단점**: "사용자에게 절대 시간 보장"은 아님 (작은 요청은 빨라도, 큰 요청은 느려도 ratio 같으면 admit).

### 방식 C: 둘 다 함께 (whichever fires first)

```bash
export SGLANG_ADMISSION_TTFT_SLO_MS=30000      # 절대 hard cap (catastrophic 방지)
export SGLANG_ADMISSION_TTFT_SLO_RATIO=2.0     # 정상 fairness 기준
```

→ ratio 2배 넘거나 절대 30초 넘으면 거절. **권장 패턴**: ratio가 정상 운영, 절대 SLO는 안전망.

### 어떤 게 적합한가?

| 상황 | 추천 |
|---|---|
| 사용자에게 "응답 시간 X초 보장" SLA 있음 | **절대값**. Stage 3 EWMA도 자동 활성. |
| 요청 크기가 매우 다양 (500 ~ 60k 토큰 등) | **비율**. fairness 자동 조정. |
| Agent 워크로드 (긴 prompt + 누적 대화) | **둘 다**. ratio가 정상, 절대값이 안전망. |
| Stress test로 서버 보호만 원함 | **절대값** (간단함). |

이 프로젝트의 워크로드 (parallel_tool agent, prompt 18-77k): **방식 C 권장**.

---

## 3. 5단계 결정 정책 — 정확히 어떻게 거절을 결정하는가

요청이 도착하면 controller.decide()가 다음 순서로 검사:

```
                    [요청 도착]
                         │
                         ▼
       ┌──────────────────────────────────────────┐
       │ Stage 1a: TTFT 절대값                      │
       │   pred_TTFT = t_queue + t_prefill         │
       │   pred_TTFT > ttft_slo_ms ? → REJECT       │  ← TTFT_PREDICTED
       └──────────────────────────────────────────┘
                         │ pass
                         ▼
       ┌──────────────────────────────────────────┐
       │ Stage 1b: TTFT 비율                        │
       │   pred_TTFT / solo_TTFT > ttft_ratio ?     │  ← TTFT_RATIO
       │   (단, solo_TTFT < 1ms 이면 skip)            │
       └──────────────────────────────────────────┘
                         │ pass
                         ▼
       ┌──────────────────────────────────────────┐
       │ Stage 2a: TBT 절대값                       │
       │   pred_TBT = TBT_cost(bs+1, kv+n)           │
       │   pred_TBT > tbt_slo_ms ? → REJECT         │  ← TBT_PREDICTED
       └──────────────────────────────────────────┘
                         │ pass
                         ▼
       ┌──────────────────────────────────────────┐
       │ Stage 2b: TBT 비율                          │
       │   pred_TBT / solo_TBT > tbt_ratio ?         │  ← TBT_RATIO
       │   (단, solo_TBT < 1ms 이면 skip)             │
       └──────────────────────────────────────────┘
                         │ pass
                         ▼
       ┌──────────────────────────────────────────┐
       │ Stage 3: TBT 반응형 안전망 (EWMA)            │
       │   recent_TBT = 측정 EWMA over 100+ steps     │
       │   recent_TBT > tbt_slo_ms × 0.9 ? → REJECT │  ← TBT_REACTIVE
       │   (절대값 SLO 설정될 때만 동작)               │
       └──────────────────────────────────────────┘
                         │ pass
                         ▼
                       [ADMIT]
```

**규칙**: 첫 번째 위반에서 즉시 거절. 같은 stage 내에서는 절대값 → 비율 순서 (절대값이 더 "심각"한 cap이라 그 사유가 우선).

각 stage는 **독립적으로 enable됩니다** — 절대값만 설정하면 1a, 2a, 3만 동작; 비율만 설정하면 1b, 2b만 동작; 둘 다 설정하면 5개 모두.

---

## 4. 무엇을 프로파일링했나 — Cost model 두 개

### 4-A. Prefill cost model

```
T_prefill_ms = α·d² + β·d + γ + δ·p
```

| 항 | 의미 |
|---|---|
| `α·d²` | Transformer attention의 quadratic 비용 (토큰 수 제곱에 비례) |
| `β·d` | FFN 등 linear 부분 |
| `γ` | 요청당 고정 오버헤드 (kernel launch, scheduling 등) |
| `δ·p` | radix cache에서 prefix p 토큰을 GPU로 로드하는 비용 (~17μs/token on B200) |

여기서 `n`= 총 prompt 길이, `p`= prefix cache hit 길이, `d = n - p`= 실제로 prefill 해야 하는 토큰 수.

**측정 방법** (`fit_cost_model.py prefill`, solo-run):

1. 서버를 idle 상태로 띄움 (다른 트래픽 없음)
2. 다양한 (n, p) 조합 (default 27 cells × 2 repeats = 54 측정):
   - `d ∈ {0, 128, 256, 512, 1k, 2k, 4k, 8k, 16k}` × `p ∈ {0, 8k, 32k}`
3. 각 cell:
   - `/flush_cache` (선택)
   - `p`개 random token id로 prefix 워밍업 요청 → radix cache에 올라감
   - `prefix + d개 random suffix` 로 측정 요청 (max_new_tokens=1, streaming)
   - **TTFT 측정** = 첫 토큰 도착까지 시간
   - 응답 `meta_info.cached_tokens`로 "정말로 p개 hit 했는지" 검증
4. 모든 (d, p, t) 샘플에 lstsq로 4-coef linear regression

**왜 solo-run?** 모델은 "이 요청 단독으로 prefill하면 X ms 걸린다"를 답해야 함. 다른 요청과 섞이면 그건 큐 contention → 별도로 `t_queue = sum(다른 큐 요청들의 예측 prefill)`로 더해서 처리.

**B200×4 / Llama-3.3-70B 결과**: `α=1.47e-6, β=0.073, γ=-145, δ=0.0166, RMSE=229ms`

### 4-B. TBT cost model

```
TBT_ms ≈ a + b·batch_size + c·total_kv_tokens
```

| 항 | 의미 |
|---|---|
| `a` | 요청 0개일 때 (이론적) base 비용 |
| `b·bs` | batch size에 따라 늘어나는 비용 (대체로 작음, 때로는 음수 — heterogeneous batch에서 평균화 효과) |
| `c·kv` | KV cache 총 점유에 따른 메모리 bandwidth 비용 |

**측정 방법** (`fit_cost_model.py tbt`, **batch-run**, 동시 측정):

1. 다양한 (bs, prompt_len) 조합 (default 15 cells):
   - `bs ∈ {1, 4, 8, 16, 32}` × `prompt_len ∈ {1k, 4k, 16k}`
2. 각 cell:
   - `bs`개의 distinct random prompt를 **동시에** ThreadPoolExecutor로 띄움
   - 각 stream의 inter-token 인터벌 측정
   - 처음 10개 인터벌 버림 (prefill + ramp-up 단계)
   - 다음 30개 인터벌 keep
3. 모든 cell의 (bs, kv≈bs·prompt_len, median_tbt) 샘플에 lstsq

**왜 batch-run?** TBT는 batch 구성의 함수. `TBT(bs=1)`과 `TBT(bs=32)`는 완전히 다름 (continuous batching의 본질). Stage 2가 admission 시점에 "지금 running batch에 추가하면 TBT가 어떻게 될까?" 예측하려면 (bs, kv) 함수로 fit한 데이터가 필요.

**B200×4 / Llama-3.3-70B 결과**: `a=16.84, b=-0.28, c=4.81e-5, RMSE=4.66ms`

### 4-C. 한계 (꼭 알아둘 것)

1. **Prefill fit은 idle 서버에서**. 운영 중 chunked prefill이 decoding과 같은 GPU에서 섞여 돌면 둘 다 느려져서 **운영 중 cost model은 약간 underestimate**.
2. **TBT fit은 균질 batch**로만. 운영은 heterogeneous → fit 결과의 b 계수가 음수가 나오는 등 비직관적인 현상. 그래서 **Stage 3 (반응형 EWMA)가 필수 안전망**.
3. **워크로드/HW 한정**. 다른 모델/GPU/TP 크기에 그대로 쓰면 안 됨.

---

## 5. 거절될 때 클라이언트가 보는 것

거절은 HTTP 429 의미를 갖지만, **streaming endpoint는 응답 헤더가 이미 200으로 떴기 때문에** 실제 전송은 다음 형태:

```json
{
  "text": "",
  "output_ids": [],
  "meta_info": {
    "id": "<rid>",
    "finish_reason": {
      "type": "abort",
      "status_code": 429,
      "message": "Admission rejected: TTFT_RATIO (predicted_ttft=410.0ms ttft_slo_ratio=2.0 ...)",
      "admission_reason": "TTFT_RATIO"
    },
    "completion_tokens": 0,
    "e2e_latency": 0.012
  }
}
```

### Agent 코드에서 분기하는 법

```python
resp = response.json()  # 또는 streaming 마지막 청크
fr = resp["meta_info"]["finish_reason"]

if fr.get("type") == "abort" and fr.get("admission_reason"):
    reason = fr["admission_reason"]
    # 가능한 값:
    # - TTFT_PREDICTED, TTFT_RATIO  → 큐 길이 문제, exponential backoff 후 재시도
    # - TBT_PREDICTED, TBT_RATIO    → batch 빡빡, 잠시 대기 / 다른 worker로
    # - TBT_REACTIVE                → 시스템 오버로드, 더 긴 backoff
    handle_admission_reject(reason)
```

⚠️ **중요**: HTTP status 자체는 200 (streaming 응답이라). 429는 finish_reason.status_code 필드에. agent가 HTTP status로만 분기한다면 finish_reason도 같이 보도록 수정 필요.

---

## 6. 로깅 — 세션마다 어디에 어떻게 저장되나

`run_experiment.py --mode single`로 실행하면 한 세션의 모든 admission 정보가
세션 폴더(`ms_dev/runtime/sessions/<name>/`) 안에 자동으로 정리됩니다:

```
ms_dev/runtime/sessions/<session_name>/
├── meta/run_meta.json                ← admission_config 블록 (★ SLO 스냅샷)
├── metrics/server_metrics.jsonl      ← Prometheus 5개 메트릭 자동 스크랩
├── process_logs/
│   ├── server.stdout.log             ← [admission] REJECT / WOULD_REJECT 로그
│   └── server.stderr.log             ← startup ServerArgs 풀덤프 + cost model 로드
└── admission_decisions.jsonl         ← (★) 모든 결정 한 줄씩, replay 입력
```

이외에 `/server_info` 라이브 endpoint로 현재 상태 조회 가능.

총 6개 채널 — 자동 정리되는 5개 + 라이브 조회 1개.

### 6-1. `meta/run_meta.json` 의 `admission_config` (★ 가장 깔끔한 단일 진입점)

세션 시작 시 SGLANG_ADMISSION_* env 스냅샷을 자동 캡처:

```json
{
  "session_name": "20260508_022300",
  "admission_config": {
    "enabled": true,
    "applied_in_mode": true,
    "env": {
      "SGLANG_ADMISSION_TTFT_SLO_RATIO": "2.0",
      "SGLANG_ADMISSION_TBT_SLO_MS": "500",
      "SGLANG_ADMISSION_TBT_SLO_RATIO": "3.0",
      "SGLANG_ADMISSION_PREFILL_COST_MODEL": ".../prefill_llama3-70b_b200x4.json",
      "SGLANG_ADMISSION_TBT_COST_MODEL": ".../tbt_llama3-70b_b200x4.json",
      "SGLANG_ADMISSION_DRY_RUN": "1"
    },
    "decision_log_path": ".../sessions/20260508_022300/admission_decisions.jsonl"
  }
}
```

→ "이 세션 어떤 SLO로 돌렸지?" 30초 안에 답.

### 6-2. `admission_decisions.jsonl` — 매 결정 한 줄

자동으로 세션 폴더에 라우팅됨 (사용자가 `SGLANG_ADMISSION_DECISION_LOG`을
명시 안 한 경우). **rank-0 scheduler만** 씁니다 (TP=N에서 N번 중복 X).

```json
{
  "ts": 1778168774.58,
  "rid": "19246fa8...",
  "admit": true,
  "reason": "TTFT_RATIO",
  "dry_run_would_reject": true,
  "predicted_ttft_ms": 410.0,
  "predicted_tbt_ms": null,
  "solo_ttft_ms": 100.0,
  "solo_tbt_ms": null,
  "tbt_ewma_ms": null,
  "queue_predicted_ms": 310.0,
  "ttft_slo_ms": null,
  "tbt_slo_ms": null,
  "ttft_slo_ratio": 2.0,
  "tbt_slo_ratio": 3.0,
  "predicted_prefill_ms": 100.0,
  "prompt_len": 4000,
  "prefix_len": 0,
  "running_batch_size": 12,
  "running_batch_total_kv_tokens": 145000
}
```

`replay_admission.py`가 이 파일을 읽고 다른 SLO로 재평가.

### 6-3. `process_logs/server.*` — Python logger

`server.stderr.log` 시작 부분에 ServerArgs 풀덤프 (admission 필드 포함):
```
admission_ttft_slo_ms=None, admission_tbt_slo_ms=500.0,
admission_ttft_slo_ratio=2.0, admission_tbt_slo_ratio=3.0, ...
```

이어서 controller 활성화 로그:
```
[TP0] admission control: enabled (ttft_slo_ms=None tbt_slo_ms=500.0
      ttft_slo_ratio=2.0 tbt_slo_ratio=3.0 dry_run=True ...)
```

`server.stdout.log` 에는 매 거절 (또는 dry-run would-reject) 한 줄:
```
[admission] REJECT rid=abc123 Admission rejected: TTFT_RATIO
            (predicted_ttft=410.0ms ttft_slo_ratio=2.0 ...)
[admission/dry-run] WOULD_REJECT rid=abc123 reason=TTFT_RATIO ...
```

거절은 항상 INFO 레벨. admit은 무음 (스팸 방지).

### 6-4. Prometheus 메트릭 (`metrics/server_metrics.jsonl`로 자동 스크랩)

```
sglang:admission_decisions_total{decision, reason}    counter
sglang:admission_predicted_ttft_ms                    histogram
sglang:admission_predicted_tbt_ms                     histogram
sglang:admission_tbt_ewma_ms                          gauge
sglang:admission_queue_predicted_ms                   gauge
```

**Counter 라벨**:
- `decision ∈ {admit, reject, dryrun_would_reject}`
- `reason ∈ {ADMIT, DISABLED, TTFT_PREDICTED, TTFT_RATIO, TBT_PREDICTED, TBT_RATIO, TBT_REACTIVE}`

→ Grafana / 자체 분석으로 거절률 시계열, reason 분포 모두 시각화 가능.

### 6-5. `/server_info` 엔드포인트 (라이브 디버깅, 세션 끝나기 전에만)

```bash
curl http://host:31000/server_info | jq '.internal_states[0].admission_state'
```

```json
{
  "enabled": true,
  "dry_run": true,
  "ttft_slo_ms": 2000.0,
  "tbt_slo_ms": 100.0,
  "ttft_slo_ratio": null,
  "tbt_slo_ratio": null,
  "tbt_reactive_ratio": 0.9,
  "tbt_ewma_ms": 145.3,
  "tbt_ewma_warm": true,
  "tbt_ewma_samples": 1024,
  "prefill_cost_loaded": true,
  "tbt_cost_loaded": true,
  "decisions_recent": [...최근 32개 결정...]
}
```

### 6-6. expctl status panel (라이브 모니터)

```
features: server_hicache_l2=ON | server_hicache_l3=OFF | admission=DRY_RUN ttft=2s tbt=100ms
```

---

## 7. 환경변수 / CLI 플래그 전체

### 어디에 두나 (★)

세 가지 패턴, 용도별로 선택:

| 방법 | 어디에 | 용도 |
|---|---|---|
| **A. 셸에서 `export`** | 직접 입력 | 일회성 디버그 |
| **B. `ms_dev/experiments/<name>.sh`** | 미리 만들고 `source` | **재현 가능한 실험 비교** ★ |
| **C. `ms_dev/env.local.sh`** | gitignored, `env.sh`가 자동 source | **호스트별 personal default** |

**`env.common.sh` / `env.single.sh` / `env.pd.sh`는 수정하지 마세요** — upstream과 충돌, 실험 비교 어려움.

#### B. Experiment wrapper 패턴 (권장)

`ms_dev/experiments/`에 미리 만들어 둔 wrapper들:
- `admission_ratio_only.sh` — 비율 SLO만
- `admission_ratio_with_safety.sh` — 비율 + 절대 안전망 (★ 운영 권장)
- `admission_absolute_only.sh` — 절대 SLO만
- `admission_dryrun.sh` — 위와 같지만 dry-run

```bash
source ms_dev/experiments/admission_ratio_with_safety.sh
python3 ms_dev/expctl/server_run_experiment.py --mode single
```

새 실험이 필요하면 같은 패턴으로 추가. `ms_dev/experiments/README.md` 참고.

#### C. `env.local.sh` 패턴 (개인용 default)

`ms_dev/env.local.sh` 만들면 `env.sh`가 자동으로 source. gitignored이니까 부담 없이 호스트별 설정 가능:

```bash
# ms_dev/env.local.sh
export SGLANG_ADMISSION_PREFILL_COST_MODEL=/abs/path/prefill.json
export SGLANG_ADMISSION_TBT_COST_MODEL=/abs/path/tbt.json
# ... 매번 export하기 귀찮은 것들 ...
```

### 전체 표

| Env var | CLI flag | Default | 의미 |
|---|---|---|---|
| `SGLANG_ADMISSION_TTFT_SLO_MS` | `--admission-ttft-slo-ms` | 미설정(off) | TTFT 절대값 SLO (ms) |
| `SGLANG_ADMISSION_TBT_SLO_MS` | `--admission-tbt-slo-ms` | 미설정(off) | TBT 절대값 SLO (ms) |
| `SGLANG_ADMISSION_TTFT_SLO_RATIO` | `--admission-ttft-slo-ratio` | 미설정(off) | TTFT 비율 (e.g. 2.0) |
| `SGLANG_ADMISSION_TBT_SLO_RATIO` | `--admission-tbt-slo-ratio` | 미설정(off) | TBT 비율 |
| `SGLANG_ADMISSION_PREFILL_COST_MODEL` | `--admission-prefill-cost-model-path` | unset | prefill JSON 경로 |
| `SGLANG_ADMISSION_TBT_COST_MODEL` | `--admission-tbt-cost-model-path` | unset | TBT JSON 경로 |
| `SGLANG_ADMISSION_TBT_EWMA_ALPHA` | `--admission-tbt-ewma-alpha` | 0.1 | EWMA 평활 (작을수록 부드러움) |
| `SGLANG_ADMISSION_TBT_REACTIVE_RATIO` | `--admission-tbt-reactive-ratio` | 0.9 | Stage 3 trip = tbt_slo × 이 값 |
| `SGLANG_ADMISSION_DRY_RUN` | `--admission-dry-run` | 0 | 1이면 거절 안 함, 결정만 기록 |
| `SGLANG_ADMISSION_DECISION_LOG` | `--admission-decision-log` | unset | JSONL 경로 |

**활성화 규칙**: TTFT/TBT의 절대값 또는 비율 중 **하나라도** 설정되면 controller가 활성화. 둘 다 미설정이면 비활성 (server overhead 0).

---

## 8. SLO 튜닝 실전 가이드

### 단계 1 — 평소 워크로드를 dry-run으로 흘림

```bash
source ms_dev/experiments/admission_dryrun.sh
python3 ms_dev/expctl/server_run_experiment.py --mode single --session-name slo_tune_$(date +%Y%m%d_%H%M%S)
# (평소 stress test 워크로드 발사 — 최소 1000건+ 결정 모음)
```

세션이 끝나면 `<session_dir>/admission_decisions.jsonl`에 모든 결정이 쌓여있음.

(SLO를 너무 빡빡하게 잡으면 dry-run 큐에 모든 게 쌓여서 뒤따른 모든 요청이 reject되어 보임 — 결과 분석이 어려워짐. `admission_dryrun.sh`는 의도적으로 느슨하게 설정됨.)

### 단계 2 — Replay로 sweep

```bash
python tools/admission_control/replay_admission.py \
    --decision-log <log>.jsonl \
    --ttft-slo 0 1000 2000 5000 30000 \
    --ttft-slo-ratio 1.5 2.0 3.0 5.0 100 \
    --tbt-slo 0 \
    --tbt-slo-ratio 1.0 \
    --csv-out report.csv
```

(0이나 1.0으로 disable 가능. `100` 같이 큰 값은 사실상 disable.)

### 단계 3 — 표 보고 결정

```
TTFT_SLO  TTFTrx  total  admit  rTTFT  rTTFTr  rej_pct
    1000    1.50    25      2     22       1   92.00%   ← 너무 빡빡, 재검토
    1000    2.00    25      5     19       1   80.00%
    2000    2.00    25     10     14       1   60.00%
    5000    2.00    25     14      8       3   44.00%   ← 합리적
   30000    5.00    25     20      0       5   20.00%   ← 너무 느슨
```

→ 운영 목표 (예: 목표 거절률 ≤ 30%, 또는 p99 latency ≤ 5s) 따라 적정점 선택.

### 단계 4 — 본 운영

```bash
unset SGLANG_ADMISSION_DRY_RUN
export SGLANG_ADMISSION_TTFT_SLO_MS=5000
export SGLANG_ADMISSION_TTFT_SLO_RATIO=2.0
bash ms_dev/start_server_no_pd.sh --metrics
```

⚠️ **dry-run 거절률은 운영 거절률의 상한**. 진짜 거절 모드에선 큰 요청이 미리 거절되니 큐가 안 쌓이고 → 후속 요청들은 admit. 운영 시 거절률은 dry-run 추정치보다 보통 적게 나옴.

---

## 9. 동작 위치 — 요청 처리 파이프라인 어디

```
HTTP POST /generate
    │
    ▼  [http_server.py]
TokenizerManager.generate_request()
    │
    ├─ 토큰화 (text → input_ids)
    │
    ▼  [ZMQ → scheduler 프로세스]
Scheduler.recv()
    │
    ▼  [scheduler.py:_add_request_to_queue]
    ├─ priority 검증
    ├─ max_queued_requests 검사 (큐 길이) ─ HTTP 503
    ├─ ★ admission control 게이트 ──── HTTP 429
    │     ├─ controller.decide(prompt_len, prefix_len, snapshot)
    │     ├─ Prometheus metrics.record_decision()
    │     ├─ DecisionLogger.record() (rank 0만)
    │     └─ admit=False면 AbortReq 송신
    ├─ HiCache 프리페치
    └─ waiting_queue.append(req)        ← 정식 진입
    │
    ▼  [scheduler 메인 루프]
event_loop_normal() → forward_step() → response
```

**왜 HTTP 진입점이 아니라 scheduler 내부?** 결정에 scheduler state (running batch, tree_cache, queue 누적 비용)가 필요해서. HTTP 레이어는 그 정보를 모름.

**거절 시 토큰화 비용은?** 이미 지불됨 (tokenizer 빠르긴 함). prefix 매치 등 복잡한 admission 로직을 하려면 어차피 토큰화 후가 자연스러움. Mooncake 논문도 같은 위치 (Conductor).

---

## 10. 의존성과 한계

### 의존성

- **Cost model JSON 두 개**: 이게 없거나 잘못된 경우 controller는 자동으로 해당 stage를 disable + WARN 로그. 서버는 정상 부팅.
- **`--enable-metrics`**: Prometheus 메트릭 등록은 metrics가 켜졌을 때만 (overhead 0 보장).
- **DisaggregationMode.NULL**: Phase A는 단일 인스턴스만. PD 모드면 controller가 자동 disable + WARN 1회.

### 한계 (꼭 알아둘 것)

1. **Cost model 부정확** → predicted vs actual TTFT/TBT 오차 있음. RMSE 200~300ms 수준. 그래서 Stage 3 EWMA 안전망 필수.
2. **dry-run 거절률 = upper bound**. 진짜 enforcement에서 조절됨.
3. **Heterogeneous batch에서 TBT model 정확도 떨어짐**. Stage 3가 보조.
4. **Cost model 한 번 fit하면 모델/HW 안 바뀌는 동안만 유효**.
5. **HTTP status 코드 자체는 200** (streaming). 429는 finish_reason.status_code 필드로만.

### 향후 개선 가능

- TBT 모델에 max_seqlen, mean_seqlen 항 추가 (heterogeneous batch 정확도 개선)
- prefill cost model에 d×p 교차항 (긴 prefix + 긴 suffix 조합 정확도)
- Online EWMA fit (cost model을 운영 중 자동 보정)
- PD-disaggregated 모드 지원 (Phase B — Conductor 패턴)

---

## 11. 관련 파일 위치 (빠른 참조)

| 컴포넌트 | 위치 |
|---|---|
| 모듈 핵심 (controller, models, tracker) | `python/sglang/srt/managers/admission_control/` |
| 모듈 설계 문서 | 위 디렉터리의 `CLAUDE.md` |
| Scheduler 통합 | `python/sglang/srt/managers/scheduler.py` (`init_admission_control`, `_abort_on_predicted_slo_violation`, `_build_admission_snapshot`) |
| CLI 플래그 | `python/sglang/srt/server_args.py` (`admission_*` 필드들) |
| EWMA hook | `python/sglang/srt/observability/scheduler_metrics_mixin.py` |
| Fit 도구 | `tools/admission_control/fit_cost_model.py` |
| Replay 도구 | `tools/admission_control/replay_admission.py` |
| Cost model JSON | `ms_dev/runtime/cost_models/` (gitignored) |
| Cost model 결과 노트 | `ms_dev/runtime/cost_models/README.md` |
| Shell 와이어링 | `ms_dev/env.common.sh`, `ms_dev/lib_server.sh`, `ms_dev/env.sh` |
| Experiment wrappers | `ms_dev/experiments/admission_*.sh` |
| Personal default (gitignored) | `ms_dev/env.local.sh` |
| Per-session 자동 라우팅 | `ms_dev/expctl/server_run_experiment.py` (`admission_config` block, decision log) |
| 단위 테스트 | `test/registered/admission/test_admission_control.py` (72 cases) |

---

## 12. FAQ

**Q. 왜 dry-run 모드에서 짧은 요청도 거절되었나?**
A. dry-run은 거절 안 하고 큐에 쌓아서, 큰 요청이 누적된 t_queue가 후속 짧은 요청의 pred_TTFT를 부풀려 reject로 분류. 실제 enforcement에서는 큰 요청이 진짜 거절되어 큐가 안 쌓이고 후속 admits.

**Q. Cost model 다시 만들어야 하는 시점은?**
A. (a) 모델 / (b) GPU 종류 / (c) TP 크기 / (d) attention backend / (e) quantization / (f) speculative decoding / (g) mem-fraction-static — 이 중 하나라도 바뀌면.

**Q. PD-disaggregation 모드에서 켜면?**
A. Stage 진입조차 안 함 (`disaggregation_mode != NULL`이면 controller가 ADMIT(DISABLED) 반환 + 첫 호출 시 WARN 1회). Phase A 범위는 단일 인스턴스.

**Q. HTTP 429를 진짜 status code로 받고 싶다.**
A. 현재 SGLang의 streaming generate endpoint는 응답 헤더 200 후에 finish_reason으로 에러 표현. 진짜 HTTP 429를 원하면 OpenAI-compat endpoint 라우터 단에서 변환 필요 (Phase A 범위 밖).

**Q. 비율 기반 SLO에서 Stage 3 (반응형 EWMA)는?**
A. EWMA 안전망은 절대값 ms 임계가 필요해서, 절대 TBT SLO가 설정될 때만 작동. 비율 단독 모드에서는 Stage 3 비활성. 강력한 안전망이 필요하면 절대값과 함께 사용 권장.

**Q. 큐 길이 max_queued_requests와 동시 사용?**
A. 호환됩니다. max_queued_requests가 먼저 (HTTP 503), admission이 그 다음 (HTTP 429). 둘이 다른 이유로 거절.
