# Halo Step Cost Model — design + implementation plan

> 이 문서는 Phase 1 후속 (§18 cost-model cliff) 작업으로 도입할 **Halo Step Cost
> Model** 의 설계, 식, 변수 의미, 데이터 수집·fit 절차, 변경 파일, 실행 순서를
> 정리한 단일 출처. `ms_dev/halo_dev/CLAUDE.md` §18·§20 에서 가리키는 상세 문서.

## 1. 배경 — 왜 새 모델인가

`runtime/cost_models/README.md` item 3 + `ms_dev/halo_dev/CLAUDE.md` §18 에 정리된
TBT cost-model cliff:

- 실측 TBT mean 50.8 ms (λ=0.3, parallel_tool_delay) vs 예측 11.4 ms — ~5x
  underestimate
- 옛 식 `TBT ≈ a + b·bs + c·per_req_kv` 가 `per_req_kv = total_kv / bs` 로
  reparam 되어 두 신호가 한 변수로 합쳐짐 → `b` 가 음수로 fit, batching 효과 무력화
- 결과적으로 Halo slowdown ratio (= actual / predicted) 가 인플레이션

§18 의 변수 후보 (`total_batch_kv`, `num_running_reqs`, `max_seqlen`,
`prefill_tokens`) 는 우연이 아니라 *근거 있는 직관* 이었음 — 같은 변수 묶음이
**MuxWise (ASPLOS '26, "Towards High-Goodput LLM Serving with Prefill-decode
Multiplexing", arXiv 2504.14489)** 의 contention-tolerant estimator 식과
거의 일치. 그 식을 우리 SGLang single-instance 환경에 맞춰 차용한 것이 본
**Halo Step Cost Model**.

참고로 DistServe (OSDI '24) 의 simulator 는 *분리(disaggregated)* 환경에서만
유효한 M/D/1 queueing 가정을 사용하므로 우리 colocated single-instance 에 직접
이식하지 않음. 큐 효과는 admission_control 의 EWMA 안전망(Stage 3) 에 맡김.

## 2. 결정 요약

| ID | 내용 |
|---|---|
| 모델 형태 | Halo Step Cost Model (6 계수, mixed prefill+decode step latency 회귀) |
| Prefill cross-term `Σ(nᵢ·rᵢ)` 포함 | **포함** — 우리 workload 가 cache hit ≥ 90% 라 비중 큼 |
| Contention guard | **없음** — drift 보이면 추후 EWMA 보정 추가 |
| Fit 단위 | **per-forward-step** 으로 회귀. 요청 단위 latency 는 step 합성으로 도출 |
| 모델 구성 | **prefill 항 + decode 항 통합 1 식 / 1 set 의 6 계수** (분리 fit 아님) |
| Solo vs batched | **같은 모델·같은 계수, 입력 모드만 다르게**. caller 가 `Σ` 안에 무엇을 넣는지로 분기 |
| admission_control 적용 | **이번 작업에서는 안 함** — 옛 식 (Mooncake-like, legacy) 그대로 유지. 미래 Phase 2 작업에서 검토 |
| Legacy 공존 | **새 path 가 set 되어 있으면 new, 아니면 legacy** 자동 선택. 양 모드 모두 유지 |
| Naming | **Halo Step Cost Model** (외부 시스템 이름 사용 안 함) |

## 3. 식 정의

### Per-step (학습 대상)

```
T_step ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ
       + θ_d1·Σrⱼ + θ_d2·bs_d
       + θ_c
```

| 변수 | 의미 | 수집 위치 |
|---|---|---|
| `nᵢ` | step 에 포함된 prefill 요청 i 의 *uncached* 새 토큰 수 (chunked prefill 의 경우 이번 step 의 chunk 크기) | scheduler running batch (prefill 진행 중) |
| `rᵢ` | prefill 요청 i 의 *cached prefix* 길이 | `match_prefix_len` 결과 |
| `rⱼ` | decode 요청 j 의 현재 KV 길이 | `req.kv_len_now` (Halo `_halo_build_request_execution_infos` 가 이미 모음) |
| `bs_d` | decode 요청 개수 | `len(running_batch.decode_reqs)` |

### 항별 물리적 의미

| 항 | 의미 | 옛 식에 대응 |
|---|---|---|
| `θ_p1·Σnᵢ²` | 새 토큰끼리 self-attention 의 quadratic 비용. 요청 간 attention 없음 → `(Σnᵢ)²` 아닌 `Σnᵢ²` | `α·d²` (단일 요청만) |
| `θ_p2·Σ(nᵢ·rᵢ)` | 새 토큰 ↔ cached prefix 의 cross-attention. 새 토큰 수가 늘면 prefix 비용도 비례 증가 — 옛 `δ·p` 가 못 잡던 부분 | `δ·p` (linear only) |
| `θ_p3·Σnᵢ` | per-token MLP/FFN compute (attention 외) | `β·d` |
| `θ_d1·Σrⱼ` | decode 한 step 의 attention memory BW 비용. **batch 합산**이 핵심 — 옛 식이 `per_req_kv` 로 *나눠* 잡아 cliff 발생 | `c·per_req_kv` (잘못된 reparam) |
| `θ_d2·bs_d` | decode 요청당 sampler/kernel-launch/output 오버헤드 | `b·bs` |
| `θ_c` | step 고정 오버헤드 (scheduler, sync) | `a` + `γ` 합 |

### Solo 모드 (Halo R1 — slowdown 분모용)

같은 식에 *batch 안에 그 요청 1개만* 가정:

```
solo_T_prefill_total(n, r)  = θ_p1·n² + θ_p2·n·r + θ_p3·n + θ_c
solo_TBT_per_step(r)        = θ_d1·r + θ_d2·1 + θ_c
```

위 두 식은 별도 학습이 *아님* — 6 계수 fit 후 추론 시점에 Σ 를 1요소로 축약한
shortcut. solo 와 batched 호출은 동일 모델·계수.

### Batched 모드 (admission stage 2, Phase 2 R3 lookahead)

식 그대로, caller 가 시나리오에 맞춰 Σ 와 bs_d 를 구성해 호출.
**이번 작업 범위에서는 사용처 없음** — admission_control 은 legacy 그대로,
R3 는 Phase 2 작업. 구조와 메서드는 마련해 두지만 caller 는 없음.

## 4. Slowdown 계산이 cost model 호출의 조합인 이유

Cost model 은 *step latency* 만 답함. Slowdown 은 caller 가 *대수적으로 조립* :

### Phase 1 R1 (현재 진행 중 — 이번에 활성화되는 경로)
```
slowdown_now(i, t) = actual_elapsed_so_far(i, t)            ← 실측 (시계)
                   ──────────────────────────────────────
                     solo_predicted_elapsed_so_far(i, t)    ← cost model SOLO 모드
```
새 job 의 영향은 *분자 (실측)* 에 자연 반영. 분모는 요청 자기 특성만 의존하므로
변하지 않음.

### Phase 2 R3 (lookahead — 이번 범위 아님)
```
expected_slowdown_if_admit(i) = [ actual_so_far(i) + Σ_step batched.step_ms(시나리오) ] / solo_total(i)
```
시나리오마다 batched 모드를 다르게 호출. 같은 6 계수 사용.

## 5. 코드 인터페이스 (`HaloStepCostModel`)

위치: `python/sglang/srt/managers/admission_control/cost_model.py` (기존 공유
파일 안에 *추가*. admission_control 모듈에 있는 이유는 Halo 가 이미 reuse 해온
패턴 유지 — 모듈 분리는 하지 않음).

```python
@dataclass(frozen=True)
class HaloStepCostModel:
    theta_p1: float  # Σnᵢ²
    theta_p2: float  # Σ(nᵢ·rᵢ)
    theta_p3: float  # Σnᵢ
    theta_d1: float  # Σrⱼ
    theta_d2: float  # bs_d
    theta_c:  float  # const
    metadata: Dict[str, Any]

    # caller가 직접 batch composition을 구성해 호출 (Phase 2용 — 지금은 미사용)
    def estimate_step_ms(
        self,
        prefill_infos: List[Tuple[int, int]],  # [(n, r), ...]
        decode_infos: List[int],               # [r, ...]
    ) -> float: ...

    # Halo R1용 shortcut (분모 — 1 요청, batch 비어있다고 가정)
    def estimate_solo_prefill_total_ms(self, n: int, r: int) -> float: ...
    def estimate_solo_tbt_ms(self, r: int) -> float: ...

    @classmethod
    def from_json(cls, path) -> "HaloStepCostModel": ...

def try_load_halo_step_cost_model(path: Optional[str]) -> Optional[HaloStepCostModel]: ...
```

옛 `PrefillCostModel`, `TBTCostModel` 은 그대로 둠.

## 6. JSON 스키마

```json
{
  "form": "halo_step_v1",
  "theta_p1": 1.23e-7,
  "theta_p2": 4.56e-7,
  "theta_p3": 0.078,
  "theta_d1": 5.67e-5,
  "theta_d2": 0.34,
  "theta_c":  9.81,
  "fit_metadata": {
    "model": "Llama-3.3-70B-Instruct",
    "hw": "B200x4 TP=4",
    "n_samples": 12345,
    "rmse_ms": 3.2,
    "r2": 0.96,
    "fit_at": "2026-05-13T16:30:00Z",
    "source_jsonl": "runtime/sessions/<id>/halo_cost_samples.jsonl"
  }
}
```

`form` 필드는 미래 호환성용 (`halo_step_v2` 등 추가 가능). 옛 prefill / tbt
JSON 은 `form` 필드 없음 — 그쪽 loader 가 그대로 사용.

## 7. CLI 플래그 / env vars

신규 (모두 unset 시 자동 legacy fallback):

| 플래그 | env var | 효과 |
|---|---|---|
| `--halo-step-cost-model-path <path>` | `SGLANG_HALO_STEP_COST_MODEL` | Halo 가 이 모델 사용. set 되면 legacy `--halo-prefill-cost-model-path` / `--halo-tbt-cost-model-path` 무시 (INFO 로그) |
| `--halo-cost-model-sample-log <path>` | `SGLANG_HALO_COST_MODEL_SAMPLE_LOG` | per-step 샘플 JSONL 출력. 데이터 수집 단계에서만 켬 |

legacy 플래그 (`--halo-prefill-cost-model-path`, `--halo-tbt-cost-model-path`,
admission_control 의 4개) 는 **그대로 살아있음**. 새 path 가 없으면 자동
legacy 동작.

우선순위:
```
--halo-step-cost-model-path 있음     → HaloStepCostModel 사용
없음 + legacy 두 path 있음            → 옛 PrefillCostModel + TBTCostModel 사용
없음 + legacy 도 없음                 → slowdown sweep no-op (현 동작)
```

## 8. Legacy 공존 규칙

- admission_control 의 cost-model 로드 경로·식·decide() 는 **이번 작업에서 손대지 않음**. Mooncake-like (α·d² + β·d + γ + δ·p, a + b·bs + c·per_req_kv) 가 admission 의 "legacy 모드" 로 그대로 살아 있음.
- Halo 는 새 path 가 set 되면 자동 전환. 옛 path 만 set 이면 옛 동작 그대로.
- 두 경로 다 set → 새 모델 우선, 옛 path 는 INFO 로그로 무시 통지.
- 미래 admission_control 도 unified 식으로 가고 싶을 때를 위한 구조 분기는 **이번 작업에서 만들지 않음** (사용자 결정 — 단순히 손도 안 댐).

## 9. Per-step 데이터 수집 (instrumentation)

위치: `observability/scheduler_metrics_mixin.py` 의 step_time_dict 업데이트
지점 근처. admission_control 의 TBT EWMA tracker 도 거기서 update 되므로 자연
스러운 지점.

조건:
- `attn_tp_rank == 0` 만 기록 (TP-dedup, 기존 Halo / admission 패턴과 동일)
- `--halo-cost-model-sample-log` 가 set 된 경우에만 동작
- 옛 / 새 cost-model 사용 여부와 *독립* — 데이터만 흘림

라인 포맷 (rank-0, 한 step 당 1 라인):
```json
{
  "step_idx": 12345,
  "ts_ns": 1717800000000000,
  "sum_n_sq": 1048576.0,
  "sum_nr":   2097152.0,
  "sum_n":    1024,
  "sum_r":    13000,
  "bs_d":     2,
  "step_time_ms": 51.2,
  "num_prefill_reqs": 1,
  "num_decode_reqs": 2,
  "max_kv": 8000
}
```

`num_prefill_reqs`, `num_decode_reqs`, `max_kv` 는 옵션 필드 (fit 자체에는 안 쓰이지만 진단·다항식 확장 시 유용).

`ms_dev/expctl/server_run_experiment.py` 가 session 시작 시 `SGLANG_HALO_COST_MODEL_SAMPLE_LOG` 를 `<session_dir>/halo_cost_samples.jsonl` 로 auto-route (pin 안 됐을 때).

## 10. Fit 절차

새 스크립트: `tools/halo/fit_halo_cost_model.py`

```bash
python tools/halo/fit_halo_cost_model.py \
    --samples runtime/sessions/<id>/halo_cost_samples.jsonl \
    --output  runtime/cost_models/halo_step_llama3-70b_b200x4.json \
    --model   "Llama-3.3-70B-Instruct" \
    --hw      "B200x4 TP=4"
```

알고리즘:
- numpy OLS:
  ```
  X = [[sum_n_sq, sum_nr, sum_n, sum_r, bs_d, 1.0], ...]
  y = [step_time_ms, ...]
  θ = np.linalg.lstsq(X, y, rcond=None)[0]
  ```
- 잔차 RMSE, R² 산출
- 옵션: `--filter "bs_d>=1"` 같은 식으로 prefill-only / decode-only 부분
  데이터로 회귀해 계수 sanity check 가능
- 옵션: `--bootstrap-ci 1000` (계수 CI)

산출물 위치: `runtime/cost_models/halo_step_<model>_<hw>.json`
(기존 `prefill_*.json`, `tbt_*.json` 와 같은 디렉터리, **gitignored**)

## 11. Halo 측 로딩·호출 변경

`HaloConfig`:
```
step_cost_model_path: Optional[str]   ← 새 필드
prefill_cost_model_path: Optional[str]  (옛, 유지)
tbt_cost_model_path: Optional[str]      (옛, 유지)
```

`HaloController.__init__`:
```
step = try_load_halo_step_cost_model(config.step_cost_model_path)
if step is not None:
    self.tracker = SlowdownTracker(step_model=step, registry=...)
else:
    prefill = try_load_prefill_cost_model(config.prefill_cost_model_path)
    tbt     = try_load_tbt_cost_model(config.tbt_cost_model_path)
    self.tracker = SlowdownTracker(legacy_prefill=prefill, legacy_tbt=tbt, ...)
```

`SlowdownTracker`:
- `use_step: bool`, `step_model`, `legacy_prefill`, `legacy_tbt`
- `compute_request_slowdown(...)` 안:
  ```
  if self.use_step:
      solo_prefill_ms = self.step_model.estimate_solo_prefill_total_ms(n, r)
      solo_tbt_ms     = self.step_model.estimate_solo_tbt_ms(r_now)
  else:
      solo_prefill_ms = self.legacy_prefill.estimate_ms(n, r)
      solo_tbt_ms     = self.legacy_tbt.estimate_ms(bs=1, per_req_kv=r_now)
  ```
- slowdown 정의·기록 포맷·JSONL row 는 **변경 없음** → monitoring panel / verify
  스크립트 그대로 동작

## 12. 변경 파일 목록 (예정)

| 파일 | 변경 종류 |
|---|---|
| `python/sglang/srt/managers/admission_control/cost_model.py` | `HaloStepCostModel` + loader 추가 (옛 클래스 그대로) |
| `python/sglang/srt/server_args.py` | `--halo-step-cost-model-path`, `--halo-cost-model-sample-log` 두 플래그 추가 |
| `python/sglang/srt/managers/halo/controller.py` | `step_cost_model_path` 로드 분기 |
| `python/sglang/srt/managers/halo/slowdown_tracker.py` | `use_step` 분기, solo shortcut 호출 |
| `python/sglang/srt/observability/scheduler_metrics_mixin.py` (또는 별도 모듈) | per-step JSONL writer (rank-0 only) |
| `tools/halo/fit_halo_cost_model.py` | 신규 fit 스크립트 |
| `tools/halo/__init__.py`, `tools/halo/CLAUDE.md` | 디렉터리 정비 |
| `test/registered/halo/test_halo_phase1.py` 또는 신규 | `HaloStepCostModel` 단위 테스트 (식 sanity, JSON roundtrip, solo↔batched 일관성) |
| `ms_dev/env.common.sh`, `ms_dev/lib_server.sh` | 두 env var → 두 CLI 플래그 변환 |
| `ms_dev/expctl/server_run_experiment.py` | `halo_cost_samples.jsonl` auto-route + meta 기록 |
| `ms_dev/expctl/monitoring_view.py` | runtime feature-flag 행에 `halo_step_cost_model=loaded/legacy/off` 표시 |
| `runtime/cost_models/README.md` | 새 `halo_step_*.json` 형식 인덱스 + cliff 해결 노트 |
| `python/sglang/srt/managers/halo/CLAUDE.md` | "Cost model brittleness inherited" 섹션 갱신, prediction_model.md 참조 |
| `ms_dev/halo_dev/CLAUDE.md` | §20 (요약 + 본 문서 참조) 추가 |

## 13. 실행 순서 (단계별 PR-sized)

현재 실험에 영향 없는 안전 단계부터:

1. ✅ **`HaloStepCostModel` 클래스 + loader + 단위 테스트** (cost_model.py)
   — admission_control 모듈·Halo controller 둘 다 안 건드림. fixture JSON 1개로 roundtrip/estimate 동작 검증. 22 tests in `test/registered/halo/test_halo_step_cost_model.py`.

2. ✅ **per-step instrumentation + 단위 테스트**
   — `--halo-cost-model-sample-log` 플래그 추가, `managers/halo/cost_model_sampler.py` 신규. 라인 포맷 + rank-0 dedup + 파일 path 비어있을 때 no-op 검증. 13 tests in `test/registered/halo/test_halo_cost_model_sampler.py`. overlap loop 에서는 sampler 자동 비활성 + WARN — 사용자는 `--disable-overlap-schedule` 로 fit data 수집.

3. ✅ **`tools/halo/fit_halo_cost_model.py` + JSONL fixture 단위 테스트**
   — synthetic 데이터로 OLS 회귀가 known θ 복원하는지 확인. 7 tests in `test/registered/halo/test_fit_halo_cost_model.py`. Noiseless 데이터에서 6 계수 정확 복원.

4. ✅ **Halo 측 로딩·호출 분기 + 단위 테스트**
   — `controller.py`, `slowdown_tracker.py` 변경. `--halo-step-cost-model-path` set 되면 새 경로, 아니면 legacy. 4 new tests in `test/registered/halo/test_halo_phase1.py` (43 total).

5. ⏳ **데이터 수집 1회** (사용자 실행)
   — production-like workload (parallel_tool_delay λ=0.3 sweep 등) 짧게 (예: 5–10 min) 돌려 `halo_cost_samples.jsonl` 생성. **반드시 `--disable-overlap-schedule` 켤 것** (overlap mode 면 sampler 자동 비활성).

6. ⏳ **Fit → 신 JSON 생성 → 실험 검증**
   — 5 단계 결과물로 Halo unified mode 켜고 짧은 sweep. slowdown ratio 가 cliff (5x) 없이 정상 분포 나오는지.

7. ✅ **문서 갱신** (`runtime/cost_models/README.md`, `tools/halo/CLAUDE.md`, `python/sglang/srt/managers/halo/CLAUDE.md`, 본 문서) — monitor panel 의 step-model 표시는 minor item 으로 보류.

코드 단계 (1–4 + 7) 는 끝났고 현재 실험에 영향 0. 데이터 수집·검증 (5–6) 은 사용자 환경에서 진행.

## 14. 미정·주의 사항

- **chunked prefill 토큰 정의**: 한 step 에 들어가는 `nᵢ` 는 *그 step 의 chunk* 크기. fit·sample 수집 모두 step 단위라 정합 — 따로 정규화 불필요. 다만 instrumentation 구현 시 `chunked_prefill_size` cap 가 정확히 반영됐는지 검증 필요.
- **Spec decoding / quantization 변경 시 재fit 필수**. JSON `fit_metadata` 에 model/hw/serve_args 핵심 항목 기록.
- **Mixed prefill+decode step 비중**: 데이터 수집 시 workload 가 두 종류 step 을 골고루 생성하는지 확인. parallel_tool_delay λ=0.3 sweep 은 충분히 mix 되리라 예상.
- **Contention guard / EWMA 보정**: 본 작업에서는 미포함. 6 단계 검증에서 drift 가 SLO 가까이 보이면 그때 추가 결정.
- **Phase 2 admission 통합**: 본 작업 범위 아님. 별도 작업으로 결정.

## 15. First-fit validation result (2026-05-13)

§13 의 단계 6 "Fit → 짧은 검증 실험" 의 결과 기록.

### 실험 조건

| 항목 | 값 |
|---|---|
| Fit 데이터 | `260513_2014_halo_cost_fit_lambda_0p3` (legacy halo run + sampler) |
|   샘플 수 | 6,982 step (DECODE 6337 / EXTEND 645) |
|   부하 | parallel_tool_delay λ=0.3, 10 분 |
| Fit 결과 | `runtime/cost_models/halo_step_llama3-70b_b200x4.json` |
|   RMSE | 22.99 ms (mean step time 85 ms 의 27%) |
|   R² | 0.9636 |
|   계수 | θ_p1=2.23e-7, θ_p2=5.13e-6, θ_p3=0.0197, θ_d1=7.03e-5, **θ_d2=−1.10**, θ_c=34.41 |
| 검증 세션 (step 모델) | `260513_2147_halo_step_verify_lambda_0p3` (같은 λ=0.3, 10 분) |
| Baseline (ground truth) | `runtime/baseline/baseline_20260424-180204` (concurrency=1) |

### Ground truth 와의 비교 (per-request slowdown)

`real_slowdown = verify_latency / baseline_latency`, 매칭된 chain_call = 688건.
Halo report 는 active-job snapshot 의 `slowdown_max` (n≈5,500).

| 통계 | **REAL (ground truth)** | OLD model (legacy) | NEW model (step) |
|---|---|---|---|
| mean | **3.44** | 4.34 (+26 %) | 2.96 (−14 %) |
| p50  | **3.06** | 4.20 (+38 %) | 3.00 (≈정확) |
| p75  | **4.59** | 5.27 (+15 %) | 3.85 (−16 %) |
| p90  | **6.13** | 6.47 (+6 %)  | 4.47 (−27 %) |
| p95  | **7.05** | 7.07 (≈정확) | 4.73 (−33 %) |
| p99  | **8.98** | 7.84 (−13 %) | 4.99 (−44 %) |
| max  | **12.24** | **23.19 (+89 %)** | 5.23 (−57 %) |

### 결론 — 어디까지 정확해졌나

| 차원 | 옛 식 (legacy) | 새 식 (step) — 본 작업 결과 |
|---|---|---|
| 평균 정확도 | +26 % 인플레이션 | **−14 % deflation, 약 2x 개선** |
| 중앙값 (p50) | +38 % 인플레이션 | **≈정확** |
| Tail (p99 / max) | **+89 % 인플레이션 (cliff)** | **−44 % ~ −57 % deflation** |
| 분포 분산 | broad-tail (max 23) | tight (max 5.23) |

핵심 정리:
- ✅ **§18 의 worst-case cliff (+89 % 인플레이션) 가 해소됨** — Halo R1 의 가장 큰 결함이었던 부분.
- ✅ **평균·중앙값 정확도 크게 향상** (옛 식 p50 +38% → 새 식 p50 ≈정확).
- ⚠️ **반대 방향의 새로운 오차: tail underestimate**. 진짜 worst-case 12.2× 를 5.2× 로만 보고 (−57 %). 즉 *진짜 SLO 위반* 을 *없는 것처럼* 가릴 위험.

### 한 줄로

> 옛 식이 *없는 SLO 위반을 만들어내던* 문제는 사라졌지만, 새 식은 *진짜 SLO 위반을 가릴* 수 있다.
> 두 오차의 *방향* 이 반대이며, 평균적으로는 새 식이 ground truth 에 약 2 배 더 가깝다.

### 왜 tail 을 underestimate 하나 — 추측 원인

1. **Fit 데이터의 부하 spectrum 부족.** λ=0.3 단일 10 분 sweep 으로는 진짜 peak load step 이 충분히 모이지 않음. 결과적으로 high-`Σrⱼ` 영역의 step time 을 fit 하지 못함.
2. **θ_d2 = −1.10 (음수).** `Σrⱼ` 와 `bs_d` 의 multicollinearity 때문에 OLS 가 *batch 클수록 step time 깎는 방향* 으로 계수를 분배. 진짜 peak 부하 (batch 큰 + KV 큰) 에서 예측이 낮게 나옴. 옛 식의 `b=−0.28` 과 같은 증상의 변형.
3. **회귀 손실이 단순 squared-error.** Tail 의 큰 step time 샘플은 수가 적어 회귀가 무시함.

### 얼마나 더 정확하게 할 수 있나 — 개선 경로 (효과 큰 순)

| ID | 방법 | 예상 효과 | 비용 |
|---|---|---|---|
| **V1** | **부하 spectrum 확장 + 긴 sweep**: λ=[0.1, 0.3, 0.5, 0.7] × 각 30~60 분 → high-load step 충분 수집 → refit. θ_d2 가 양수로 수렴할 가능성 큼. | tail underestimate 대폭 완화 예상 (가장 큰 효과) | 데이터 수집 ~3 시간 + refit |
| **V2** | **Ridge regression** 또는 `bs_d` 항 제거. multicollinearity 흡수. | 계수의 *합리성* 향상 — θ_d2 음수 사라짐. tail 정확도도 따라 향상 가능. | 코드 ~30 LoC + refit |
| **V3** | **Tail-weighted loss**: 큰 step time 샘플에 가중치 (예: `weight = step_time_ms^α`, α=1). | tail 정확도 직접 향상. 그러나 평균 정확도 trade-off 가능. | 코드 ~30 LoC + refit |
| **V4** | **§18 추가 변수**: `max_seqlen`, `num_prefill_in_progress` 도입 → 6→8 계수 모델로 확장. | 잔차 흡수, 잠재력 있지만 multicollinearity 가중 위험 | 코드 ~50 LoC + fit 데이터 재수집 |
| **V5** | **Contention guard (×1.2~1.3)**: 보수적 배수로 tail 한 차례 보정. | 직관적 patch 지만 분포 *전체* 를 위로 밀어서 정확도는 못 보장 | 코드 ~5 LoC |

추천: **V1 → (필요 시) V2 → 필요 시 V3** 순서. V1 만으로도 tail 정확도 크게 좋아질 가능성이 큼.

### 사용 결정 (당장)

현 step model 은 **Phase 1 R1 (관측 only) 의 분모로 사용 가능** — 평균·중앙값이 정확하므로 *대다수 정상 시나리오의 slowdown 판정* 에 옛 식보다 안정적. 단 **Phase 2 R3 (admission decision) 에서 lookahead 로 쓰기 전엔 V1 refit 필수** — admission 결정이 tail underestimate 에 의존하면 *진짜 SLO 위반인 요청을 admit* 할 위험.

## 16. MuxWise 식과의 관계 — 무엇을 차용하고 무엇이 다른가

§1 에서 MuxWise (ASPLOS '26, arXiv 2504.14489) 의 식이 *영감의 출처* 라고 적었음. 본 절은 *그대로 쓴 게 아니라 어떤 점이 같고 어떤 점이 우리 환경에 맞춰 다른가* 를 명시.

### 같은 부분 (식 구조 자체)

| MuxWise (원본) | 우리 Halo Step Cost Model |
|---|---|
| Prefill 항: `Σnᵢ²`, `Σ(nᵢrᵢ)`, `Σnᵢ`, 상수 | **동일 변수 묶음 — 그대로 차용** |
| Decode 항: `Σrⱼ`, `bs`, 상수 | **동일 변수 묶음 — 그대로 차용** |
| 변수 정의 (n=새 토큰, r=cached prefix 또는 KV 길이) | **동일 정의** |

→ §18 의 cliff 진단 (`total_batch_kv`, `num_running_reqs`, `Σnᵢ²`, `Σ(nᵢrᵢ)`) 결론이
MuxWise 식과 정확히 일치한다는 것을 §1 에서 확인했고, 그 *변수·항 구조* 는 그대로 사용.

### 다른 부분 (우리 환경에 맞춰 재설계)

| 차원 | MuxWise | Halo Step Cost Model | 왜 다른가 |
|---|---|---|---|
| **모델 수** | 두 개 — Prefill 모델 + Decode 모델 (각자 θ set) | **하나 — 통합 6 계수 (prefill 항 + decode 항이 같은 식)** | MuxWise 는 SM-partition 으로 prefill/decode 를 *물리적 분리* 해서 두 식을 따로 학습. SGLang continuous batching 은 한 forward step 안에 prefill+decode 가 섞여 (`ForwardMode.MIXED`) 분리 불가 — 통합 모델 필수 |
| **Fit 단위** | per-request 완료 시간 | **per-forward-step latency** | MuxWise: SM partition 안에서 요청 끝까지. Halo: chunked prefill 때문에 한 요청이 여러 step 으로 쪼개짐. step 단위 회귀 + 추론 시 step 합성 |
| **Solo↔batched 분리** | 별도 호출 경로 | **같은 식, 입력 모드만 다름**. caller 가 Σ 안에 1 요소 넣으면 solo, 실제 batch 넣으면 batched. 같은 6 계수 공유 | Halo R1 (분모, solo) + admission stage 2 (batched) + Phase 2 R3 (lookahead, batched) 세 경로가 같은 cost model 을 다른 입력으로 호출하는 것이 자연 |
| **Contention guard** | offline-profile 한 worst-case 배수 (A100 ×1.20, H100 ×1.30) 곱함 | **없음** (본 작업 결정 — §14). 필요 시 admission 의 EWMA 안전망 활용 | MuxWise 는 SM partition 내 *unmodeled* contention 보정용. 우리는 single-instance 라 모든 contention 이 batch composition 에 이미 인코딩됨 — 같은 guard 가 부적절 |
| **사용처** | MuxWise dispatcher 의 *intra-GPU prefill-decode multiplexing* 결정 | Halo *job-level slowdown 측정의 분모* (Phase 1 R1) + 미래의 R3 lookahead admission | 시스템 목적이 완전히 다름 — MuxWise 는 *스케줄러*, Halo 는 *job-level SLO 추적기* |
| **검증 지표** | prefill 최대 오차 8.16 %, decode 8.84 % (논문) | mean −14 %, p99 −44 % (§15) — workload·HW·fit 데이터 양 다름, 1:1 비교 부적절 | 직접 비교 불가하나 *방향은 비슷* (cliff 해소). 정확도 격차는 V1 refit 으로 좁힐 여지 |

### 한 줄 요약

> 식의 *변수와 항 구조* 는 MuxWise 의 contention-tolerant estimator 에서 가져왔지만,
> *모델 수 (1개 통합)*, *fit 단위 (per-step)*, *호출 모드 (solo↔batched 통합)*, *guard 정책 (없음)*,
> *사용처 (slowdown 분모)* 는 SGLang single-instance + continuous batching 환경에 맞춰 재설계.

## 17. Split 모드 — prefill / decode 분리 회귀 (2026-05-13 follow-up)

§15 의 한계 (특히 θ_d2 = −1.10 음수, tail underestimate) 가
"통합 식의 multicollinearity" 때문이 아닐까 하는 의심에서 출발한 follow-up.

### 17.1 동기 — 우리 환경에선 분리가 자연스럽다

수집된 6,982 step 의 forward_mode 분포 (260513_2014 세션):

| forward_mode | 개수 | 비고 |
|---|---|---|
| EXTEND | 645 | 모두 *pure prefill* (`bs_d=0`). decode 안 섞임 |
| DECODE | 6,337 | 모두 *pure decode* (`num_prefill_reqs=0`). prefill 안 섞임 |
| **MIXED** | **0** | `--enable-mixed-chunk` 미사용 환경 |

즉 우리 워크로드는 **chunked prefill 중 decode 가 같은 step 에 끼는 경우가 없음** — prefill step 과 decode step 이 시간적으로 완전히 분리. §16 에서 "통합이 필수" 라고 적은 건 (MIXED 발생 시 한 식이 두 케이스 다 잡아야 한다는) 보수적 가정이었고, 실제로는 MIXED 가 없어 **분리해도 데이터 누락 없음**.

### 17.2 식 정의

**Prefill 식 (EXTEND step 만 fit, 4 계수):**
```
T_prefill_step ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ + θ_c_p
```

**Decode 식 (DECODE step 만 fit, 3 계수):**
```
T_decode_step ≈ θ_d1·Σrⱼ + θ_d2·bs_d + θ_c_d
```

총 7 계수 — 통합 식 (6 계수, 공통 θ_c 하나) 대비 상수가 prefill / decode 로 *분리*. MuxWise 원본 식과 정확히 동일한 구조.

### 17.3 Split fit 결과 (260513_2014 데이터 재활용)

| 계수 | UNIFIED (§15) | **SPLIT** | 변화 |
|---|---|---|---|
| θ_p1 (Σnᵢ²) | +2.23e−7 | +9.21e−7 | ×4.13 |
| θ_p2 (Σnᵢrᵢ) | +5.13e−6 | +4.92e−6 | ×0.96 |
| θ_p3 (Σnᵢ) | +0.0197 | +0.0130 | ×0.66 |
| θ_d1 (Σrⱼ) | +7.03e−5 | +1.51e−5 | ×0.22 |
| **θ_d2 (bs_d)** | **−1.104 (음수, 비물리적)** | **+0.183 (★ 양수, 물리적)** | **부호 flip** |
| 상수 | θ_c = +34.41 | θ_c_p = +70.0, θ_c_d = +21.8 | 분리 |

| 지표 | UNIFIED | SPLIT |
|---|---|---|
| 결합 RMSE | 22.99 ms | **21.33 ms** (−7 %) |
| Decode-side RMSE | (혼합) | **6.46 ms** (DECODE step 6337 개에 대해) |
| Prefill-side RMSE | (혼합) | 67.21 ms (EXTEND step 645 개 — 적어 noise 큼) |
| Decode-side R² | (혼합) | **0.9497** |

### 17.4 핵심 예측 변화 — 분모가 더 작아짐

같은 입력에 대한 unified vs split 의 *solo prediction* 차이 (Halo R1 의 분모):

| 시나리오 | UNIFIED 예측 | SPLIT 예측 | 변화 |
|---|---|---|---|
| Solo prefill (n=2000, r=18000) | 259.4 ms | 276.8 ms | +7 % |
| **Solo TBT (r=1000, bs=1)** | **33.4 ms** | **22.0 ms** | **−34 %** |
| Solo TBT (r=20000, bs=1) | 34.7 ms | 22.2 ms | −36 % |
| Decode step (bs=16, Σr=200k) | 30.8 ms | 27.7 ms | −10 % |

**의미**:
- Decode 분모가 **−34 % 작아짐** → Halo R1 의 slowdown ratio (= actual / predicted_solo) 가 **더 크게** 보고됨
- §15 의 tail underestimate (실제 max 12.24 → unified 보고 5.23) 가 **split 으로 완화될 가능성**: 분모 작아지면 max 도 위로 이동
- 예상 (보수적): split 모드 mean smax ≈ 4~5 (ground truth 3.44 위로, unified 2.96 아래), max ≈ 7~10 (ground truth 12.24 향으로)

### 17.5 무엇이 더 좋아졌나 — 한 줄 결론

> Split 모드는 **(1) θ_d2 부호를 정상화** (multicollinearity 해소) 하고 **(2) decode 분모를 작게 다시 잡아** §15 의 *tail underestimate* 를 어느 정도 완화함. Combined RMSE 도 −7 % 개선.
> Prefill 쪽 RMSE 가 큰 건 EXTEND sample 수 (645) 가 작은 탓 — 더 모으면 함께 좋아짐.

### 17.6 언제 unified vs split 을 써야 하나

| 조건 | 권장 form |
|---|---|
| `--enable-mixed-chunk` 켜져 있거나 MIXED step 발생 ≥ ~1% | **Unified** (한 식이 두 케이스 다 흡수) |
| MIXED step 거의 0%, prefill / decode 가 시간적으로 분리 | **Split** ★ 권장 (multicollinearity 회피, 상수 분리) |
| 부하 spectrum 다양 (λ 여러 + 긴 sweep) + fit data 충분 | 둘 다 비슷할 가능성 |

본 호스트의 기본 구성 (mixed_chunk off) 에서는 **split 이 default**.
2026-05-14 이후 default 로 채택:

- `tools/halo/fit_halo_cost_model.py --form` default = `split`.
  unified 로 fit 하려면 명시적으로 `--form unified` 필요.
- `ms_dev/experiments/halo_base.sh` 가 `SGLANG_HALO_STEP_COST_MODEL`
  을 split JSON path 로 default export. legacy 두 모델 path 도 fallback
  으로 살아 있지만, step 모델이 set 되어 있으면 server 가 그걸 우선 사용.

### 17.7 검증 명령어 (사용자 환경)

새 split JSON 으로 서버 다시 띄워서 §15 와 같은 방식의 ground-truth 비교 진행:

```bash
python run_experiment.py \
    --workload swe_bench_coding_parallel_tool_delay \
    --mode poisson-sweep \
    --baseline-dir results/baseline_20260424-180204 \
    --lambda-list 0.3 --duration-min 10 \
    --halo-enabled --tau 5.0 \
    --session-name halo_step_split_verify \
    --restart-server \
    --sglang-start-cmd "cd /home/nxclab/sglang/ms_dev/expctl && SGLANG_HALO_STEP_COST_MODEL=/home/nxclab/sglang/ms_dev/runtime/cost_models/halo_step_split_llama3-70b_b200x4.json python3 run_experiment.py --mode single --single-port 31000"
```

이후 분석:
- `halo_jobs.jsonl` 의 slowdown_max 분포 vs `260513_2147_halo_step_verify_lambda_0p3` (unified) vs ground truth
- 기대: split 의 mean / p99 / max 가 ground truth (3.44 / 8.98 / 12.24) 에 더 가까워질 것

### 17.8 코드 인터페이스

같은 `HaloStepCostModel` 클래스가 두 form 모두 처리 — caller 코드 변경 없음.

`from_json` 이 `form` 필드를 보고 자동 분기:
- `halo_step_v1` → unified mode (θ_c 단일)
- `halo_step_split_v1` → split mode (θ_c_p / θ_c_d)

`estimate_solo_prefill_total_ms`, `estimate_solo_tbt_ms`, `estimate_step_ms` 가 내부적으로 form 따라 올바른 intercept 사용. 외부 코드 (SlowdownTracker, admission controller) 는 그대로.

Fit 스크립트 사용법:
```bash
# Split form — default since 2026-05-14
python tools/halo/fit_halo_cost_model.py \
    --samples <jsonl> --output <json> \
    --model "<name>" --hw "<descriptor>"

# Unified form (explicit opt-in)
python tools/halo/fit_halo_cost_model.py \
    --samples <jsonl> --output <json> \
    --model "<name>" --hw "<descriptor>" \
    --form unified
```
