# PD Experiment Controller (`expctl`)

`run_pd_experiment.py`는 `prefill/decode/router`를 함께 실행하고,
실행 시작 시각 기반 세션 폴더에 프로세스 로그와 메트릭을 저장합니다.

- 기본 세션 폴더: `/workspace/sglang/ms_dev/runtime/sessions/YYYYMMDD_HHMMSS`
- 종료: `Ctrl+C` (graceful stop)

## What It Logs

- 프로세스 로그:
  - `process_logs/prefill.stdout.log`, `prefill.stderr.log`
  - `process_logs/decode.stdout.log`, `decode.stderr.log`
  - `process_logs/router.stdout.log`, `router.stderr.log`
- 메트릭(JSONL):
  - `metrics/prefill_metrics.jsonl`
  - `metrics/decode_metrics.jsonl`
  - `metrics/router_metrics.jsonl`
  - `metrics/gpu_metrics.jsonl` (`nvidia-smi` 사용 가능 시)
- 메타 정보:
  - `meta/run_meta.json`, `meta/readiness.json`, `meta/run_end.json`

포함되는 대표 메트릭:
- Decode throughput: `sglang:gen_throughput`
- Queue: `sglang:num_queue_reqs`, `sglang:num_decode_prealloc_queue_reqs`, `sglang:num_decode_transfer_queue_reqs` 등
- TTFT/TBT: `sglang:time_to_first_token_seconds*`, `sglang:inter_token_latency_seconds*`
- KV/캐시: `sglang:cache_hit_rate`, `sglang:kv_transfer_*`, `sglang:hicache_host_*`(HiCache 사용 시)
- GPU 메모리/활용률: `gpu_metrics.jsonl` (`nvidia-smi` 수집)

## Live Terminal Monitor

기본적으로 실행 중 터미널에 상태판이 주기적으로 갱신됩니다.

- 프로세스 상태: prefill/decode/router `RUNNING/EXIT/DISABLED`
- prefill/decode: running/queue/prealloc/transfer, throughput, token usage, cache hit, TTFT/TBT 평균
- router: http/worker active, pool size, circuit breaker 상태(open/half-open), router TTFT/TPOT 평균(노출 시)
- GPU: 총 메모리 사용량, 평균 util, 최대 온도
- TTY 환경에서는 상태/핵심 숫자 하이라이트 컬러가 자동 적용 (`NO_COLOR` 또는 `--no-color`로 비활성화)
- IDE 실행처럼 non-TTY 환경이면 `--force-color`로 강제 컬러 가능

## Quick Start

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py
```

## Decode Offload Default

현재 `ms_dev/env.sh` 기본값은 **PD decode GPU KV를 host/storage로 offload**하도록 설정되어 있습니다.

- `SGLANG_PD_DECODE_HICACHE_ENABLE=0` (decode L2 hierarchical-cache 플래그는 비활성)
- `SGLANG_PD_DECODE_OFFLOAD_ENABLE=1`
- `SGLANG_PD_DECODE_HICACHE_STORAGE_BACKEND=file`
- `SGLANG_PD_DECODE_HICACHE_SIZE=256` (GB)
- decode launch 시 `--disaggregation-decode-enable-offload-kvcache` 자동 포함

해제하려면 실행 전 아래를 설정하세요.

```bash
export SGLANG_PD_DECODE_OFFLOAD_ENABLE=0
```

## Usage Examples

### 1) 기본 실행 (기본 포트)

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py
```

기본 포트:
- prefill: `30002`
- decode: `30001`
- router API: `30000`
- router metrics: `29000`

### 2) 포트 변경

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py \
  --prefill-port 31002 \
  --decode-port 31001 \
  --router-port 31000 \
  --router-metrics-port 39000
```

### 3) 세션 이름/저장 경로 지정

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py \
  --session-name long_stress_v1 \
  --session-root /workspace/sglang/ms_dev/runtime/sessions
```

### 4) 메트릭 폴링/상태판 주기 조정

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py \
  --scrape-interval 1.0 \
  --status-interval 3.0 \
  --metrics-timeout 5.0
```

### 5) 상태판 출력 끄기 (로그만 수집)

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --quiet-status
```

### 5-1) 컬러 출력 비활성화

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --no-color
```

### 5-2) 컬러 출력 강제

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --force-color
```

### 6) 일부 역할 제외 실행

```bash
# router 없이 prefill+decode만
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --no-router

# decode 없이 prefill+router만
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --no-decode
```

### 7) 메트릭/라우터 호스트 분리

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py \
  --metrics-host 127.0.0.1 \
  --router-host 127.0.0.1
```

## CLI Options

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --help
```

주요 옵션:
- `--session-root`
- `--session-name`
- `--scrape-interval`
- `--status-interval`
- `--quiet-status`
- `--no-color`
- `--force-color`
- `--startup-timeout`
- `--metrics-timeout`
- `--metrics-host`
- `--router-host`
- `--prefill-port`
- `--decode-port`
- `--router-port`
- `--router-metrics-port`
- `--no-router`, `--no-prefill`, `--no-decode`
