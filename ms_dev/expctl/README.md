# Experiment Controller (`expctl`)

`run_pd_experiment.py`는 세션 단위로 프로세스를 띄우고,
프로세스 로그 + Prometheus 스크랩 메트릭 + GPU/시스템 메트릭을 함께 저장합니다.

- 기본 세션 폴더: `/workspace/sglang/ms_dev/runtime/sessions/YYYYMMDD_HHMMSS`
- 종료: `Ctrl+C` (graceful stop)

## Code Layout

- `run_pd_experiment.py`: 엔트리포인트/오케스트레이터. 프로세스 실행/종료, readiness 확인, 세션 메타 기록, 루프 제어를 담당합니다.
- `monitoring_metrics.py`: 메트릭 수집기(RuntimeMetricsCollector). Prometheus + GPU + 시스템 메트릭을 주기 수집하고 JSONL로 저장하며 최신 스냅샷을 제공합니다.
- `monitoring_view.py`: 상태판 렌더러. 수집 스냅샷을 받아 PD/Single 모니터 화면 문자열을 만들고 feature ON/OFF를 표시합니다.

관계:
- `run_pd_experiment.py` -> `RuntimeMetricsCollector`(수집 시작/종료, 스냅샷 조회)
- `run_pd_experiment.py` -> `render_status*`(상태판 출력)
- `monitoring_view.py`는 수집/프로세스 제어를 모르고, 입력 스냅샷만 렌더합니다.
- `monitoring_metrics.py`는 출력 렌더를 모르고, 수집/저장/상태 제공만 담당합니다.


## Modes

- `--mode pd` (기본): `prefill/decode/router` 실행
- `--mode single`: 단일 SGLang 서버 1개만 실행 (`router` 완전 비활성)

## What It Logs

- 프로세스 로그:
  - PD: `process_logs/prefill.*`, `decode.*`, `router.*`
  - Single: `process_logs/server.*`
- 메트릭(JSONL):
  - PD: `metrics/prefill_metrics.jsonl`, `decode_metrics.jsonl`, `router_metrics.jsonl`
  - Single: `metrics/server_metrics.jsonl`
  - 공통: `metrics/gpu_metrics.jsonl`, `metrics/system_metrics.jsonl`
- 메타 정보:
  - `meta/run_meta.json`, `meta/readiness.json`, `meta/run_end.json`

포함되는 대표 메트릭:
- Throughput: `sglang:gen_throughput`
- Queue: `sglang:num_queue_reqs`, `sglang:num_decode_prealloc_queue_reqs`, `sglang:num_decode_transfer_queue_reqs` 등
- TTFT/TBT: `sglang:time_to_first_token_seconds*`, `sglang:inter_token_latency_seconds*`
- KV/캐시: `sglang:cache_hit_rate`, `sglang:kv_transfer_*`, `sglang:hicache_host_*`

## Live Terminal Monitor

실행 중 상태판이 주기적으로 갱신됩니다.

- PD: prefill/decode/router 상태 + 큐/throughput/TTFT/TBT + GPU/SYS
- Single: server 상태 + 큐/throughput/TTFT/TBT + GPU/SYS
- TTY에서는 컬러 하이라이트 자동 적용 (`--no-color`로 비활성)

## Quick Start

### 1) PD 모드 (기본)

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py
```

### 2) Single 모드

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --mode single
```

## Decode Offload Default (PD)

현재 `ms_dev/env.sh` 기본값은 **PD decode GPU KV를 host/storage로 offload**하도록 설정되어 있습니다.

- `SGLANG_PD_DECODE_HICACHE_ENABLE=0`
- `SGLANG_PD_DECODE_OFFLOAD_ENABLE=1`
- `SGLANG_PD_DECODE_HICACHE_STORAGE_BACKEND=file`
- `SGLANG_PD_DECODE_HICACHE_SIZE=256`

해제:

```bash
export SGLANG_PD_DECODE_OFFLOAD_ENABLE=0
```

## Usage Examples

### 1) PD 기본 포트

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py
```

기본 포트:
- prefill: `30002`
- decode: `30001`
- router API: `30000`
- router metrics: `29000`

### 2) PD 포트 변경

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py \
  --prefill-port 31002 \
  --decode-port 31001 \
  --router-port 31000 \
  --router-metrics-port 39000
```

### 3) Single 모드 + 포트 지정

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py \
  --mode single \
  --single-port 31000
```

### 4) 세션 이름/저장 경로 지정

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py \
  --session-name long_stress_v1 \
  --session-root /workspace/sglang/ms_dev/runtime/sessions
```

### 5) 폴링/상태판 주기 조정

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py \
  --scrape-interval 1.0 \
  --status-interval 3.0 \
  --metrics-timeout 5.0
```

### 6) 상태판 끄기 / 컬러 제어

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --quiet-status
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --no-color
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --force-color
```

### 7) PD에서 일부 역할 제외

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --no-router
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --no-decode
```

## CLI

```bash
python3 /workspace/sglang/ms_dev/expctl/run_pd_experiment.py --help
```

주요 옵션:
- `--mode {pd,single}`
- `--single-port`
- `--session-root`, `--session-name`
- `--scrape-interval`, `--status-interval`, `--metrics-timeout`, `--startup-timeout`
- `--quiet-status`, `--no-color`, `--force-color`
- `--metrics-host`
- PD 전용: `--prefill-port`, `--decode-port`, `--router-port`, `--router-metrics-port`, `--no-router`, `--no-prefill`, `--no-decode`
