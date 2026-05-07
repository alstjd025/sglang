# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

This is a multi-package monorepo, not a flat Python project:

- `python/sglang/` — main Python package. Two important subtrees:
  - `srt/` — **SGLang Runtime (SRT)**, the actual serving engine. This is where most server-side work happens.
  - `lang/` — the SGL frontend DSL (`api.py`, `interpreter.py`, `tracer.py`).
  - `jit_kernel/` — lightweight JIT (Triton/CuTeDSL) kernels, plus their tests/benchmarks.
- `sgl-kernel/` — separately published `sglang-kernel` package (heavyweight AOT C++/CUDA kernels). Built with CMake + scikit-build-core; has its own `Makefile` and `pyproject.toml`. Imported as `sgl_kernel`.
- `sgl-model-gateway/` — Rust + Python control/data plane router (worker registry, load balancing, OpenAI-compat gateway, gRPC pipeline). Has its own README and Cargo workspace.
- `rust/sglang-grpc/` — Rust gRPC bindings, compiled by setuptools-rust into `sglang.srt.grpc._core` (PyO3 extension declared in [python/pyproject.toml](python/pyproject.toml)).
- `test/` — CI tests. `registered/` is auto-discovered by `run_suite.py`; `manual/` is for local-only tests; `srt/` is legacy and being deprecated.
- `docs/` — **legacy Sphinx docs, frozen.** A pre-commit hook ([scripts/ci/check_no_docs_changes.py](scripts/ci/check_no_docs_changes.py)) rejects any change here. New docs go to `docs_new/` (Mintlify).
- `docs_new/` — new Mintlify docs site. See [docs_new/AGENTS.md](docs_new/AGENTS.md) for writing rules.
- `benchmark/`, `examples/`, `scripts/` — what the names suggest.
- `ms_dev/` — **user-local dev scripts (not upstream).** Helpers for installing in this host's venv and running PD experiments. Treat as scratch tooling that may diverge from main.
- `.claude/skills/` — project-specific skills (see "Skills" below). Always consult these before doing the corresponding task; they are the authoritative how-to docs.

## SGLang Runtime (SRT) architecture

SRT is a multi-process engine, not a single FastAPI app. The big picture:

```
HTTP/gRPC entrypoint  →  TokenizerManager  →  Scheduler (per TP rank)  →  TpWorker
                                                    ↓
                                          DetokenizerManager  →  client
```

- **Entrypoints** ([python/sglang/srt/entrypoints/](python/sglang/srt/entrypoints/)): `http_server.py` (FastAPI), `grpc_server.py`, `engine.py` (in-process). `python/sglang/launch_server.py` dispatches between them based on `--grpc-mode` / `--encoder-only`.
- **TokenizerManager** ([managers/tokenizer_manager.py](python/sglang/srt/managers/tokenizer_manager.py)): receives HTTP requests, tokenizes, forwards to schedulers via zmq, streams responses back. Uses many mixins (`TokenizerControlMixin`, `TokenizerManagerScoreMixin`).
- **Scheduler** ([managers/scheduler.py](python/sglang/srt/managers/scheduler.py)): the central batching loop. One process per TP rank. Heavy use of mixin composition (`SchedulerDisaggregationDecodeMixin`, `SchedulerDisaggregationPrefillMixin`, `SchedulerDpAttnMixin`, `SchedulerPpMixin`, `SchedulerProfilerMixin`, `SchedulerUpdateWeightsMixin`, etc.). When changing scheduling behavior, find the right mixin instead of bloating `scheduler.py`.
- **TpWorker** ([managers/tp_worker.py](python/sglang/srt/managers/tp_worker.py)): owns the model and runs forward passes.
- **DetokenizerManager**: separate process, decodes token IDs to text and streams.

Cross-cutting subsystems under `srt/`:

- `mem_cache/` — RadixAttention prefix caching (`radix_cache.py`, `radix_cache_cpp.py`, hierarchical `hiradix_cache.py`), KV memory pools, hicache offload, mamba/SWA variants.
- `disaggregation/` — **PD (Prefill/Decode) disaggregation.** Transports: `mooncake/`, `nixl/`, `mori/`, plus `fake/` for testing. Also hosts encode disaggregation (`encode_*`).
- `speculative/` — EAGLE (v1/v2/draft), ngram, DFlash speculative decoding.
- `layers/` — attention backends (`attention/`), MoE (`moe/`), quantization (`quantization/`, including FP4/FP8/INT4/AWQ/GPTQ), TP/DP/PP collectives, sampler.
- `models/` — ~185 model implementations, one file per architecture family.
- `eplb/` — expert-parallelism load balancing.
- `multimodal/`, `multimodal_gen/` — VLM and diffusion (image/video) generation.
- `lora/`, `weight_sync/`, `checkpoint_engine/` — RL/post-training integrations and live weight updates.

`server_args.py` is the single source of truth for CLI flags; mirror new options through there.

## Common commands

### Build and install (Python runtime)

```bash
# Editable install of the Python runtime (from python/)
pip install -e "python[dev]" --no-build-isolation
```

This pulls in `sglang-kernel` from PyPI by default. To use a locally built kernel instead, install `sgl-kernel/` first.

### Build sgl-kernel (C++/CUDA)

```bash
cd sgl-kernel
make install           # editable install (pip install -e .)
make build             # build wheel + reinstall, parallelism via MAX_JOBS
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"  # constrained
make format            # clang-format C++/CUDA + isort/black Python + pre-commit
```

### Launch a server

```bash
python -m sglang.launch_server --model-path <hf-id-or-path> --tp <N> --port 30000
```

### Run tests

The CI runner is [test/run_suite.py](test/run_suite.py); see [test/README.md](test/README.md) for the full layout.

```bash
# Single file
python3 test/registered/core/test_srt_endpoint.py

# Single test method (unittest path syntax)
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode

# Single JIT kernel test (these live OUTSIDE test/registered/)
python3 python/sglang/jit_kernel/tests/test_add_constant.py

# Suite (matches what CI runs)
python3 test/run_suite.py --hw cuda --suite stage-b-test-1-gpu-small
python3 test/run_suite.py --hw cpu  --suite stage-a-test-cpu
```

The CI runner appends `-f` (failfast). Test files must end with the canonical `if __name__ == "__main__": unittest.main()` (or `pytest.main([__file__])`) — do not add custom `argparse` or mutate `sys.argv` before that, it breaks failfast.

### Lint / format

Pre-commit is the gate ([.pre-commit-config.yaml](.pre-commit-config.yaml)):

```bash
pre-commit install
pre-commit run --all-files                   # run everything
SKIP=no-commit-to-branch pre-commit run --all-files   # what CI runs
pre-commit run isort --files <file>          # one hook on specific files
```

Stack: isort, black, ruff (only `F401,F821`), clang-format (C++/CUDA in `sgl-kernel/`), codespell, plus several local repo hooks (`check-registered-tests`, `check-no-docs-changes`, `sort-ci-permissions`, `check-workflow-job-names`).

## CI registration (mandatory for new tests)

Every file under [test/registered/](test/registered/) and every JIT kernel test/benchmark must call a registration function at module level:

```python
from sglang.test.ci.ci_register import register_cuda_ci
register_cuda_ci(est_time=80, suite="stage-b-test-1-gpu-small")
```

`run_suite.py` parses these via AST, so `est_time` and `suite` must be **literal values** — no variables, no f-strings. Suites: `stage-a-*` (CPU/preflight), `stage-b-*` (basic GPU), `stage-c-*` (multi-GPU/advanced), `nightly-*`. Pick the lightest suite that still exercises the test. JIT kernel correctness tests go to `stage-b-kernel-unit-1-gpu-large`; benchmarks to `stage-b-kernel-benchmark-1-gpu-large`.

For full templates, fixtures, model picks, and a checklist, use the `write-sglang-test` skill instead of guessing.

## Generated files (do not hand-edit, do not format)

- `python/sglang/srt/grpc/*_pb2.py`, `*_pb2_grpc.py`, `*.pyi` — generated from [proto/](proto/). Excluded from isort/black.
- `python/sglang/_version.py` — written by setuptools-scm; version comes from git tags via [python/tools/get_version_tag.py](python/tools/get_version_tag.py).

## Skills

`.claude/skills/` contains step-by-step workflows tuned for this repo. **Use them when the task matches** — they are more current and detailed than this file:

| Task | Skill |
|---|---|
| Add a Triton/CuTeDSL JIT kernel | `add-jit-kernel` |
| Add a heavyweight AOT CUDA/C++ kernel | `add-sgl-kernel` |
| Write/register a CI test | `write-sglang-test` |
| Understand or modify CI workflows | `ci-workflow-guide` |
| Debug a CUDA crash | `debug-cuda-crash` |
| Debug a hang in TP/PP/DP/EP | `debug-distributed-hang` |
| Triage a server incident from a replay | `sglang-prod-incident-triage` |
| Hit SOTA perf vs vLLM/TRT-LLM | `sglang-sota-performance` |
| Capture an e2e profile trace | `generate-profile` |
| Analyze an existing profile | `llm-torch-profiler-analysis` |
| Bisect a flaky/regressed CI test | `sglang-bisect-ci-regression` |
| Compare serving frameworks under SLA | `llm-serving-auto-benchmark` |

## Host-specific notes

- This host has `sudo` disabled — use `gcsudo` (`/engrid/ensh/gpubin/ctn_gcsudo`) when escalation is needed; the install script picks it up via `${GCSUDO}`.
- A venv at `.venv/` is used by `ms_dev/sglang_install_dev.sh`.

## User preferences

- **Clarify before coding.** When a request looks like it will result in code changes, scan it for ambiguous, vague, or unspecified parts (target file/module, expected behavior on edge cases, API shape, naming, scope of change, which existing pattern to follow, etc.). Collect those into a short list and **ask the user before writing code**. Do not silently pick a "reasonable default" and start editing — get explicit answers first, then implement. If the request is unambiguous, proceed without asking. This applies even in auto mode.
- **Confirm the design level before coding.** Decide whether the change calls for a maintainable / modular / OO design (clear classes, separation of concerns, extension points) **or** a simple inline / functional addition (a few lines, no new abstractions), and **ask the user which one they want** when it isn't obvious from the request. Cues that suggest OO/modular: long-lived feature, multiple call sites, expected variants, touches a layer that already follows that style. Cues that suggest "just keep it simple": one-shot script, prototype, single call site, throwaway debug helper, or the surrounding code is already plain functions. When in doubt, ask — do not over-engineer a small task or under-engineer a long-lived one.
