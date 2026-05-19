#!/usr/bin/env python3
"""HALO microbench: register/admit hot-path + sweep cost.

Standalone — does NOT need a running sglang server. We instantiate a
HaloController in-process, build synthetic RequestExecutionInfo lists,
and time the public APIs the scheduler hits.

The numbers reported are pure CPU cost of Halo's logic. Real-world
overhead also includes zmq IPC for register_program; that's measured
separately in `e2e_smoke.sh`.

Run:
    .venv/bin/python3 ms_dev/halo_dev/verify/microbench_overhead.py
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Callable, List, Tuple

from sglang.srt.managers.admission_control.cost_model import (
    PrefillCostModel,
    TBTCostModel,
)
from sglang.srt.managers.halo import (
    HaloConfig,
    HaloController,
    RequestExecutionInfo,
)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = int(round((p / 100.0) * (len(s) - 1)))
    return s[k]


def fmt_us(seconds_list: List[float]) -> str:
    us = [s * 1e6 for s in seconds_list]
    return (
        f"mean={statistics.mean(us):7.2f} µs  "
        f"median={statistics.median(us):7.2f}  "
        f"p99={percentile(us, 99):7.2f}  "
        f"max={max(us):7.2f}"
    )


def time_block(fn: Callable[[], None], iterations: int) -> List[float]:
    """Run fn() iterations times, return per-iteration durations (s).

    Uses perf_counter_ns for resolution; warmup of 100 ignored.
    """
    samples: List[float] = []
    warmup = min(100, iterations // 10) if iterations > 100 else 0
    for _ in range(warmup):
        fn()
    for _ in range(iterations):
        t0 = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - t0) / 1e9)
    return samples


@dataclass
class Bench:
    name: str
    samples: List[float]

    def line(self) -> str:
        return f"  {self.name:<48s} {fmt_us(self.samples)}"


def make_controller(with_models: bool) -> HaloController:
    config = HaloConfig(enabled=True, default_slo=5.0, tick_interval_ms=100.0)
    c = HaloController(config, is_rank0=True)
    if with_models:
        # Inject deterministic cost models so sweep does real math.
        c.tracker.prefill_cost = PrefillCostModel(alpha=0.0, beta=0.1, gamma=10.0)
        c.tracker.tbt_cost = TBTCostModel(a=20.0, b=0.5, c=0.001)
    return c


# --------------------------------------------------------------------------
# Sweep overhead — main concern for tick interval safety
# --------------------------------------------------------------------------

def build_infos_matrix(
    controller: HaloController, num_jobs: int, reqs_per_job: int
) -> List[RequestExecutionInfo]:
    """Pre-register num_jobs jobs and build reqs_per_job synthetic infos each."""
    infos: List[RequestExecutionInfo] = []
    for j in range(num_jobs):
        jid = f"j{j}"
        controller.register_program(jid, slo=5.0, total_calls=reqs_per_job)
        for r in range(reqs_per_job):
            rid = f"{jid}-r{r}"
            controller.register_request(rid=rid, halo_job_id=jid, halo_slo=5.0)
            infos.append(
                RequestExecutionInfo(
                    rid=rid, job_id=jid, prompt_len=100,
                    prefix_len_at_admission=0, decoded_tokens_so_far=10,
                    kv_len_now=200, elapsed_ms=300.0 + r * 5,
                )
            )
    return infos


def bench_sweep(num_jobs: int, reqs_per_job: int, iterations: int = 500) -> Bench:
    controller = make_controller(with_models=True)
    infos = build_infos_matrix(controller, num_jobs, reqs_per_job)
    name = (
        f"sweep  jobs={num_jobs:>4d}  reqs/job={reqs_per_job:>3d}  "
        f"total_reqs={num_jobs*reqs_per_job:>5d}"
    )
    samples = time_block(lambda: controller.tracker.sweep(infos), iterations)
    return Bench(name=name, samples=samples)


# --------------------------------------------------------------------------
# Hot-path: register_program / admit_to_job (via controller's
# register_request) / on_request_finished
# --------------------------------------------------------------------------

def bench_register_program(iterations: int = 5000) -> Bench:
    controller = make_controller(with_models=False)
    counter = {"i": 0}

    def go():
        counter["i"] += 1
        controller.register_program(
            job_id=f"j{counter['i']}",
            slo=5.0,
            total_calls=8,
            stage_sequence=["U", "L", "P", "I", "V", "D", "I", "V"],
        )
    return Bench("register_program (fresh job each call)",
                 time_block(go, iterations))


def bench_admit_hot_path(iterations: int = 5000) -> Bench:
    """Steady-state: program already registered, admit N rids onto it."""
    controller = make_controller(with_models=False)
    controller.register_program("steady-job", slo=5.0, total_calls=1)
    counter = {"i": 0}

    def go():
        counter["i"] += 1
        controller.register_request(
            rid=f"rid-{counter['i']}",
            halo_job_id="steady-job",
            halo_slo=5.0,
        )
    samples = time_block(go, iterations)
    return Bench("register_request (admit to existing job)", samples)


def bench_finish_hot_path(iterations: int = 5000) -> Bench:
    controller = make_controller(with_models=False)
    controller.register_program("steady-job", slo=5.0)
    # pre-admit iterations + warmup rids so finish has something to drop
    total = iterations + 200
    for i in range(total):
        controller.register_request(
            rid=f"rid-{i}", halo_job_id="steady-job", halo_slo=5.0
        )
    counter = {"i": 0}

    def go():
        controller.on_request_finished(f"rid-{counter['i']}")
        counter["i"] += 1
    return Bench("on_request_finished", time_block(go, iterations))


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    print()
    print("=" * 100)
    print("HALO microbench — Phase 1 overhead (CPU only, no IPC/HTTP)")
    print("=" * 100)
    print()
    print("Acceptance: hot-path < 50 µs mean; sweep < 1 ms mean for")
    print("(jobs × reqs_per_job ≤ 1000) so tick=100ms stays safe (< 1%).")
    print()

    print("[1] Hot-path (per-call) overhead")
    print("-" * 100)
    for b in [
        bench_register_program(),
        bench_admit_hot_path(),
        bench_finish_hot_path(),
    ]:
        print(b.line())
    print()

    print("[2] Sweep overhead — varies with (num_jobs × reqs_per_job)")
    print("-" * 100)
    for jobs, reqs in [
        (1, 1), (10, 1), (10, 10),
        (100, 1), (100, 10),
        (1000, 1), (100, 50),  # 100 jobs × 50 reqs = 5000 reqs
    ]:
        try:
            b = bench_sweep(jobs, reqs, iterations=200)
        except Exception as e:  # pragma: no cover
            print(f"  sweep jobs={jobs} reqs={reqs}: skipped ({e})")
            continue
        print(b.line())
    print()

    print("[3] Tick interval safety check")
    print("-" * 100)
    tick_ms = 100.0
    b = bench_sweep(100, 10, iterations=200)
    mean_ms = statistics.mean(b.samples) * 1e3
    pct = mean_ms / tick_ms * 100.0
    print(
        f"  sweep(100 jobs × 10 reqs) mean = {mean_ms:.3f} ms vs "
        f"tick={tick_ms:.0f} ms → {pct:.4f} %"
    )
    print(f"  {'PASS — well below 1% budget' if pct < 1.0 else 'FAIL — exceeds 1% budget':<60s}")
    print()


if __name__ == "__main__":
    main()
