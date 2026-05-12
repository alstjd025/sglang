"""Prometheus metrics for Halo (Project Halo Phase 1).

Mirrors `admission_control/metrics.py`. Instantiated once per scheduler
process by `scheduler.init_halo`, gated on `--enable-metrics` and
rank-0 the same way (so a TP=N deployment doesn't inflate counter
values by tp_size).

See managers/halo/CLAUDE.md and ms_dev/expctl/CLAUDE.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Optional

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.managers.halo.job import Job


class HaloMetrics:
    """Prometheus metric set for Halo.

    The collector uses the same allowlist pattern as admission_control —
    metric names are exact-matched in run_experiment.py so they get
    persisted into `<session>/metrics/server_metrics.jsonl` and rendered
    by `monitoring_view`.
    """

    def __init__(self, labels: Dict[str, str], registry=None) -> None:
        from prometheus_client import Counter, Gauge

        self._labels = labels
        label_keys = list(labels.keys())
        common_kwargs = {"registry": registry} if registry is not None else {}
        gauge_extra = (
            {} if registry is not None else {"multiprocess_mode": "mostrecent"}
        )

        # ---- counters (lifecycle events) ----
        self._programs_registered_total = Counter(
            name="sglang:halo_programs_registered_total",
            documentation=(
                "Halo: number of successful POST /halo/programs registrations."
            ),
            labelnames=label_keys,
            **common_kwargs,
        )
        self._programs_rejected_total = Counter(
            name="sglang:halo_programs_rejected_total",
            documentation=(
                "Halo: rejected POST /halo/programs requests. "
                "labels: reason in {JOB_ID_ALREADY_REGISTERED, HALO_DISABLED}."
            ),
            labelnames=label_keys + ["reason"],
            **common_kwargs,
        )
        self._requests_admitted_total = Counter(
            name="sglang:halo_requests_admitted_total",
            documentation=(
                "Halo: LLM requests admitted into a registered job."
            ),
            labelnames=label_keys,
            **common_kwargs,
        )
        self._requests_rejected_total = Counter(
            name="sglang:halo_requests_rejected_total",
            documentation=(
                "Halo: LLM requests rejected by the job-level admission gate. "
                "labels: reason in {HALO_NO_JOB_ID, HALO_PROGRAM_NOT_REGISTERED}."
            ),
            labelnames=label_keys + ["reason"],
            **common_kwargs,
        )
        self._slo_violations_total = Counter(
            name="sglang:halo_slo_violations_total",
            documentation=(
                "Halo: per-sweep increments where a job's slowdown_max "
                "exceeded its SLO bound."
            ),
            labelnames=label_keys,
            **common_kwargs,
        )

        # ---- gauges (sweep-derived live state) ----
        self._active_jobs = Gauge(
            name="sglang:halo_active_jobs",
            documentation="Halo: number of jobs in QUEUED or RUNNING state.",
            labelnames=label_keys,
            **gauge_extra,
            **common_kwargs,
        )
        self._total_known_jobs = Gauge(
            name="sglang:halo_total_known_jobs",
            documentation=(
                "Halo: total jobs the registry currently knows (active + "
                "completed-but-not-GC'd)."
            ),
            labelnames=label_keys,
            **gauge_extra,
            **common_kwargs,
        )
        self._mean_slowdown_max = Gauge(
            name="sglang:halo_mean_slowdown_max",
            documentation=(
                "Halo: mean over active jobs of each job's slowdown_max — "
                "rises when worst-case slowdown is bad across the fleet."
            ),
            labelnames=label_keys,
            **gauge_extra,
            **common_kwargs,
        )
        self._mean_slowdown_mean = Gauge(
            name="sglang:halo_mean_slowdown_mean",
            documentation=(
                "Halo: mean over active jobs of each job's slowdown_mean — "
                "rises when fleet-average slowdown is bad."
            ),
            labelnames=label_keys,
            **gauge_extra,
            **common_kwargs,
        )
        self._max_slowdown_max = Gauge(
            name="sglang:halo_max_slowdown_max",
            documentation=(
                "Halo: max over active jobs of slowdown_max — the worst "
                "single job seen this sweep."
            ),
            labelnames=label_keys,
            **gauge_extra,
            **common_kwargs,
        )

    # ------------------------------------------------------------------
    # Counter increments (called from controller)
    # ------------------------------------------------------------------

    def record_program_registered(self) -> None:
        self._programs_registered_total.labels(**self._labels).inc()

    def record_program_rejected(self, reason: str) -> None:
        self._programs_rejected_total.labels(**self._labels, reason=reason).inc()

    def record_request_admitted(self) -> None:
        self._requests_admitted_total.labels(**self._labels).inc()

    def record_request_rejected(self, reason: str) -> None:
        self._requests_rejected_total.labels(**self._labels, reason=reason).inc()

    # ------------------------------------------------------------------
    # Sweep — gauges from the active-jobs snapshot
    # ------------------------------------------------------------------

    def update_from_sweep(
        self,
        active_jobs: Iterable["Job"],
        total_known: int,
        slo_violations_delta: int,
    ) -> None:
        """Called after every sweep with the current set of active jobs."""
        actives = list(active_jobs)
        n = len(actives)
        self._active_jobs.labels(**self._labels).set(n)
        self._total_known_jobs.labels(**self._labels).set(total_known)

        if slo_violations_delta > 0:
            self._slo_violations_total.labels(**self._labels).inc(
                slo_violations_delta
            )

        if n == 0:
            # No active jobs — zero out the slowdown gauges so the panel
            # doesn't show a stale spike.
            self._mean_slowdown_max.labels(**self._labels).set(0.0)
            self._mean_slowdown_mean.labels(**self._labels).set(0.0)
            self._max_slowdown_max.labels(**self._labels).set(0.0)
            return

        sum_max = 0.0
        sum_mean = 0.0
        peak_max = 0.0
        for j in actives:
            sum_max += j.slowdown_max
            sum_mean += j.slowdown_mean
            if j.slowdown_max > peak_max:
                peak_max = j.slowdown_max
        self._mean_slowdown_max.labels(**self._labels).set(sum_max / n)
        self._mean_slowdown_mean.labels(**self._labels).set(sum_mean / n)
        self._max_slowdown_max.labels(**self._labels).set(peak_max)
