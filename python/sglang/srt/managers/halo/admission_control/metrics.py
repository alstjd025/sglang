"""Prometheus metrics for admission control.

See managers/halo/admission_control/CLAUDE.md.

Metric registration is gated on the caller (scheduler.init_admission_control).
When admission control is off OR --enable-metrics is off we never instantiate
this class, so prometheus_client never sees the metric names.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.managers.halo.admission_control.controller import AdmissionDecision


class AdmissionMetrics:
    """Wrapper around the admission-control Prometheus metric set.

    Constructed once per scheduler process. record_decision() is called from
    Scheduler._abort_on_predicted_slo_violation for every decision (admit
    and reject alike).
    """

    def __init__(self, labels: Dict[str, str], registry=None) -> None:
        # Import inside the constructor: prometheus_client must be imported
        # AFTER PROMETHEUS_MULTIPROC_DIR is set in the scheduler process.
        # `registry` is for test isolation; production path leaves it None
        # so prometheus_client uses the global registry like every other
        # collector in SGLang.
        from prometheus_client import Counter, Gauge, Histogram

        self._labels = labels
        label_keys = list(labels.keys())
        common_kwargs = {"registry": registry} if registry is not None else {}

        self._decisions_total = Counter(
            name="sglang:admission_decisions_total",
            documentation=(
                "Admission control decisions. labels: decision in "
                "{admit, reject, dryrun_would_reject}, reason in "
                "{ADMIT, DISABLED, TTFT_PREDICTED, TBT_PREDICTED, TBT_REACTIVE}."
            ),
            labelnames=label_keys + ["decision", "reason"],
            **common_kwargs,
        )

        ttft_buckets = (
            10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000,
        )
        tbt_buckets = (
            5, 10, 20, 50, 100, 150, 200, 300, 500, 1000, 2000,
        )
        self._predicted_ttft_ms = Histogram(
            name="sglang:admission_predicted_ttft_ms",
            documentation="Predicted TTFT (ms) at admission time.",
            labelnames=label_keys,
            buckets=ttft_buckets,
            **common_kwargs,
        )
        self._predicted_tbt_ms = Histogram(
            name="sglang:admission_predicted_tbt_ms",
            documentation=(
                "Predicted TBT (ms) at admission time, current-batch approximation."
            ),
            labelnames=label_keys,
            buckets=tbt_buckets,
            **common_kwargs,
        )

        # Gauge multiprocess_mode is only valid with the multiprocess registry,
        # which is the default global registry used by SGLang. Tests pass
        # registry=CollectorRegistry() and must omit multiprocess_mode.
        gauge_extra = {} if registry is not None else {"multiprocess_mode": "mostrecent"}
        self._tbt_ewma_ms = Gauge(
            name="sglang:admission_tbt_ewma_ms",
            documentation="Reactive EWMA of measured per-step TBT (ms).",
            labelnames=label_keys,
            **gauge_extra,
            **common_kwargs,
        )
        self._queue_predicted_ms = Gauge(
            name="sglang:admission_queue_predicted_ms",
            documentation=(
                "Predicted total prefill ms already queued ahead of any incoming request."
            ),
            labelnames=label_keys,
            **gauge_extra,
            **common_kwargs,
        )

    def record_decision(self, decision: "AdmissionDecision") -> None:
        if decision.admit:
            decision_label = "dryrun_would_reject" if decision.dry_run_would_reject else "admit"
        else:
            decision_label = "reject"
        self._decisions_total.labels(
            **self._labels, decision=decision_label, reason=decision.reason
        ).inc()

        if decision.predicted_ttft_ms is not None:
            self._predicted_ttft_ms.labels(**self._labels).observe(
                decision.predicted_ttft_ms
            )
        if decision.predicted_tbt_ms is not None:
            self._predicted_tbt_ms.labels(**self._labels).observe(
                decision.predicted_tbt_ms
            )
        if decision.tbt_ewma_ms is not None:
            self._tbt_ewma_ms.labels(**self._labels).set(decision.tbt_ewma_ms)
        if decision.queue_predicted_ms is not None:
            self._queue_predicted_ms.labels(**self._labels).set(
                decision.queue_predicted_ms
            )

    def update_tbt_ewma(self, value_ms: Optional[float]) -> None:
        """Push the latest EWMA into the gauge between admission decisions."""
        if value_ms is None:
            return
        self._tbt_ewma_ms.labels(**self._labels).set(value_ms)
