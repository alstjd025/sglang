"""Prometheus metrics for Halo (request-level).

Instantiated once per scheduler process by `scheduler.init_halo`, gated on
`--enable-metrics` + rank-0 the same way as `admission_control/metrics.py`
(so a TP=N deployment doesn't inflate counter values by tp_size).

Per-request quantities (TTFT, TBT, e2e, e2e-slowdown) are *histograms* —
one observation per finished request — so percentiles come for free.
Only genuinely-current state (active request count) is a gauge.

See managers/halo/CLAUDE.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.managers.halo.request_tracker.record import RequestRecord

# Slowdown-ratio histogram buckets (dimensionless: actual / solo).
_SLOWDOWN_BUCKETS = (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 50.0, 100.0)
# Latency histogram buckets (seconds).
_LATENCY_BUCKETS = (
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
)


class HaloMetrics:
    """Prometheus metric set for request-level Halo."""

    def __init__(self, labels: Dict[str, str], registry=None) -> None:
        from prometheus_client import Counter, Gauge, Histogram

        self._labels = labels
        label_keys = list(labels.keys())
        common = {"registry": registry} if registry is not None else {}
        gauge_extra = (
            {} if registry is not None else {"multiprocess_mode": "mostrecent"}
        )

        # ---- counters (admission events) ----
        self._admitted_total = Counter(
            name="sglang:halo_requests_admitted_total",
            documentation="Halo: requests admitted by the admission gate.",
            labelnames=label_keys,
            **common,
        )
        self._rejected_total = Counter(
            name="sglang:halo_requests_rejected_total",
            documentation=(
                "Halo: requests rejected by the admission gate. labels: "
                "reason in {HALO_KV_CAP, MOONCAKE_TTFT, MOONCAKE_TBT, "
                "MOONCAKE_TBT_REACTIVE, HALO_VSS_PREDICTED}."
            ),
            labelnames=label_keys + ["reason"],
            **common,
        )

        # ---- histograms (one observation per finished request) ----
        self._ttft_seconds = Histogram(
            name="sglang:halo_request_ttft_seconds",
            documentation="Halo: measured per-request time-to-first-token.",
            labelnames=label_keys,
            buckets=_LATENCY_BUCKETS,
            **common,
        )
        self._tbt_seconds = Histogram(
            name="sglang:halo_request_tbt_seconds",
            documentation="Halo: measured per-request mean time-between-tokens.",
            labelnames=label_keys,
            buckets=_LATENCY_BUCKETS,
            **common,
        )
        self._e2e_seconds = Histogram(
            name="sglang:halo_request_e2e_seconds",
            documentation="Halo: measured per-request end-to-end latency.",
            labelnames=label_keys,
            buckets=_LATENCY_BUCKETS,
            **common,
        )
        self._e2e_slowdown = Histogram(
            name="sglang:halo_request_e2e_slowdown",
            documentation=(
                "Halo: measured per-request end-to-end slowdown "
                "(actual e2e / solo-run e2e)."
            ),
            labelnames=label_keys,
            buckets=_SLOWDOWN_BUCKETS,
            **common,
        )

        # ---- gauge (current state) ----
        self._active_requests = Gauge(
            name="sglang:halo_active_requests",
            documentation="Halo: requests currently QUEUED or RUNNING.",
            labelnames=label_keys,
            **gauge_extra,
            **common,
        )

    # ------------------------------------------------------------------
    # Counter increments — called from the controller's admission hook.
    # ------------------------------------------------------------------

    def record_request_admitted(self) -> None:
        self._admitted_total.labels(**self._labels).inc()

    def record_request_rejected(self, reason: str) -> None:
        self._rejected_total.labels(**self._labels, reason=reason).inc()

    # ------------------------------------------------------------------
    # Per-finished-request observations.
    # ------------------------------------------------------------------

    def observe_finished(self, record: "RequestRecord") -> None:
        """Record the terminal latency + slowdown of one finished request."""
        if record.ttft_ms is not None:
            self._ttft_seconds.labels(**self._labels).observe(record.ttft_ms / 1e3)
        if record.tbt_mean_ms is not None:
            self._tbt_seconds.labels(**self._labels).observe(record.tbt_mean_ms / 1e3)
        if record.e2e_ms is not None:
            self._e2e_seconds.labels(**self._labels).observe(record.e2e_ms / 1e3)
        if record.e2e_slowdown is not None:
            self._e2e_slowdown.labels(**self._labels).observe(record.e2e_slowdown)

    # ------------------------------------------------------------------
    # Gauge refresh — called from the periodic tick.
    # ------------------------------------------------------------------

    def update_gauges(self, *, active_requests: int) -> None:
        self._active_requests.labels(**self._labels).set(active_requests)
