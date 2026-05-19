"""Reactive TBT safety net for admission control.

See managers/halo/admission_control/CLAUDE.md for the full design.

EWMA over per-decode-step latency. Used as Stage 3 — even if Stage 2's predictor
underestimates, this catches sustained SLO violations: when the smoothed TBT
exceeds tbt_slo_ms * reactive_ratio, new requests are rejected until the
system cools down.

Cold-start: until `warm_up_steps` updates have arrived, `is_warm()` returns
False and the controller skips Stage 3 entirely (no false rejects from a
single early measurement).
"""

from __future__ import annotations

import threading


class TBTEwmaTracker:
    """Thread-safe exponentially weighted moving average of per-step latency.

    The scheduler is single-threaded so update/get races are not expected, but
    the lock keeps `get_internal_state` snapshots consistent.
    """

    def __init__(self, alpha: float = 0.1, warm_up_steps: int = 100) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(
                f"alpha must be in (0, 1]; got {alpha}"
            )
        if warm_up_steps < 0:
            raise ValueError(
                f"warm_up_steps must be >= 0; got {warm_up_steps}"
            )
        self._alpha = float(alpha)
        self._warm_up_steps = int(warm_up_steps)
        self._value = 0.0
        self._n = 0
        self._lock = threading.Lock()

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def warm_up_steps(self) -> int:
        return self._warm_up_steps

    def update(self, latency_ms: float) -> None:
        if latency_ms < 0:
            return  # ignore garbage; do not poison the EWMA
        with self._lock:
            if self._n == 0:
                self._value = float(latency_ms)
            else:
                self._value = (
                    self._alpha * float(latency_ms)
                    + (1.0 - self._alpha) * self._value
                )
            self._n += 1

    def get(self) -> float:
        with self._lock:
            return self._value

    def is_warm(self) -> bool:
        with self._lock:
            return self._n >= self._warm_up_steps

    def sample_count(self) -> int:
        with self._lock:
            return self._n

    def reset(self) -> None:
        with self._lock:
            self._value = 0.0
            self._n = 0
