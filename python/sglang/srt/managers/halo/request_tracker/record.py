"""RequestRecord — the canonical per-request state object.

One `RequestRecord` exists per in-flight (or recently finished) request.
`RequestTracker` is the SOLE writer; admission policies, the gate, and
metrics read but never mutate it.

All wall-clock fields are `time.monotonic()` seconds. Latency *views*
(`ttft_ms`, `e2e_ms`, `tbt_mean_ms`) are returned in milliseconds.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional


class RequestState(enum.Enum):
    """Lifecycle of a halo-tracked request."""

    QUEUED = "queued"      # passed the admission gate, waiting in the scheduler queue
    RUNNING = "running"    # in the running batch (prefill or decode)
    FINISHED = "finished"  # generation completed
    REJECTED = "rejected"  # admission gate rejected it — never queued


@dataclass
class RequestRecord:
    """Canonical per-request state. Mutated only by `RequestTracker`."""

    rid: str

    # --- request-level SLOs (from the halo_*_slo request body fields) ---
    # ttft_slo / tbt_slo are interpreted as absolute ms or as a slowdown
    # ratio depending on the server-level --halo-slo-mode flag; e2e_slo is
    # always a slowdown ratio. None ⇒ that SLO is unconstrained.
    ttft_slo: Optional[float] = None
    tbt_slo: Optional[float] = None
    e2e_slo: Optional[float] = None

    # --- static features captured at admission ---
    prompt_len: int = 0
    prefix_len: int = 0          # radix-cache match length at admission
    admitted_ts: float = 0.0     # monotonic seconds — when the gate admitted it

    # --- mutable progress (RequestTracker writes these) ---
    state: RequestState = RequestState.QUEUED
    first_token_ts: Optional[float] = None
    last_token_ts: Optional[float] = None
    finished_ts: Optional[float] = None
    decoded_tokens: int = 0
    kv_len: int = 0              # per-request KV span right now

    # --- admission-time predictions (for predicted-vs-actual evaluation) ---
    predicted_ttft_ms: Optional[float] = None
    predicted_tbt_ms: Optional[float] = None

    # --- terminal derived metrics (RequestTracker fills these at finish) ---
    solo_e2e_ms: Optional[float] = None
    e2e_slowdown: Optional[float] = None

    # --- terminal bookkeeping ---
    reject_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Live latency views — pure functions of the timestamps above.
    # ------------------------------------------------------------------

    @property
    def is_prefill(self) -> bool:
        """A running request that has not produced its first token yet."""
        return self.state is RequestState.RUNNING and self.first_token_ts is None

    @property
    def ttft_ms(self) -> Optional[float]:
        """Measured time-to-first-token (ms). None until the first token."""
        if self.first_token_ts is None:
            return None
        return (self.first_token_ts - self.admitted_ts) * 1e3

    @property
    def e2e_ms(self) -> Optional[float]:
        """Measured end-to-end latency (ms). None until finished."""
        if self.finished_ts is None:
            return None
        return (self.finished_ts - self.admitted_ts) * 1e3

    @property
    def tbt_mean_ms(self) -> Optional[float]:
        """Mean time-between-tokens (ms) over the decode phase so far.

        None until at least one inter-token interval exists (i.e. the
        request has produced ≥ 2 tokens).
        """
        if self.first_token_ts is None or self.last_token_ts is None:
            return None
        if self.decoded_tokens <= 1:
            return None
        span_ms = (self.last_token_ts - self.first_token_ts) * 1e3
        return span_ms / (self.decoded_tokens - 1)

    def to_dict(self) -> dict:
        """JSON-friendly snapshot — used by /halo/status and the decision log."""
        return {
            "rid": self.rid,
            "state": self.state.value,
            "ttft_slo": self.ttft_slo,
            "tbt_slo": self.tbt_slo,
            "e2e_slo": self.e2e_slo,
            "prompt_len": self.prompt_len,
            "prefix_len": self.prefix_len,
            "decoded_tokens": self.decoded_tokens,
            "kv_len": self.kv_len,
            "ttft_ms": self.ttft_ms,
            "tbt_mean_ms": self.tbt_mean_ms,
            "e2e_ms": self.e2e_ms,
            "solo_e2e_ms": self.solo_e2e_ms,
            "e2e_slowdown": self.e2e_slowdown,
            "predicted_ttft_ms": self.predicted_ttft_ms,
            "predicted_tbt_ms": self.predicted_tbt_ms,
            "reject_reason": self.reject_reason,
        }


@dataclass
class ServerStateSnapshot:
    """Read-only view of current server state the admission gate and the
    selected policy consult. Built by `RequestTracker.snapshot()`.

    `queued` / `running` are the live RequestRecords; a policy derives
    whatever aggregate it needs (queue backlog, batch composition, …)
    from them — the snapshot intentionally stays raw.
    """

    queued: List[RequestRecord] = field(default_factory=list)
    running: List[RequestRecord] = field(default_factory=list)
    kv_usage_ratio: float = 0.0  # current KV-cache occupancy in [0, 1]

    @property
    def running_batch_size(self) -> int:
        return len(self.running)

    @property
    def queue_depth(self) -> int:
        return len(self.queued)
