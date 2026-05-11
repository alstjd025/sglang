"""Job dataclass + JobState enum for HALO Phase 1.

HALO (Project Halo) job-level slowdown tracking. See managers/halo/CLAUDE.md
for the module design.
"""

# HALO: Phase 1 job state representation. Per spec
# (ms_dev/halo_dev/CLAUDE.md §"Job 자료구조"), this mirrors a Linux task_struct
# in spirit: identity + state + accounting + a bounded history buffer for
# debugging. Phase 2 will extend this struct (admission decisions, deadlines);
# the placeholder `dag` field is reserved for Option A interface in the future.

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Optional, Set, Tuple


class JobState(Enum):
    QUEUED = "queued"        # registered, not yet observed running
    RUNNING = "running"      # at least one of its requests is being served
    COMPLETE = "complete"    # remaining_request_number == 0
    REJECTED = "rejected"    # reserved for Phase 2 admission gate; unused in Phase 1


def _now_monotonic() -> float:
    return time.monotonic()


@dataclass
class Job:
    """Per-job state record. Single-threaded access; lives inside the scheduler.

    Notes on initial slowdown values (per spec ms_dev/halo_dev/CLAUDE.md §2 Q5):
    initial `slowdown_max` and `slowdown_mean` are set to `slo` rather than
    1.0 — meaning "no measurement yet, treat as worst-case (SLO bound) for
    Phase 2 admission decisions". This is overwritten on the first sweep.

    Note on `dag`: placeholder for Option A (Phase 2/3) program-compile
    interface. In Phase 1 we only support Option B (per-request `halo_job_id`
    + `halo_slo`), so this stays None.
    """

    job_id: str
    slo: float

    state: JobState = JobState.QUEUED

    # Slowdown observation — see SlowdownTracker.sweep().
    slowdown_max: float = 0.0     # set in __post_init__ to slo
    slowdown_mean: float = 0.0    # set in __post_init__ to slo

    # Request counters.
    total_request_number: int = 0
    remaining_request_number: int = 0
    slo_violation_count: int = 0

    # Timestamps (monotonic seconds).
    first_seen_ts: float = field(default_factory=_now_monotonic)
    last_update_ts: float = field(default_factory=_now_monotonic)

    # Member request IDs (rid). Removed when a request finishes.
    request_ids: Set[str] = field(default_factory=set)

    # Bounded (max, mean) history for debugging / /server_info exposure.
    slowdown_history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=64)
    )

    # Phase 2/3 extension placeholder; stays None in Phase 1.
    dag: Optional[Any] = None

    def __post_init__(self) -> None:
        # Honor the spec: initial slowdown == SLO (worst-case fallback).
        if self.slowdown_max == 0.0:
            self.slowdown_max = self.slo
        if self.slowdown_mean == 0.0:
            self.slowdown_mean = self.slo

    # ---- mutators (scheduler-side; no internal locking) ----

    def on_request_admitted(self, rid: str) -> None:
        """Called when a request belonging to this job enters the waiting queue."""
        self.request_ids.add(rid)
        self.total_request_number += 1
        self.remaining_request_number += 1
        self.last_update_ts = _now_monotonic()
        if self.state == JobState.COMPLETE:
            # Re-opened with a new request — back to running.
            self.state = JobState.RUNNING

    def on_request_completed(self, rid: str) -> None:
        """Called when a request finishes (success or abort)."""
        self.request_ids.discard(rid)
        if self.remaining_request_number > 0:
            self.remaining_request_number -= 1
        self.last_update_ts = _now_monotonic()
        if self.remaining_request_number == 0 and not self.request_ids:
            self.state = JobState.COMPLETE

    def record_sweep(self, max_ratio: float, mean_ratio: float) -> None:
        """Called from SlowdownTracker after each periodic sweep."""
        self.slowdown_max = max_ratio
        self.slowdown_mean = mean_ratio
        self.slowdown_history.append((max_ratio, mean_ratio))
        self.last_update_ts = _now_monotonic()
        if max_ratio > self.slo:
            self.slo_violation_count += 1
        if self.state == JobState.QUEUED and self.request_ids:
            self.state = JobState.RUNNING

    # ---- read-only helpers ----

    def to_dict(self) -> dict:
        """JSON-able representation for /server_info and jsonl logs."""
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "slo": self.slo,
            "slowdown_max": self.slowdown_max,
            "slowdown_mean": self.slowdown_mean,
            "total_request_number": self.total_request_number,
            "remaining_request_number": self.remaining_request_number,
            "slo_violation_count": self.slo_violation_count,
            "first_seen_ts": self.first_seen_ts,
            "last_update_ts": self.last_update_ts,
        }
