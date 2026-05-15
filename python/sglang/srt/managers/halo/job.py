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
from typing import Any, Deque, Dict, List, Optional, Set


class JobState(Enum):
    QUEUED = "queued"        # registered, not yet observed running
    RUNNING = "running"      # at least one of its requests is being served
    COMPLETE = "complete"    # remaining_request_number == 0
    REJECTED = "rejected"    # reserved for Phase 2 admission gate; unused in Phase 1


def _now_monotonic() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class JobCallSpan:
    """One LLM call's wall-clock span + token features, for the job-lifetime
    virtual job slowdown (VJS) computation.

    A call occupies the interval ``[admitted_ts, end_ts]``. The feature
    fields let ``SlowdownTracker.compute_job_vjs`` reconstruct the call's
    solo time (prefill + decode). For a *completed* call ``end_ts`` is its
    finish time and the features are final; for an *in-flight* call
    ``end_ts`` is "now" at computation time and the features are current.

    See ms_dev/halo_dev/prediction_model.md (VJS section) for how spans are
    merged into stages and turned into a slowdown ratio.
    """

    admitted_ts: float    # monotonic s — when this call was first admitted
    end_ts: float         # monotonic s — finish time, or "now" if in-flight
    prompt_len: int       # total prompt tokens (n + cached prefix)
    prefix_len: int       # cached prefix length at admission (radix hit)
    decoded_tokens: int   # output tokens produced so far / in total
    kv_len: int           # KV span (= prompt_len + decoded_tokens), now/final


@dataclass
class Job:
    """Per-job state record. Single-threaded access; lives inside the scheduler.

    Notes on the initial slowdown value (per spec ms_dev/halo_dev/CLAUDE.md §2 Q5):
    initial `virtual_job_slowdown` is set to `slo` rather than 1.0 — meaning
    "no measurement yet, treat as worst-case (SLO bound) for Phase 2
    admission decisions". This is overwritten on the first sweep that sees
    an in-flight call of this job.

    Option A pre-registration fields (`total_calls_expected`, `stage_sequence`,
    expected input/output token lengths, `dag`) are populated only when the
    job arrives via `POST /halo/programs`. They are stored in Phase 1 but not
    used by the slowdown sweep math — they are reserved for Phase 2
    admission/scheduling decisions that need lookahead information.

    `from_program=True` distinguishes pre-registered jobs from jobs that
    arrived via the request body (Option B). Phase 1 Q12 says: when Halo is
    enabled, a request whose job_id has not been pre-registered is rejected
    with HTTP 400. So in Phase 1 every active Job should have
    `from_program=True`. The flag exists primarily for the registry to tell
    pre-registration apart from accidental lazy-create paths during testing.
    """

    job_id: str
    slo: float

    state: JobState = JobState.QUEUED

    # Virtual job slowdown — the job's lifetime critical-path slowdown.
    # = (job critical-path actual ms) / (job critical-path solo ms),
    # recomputed every sweep by SlowdownTracker.compute_job_vjs over the
    # job's completed + in-flight calls. set in __post_init__ to slo.
    virtual_job_slowdown: float = 0.0

    # Request counters.
    total_request_number: int = 0
    remaining_request_number: int = 0
    slo_violation_count: int = 0

    # Timestamps (monotonic seconds).
    first_seen_ts: float = field(default_factory=_now_monotonic)
    last_update_ts: float = field(default_factory=_now_monotonic)

    # Member request IDs (rid). Removed when a request finishes.
    request_ids: Set[str] = field(default_factory=set)

    # Bounded VJS history for debugging / /server_info exposure.
    slowdown_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=64)
    )

    # Finished calls of this job, kept for the job-lifetime VJS computation.
    # Each request that finishes appends its frozen JobCallSpan here (see
    # HaloController.on_request_finished). Bounded by the job's call count;
    # dropped wholesale when the job is GC'd.
    completed_call_spans: List[JobCallSpan] = field(default_factory=list)

    # ---- Option A (pre-registration) metadata ----
    # All optional: None for jobs created without pre-registration (forbidden
    # in strict mode, but allowed in unit tests that build Job directly).
    # Phase 1 just stores these; Phase 2 admission/scheduling will read them.
    total_calls_expected: Optional[int] = None
    stage_sequence: Optional[List[str]] = None
    expected_input_lens: Optional[List[int]] = None
    expected_output_lens: Optional[List[int]] = None
    # Free-form JSON-able DAG description. Caps enforced at the registration
    # boundary (HTTP body parser limits to 16KB), not here.
    dag: Optional[Dict[str, Any]] = None
    # Distinguishes pre-registered jobs from jobs built directly (tests, etc).
    from_program: bool = False
    # ---- Phase 2 Stage B — concurrency cap ----
    # Application-declared max in-flight concurrent LLM calls for this job.
    # None ⇒ no cap (D4). Server enforces by counting `in_flight_count`.
    declared_max_concurrency: Optional[int] = None
    # Running count of in-flight (admitted but not yet finished) requests.
    # Updated by HaloController.register_request / on_request_finished.
    in_flight_count: int = 0

    def __post_init__(self) -> None:
        # Honor the spec: initial slowdown == SLO (worst-case fallback).
        if self.virtual_job_slowdown == 0.0:
            self.virtual_job_slowdown = self.slo

    # ---- mutators (scheduler-side; no internal locking) ----

    def on_request_admitted(self, rid: str) -> None:
        """Called when a request belonging to this job enters the waiting queue."""
        self.request_ids.add(rid)
        self.total_request_number += 1
        self.remaining_request_number += 1
        self.in_flight_count += 1
        self.last_update_ts = _now_monotonic()
        if self.state == JobState.COMPLETE:
            # Re-opened with a new request — back to running.
            self.state = JobState.RUNNING

    def on_request_completed(self, rid: str) -> None:
        """Called when a request finishes (success or abort).

        Decouples *call finish* from *job finish*. A chain with gaps
        between calls (tool delays, conditional branches, parallel
        rounds…) may have `remaining_request_number == 0` repeatedly
        during its lifetime without the job itself being done — so we
        do NOT auto-flip to COMPLETE here. The job's state moves to
        COMPLETE only when one of:
          - the client sends `halo_job_done=true` on a chat.completions
            request → `mark_done()` is called from the finish hook
          - the quiescent safety net (`gc_quiescent_jobs`) trips
        See ms_dev/halo_dev/halo_api_reference.md.
        """
        self.request_ids.discard(rid)
        if self.remaining_request_number > 0:
            self.remaining_request_number -= 1
        if self.in_flight_count > 0:
            self.in_flight_count -= 1
        self.last_update_ts = _now_monotonic()
        # NOTE: deliberately NOT transitioning to COMPLETE here.

    def mark_done(self) -> None:
        """Explicit client-side termination signal — see
        halo_api_reference.md (`halo_job_done` body field).

        Sets state to COMPLETE regardless of in-flight count. The next
        `gc_completed` sweep will GC the job after retain_seconds.
        """
        self.last_update_ts = _now_monotonic()
        self.state = JobState.COMPLETE

    def record_vjs(self, vjs: float) -> None:
        """Called from SlowdownTracker after each periodic sweep with the
        freshly computed virtual job slowdown."""
        self.virtual_job_slowdown = vjs
        self.slowdown_history.append(vjs)
        self.last_update_ts = _now_monotonic()
        if vjs > self.slo:
            self.slo_violation_count += 1
        if self.state == JobState.QUEUED and self.request_ids:
            self.state = JobState.RUNNING

    def record_completed_call(self, span: JobCallSpan) -> None:
        """Freeze a finished call's time span so it keeps contributing to
        this job's lifetime VJS after the request object is gone. Called
        from HaloController.on_request_finished."""
        self.completed_call_spans.append(span)
        self.last_update_ts = _now_monotonic()

    # ---- read-only helpers ----

    def to_dict(self) -> dict:
        """JSON-able representation for /server_info and jsonl logs."""
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "slo": self.slo,
            "virtual_job_slowdown": self.virtual_job_slowdown,
            "total_request_number": self.total_request_number,
            "remaining_request_number": self.remaining_request_number,
            "slo_violation_count": self.slo_violation_count,
            "first_seen_ts": self.first_seen_ts,
            "last_update_ts": self.last_update_ts,
            # Option A pre-registration fields (None when not pre-registered).
            "from_program": self.from_program,
            "total_calls_expected": self.total_calls_expected,
            "stage_sequence": self.stage_sequence,
            "expected_input_lens": self.expected_input_lens,
            "expected_output_lens": self.expected_output_lens,
            "dag": self.dag,
            # Phase 2 — Stage B.
            "declared_max_concurrency": self.declared_max_concurrency,
            "in_flight_count": self.in_flight_count,
        }

    def is_idle(self) -> bool:
        """True iff this job has zero in-flight requests and never had any.

        Used by `JobRegistry.gc_idle_programs()` to identify pre-registered
        programs that never received a single LLM request — candidates for
        timeout-based GC. A Job that received at least one request and then
        completed transitions to JobState.COMPLETE instead.
        """
        return (
            self.state == JobState.QUEUED
            and self.total_request_number == 0
            and not self.request_ids
        )
