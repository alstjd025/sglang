"""JobRegistry — rid ↔ Job mapping for HALO Phase 1 (Option A + B).

See managers/halo/CLAUDE.md.
"""

# HALO: Phase 1 job ownership table. The scheduler is the sole owner; methods
# are not thread-safe by design (single-threaded scheduler loop).
#
# Lifecycle:
#   - register_program(...): Option A — pre-register a Job before any LLM
#     request arrives. Carries chain length / DAG / expected token lens for
#     Phase 2 admission lookahead. Returns (newly_registered, job).
#   - record_admission(job_id, slo, rid): Option B transport — admit one
#     request into the pre-registered job. Per Q12, jobs that were NOT
#     pre-registered are rejected with `AdmissionResult(admit=False, ...)`
#     so the scheduler can convert to HTTP 400.
#   - record_completion(rid): finds job for rid, decrements counters, removes
#     mapping.
#   - active_jobs(): returns all non-COMPLETE jobs.
#   - gc_completed(retain_seconds): drops COMPLETE jobs older than threshold.
#   - gc_idle_programs(idle_seconds): drops pre-registered jobs that never
#     received a single LLM request (Q13 — idle timeout).
#
# Issues / future concerns:
#   - Concurrent rid collisions across requests of one job are fine (set
#     semantics); across different jobs is a client bug (we keep the first
#     mapping, log a warning).

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from sglang.srt.managers.halo.job import Job, JobState, _now_monotonic

logger = logging.getLogger(__name__)


# Reject reasons returned by `record_admission` to the scheduler.
# The scheduler converts these into HTTP status codes (see managers/halo/
# CLAUDE.md §13 Q12).
REASON_PROGRAM_NOT_REGISTERED = "HALO_PROGRAM_NOT_REGISTERED"


@dataclass
class AdmissionResult:
    """Result of `JobRegistry.record_admission()`.

    `admit=True`  → job is the registered Job (counters bumped, rid mapped).
    `admit=False` → reason is set to one of REASON_* constants; the scheduler
                    converts this to an HTTP 4xx reject. `job` is None.
    """

    admit: bool
    job: Optional[Job] = None
    reason: Optional[str] = None


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._rid_to_job: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Option A — pre-registration (POST /halo/programs)
    # ------------------------------------------------------------------

    def register_program(
        self,
        job_id: str,
        slo: float,
        *,
        total_calls: Optional[int] = None,
        stage_sequence: Optional[List[str]] = None,
        expected_input_lens: Optional[List[int]] = None,
        expected_output_lens: Optional[List[int]] = None,
        dag: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Job]:
        """Pre-register a job before any LLM request arrives (Option A).

        Returns `(newly_registered, job)`:
        - newly_registered=True  → fresh registration (HTTP 200)
        - newly_registered=False → job_id already exists; caller returns
                                   HTTP 409 (per Q11).
        """
        if not math.isfinite(slo) or slo <= 0.0:
            # Defensive: the HTTP layer should have validated already, but
            # we don't want a misconfigured client to poison the registry.
            raise ValueError(f"invalid slo={slo!r}; must be a positive finite float")
        existing = self._jobs.get(job_id)
        if existing is not None:
            return False, existing
        job = Job(
            job_id=job_id,
            slo=slo,
            total_calls_expected=total_calls,
            stage_sequence=stage_sequence,
            expected_input_lens=expected_input_lens,
            expected_output_lens=expected_output_lens,
            dag=dag,
            from_program=True,
        )
        self._jobs[job_id] = job
        return True, job

    # ------------------------------------------------------------------
    # Option B — per-request admission
    # ------------------------------------------------------------------

    def record_admission(self, job_id: str, slo: float, rid: str) -> AdmissionResult:
        """Admit a request into a pre-registered job.

        Strict mode per Q12: jobs that were NOT pre-registered are rejected
        (the lazy-create branch from Phase-1-only Option B is gone). The
        scheduler converts `admit=False` into HTTP 400.

        If a previous request for the same `rid` was already mapped (should
        only happen with malformed clients), we keep the original mapping
        and emit a WARN.
        """
        existing_for_rid = self._rid_to_job.get(rid)
        if existing_for_rid is not None and existing_for_rid != job_id:
            logger.warning(
                "halo: rid=%s already mapped to job_id=%s; "
                "ignoring new claim job_id=%s",
                rid, existing_for_rid, job_id,
            )
            return AdmissionResult(admit=True, job=self._jobs[existing_for_rid])

        job = self._jobs.get(job_id)
        if job is None:
            # Q12: no lazy-create. The client must call POST /halo/programs
            # first. Return a rejection; scheduler will respond HTTP 400.
            return AdmissionResult(
                admit=False, reason=REASON_PROGRAM_NOT_REGISTERED
            )

        # SLO conflict policy (Q10): pre-registered SLO wins. The request's
        # claimed slo is logged as WARN if it differs but otherwise ignored.
        if (
            job.from_program
            and math.isfinite(slo)
            and not math.isclose(slo, job.slo, rel_tol=1e-9, abs_tol=1e-9)
        ):
            logger.warning(
                "halo: rid=%s job_id=%s ignoring request slo=%.4f "
                "(pre-registered slo=%.4f wins)",
                rid, job_id, slo, job.slo,
            )

        self._rid_to_job[rid] = job_id
        job.on_request_admitted(rid)
        return AdmissionResult(admit=True, job=job)

    def record_completion(self, rid: str) -> Optional[Job]:
        """Called from the scheduler's finish path. Returns the affected job."""
        job_id = self._rid_to_job.pop(rid, None)
        if job_id is None:
            return None
        job = self._jobs.get(job_id)
        if job is None:
            return None
        job.on_request_completed(rid)
        return job

    # ------------------------------------------------------------------
    # GC
    # ------------------------------------------------------------------

    def gc_completed(self, retain_seconds: float = 5.0) -> int:
        """Drop COMPLETE jobs that have been complete for > retain_seconds.

        Returns count of dropped jobs. Called from controller.tick().
        """
        now = _now_monotonic()
        to_drop: List[str] = []
        for jid, job in self._jobs.items():
            if (
                job.state == JobState.COMPLETE
                and not job.request_ids
                and now - job.last_update_ts > retain_seconds
            ):
                to_drop.append(jid)
        for jid in to_drop:
            self._jobs.pop(jid, None)
        return len(to_drop)

    def gc_idle_programs(self, idle_seconds: float) -> int:
        """Drop pre-registered jobs that never received a single LLM request.

        Implements Q13 — a client that calls `POST /halo/programs` but
        never issues a `chat.completions` for that job should not leak
        memory indefinitely. Programs idle for > `idle_seconds` are dropped
        with a WARN log.

        `idle_seconds <= 0` disables this GC entirely. Returns the number
        of dropped jobs (zero when disabled).
        """
        if idle_seconds <= 0:
            return 0
        now = _now_monotonic()
        to_drop: List[str] = []
        for jid, job in self._jobs.items():
            if (
                job.from_program
                and job.is_idle()
                and now - job.first_seen_ts > idle_seconds
            ):
                to_drop.append(jid)
        for jid in to_drop:
            logger.warning(
                "halo: idle program job_id=%s dropped after %.1fs without any "
                "LLM request (raise --halo-program-idle-timeout to keep it longer)",
                jid, idle_seconds,
            )
            self._jobs.pop(jid, None)
        return len(to_drop)

    # ------------------------------------------------------------------
    # Read-only
    # ------------------------------------------------------------------

    def job_for_request(self, rid: str) -> Optional[Job]:
        job_id = self._rid_to_job.get(rid)
        if job_id is None:
            return None
        return self._jobs.get(job_id)

    def job_for_id(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def active_jobs(self) -> List[Job]:
        """All jobs not in COMPLETE state."""
        return [j for j in self._jobs.values() if j.state != JobState.COMPLETE]

    def all_jobs(self) -> List[Job]:
        return list(self._jobs.values())

    def snapshot_dict(self, limit: int = 32) -> dict:
        """JSON-able snapshot for /server_info. Truncates the per-job list."""
        active = self.active_jobs()
        active.sort(key=lambda j: j.slowdown_max, reverse=True)
        return {
            "active_jobs": len(active),
            "total_known_jobs": len(self._jobs),
            "jobs": [j.to_dict() for j in active[:limit]],
        }
