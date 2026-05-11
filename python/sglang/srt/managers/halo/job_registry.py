"""JobRegistry — rid ↔ Job mapping for HALO Phase 1.

See managers/halo/CLAUDE.md.
"""

# HALO: Phase 1 job ownership table. The scheduler is the sole owner; methods
# are not thread-safe by design (single-threaded scheduler loop).
#
# Lifecycle:
#   - record_admission(job_id, slo, rid): lazily creates Job if first seen,
#     bumps counters, maps rid → job_id.
#   - record_completion(rid): finds job for rid, decrements counters, removes
#     mapping.
#   - active_jobs(): returns all non-COMPLETE jobs.
#   - gc_completed(retain_seconds): drops COMPLETE jobs older than threshold
#     (called periodically from controller.tick()).
#
# Issues / future concerns:
#   - Memory unbounded if many short jobs with distinct ids: gc_completed limits this.
#   - Concurrent rid collisions across requests of one job are fine (set semantics);
#     across different jobs is a client bug (we keep the first mapping, log a warning).

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from sglang.srt.managers.halo.job import Job, JobState, _now_monotonic

logger = logging.getLogger(__name__)


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._rid_to_job: Dict[str, str] = {}

    # ---- mutators ----

    def record_admission(self, job_id: str, slo: float, rid: str) -> Job:
        """Register an incoming request. Creates the Job lazily on first sight.

        If the same rid was already mapped to a different job, the old mapping
        wins (we log a warning). This should not happen with well-formed
        clients (rid is server-generated and globally unique).
        """
        existing = self._rid_to_job.get(rid)
        if existing is not None and existing != job_id:
            logger.warning(
                "halo: rid=%s already mapped to job_id=%s; ignoring new claim job_id=%s",
                rid, existing, job_id,
            )
            return self._jobs[existing]

        job = self._jobs.get(job_id)
        if job is None:
            job = Job(job_id=job_id, slo=slo)
            self._jobs[job_id] = job
        # If a later request omits SLO but the job already exists, keep the
        # SLO recorded at first sighting. Phase 2 may revisit semantics.
        self._rid_to_job[rid] = job_id
        job.on_request_admitted(rid)
        return job

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

    # ---- read-only ----

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
