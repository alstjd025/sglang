"""HALO (Project Halo) — job-level slowdown tracking. See CLAUDE.md."""

from sglang.srt.managers.halo.controller import (
    REASON_DISABLED,
    REASON_JOB_ID_ALREADY_REGISTERED,
    REASON_NO_JOB_ID,
    REASON_PROGRAM_NOT_REGISTERED,
    HaloConfig,
    HaloController,
    HaloRegisterProgramResult,
    HaloRejectError,
    build_halo_controller_from_server_args,
)
from sglang.srt.managers.halo.job import Job, JobState
from sglang.srt.managers.halo.job_registry import AdmissionResult, JobRegistry
from sglang.srt.managers.halo.slowdown_tracker import (
    RequestExecutionInfo,
    SlowdownTracker,
)

__all__ = [
    "AdmissionResult",
    "HaloConfig",
    "HaloController",
    "HaloRegisterProgramResult",
    "HaloRejectError",
    "Job",
    "JobRegistry",
    "JobState",
    "REASON_DISABLED",
    "REASON_JOB_ID_ALREADY_REGISTERED",
    "REASON_NO_JOB_ID",
    "REASON_PROGRAM_NOT_REGISTERED",
    "RequestExecutionInfo",
    "SlowdownTracker",
    "build_halo_controller_from_server_args",
]
