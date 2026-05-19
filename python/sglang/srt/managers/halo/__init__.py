"""HALO (Project Halo) — job-level slowdown tracking. See CLAUDE.md."""

from sglang.srt.managers.halo.admission_control.vss_predictor import (
    REASON_HALO_ADMISSION_PREDICTED,
    REASON_OK,
    ActiveCallInfo,
    AdmissionDecisionResult,
    AdmissionPredictor,
    JobLookaheadInput,
    JobSlowdownAdmissionPredictor,
    LookaheadAdmissionPredictor,
    NewJobInput,
    RequestSlowdownAdmissionPredictor,
    decide_admission,
)
from sglang.srt.managers.halo.controller import (
    REASON_DISABLED,
    REASON_HALO_CONCURRENCY_CAP,
    REASON_JOB_ID_ALREADY_REGISTERED,
    REASON_NO_JOB_ID,
    REASON_PROGRAM_NOT_REGISTERED,
    HaloConfig,
    HaloController,
    HaloRegisterProgramResult,
    HaloRejectError,
    build_halo_controller_from_server_args,
)
from sglang.srt.managers.halo.cost_model_sampler import (
    HaloCostModelSampler,
    build_halo_cost_sampler_from_server_args,
)
from sglang.srt.managers.halo.job import Job, JobCallSpan, JobState
from sglang.srt.managers.halo.job_registry import JobAdmissionResult, JobRegistry
from sglang.srt.managers.halo.metrics import HaloMetrics
from sglang.srt.managers.halo.slowdown_tracker import (
    RequestExecutionInfo,
    SlowdownTracker,
)

__all__ = [
    "ActiveCallInfo",
    "AdmissionDecisionResult",
    "AdmissionPredictor",
    "HaloConfig",
    "HaloController",
    "HaloCostModelSampler",
    "HaloMetrics",
    "HaloRegisterProgramResult",
    "HaloRejectError",
    "Job",
    "JobAdmissionResult",
    "JobCallSpan",
    "JobLookaheadInput",
    "JobRegistry",
    "JobSlowdownAdmissionPredictor",
    "JobState",
    "LookaheadAdmissionPredictor",
    "NewJobInput",
    "REASON_DISABLED",
    "REASON_HALO_ADMISSION_PREDICTED",
    "REASON_HALO_CONCURRENCY_CAP",
    "REASON_JOB_ID_ALREADY_REGISTERED",
    "REASON_NO_JOB_ID",
    "REASON_OK",
    "REASON_PROGRAM_NOT_REGISTERED",
    "RequestExecutionInfo",
    "RequestSlowdownAdmissionPredictor",
    "SlowdownTracker",
    "build_halo_controller_from_server_args",
    "build_halo_cost_sampler_from_server_args",
    "decide_admission",
]
