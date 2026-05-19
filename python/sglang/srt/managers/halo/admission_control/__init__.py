"""Admission control module — see CLAUDE.md for the full design."""

from sglang.srt.managers.halo.admission_control.controller import (
    REASON_ADMIT,
    REASON_DISABLED,
    REASON_TBT_PREDICTED,
    REASON_TBT_RATIO,
    REASON_TBT_REACTIVE,
    REASON_TTFT_PREDICTED,
    REASON_TTFT_RATIO,
    REJECT_REASONS,
    AdmissionConfig,
    AdmissionController,
    AdmissionDecision,
    SchedulerSnapshot,
)
from sglang.srt.managers.halo.admission_control.cost_model import (
    HALO_STEP_FORM_SPLIT_V1,
    HALO_STEP_FORM_V1,
    HALO_STEP_FORMS,
    CostModelLoadError,
    HaloStepCostModel,
    PrefillCostModel,
    TBTCostModel,
    try_load_halo_step_cost_model,
    try_load_prefill_cost_model,
    try_load_tbt_cost_model,
)
from sglang.srt.managers.halo.admission_control.decision_log import DecisionLogger
from sglang.srt.managers.halo.admission_control.metrics import AdmissionMetrics
from sglang.srt.managers.halo.admission_control.tbt_tracker import TBTEwmaTracker

# Request-level policy layer (Phase 1 request-level refactor).
from sglang.srt.managers.halo.admission_control.gate import (
    REASON_KV_CAP,
    AdmissionResult,
    HaloAdmissionGate,
)
from sglang.srt.managers.halo.admission_control.policy import (
    AdmissionPolicy,
    NewRequestInput,
    PolicyDecision,
)
from sglang.srt.managers.halo.admission_control.policy_mooncake import MooncakePolicy
from sglang.srt.managers.halo.admission_control.policy_vss import VssPolicy

__all__ = [
    "AdmissionConfig",
    "AdmissionController",
    "AdmissionDecision",
    "AdmissionMetrics",
    "AdmissionPolicy",
    "AdmissionResult",
    "HaloAdmissionGate",
    "MooncakePolicy",
    "NewRequestInput",
    "PolicyDecision",
    "REASON_KV_CAP",
    "VssPolicy",
    "CostModelLoadError",
    "DecisionLogger",
    "HALO_STEP_FORM_SPLIT_V1",
    "HALO_STEP_FORM_V1",
    "HALO_STEP_FORMS",
    "HaloStepCostModel",
    "PrefillCostModel",
    "REASON_ADMIT",
    "REASON_DISABLED",
    "REASON_TBT_PREDICTED",
    "REASON_TBT_RATIO",
    "REASON_TBT_REACTIVE",
    "REASON_TTFT_PREDICTED",
    "REASON_TTFT_RATIO",
    "REJECT_REASONS",
    "SchedulerSnapshot",
    "TBTCostModel",
    "TBTEwmaTracker",
    "try_load_halo_step_cost_model",
    "try_load_prefill_cost_model",
    "try_load_tbt_cost_model",
]
