"""HaloAdmissionGate — the single admission gate.

One gate, one selected policy. The gate applies the policy-independent
KV-cache hard cap (Stage B′) first, then delegates to the policy. This
replaces the two serial gates of the job-level era (the standalone
mooncake `_abort_on_predicted_slo_violation` + Halo `_halo_register_or_abort`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sglang.srt.managers.halo.admission_control.policy import (
    AdmissionPolicy,
    NewRequestInput,
    PolicyDecision,
)
from sglang.srt.managers.halo.request_tracker.record import ServerStateSnapshot

# Stable reason strings — used as Prometheus labels, decision-log keys.
REASON_ADMIT = "ADMIT"
REASON_KV_CAP = "HALO_KV_CAP"


@dataclass(frozen=True)
class AdmissionResult:
    """Outcome of one gate decision."""

    admit: bool
    reason: str
    policy_decision: Optional[PolicyDecision] = None


class HaloAdmissionGate:
    """Runs the KV-cache hard cap, then the selected admission policy."""

    def __init__(
        self, policy: AdmissionPolicy, kv_cap_ratio: float = 0.0
    ) -> None:
        self.policy = policy
        # kv_cap_ratio in (0, 1]: reject when KV usage ≥ ratio. 0 disables it.
        self.kv_cap_ratio = kv_cap_ratio

    def decide(
        self, new_req: NewRequestInput, state: ServerStateSnapshot
    ) -> AdmissionResult:
        # ── Stage B′: KV-cache hard cap (policy-independent) ──────────────
        if (
            0.0 < self.kv_cap_ratio <= 1.0
            and state.kv_usage_ratio >= self.kv_cap_ratio
        ):
            return AdmissionResult(admit=False, reason=REASON_KV_CAP)

        # ── Selected policy ─────────────────────────────────────────────
        decision = self.policy.decide(new_req, state)
        return AdmissionResult(
            admit=decision.admit,
            reason=REASON_ADMIT if decision.admit else decision.reason,
            policy_decision=decision,
        )
