"""Admission policy abstraction — the pluggable decision algorithm.

A `HaloAdmissionGate` runs exactly one `AdmissionPolicy`. The policy rules
admit/reject on an arriving request given current server state; the gate
adds the policy-independent KV-cache hard cap on top.

Policies (selected by `--halo-admission-policy`):
  - "mooncake" — per-request predictive TTFT/TBT cost-model gate.
  - "vss"      — per-request memoryless virtual-server-slowdown gate.
  - "reactive" — measured, queueing-inclusive reactive gate (Phase 3).

Design note (D5): all three request SLOs (ttft / tbt / e2e) are always
active; each policy uses whichever ones its signal can speak to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from sglang.srt.managers.halo.request_tracker.record import ServerStateSnapshot


@dataclass(frozen=True)
class NewRequestInput:
    """The arriving request the admission gate must rule on."""

    rid: str
    prompt_len: int
    prefix_len: int  # radix-cache match length at admission
    ttft_slo: Optional[float] = None
    tbt_slo: Optional[float] = None
    e2e_slo: Optional[float] = None


@dataclass(frozen=True)
class PolicyDecision:
    """A policy's verdict on one arrival.

    `reason` is a policy-specific string ("ADMIT" when admitted). The
    `predicted_*` fields are best-effort and feed predicted-vs-actual
    evaluation + the decision log; a policy leaves them None when its
    signal does not produce that quantity.
    """

    admit: bool
    reason: str
    predicted_ttft_ms: Optional[float] = None
    predicted_tbt_ms: Optional[float] = None
    predicted_e2e_slowdown: Optional[float] = None
    detail: dict = field(default_factory=dict)


class AdmissionPolicy(ABC):
    """Pluggable admission decision algorithm."""

    name: str = "abstract"

    @abstractmethod
    def decide(
        self, new_req: NewRequestInput, state: ServerStateSnapshot
    ) -> PolicyDecision:
        """Rule on `new_req` given current server `state`."""
