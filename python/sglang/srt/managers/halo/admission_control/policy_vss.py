"""VssPolicy — per-request memoryless virtual-server-slowdown admission.

Carried over from the job-level era (the request-scoped VSS arm). Every
in-flight request is scored with the predicted virtual server slowdown it
would experience if the arrival is admitted; the gate rejects when the
violation ratio exceeds the threshold. VSS is a per-step *contention*
ratio, so each running request is scored against its `e2e_slo` (a
slowdown bound).

The batch-slowdown kernel (augmented step times, chunk-bounded EXTEND
step) is reused here and is also intended for the Phase-3 reactive
policy's *interference* term.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.managers.halo.admission_control.cost_model import HaloStepCostModel
from sglang.srt.managers.halo.admission_control.policy import (
    AdmissionPolicy,
    NewRequestInput,
    PolicyDecision,
)
from sglang.srt.managers.halo.admission_control.vss_predictor import (
    ActiveCallInfo,
    ArrivalInput,
    BatchSnapshot,
    RequestSlowdownAdmissionPredictor,
    decide_admission,
)
from sglang.srt.managers.halo.request_tracker.record import ServerStateSnapshot

REASON_ADMIT = "ADMIT"
REASON_VSS = "HALO_VSS_PREDICTED"


class VssPolicy(AdmissionPolicy):
    """Memoryless per-request virtual-server-slowdown gate."""

    name = "vss"

    def __init__(
        self,
        *,
        cost_model: HaloStepCostModel,
        violation_threshold: float = 0.2,
        chunked_prefill_size: Optional[int] = None,
    ) -> None:
        self.cost_model = cost_model
        self.violation_threshold = violation_threshold
        self.chunked_prefill_size = chunked_prefill_size
        self._predictor = RequestSlowdownAdmissionPredictor(cost_model)

    def decide(
        self, new_req: NewRequestInput, state: ServerStateSnapshot
    ) -> PolicyDecision:
        # The running batch is the set of "active calls" VSS scores.
        active_calls = tuple(
            ActiveCallInfo(
                rid=r.rid,
                is_prefill=r.is_prefill,
                prompt_len=r.prompt_len,
                prefix_len=r.prefix_len,
                decoded_tokens=r.decoded_tokens,
                kv_len_now=r.kv_len or (r.prompt_len + r.decoded_tokens),
            )
            for r in state.running
        )
        batch = BatchSnapshot(slo=0.0, active_calls=active_calls)
        arrival = ArrivalInput(
            slo=0.0,
            first_call_input_len=new_req.prompt_len,
            first_call_prefix_len=new_req.prefix_len,
        )
        predicted = self._predictor.predict(
            batch, arrival, chunked_prefill_size=self.chunked_prefill_size
        )
        # Score each running request's predicted VSS against its e2e_slo;
        # decide_admission treats a missing SLO as always satisfied.
        slos = {r.rid: r.e2e_slo for r in state.running if r.e2e_slo is not None}
        result = decide_admission(predicted, slos, self.violation_threshold)
        detail = {
            "violation_ratio": result.violation_ratio,
            "violation_count": result.violation_count,
            "scored_units": result.scored_total,
        }
        return PolicyDecision(
            admit=result.admit,
            reason=REASON_ADMIT if result.admit else REASON_VSS,
            detail=detail,
        )
