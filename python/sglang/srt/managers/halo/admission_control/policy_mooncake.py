"""MooncakePolicy — per-request predictive TTFT/TBT admission.

Mooncake-style: predict the arriving request's TTFT and TBT from cost
models + current server state and reject if either would breach the
request's SLO. This is the historical 3-stage check (Stage 1 TTFT,
Stage 2 TBT predicted, Stage 3 TBT reactive-EWMA), now applied
*per-request* against each request's own `ttft_slo` / `tbt_slo`.

SLO interpretation follows the server-level `--halo-slo-mode` flag:
  - "absolute": ttft_slo / tbt_slo are millisecond caps.
  - "ratio":    they are slowdown bounds vs the request's solo-run baseline.

It is the validation baseline for the request-level refactor — the proven
gate that the new reactive policy is measured against.
"""

from __future__ import annotations

from typing import Optional

from sglang.srt.managers.halo.admission_control.cost_model import (
    PrefillCostModel,
    TBTCostModel,
)
from sglang.srt.managers.halo.admission_control.policy import (
    AdmissionPolicy,
    NewRequestInput,
    PolicyDecision,
)
from sglang.srt.managers.halo.admission_control.tbt_tracker import TBTEwmaTracker
from sglang.srt.managers.halo.request_tracker.record import ServerStateSnapshot

REASON_ADMIT = "ADMIT"
REASON_TTFT = "MOONCAKE_TTFT"
REASON_TBT = "MOONCAKE_TBT"
REASON_TBT_REACTIVE = "MOONCAKE_TBT_REACTIVE"

# Below this solo baseline the ratio is numerical noise — skip the ratio check.
_RATIO_SOLO_FLOOR_MS = 1.0


class MooncakePolicy(AdmissionPolicy):
    """Predictive per-request TTFT/TBT gate."""

    name = "mooncake"

    def __init__(
        self,
        *,
        prefill_cost: Optional[PrefillCostModel],
        tbt_cost: Optional[TBTCostModel],
        slo_mode: str = "ratio",
        tbt_tracker: Optional[TBTEwmaTracker] = None,
        tbt_reactive_ratio: float = 0.9,
    ) -> None:
        self.prefill_cost = prefill_cost
        self.tbt_cost = tbt_cost
        self.slo_mode = slo_mode
        self.tbt_tracker = tbt_tracker
        self.tbt_reactive_ratio = tbt_reactive_ratio

    def decide(
        self, new_req: NewRequestInput, state: ServerStateSnapshot
    ) -> PolicyDecision:
        # ── Predict TTFT: queue backlog + this request's solo prefill ────
        solo_ttft: Optional[float] = None
        pred_ttft: Optional[float] = None
        if self.prefill_cost is not None:
            solo_ttft = self.prefill_cost.estimate_ms(
                new_req.prompt_len, new_req.prefix_len
            )
            t_queue = sum(
                self.prefill_cost.estimate_ms(r.prompt_len, r.prefix_len)
                for r in state.queued
            )
            pred_ttft = t_queue + solo_ttft

        # ── Predict TBT: cost model on the running batch + this request ──
        solo_tbt: Optional[float] = None
        pred_tbt: Optional[float] = None
        if self.tbt_cost is not None:
            solo_tbt = self.tbt_cost.estimate_ms(1, new_req.prompt_len)
            bs = len(state.running) + 1
            total_kv = sum(r.kv_len for r in state.running) + new_req.prompt_len
            pred_tbt = self.tbt_cost.estimate_ms(bs, total_kv // bs)

        detail = {
            "solo_ttft_ms": solo_ttft,
            "pred_ttft_ms": pred_ttft,
            "solo_tbt_ms": solo_tbt,
            "pred_tbt_ms": pred_tbt,
        }

        # ── Stage 1: TTFT ────────────────────────────────────────────────
        if (
            new_req.ttft_slo is not None
            and pred_ttft is not None
            and self._violates(pred_ttft, solo_ttft, new_req.ttft_slo)
        ):
            return PolicyDecision(
                False, REASON_TTFT, pred_ttft, pred_tbt, detail=detail
            )

        # ── Stage 2: TBT predicted ───────────────────────────────────────
        if (
            new_req.tbt_slo is not None
            and pred_tbt is not None
            and self._violates(pred_tbt, solo_tbt, new_req.tbt_slo)
        ):
            return PolicyDecision(
                False, REASON_TBT, pred_ttft, pred_tbt, detail=detail
            )

        # ── Stage 3: TBT reactive EWMA (absolute SLO only) ───────────────
        if (
            self.slo_mode == "absolute"
            and new_req.tbt_slo is not None
            and self.tbt_tracker is not None
            and self.tbt_tracker.is_warm()
        ):
            ewma = self.tbt_tracker.get()
            detail["tbt_ewma_ms"] = ewma
            if ewma > new_req.tbt_slo * self.tbt_reactive_ratio:
                return PolicyDecision(
                    False, REASON_TBT_REACTIVE, pred_ttft, pred_tbt, detail=detail
                )

        return PolicyDecision(
            True, REASON_ADMIT, pred_ttft, pred_tbt, detail=detail
        )

    def _violates(
        self, predicted: float, solo: Optional[float], slo: float
    ) -> bool:
        """True if `predicted` breaches `slo` under the configured slo_mode."""
        if self.slo_mode == "absolute":
            return predicted > slo
        # ratio mode — predicted / solo > slo
        if solo is None or solo <= _RATIO_SOLO_FLOOR_MS:
            return False
        return predicted / solo > slo
