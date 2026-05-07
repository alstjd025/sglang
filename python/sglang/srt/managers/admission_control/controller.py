"""Admission controller — hybrid 3-stage SLO-based admission policy.

See managers/admission_control/CLAUDE.md for the full design.

Stage 1: TTFT predicted = sum of queued T_prefill + this request's T_prefill
Stage 2: TBT predicted with current running batch + this request
Stage 3: TBT reactive — recent EWMA over per-step latency

The controller is intentionally decoupled from the Scheduler. The caller
provides a lightweight `SchedulerSnapshot` and the controller returns a
`AdmissionDecision`. The caller is responsible for:
  - constructing the snapshot from real scheduler state
  - sending the 429 AbortReq when admit=False
  - storing decision.predicted_prefill_ms on the request for the next call
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from sglang.srt.managers.admission_control.cost_model import (
    PrefillCostModel,
    TBTCostModel,
)
from sglang.srt.managers.admission_control.tbt_tracker import TBTEwmaTracker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerSnapshot:
    """Read-only view of scheduler state at the moment of admission.

    Keeps the controller testable without importing Scheduler/Req.
    """

    # Sum of `_predicted_prefill_ms` for every request currently in waiting_queue.
    # The caller computes this; the controller does not iterate scheduler state.
    waiting_queue_predicted_prefill_ms: float

    # Running (decoding) batch composition right now.
    running_batch_size: int
    running_batch_total_kv_tokens: int

    # Phase A only supports DisaggregationMode.NULL. Caller passes False for
    # PD/encode modes; controller short-circuits to ADMIT(reason=DISABLED).
    disaggregation_mode_is_null: bool


# Decision reasons. Stable strings — used as Prometheus label values, log keys,
# and JSONL log fields.
REASON_ADMIT = "ADMIT"
REASON_DISABLED = "DISABLED"
REASON_TTFT_PREDICTED = "TTFT_PREDICTED"        # absolute TTFT > ttft_slo_ms
REASON_TTFT_RATIO = "TTFT_RATIO"                # pred_ttft / solo_ttft > ratio
REASON_TBT_PREDICTED = "TBT_PREDICTED"          # absolute TBT > tbt_slo_ms
REASON_TBT_RATIO = "TBT_RATIO"                  # pred_tbt / solo_tbt > ratio
REASON_TBT_REACTIVE = "TBT_REACTIVE"            # measured EWMA > tbt_slo_ms * react

REJECT_REASONS = (
    REASON_TTFT_PREDICTED,
    REASON_TTFT_RATIO,
    REASON_TBT_PREDICTED,
    REASON_TBT_RATIO,
    REASON_TBT_REACTIVE,
)


# Below this floor (ms) the solo-run baseline is too small to give a meaningful
# slowdown ratio (numerical noise dominates). Skip ratio checks in that regime.
_RATIO_SOLO_FLOOR_MS = 1.0


@dataclass(frozen=True)
class AdmissionDecision:
    """Outcome of one admission decision.

    `admit` reflects the actual outcome (after dry-run handling).
    `dry_run_would_reject` is True iff the controller is in dry-run mode AND
    the underlying policy would have rejected. Use this to drive log/metric
    bookkeeping that distinguishes "shadow reject" from genuine admit.

    Solo-run baselines (`solo_ttft_ms`, `solo_tbt_ms`) are the model's estimate
    of "this request alone, on an idle server". The ratio policy compares the
    under-load prediction against this baseline.
    """

    admit: bool
    reason: str
    predicted_ttft_ms: Optional[float] = None
    predicted_tbt_ms: Optional[float] = None
    solo_ttft_ms: Optional[float] = None
    solo_tbt_ms: Optional[float] = None
    queue_predicted_ms: Optional[float] = None
    tbt_ewma_ms: Optional[float] = None
    ttft_slo_ms: Optional[float] = None
    tbt_slo_ms: Optional[float] = None
    ttft_slo_ratio: Optional[float] = None
    tbt_slo_ratio: Optional[float] = None
    # Echo of what the new request's T_prefill prediction was. Caller stores
    # this on the request so the next admission can include it in t_queue.
    predicted_prefill_ms: float = 0.0
    dry_run_would_reject: bool = False


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


@dataclass
class AdmissionConfig:
    """Plain-data config injected at scheduler `__init__`.

    Two parallel SLO axes — set either, or both:

    - **Absolute** (`ttft_slo_ms`, `tbt_slo_ms`): hard latency caps in ms.
      Reject if predicted latency exceeds the cap.
    - **Ratio** (`ttft_slo_ratio`, `tbt_slo_ratio`): per-request fairness bound,
      `pred / solo_run_baseline`. Reject if a request would be slowed down
      more than N× compared to running alone on an idle server. Adapts to
      heterogeneous request sizes (a 50k prompt has a 500ms baseline, so
      1000ms is fine; a 500-tok prompt has a 30ms baseline, so 60ms is the
      bound — both at ratio=2).

    When both are set, whichever fires first wins; absolute is checked
    before ratio so the reason is the more "severe" cap.
    """

    ttft_slo_ms: Optional[float] = None
    tbt_slo_ms: Optional[float] = None
    ttft_slo_ratio: Optional[float] = None
    tbt_slo_ratio: Optional[float] = None
    tbt_ewma_alpha: float = 0.1
    tbt_warm_up_steps: int = 100
    tbt_reactive_ratio: float = 0.9
    dry_run: bool = False

    @property
    def ttft_abs_enabled(self) -> bool:
        return self.ttft_slo_ms is not None and self.ttft_slo_ms > 0

    @property
    def ttft_ratio_enabled(self) -> bool:
        return self.ttft_slo_ratio is not None and self.ttft_slo_ratio > 1.0

    @property
    def tbt_abs_enabled(self) -> bool:
        return self.tbt_slo_ms is not None and self.tbt_slo_ms > 0

    @property
    def tbt_ratio_enabled(self) -> bool:
        return self.tbt_slo_ratio is not None and self.tbt_slo_ratio > 1.0

    @property
    def stage1_enabled(self) -> bool:
        """Stage 1 (TTFT) runs iff at least one TTFT criterion is set."""
        return self.ttft_abs_enabled or self.ttft_ratio_enabled

    @property
    def stage2_enabled(self) -> bool:
        """Stage 2 (TBT predicted) runs iff at least one TBT criterion is set."""
        return self.tbt_abs_enabled or self.tbt_ratio_enabled

    @property
    def stage3_enabled(self) -> bool:
        """Stage 3 (TBT reactive) needs the absolute TBT SLO as a ms threshold."""
        return self.tbt_abs_enabled

    @property
    def any_stage_enabled(self) -> bool:
        return self.stage1_enabled or self.stage2_enabled


class AdmissionController:
    """Hybrid 3-stage SLO admission policy.

    All fields are optional; the controller silently degrades when a piece is
    missing:
      - prefill_cost=None  → Stage 1 disabled
      - tbt_cost=None      → Stage 2 disabled
      - tbt_tracker=None   → Stage 3 disabled (also disabled until warm)
    If neither SLO is configured, decide() short-circuits to ADMIT(DISABLED).
    """

    def __init__(
        self,
        config: AdmissionConfig,
        prefill_cost: Optional[PrefillCostModel] = None,
        tbt_cost: Optional[TBTCostModel] = None,
        tbt_tracker: Optional[TBTEwmaTracker] = None,
    ) -> None:
        self.config = config
        self.prefill_cost = prefill_cost
        self.tbt_cost = tbt_cost
        self.tbt_tracker = tbt_tracker

        # Lightweight in-memory ring buffer of recent decisions for
        # internal-state introspection (Step 4 wires it into get_internal_state).
        self._recent_decisions: List[AdmissionDecision] = []
        self._recent_decisions_max = 32
        self._disagg_warned = False

        if not config.any_stage_enabled:
            logger.info("admission control: disabled (no SLO configured)")
        else:
            logger.info(
                "admission control: enabled "
                "(ttft_slo_ms=%s tbt_slo_ms=%s ttft_slo_ratio=%s tbt_slo_ratio=%s "
                "dry_run=%s prefill_cost=%s tbt_cost=%s tbt_tracker=%s)",
                config.ttft_slo_ms,
                config.tbt_slo_ms,
                config.ttft_slo_ratio,
                config.tbt_slo_ratio,
                config.dry_run,
                prefill_cost is not None,
                tbt_cost is not None,
                tbt_tracker is not None,
            )

    # ---- public API -------------------------------------------------------

    def is_active(self) -> bool:
        return self.config.any_stage_enabled

    def decide(
        self,
        prompt_len: int,
        prefix_match_len: int,
        snapshot: SchedulerSnapshot,
    ) -> AdmissionDecision:
        """Run the 3-stage policy and return a decision."""
        if not self.config.any_stage_enabled:
            return self._record(
                AdmissionDecision(admit=True, reason=REASON_DISABLED)
            )

        if not snapshot.disaggregation_mode_is_null:
            if not self._disagg_warned:
                logger.warning(
                    "admission control: disabled in disaggregation mode "
                    "(Phase A is single-instance only)"
                )
                self._disagg_warned = True
            return self._record(
                AdmissionDecision(admit=True, reason=REASON_DISABLED)
            )

        # ---- Compute baselines + predictions ----------------------------
        # Solo-run TTFT = T_prefill on an idle server (queue empty). This is
        # also the per-request "predicted prefill" we attach back to the req
        # so the next admission's t_queue includes it.
        solo_ttft = (
            self.prefill_cost.estimate_ms(prompt_len, prefix_match_len)
            if self.prefill_cost is not None
            else None
        )
        t_prefill = solo_ttft if solo_ttft is not None else 0.0
        t_queue = snapshot.waiting_queue_predicted_prefill_ms
        pred_ttft = t_queue + t_prefill

        # Solo-run TBT = TBT cost model at bs=1 with just this request's KV.
        # Predicted TBT = same model with running_batch + this request added.
        solo_tbt: Optional[float] = None
        pred_tbt: Optional[float] = None
        if self.tbt_cost is not None:
            solo_tbt = self.tbt_cost.estimate_ms(1, prompt_len)
            bs = snapshot.running_batch_size + 1
            kv = snapshot.running_batch_total_kv_tokens + prompt_len
            pred_tbt = self.tbt_cost.estimate_ms(bs, kv)

        # ---- Stage 1a: TTFT absolute -------------------------------------
        if (
            self.config.ttft_abs_enabled
            and solo_ttft is not None
            and pred_ttft > self.config.ttft_slo_ms
        ):
            return self._record(
                self._reject_or_dryrun(
                    reason=REASON_TTFT_PREDICTED,
                    predicted_ttft_ms=pred_ttft,
                    predicted_tbt_ms=pred_tbt,
                    solo_ttft_ms=solo_ttft,
                    solo_tbt_ms=solo_tbt,
                    predicted_prefill_ms=t_prefill,
                    queue_predicted_ms=t_queue,
                )
            )

        # ---- Stage 1b: TTFT ratio (slowdown vs solo-run) -----------------
        if (
            self.config.ttft_ratio_enabled
            and solo_ttft is not None
            and solo_ttft > _RATIO_SOLO_FLOOR_MS
            and pred_ttft / solo_ttft > self.config.ttft_slo_ratio
        ):
            return self._record(
                self._reject_or_dryrun(
                    reason=REASON_TTFT_RATIO,
                    predicted_ttft_ms=pred_ttft,
                    predicted_tbt_ms=pred_tbt,
                    solo_ttft_ms=solo_ttft,
                    solo_tbt_ms=solo_tbt,
                    predicted_prefill_ms=t_prefill,
                    queue_predicted_ms=t_queue,
                )
            )

        # ---- Stage 2a: TBT absolute --------------------------------------
        if (
            self.config.tbt_abs_enabled
            and pred_tbt is not None
            and pred_tbt > self.config.tbt_slo_ms
        ):
            return self._record(
                self._reject_or_dryrun(
                    reason=REASON_TBT_PREDICTED,
                    predicted_ttft_ms=pred_ttft,
                    predicted_tbt_ms=pred_tbt,
                    solo_ttft_ms=solo_ttft,
                    solo_tbt_ms=solo_tbt,
                    predicted_prefill_ms=t_prefill,
                    queue_predicted_ms=t_queue,
                )
            )

        # ---- Stage 2b: TBT ratio -----------------------------------------
        if (
            self.config.tbt_ratio_enabled
            and pred_tbt is not None
            and solo_tbt is not None
            and solo_tbt > _RATIO_SOLO_FLOOR_MS
            and pred_tbt / solo_tbt > self.config.tbt_slo_ratio
        ):
            return self._record(
                self._reject_or_dryrun(
                    reason=REASON_TBT_RATIO,
                    predicted_ttft_ms=pred_ttft,
                    predicted_tbt_ms=pred_tbt,
                    solo_ttft_ms=solo_ttft,
                    solo_tbt_ms=solo_tbt,
                    predicted_prefill_ms=t_prefill,
                    queue_predicted_ms=t_queue,
                )
            )

        # ---- Stage 3: TBT reactive EWMA safety net (absolute SLO only) ---
        ewma_ms: Optional[float] = None
        if self.config.stage3_enabled and self.tbt_tracker is not None:
            if self.tbt_tracker.is_warm():
                ewma_ms = self.tbt_tracker.get()
                threshold = self.config.tbt_slo_ms * self.config.tbt_reactive_ratio
                if ewma_ms > threshold:
                    return self._record(
                        self._reject_or_dryrun(
                            reason=REASON_TBT_REACTIVE,
                            predicted_ttft_ms=pred_ttft,
                            predicted_tbt_ms=pred_tbt,
                            solo_ttft_ms=solo_ttft,
                            solo_tbt_ms=solo_tbt,
                            tbt_ewma_ms=ewma_ms,
                            predicted_prefill_ms=t_prefill,
                            queue_predicted_ms=t_queue,
                        )
                    )
            else:
                ewma_ms = self.tbt_tracker.get()  # informational only

        # ---- Admit -------------------------------------------------------
        return self._record(
            AdmissionDecision(
                admit=True,
                reason=REASON_ADMIT,
                predicted_ttft_ms=pred_ttft,
                predicted_tbt_ms=pred_tbt,
                solo_ttft_ms=solo_ttft,
                solo_tbt_ms=solo_tbt,
                queue_predicted_ms=t_queue,
                tbt_ewma_ms=ewma_ms,
                ttft_slo_ms=self.config.ttft_slo_ms,
                tbt_slo_ms=self.config.tbt_slo_ms,
                ttft_slo_ratio=self.config.ttft_slo_ratio,
                tbt_slo_ratio=self.config.tbt_slo_ratio,
                predicted_prefill_ms=t_prefill,
            )
        )

    def recent_decisions(self) -> List[AdmissionDecision]:
        """Snapshot of the recent-decisions ring buffer."""
        return list(self._recent_decisions)

    # ---- internals --------------------------------------------------------

    def _reject_or_dryrun(
        self,
        *,
        reason: str,
        predicted_ttft_ms: Optional[float] = None,
        predicted_tbt_ms: Optional[float] = None,
        solo_ttft_ms: Optional[float] = None,
        solo_tbt_ms: Optional[float] = None,
        tbt_ewma_ms: Optional[float] = None,
        queue_predicted_ms: Optional[float] = None,
        predicted_prefill_ms: float = 0.0,
    ) -> AdmissionDecision:
        admit = self.config.dry_run  # in dry-run we always admit
        return AdmissionDecision(
            admit=admit,
            reason=reason,
            predicted_ttft_ms=predicted_ttft_ms,
            predicted_tbt_ms=predicted_tbt_ms,
            solo_ttft_ms=solo_ttft_ms,
            solo_tbt_ms=solo_tbt_ms,
            queue_predicted_ms=queue_predicted_ms,
            tbt_ewma_ms=tbt_ewma_ms,
            ttft_slo_ms=self.config.ttft_slo_ms,
            tbt_slo_ms=self.config.tbt_slo_ms,
            ttft_slo_ratio=self.config.ttft_slo_ratio,
            tbt_slo_ratio=self.config.tbt_slo_ratio,
            predicted_prefill_ms=predicted_prefill_ms,
            dry_run_would_reject=self.config.dry_run,
        )

    def _record(self, decision: AdmissionDecision) -> AdmissionDecision:
        self._recent_decisions.append(decision)
        if len(self._recent_decisions) > self._recent_decisions_max:
            del self._recent_decisions[
                : len(self._recent_decisions) - self._recent_decisions_max
            ]
        return decision
