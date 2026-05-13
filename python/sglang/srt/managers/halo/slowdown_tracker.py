"""SlowdownTracker — periodic per-request slowdown sweep + per-job aggregation.

See managers/halo/CLAUDE.md and ms_dev/halo_dev/prediction_model.md.
"""

# HALO: Phase 1 slowdown derivation.
#
# Per-request slowdown = actual_elapsed_ms / solo_run_elapsed_ms.
#
# Two paths for the solo estimate, chosen by which cost model is loaded:
#
#   (A) Halo Step Cost Model (new — see prediction_model.md). When present, it
#       supersedes the legacy two-model path.
#         solo_prefill_ms = step_cost.estimate_solo_prefill_total_ms(n, r)
#         solo_tbt_ms     = step_cost.estimate_solo_tbt_ms(r_now)
#
#   (B) Legacy Mooncake-like pair. When step_cost is None and at least one of
#       the legacy models is loaded:
#         solo_prefill_ms = prefill_cost.estimate_ms(prompt_len, prefix_len)
#         solo_tbt_ms     = tbt_cost.estimate_ms(batch_size=1, per_req_kv=r_now)
#
#   solo_elapsed_ms = solo_prefill_ms + solo_tbt_ms * decoded_tokens_so_far
#
# Per-job aggregation: max + mean over its active requests (both kept on Job).
# The scheduler is responsible for assembling the RequestExecutionInfo list each
# tick (it has direct access to Req metadata; we don't want this module to
# import scheduler internals).
#
# Issues / future concerns:
#   - When decoded_tokens_so_far == 0 (still in prefill), solo_elapsed_ms is just
#     solo_prefill_ms; if actual_elapsed is also small the ratio is noisy. We
#     guard with MIN_SOLO_MS and MIN_DECODED_BEFORE_TRUST.
#   - The legacy TBT model has a narrow prediction range (~55ms regardless of
#     batch composition) — see cost_models/README.md §3. That's the original
#     motivation for path (A); see ms_dev/halo_dev/prediction_model.md §1.

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from sglang.srt.managers.admission_control.cost_model import (
    HaloStepCostModel,
    PrefillCostModel,
    TBTCostModel,
)
from sglang.srt.managers.halo.job_registry import JobRegistry

logger = logging.getLogger(__name__)

MIN_SOLO_MS = 1.0
MIN_DECODED_BEFORE_TRUST = 0   # 0 = trust from the first sweep; raise if noisy


@dataclass
class RequestExecutionInfo:
    """Per-request snapshot the scheduler builds each tick. Pure data."""

    rid: str
    job_id: Optional[str]                 # None means request was not Halo-tracked
    prompt_len: int                       # tokens in prompt
    prefix_len_at_admission: int          # radix-cache match length at admit time
    decoded_tokens_so_far: int
    kv_len_now: int                       # per-request KV span right now
    elapsed_ms: float                     # actual wall-clock since admission


class SlowdownTracker:
    """Pure compute. Given execution infos and the job registry, updates each
    affected Job's slowdown_max / slowdown_mean.

    Cost-model selection (mutually exclusive at construction time):
      - `step_cost` not None         → Halo Step Cost Model path (new).
      - `step_cost` None, legacy set → legacy two-model path (Phase 1).
      - All three None               → no-op tracker.

    The controller is responsible for not passing both. If a caller does pass
    all three, `step_cost` wins and the legacy pair is ignored.
    """

    def __init__(
        self,
        prefill_cost: Optional[PrefillCostModel],
        tbt_cost: Optional[TBTCostModel],
        registry: JobRegistry,
        step_cost: Optional[HaloStepCostModel] = None,
    ) -> None:
        self.prefill_cost = prefill_cost
        self.tbt_cost = tbt_cost
        self.step_cost = step_cost
        self.registry = registry

    @property
    def has_cost_model(self) -> bool:
        """True when any cost model is loaded — used by introspection and tests."""
        return (
            self.step_cost is not None
            or self.prefill_cost is not None
            or self.tbt_cost is not None
        )

    # ---- per-request ----

    def compute_request_slowdown(self, info: RequestExecutionInfo) -> Optional[float]:
        """Returns the slowdown ratio or None if no cost model is loaded."""
        if self.step_cost is not None:
            solo_prefill_ms = self.step_cost.estimate_solo_prefill_total_ms(
                info.prompt_len, info.prefix_len_at_admission
            )
            solo_tbt_ms = self.step_cost.estimate_solo_tbt_ms(
                max(info.kv_len_now, 0)
            )
        elif self.prefill_cost is not None or self.tbt_cost is not None:
            solo_prefill_ms = 0.0
            if self.prefill_cost is not None:
                solo_prefill_ms = self.prefill_cost.estimate_ms(
                    info.prompt_len, info.prefix_len_at_admission
                )
            solo_tbt_ms = 0.0
            if self.tbt_cost is not None:
                solo_tbt_ms = self.tbt_cost.estimate_ms(
                    batch_size=1, per_req_kv=max(info.kv_len_now, 0)
                )
        else:
            return None

        solo_total_ms = solo_prefill_ms + solo_tbt_ms * max(
            info.decoded_tokens_so_far, 0
        )
        solo_total_ms = max(solo_total_ms, MIN_SOLO_MS)

        ratio = info.elapsed_ms / solo_total_ms
        # Defensive: NaN/inf guard if anything upstream goes weird.
        if not math.isfinite(ratio):
            return None
        return ratio

    # ---- per-tick ----

    def sweep(self, infos: List[RequestExecutionInfo]) -> None:
        """Aggregate per-request slowdowns into each affected Job."""
        if not infos:
            return
        # Bucket requests by job_id.
        buckets: Dict[str, List[float]] = {}
        for info in infos:
            if info.job_id is None:
                continue
            if info.decoded_tokens_so_far < MIN_DECODED_BEFORE_TRUST:
                continue
            ratio = self.compute_request_slowdown(info)
            if ratio is None:
                continue
            buckets.setdefault(info.job_id, []).append(ratio)

        for job_id, ratios in buckets.items():
            if not ratios:
                continue
            job = self.registry.job_for_id(job_id)
            if job is None:
                continue
            mx = max(ratios)
            mn = sum(ratios) / len(ratios)
            job.record_sweep(mx, mn)
