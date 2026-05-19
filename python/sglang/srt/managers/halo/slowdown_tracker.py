"""SlowdownTracker — job-lifetime virtual job slowdown (VJS) computation.

See managers/halo/CLAUDE.md and ms_dev/halo_dev/prediction_model.md.
"""

# HALO: virtual job slowdown (VJS).
#
# VJS = (job critical-path actual ms) / (job critical-path solo ms),
# accumulated over the job's whole life so far — every completed call plus
# every in-flight call. Tool-delay gaps between calls are excluded;
# concurrent calls are merged by critical path (max).
#
# compute_job_vjs() merges a job's call spans into STAGES by wall-clock
# interval overlap (a stage = one connected component of overlapping
# intervals, which is itself one contiguous time span), then per stage:
#
#   stage_actual = max(end_ts) - min(admitted_ts)            [interval union]
#   stage_solo   = max over the stage's calls of
#                  ( prefill_solo + decoded_tokens * decode_step )
#                  where decode_step is the cost-model DECODE-step time for
#                  the stage's decode-phase calls batched together — i.e.
#                  the job's own self-batching, which is present in BOTH the
#                  actual and the solo run, so it does not count as slowdown.
#
#   job_actual = Σ stage_actual,  job_solo = Σ stage_solo,  VJS = ratio.
#
# Both SlowdownTracker.sweep and the Phase-2 admission gate call
# compute_job_vjs — one source of truth. Admission then multiplies the
# result by a per-phase stretch (see managers/halo/admission_decision.py).
#
# Solo estimate — two cost-model paths:
#   (A) Halo Step Cost Model (default; see prediction_model.md):
#         prefill_solo = step_cost.estimate_solo_prefill_total_ms(n_new, r)
#         decode_step  = step_cost.estimate_step_ms([], kv_list)
#       where n_new = prompt_len - prefix_len  (the radix-cache hit `r` is
#       excluded from prefill compute — KV caching effect).
#   (B) Legacy Mooncake-like pair (fallback when no step model):
#         prefill_solo = prefill_cost.estimate_ms(prompt_len, prefix_len)
#         decode_step  = tbt_cost.estimate_ms(batch_size=K, per_req_kv=avg)

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional

from sglang.srt.managers.halo.admission_control.cost_model import (
    HaloStepCostModel,
    PrefillCostModel,
    TBTCostModel,
)
from sglang.srt.managers.halo.job import JobCallSpan
from sglang.srt.managers.halo.job_registry import JobRegistry

logger = logging.getLogger(__name__)

# Floor on a job's total solo time — guards the VJS division.
MIN_SOLO_MS = 1.0


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
    admitted_ts: float                    # monotonic s — when first admitted


class SlowdownTracker:
    """Pure compute. Turns request snapshots into each job's virtual job
    slowdown via compute_job_vjs (the stage-merge model documented above).

    Cost-model selection (mutually exclusive at construction time):
      - `step_cost` not None         → Halo Step Cost Model path (default).
      - `step_cost` None, legacy set → legacy two-model path.
      - All three None               → no-op tracker (VJS stays at SLO).

    If a caller passes all three, `step_cost` wins and the legacy pair is
    ignored.
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

    # ------------------------------------------------------------------
    # Per-call solo primitives (cost-model path A or B)
    # ------------------------------------------------------------------

    def prefill_solo_ms(self, prompt_len: int, prefix_len: int) -> float:
        """Solo prefill time for one call.

        `n_new = prompt_len - prefix_len` so the radix-cache hit (`prefix_len`
        cached tokens) is excluded from prefill compute — only the new tokens
        are prefilled. Returns 0.0 when no cost model is loaded.
        """
        n_new = max(0, prompt_len - prefix_len)
        r = max(0, prefix_len)
        if self.step_cost is not None:
            return self.step_cost.estimate_solo_prefill_total_ms(n_new, r)
        if self.prefill_cost is not None:
            # PrefillCostModel.estimate_ms(n, p) computes d = n - p itself.
            return self.prefill_cost.estimate_ms(prompt_len, prefix_len)
        return 0.0

    def decode_step_ms(self, kv_lens: List[int]) -> float:
        """One DECODE step's wall-clock for a batch with the given per-request
        KV spans. Empty list ⇒ 0.0 (no decode-phase call in the batch).

        Used both for a job's own concurrent self-batching (stage solo) and,
        with a single-element list, for one solo decoder's per-token time.
        """
        kvs = [max(0, k) for k in kv_lens]
        if not kvs:
            return 0.0
        if self.step_cost is not None:
            return self.step_cost.estimate_step_ms([], kvs)
        if self.tbt_cost is not None:
            return self.tbt_cost.estimate_ms(
                batch_size=len(kvs), per_req_kv=sum(kvs) // len(kvs)
            )
        return 0.0

    def span_from_info(self, info: RequestExecutionInfo) -> JobCallSpan:
        """Build a JobCallSpan from an in-flight (or just-finished) request
        snapshot. `end_ts = admitted_ts + elapsed` — for an in-flight request
        this is "now"; the scheduler builds the finish snapshot with the
        final elapsed so end_ts becomes the finish time.
        """
        return JobCallSpan(
            admitted_ts=info.admitted_ts,
            end_ts=info.admitted_ts + max(0.0, info.elapsed_ms) / 1000.0,
            prompt_len=info.prompt_len,
            prefix_len=info.prefix_len_at_admission,
            decoded_tokens=max(0, info.decoded_tokens_so_far),
            kv_len=max(0, info.kv_len_now),
        )

    # ------------------------------------------------------------------
    # Job-lifetime VJS
    # ------------------------------------------------------------------

    def compute_job_vjs(
        self, spans: List[JobCallSpan], fallback_slo: float
    ) -> float:
        """Virtual job slowdown over the given call spans (a job's completed
        spans + its current in-flight spans). See the module docstring and
        ms_dev/halo_dev/prediction_model.md.

        Returns `fallback_slo` when there are no spans or no cost model —
        matching Job's initial-VJS = SLO convention.
        """
        if not spans or not self.has_cost_model:
            return fallback_slo

        # 1. Merge spans into stages by wall-clock interval overlap. Sorting
        #    by start time and sweeping merges every connected overlap chain;
        #    each resulting stage is one contiguous time span.
        ordered = sorted(spans, key=lambda s: s.admitted_ts)
        stages: List[List[JobCallSpan]] = []
        cur: List[JobCallSpan] = []
        cur_end = float("-inf")
        for sp in ordered:
            if cur and sp.admitted_ts <= cur_end:
                cur.append(sp)
                cur_end = max(cur_end, sp.end_ts)
            else:
                if cur:
                    stages.append(cur)
                cur = [sp]
                cur_end = sp.end_ts
        if cur:
            stages.append(cur)

        # 2-4. Per stage: actual = interval span; solo = critical path (max)
        #      over the stage's calls, each call's decode steps charged at the
        #      stage's batched decode-step time. Sum over stages.
        job_actual_ms = 0.0
        job_solo_ms = 0.0
        for stage in stages:
            stage_actual_s = max(s.end_ts for s in stage) - min(
                s.admitted_ts for s in stage
            )
            job_actual_ms += max(0.0, stage_actual_s) * 1000.0

            # The stage's decode-phase calls run as one batch (the job alone).
            decode_step = self.decode_step_ms(
                [s.kv_len for s in stage if s.decoded_tokens > 0]
            )
            stage_solo = 0.0
            for s in stage:
                call_solo = self.prefill_solo_ms(s.prompt_len, s.prefix_len)
                call_solo += s.decoded_tokens * decode_step
                if call_solo > stage_solo:
                    stage_solo = call_solo
            job_solo_ms += stage_solo

        if job_solo_ms <= 0.0:
            return fallback_slo
        vjs = job_actual_ms / max(job_solo_ms, MIN_SOLO_MS)
        if not math.isfinite(vjs):
            return fallback_slo
        return vjs

    # ------------------------------------------------------------------
    # Per-tick sweep
    # ------------------------------------------------------------------

    def sweep(self, infos: List[RequestExecutionInfo]) -> None:
        """Recompute the VJS of every job that currently has an in-flight
        call. A job with no in-flight call keeps its last VJS (it is doing
        no work, so the value is unchanged anyway).
        """
        if not infos:
            return
        by_job: Dict[str, List[RequestExecutionInfo]] = {}
        for info in infos:
            if info.job_id is None:
                continue
            by_job.setdefault(info.job_id, []).append(info)

        for job_id, job_infos in by_job.items():
            job = self.registry.job_for_id(job_id)
            if job is None:
                continue
            inflight = [self.span_from_info(i) for i in job_infos]
            vjs = self.compute_job_vjs(
                job.completed_call_spans + inflight, fallback_slo=job.slo
            )
            job.record_vjs(vjs)
