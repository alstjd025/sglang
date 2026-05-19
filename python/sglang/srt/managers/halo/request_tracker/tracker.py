"""RequestTracker — the single per-request state store.

`RequestTracker` owns one `RequestRecord` per admitted request and is the
*only* component that mutates request state. The scheduler feeds it
lifecycle events (`on_admitted`, `on_state_change`, `on_step`,
`on_finished`); the admission gate, policies, and metrics consume it
read-only via `get()` / `snapshot()`.

Single-threaded: every method is called from the scheduler process main
loop, like the rest of the halo module. No internal locking.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import List, Optional

from sglang.srt.managers.halo.admission_control.cost_model import (
    HaloStepCostModel,
    PrefillCostModel,
    TBTCostModel,
)
from sglang.srt.managers.halo.request_tracker.record import (
    RequestRecord,
    RequestState,
    ServerStateSnapshot,
)

logger = logging.getLogger(__name__)

# Floor on a solo-run estimate — guards the e2e-slowdown division.
MIN_SOLO_MS = 1.0


class RequestTracker:
    """Single source of truth for per-request state.

    Cost-model selection mirrors the legacy SlowdownTracker:
      - `step_cost` set            → Halo Step Cost Model path (preferred).
      - `step_cost` None, legacy   → legacy prefill/tbt two-model path.
      - all None                   → solo estimates return 0.0 (no cost model).
    """

    def __init__(
        self,
        *,
        step_cost: Optional[HaloStepCostModel] = None,
        prefill_cost: Optional[PrefillCostModel] = None,
        tbt_cost: Optional[TBTCostModel] = None,
        retain_finished: int = 256,
    ) -> None:
        self.step_cost = step_cost
        self.prefill_cost = prefill_cost
        self.tbt_cost = tbt_cost
        self._retain_finished = max(0, retain_finished)

        # rid → record. Active = QUEUED or RUNNING. Finished/rejected records
        # move to `_recent` (bounded ring) for introspection then are evicted.
        self._active: "OrderedDict[str, RequestRecord]" = OrderedDict()
        self._recent: "OrderedDict[str, RequestRecord]" = OrderedDict()

    @property
    def has_cost_model(self) -> bool:
        return (
            self.step_cost is not None
            or self.prefill_cost is not None
            or self.tbt_cost is not None
        )

    # ------------------------------------------------------------------
    # Event handlers — the ONLY mutators of RequestRecord.
    # ------------------------------------------------------------------

    def on_admitted(
        self,
        rid: str,
        *,
        ttft_slo: Optional[float],
        tbt_slo: Optional[float],
        e2e_slo: Optional[float],
        prompt_len: int,
        prefix_len: int,
        ts: float,
        predicted_ttft_ms: Optional[float] = None,
        predicted_tbt_ms: Optional[float] = None,
    ) -> RequestRecord:
        """Create the record for a request the admission gate just admitted."""
        record = RequestRecord(
            rid=rid,
            ttft_slo=ttft_slo,
            tbt_slo=tbt_slo,
            e2e_slo=e2e_slo,
            prompt_len=max(0, prompt_len),
            prefix_len=max(0, prefix_len),
            admitted_ts=ts,
            state=RequestState.QUEUED,
            predicted_ttft_ms=predicted_ttft_ms,
            predicted_tbt_ms=predicted_tbt_ms,
        )
        self._active[rid] = record
        return record

    def on_state_change(self, rid: str, new_state: RequestState, ts: float) -> None:
        """Move a request between lifecycle states (e.g. QUEUED → RUNNING)."""
        record = self._active.get(rid)
        if record is None:
            return
        record.state = new_state

    def on_step(
        self,
        rid: str,
        *,
        decoded_tokens: int,
        kv_len: int,
        ts: float,
        first_token_ts: Optional[float] = None,
    ) -> None:
        """Per-step progress update for a running request.

        Advances `last_token_ts` / `decoded_tokens` / `kv_len`. Stamps
        `first_token_ts` the first time the request has produced a token:
        from `first_token_ts` when the caller supplies the scheduler's
        accurate prefill-finished timestamp, else from `ts` (tick-granular,
        ±tick_interval — see managers/halo/CLAUDE.md "Known limitations").
        """
        record = self._active.get(rid)
        if record is None:
            return
        if record.state is RequestState.QUEUED:
            record.state = RequestState.RUNNING

        decoded_tokens = max(0, decoded_tokens)
        if decoded_tokens > record.decoded_tokens:
            if record.first_token_ts is None and decoded_tokens >= 1:
                record.first_token_ts = first_token_ts or ts
            record.last_token_ts = ts
            record.decoded_tokens = decoded_tokens
        record.kv_len = max(0, kv_len)

    def on_finished(self, rid: str, ts: float) -> Optional[RequestRecord]:
        """Finalize a request: stamp finish time, compute solo baseline +
        e2e slowdown, and move the record to the recent ring.
        """
        record = self._active.pop(rid, None)
        if record is None:
            return None
        record.state = RequestState.FINISHED
        record.finished_ts = ts

        # Solo-run baseline = solo prefill + solo decode. The decode term
        # integrates the per-step cost as KV grows; we approximate the
        # integral with the mean KV span (≈ prompt_len + decoded/2). The
        # solo predictor is slated for a dedicated rework (see CLAUDE.md),
        # so this stays a documented approximation.
        solo_prefill = self.prefill_solo_ms(record.prompt_len, record.prefix_len)
        solo_decode = 0.0
        if record.decoded_tokens > 0:
            mean_kv = record.prompt_len + record.decoded_tokens / 2.0
            solo_decode = record.decoded_tokens * self.decode_step_ms([int(mean_kv)])
        record.solo_e2e_ms = solo_prefill + solo_decode

        e2e_ms = record.e2e_ms
        if e2e_ms is not None and record.solo_e2e_ms is not None:
            record.e2e_slowdown = e2e_ms / max(record.solo_e2e_ms, MIN_SOLO_MS)

        self._retire(record)
        return record

    def on_rejected(self, rid: str, reason: str) -> None:
        """Record an admission-gate rejection (request never entered the queue)."""
        record = RequestRecord(rid=rid, state=RequestState.REJECTED)
        record.reject_reason = reason
        self._retire(record)

    def _retire(self, record: RequestRecord) -> None:
        """Move a terminal record into the bounded recent ring."""
        if self._retain_finished <= 0:
            return
        self._recent[record.rid] = record
        while len(self._recent) > self._retain_finished:
            self._recent.popitem(last=False)

    # ------------------------------------------------------------------
    # Read-only consumers.
    # ------------------------------------------------------------------

    def get(self, rid: str) -> Optional[RequestRecord]:
        """Look up a record — active first, then the recent ring."""
        return self._active.get(rid) or self._recent.get(rid)

    def active_count(self) -> int:
        return len(self._active)

    def snapshot(self, kv_usage_ratio: float = 0.0) -> ServerStateSnapshot:
        """Current server-state view for the admission gate / policy."""
        queued: List[RequestRecord] = []
        running: List[RequestRecord] = []
        for record in self._active.values():
            if record.state is RequestState.QUEUED:
                queued.append(record)
            elif record.state is RequestState.RUNNING:
                running.append(record)
        return ServerStateSnapshot(
            queued=queued, running=running, kv_usage_ratio=kv_usage_ratio
        )

    def recent_finished(self) -> List[RequestRecord]:
        return list(self._recent.values())

    # ------------------------------------------------------------------
    # Solo cost-model primitives (salvaged from the legacy SlowdownTracker).
    # ------------------------------------------------------------------

    def prefill_solo_ms(self, prompt_len: int, prefix_len: int) -> float:
        """Solo prefill time (ms) for one request. `n_new = prompt_len -
        prefix_len` excludes the radix-cache hit from prefill compute.
        Returns 0.0 when no cost model is loaded.
        """
        n_new = max(0, prompt_len - prefix_len)
        r = max(0, prefix_len)
        if self.step_cost is not None:
            return self.step_cost.estimate_solo_prefill_total_ms(n_new, r)
        if self.prefill_cost is not None:
            return self.prefill_cost.estimate_ms(prompt_len, prefix_len)
        return 0.0

    def decode_step_ms(self, kv_lens: List[int]) -> float:
        """One DECODE step's wall-clock (ms) for a batch with the given
        per-request KV spans. Empty list ⇒ 0.0. Returns 0.0 when no cost
        model is loaded.
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
