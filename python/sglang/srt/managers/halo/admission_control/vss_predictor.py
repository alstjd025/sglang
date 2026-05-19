"""VSS predictor — the memoryless virtual-server-slowdown kernel.

Used by `VssPolicy` (see policy_vss.py). The batch-slowdown machinery here
— augmented step times, the chunk-bounded EXTEND step — is also intended
for reuse by the Phase-3 reactive policy's interference term.

Core idea (memoryless VSS):
    Reject a new arrival if admitting it would push too many *currently
    in-flight requests* over their SLO. Each in-flight request's predicted
    slowdown is the **virtual server slowdown (VSS)** it would experience:

        predicted_VSS_i = S_phase(i)⁺ / solo_step_i

    where S_phase(i)⁺ is the Halo Step Cost Model step time of the
    *current batch plus the new arrival* and solo_step_i is request i's
    *current* solo step time. Memoryless: a pure function of the current
    batch composition and the arrival's shape — no lifetime history.

`RequestSlowdownAdmissionPredictor` returns {rid: predicted_VSS} for every
in-flight request; `decide_admission` applies the violation-ratio threshold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.managers.halo.admission_control.cost_model import HaloStepCostModel

# Reject reason — shared by the predictor + VssPolicy without importing it.
REASON_OK = "OK"
REASON_HALO_ADMISSION_PREDICTED = "HALO_ADMISSION_PREDICTED"


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ActiveCallInfo:
    """A single in-flight request. A snapshot of *current* state only —
    the predictor derives the solo step time from ``prompt_len /
    prefix_len`` (prefill phase) or ``kv_len_now`` (decode phase)."""

    rid: str
    is_prefill: bool  # True while still doing prefill (uncached new tokens)
    prompt_len: int  # total prompt tokens (n + r)
    prefix_len: int  # cached prefix (r)
    decoded_tokens: int  # tokens already produced
    kv_len_now: int  # current KV span (= prompt_len + decoded_tokens)


@dataclass(frozen=True)
class BatchSnapshot:
    """The in-flight requests the predictor scores — `active_calls` is the
    running batch. `slo` is the common SLO used for every scored request
    when the per-request SLO map omits one."""

    slo: float
    active_calls: Tuple[ActiveCallInfo, ...]


@dataclass(frozen=True)
class ArrivalInput:
    """The new arrival being scored — only its first call's shape is needed."""

    slo: float
    first_call_input_len: int
    first_call_prefix_len: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Decision result
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AdmissionDecisionResult:
    """Output of decide_admission(...)."""

    admit: bool
    reason: str  # REASON_OK or REASON_HALO_ADMISSION_PREDICTED
    violation_ratio: float
    violation_count: int
    scored_total: int  # count of scored in-flight requests
    threshold: float
    predicted_slowdowns: Dict[str, float]  # {rid: predicted VSS}


# Floor on a solo step time — guards the predicted_VSS division.
_MIN_SOLO_STEP_MS = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Predictor base
# ─────────────────────────────────────────────────────────────────────────────
class AdmissionPredictor(ABC):
    """Predicts each in-flight request's slowdown *if* the new arrival is
    admitted."""

    def __init__(self, cost_model: HaloStepCostModel) -> None:
        self.cost_model = cost_model

    @abstractmethod
    def predict(
        self,
        active_batch: BatchSnapshot,
        arrival: ArrivalInput,
        chunked_prefill_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Returns {rid: predicted_slowdown} for every in-flight request."""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _split_current_batch_features(
    active_calls: Sequence[ActiveCallInfo],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Returns (current_prefill, current_decode) — the (n, r) list for
    prefill-phase calls and the KV-length list for decode-phase calls.
    Composes the cost-model inputs for the two separate step types
    (sample env has mixed_chunk=OFF)."""
    current_prefill: List[Tuple[int, int]] = []
    current_decode: List[int] = []
    for call in active_calls:
        if call.is_prefill:
            n = max(0, call.prompt_len - call.prefix_len)
            current_prefill.append((n, call.prefix_len))
        else:
            current_decode.append(call.kv_len_now)
    return current_prefill, current_decode


def _bounded_extend_step(
    prefill_infos: List[Tuple[int, int]],
    chunked_prefill_size: Optional[int],
) -> List[Tuple[int, int]]:
    """Cap a list of (n, r) prefill entries to ONE realistic chunked-prefill
    EXTEND step.

    결함 A (2026-05-18): with chunked prefill ON, a single forward pass
    processes at most `chunked_prefill_size` new tokens *total*. Feeding the
    whole un-chunked waiting backlog into estimate_step_ms made the `Σnᵢ²`
    prefill term blow up. This caps the modelled step: each entry contributes
    at most the remaining budget; entries past the budget are not in this step.

    `chunked_prefill_size` None or <= 0 (chunked prefill disabled) → no cap.
    """
    if not chunked_prefill_size or chunked_prefill_size <= 0:
        return list(prefill_infos)
    out: List[Tuple[int, int]] = []
    remaining = chunked_prefill_size
    for n, r in prefill_infos:
        if remaining <= 0:
            break
        take = min(max(0, n), remaining)
        out.append((take, r))
        remaining -= take
    return out


def _augmented_step_times(
    cost_model: HaloStepCostModel,
    active_calls: Sequence[ActiveCallInfo],
    arrival: ArrivalInput,
    chunked_prefill_size: Optional[int] = None,
) -> Tuple[float, float]:
    """Return (S_extend⁺, S_decode⁺) — the EXTEND-step and DECODE-step
    wall-clock the *current batch plus the new arrival* would cost. These
    are the shared numerators of every in-flight request's predicted VSS;
    each request divides by its own solo step time.

    The arrival is added to BOTH halves: it prefills now (EXTEND) and, once
    prefill finishes, joins decode (DECODE). The EXTEND batch is bounded to
    one chunked-prefill step (_bounded_extend_step / 결함 A); DECODE is not
    bounded (a decode step genuinely processes every running request).
    """
    current_prefill, current_decode = _split_current_batch_features(active_calls)
    n_new = max(0, arrival.first_call_input_len - arrival.first_call_prefix_len)
    r_new = max(0, arrival.first_call_prefix_len)
    extend_batch = _bounded_extend_step(
        current_prefill + [(n_new, r_new)], chunked_prefill_size
    )
    s_extend = cost_model.estimate_step_ms(extend_batch, [])
    s_decode = cost_model.estimate_step_ms(
        [], current_decode + [max(0, arrival.first_call_input_len)]
    )
    return s_extend, s_decode


# ─────────────────────────────────────────────────────────────────────────────
# Per-request memoryless VSS predictor
# ─────────────────────────────────────────────────────────────────────────────
class RequestSlowdownAdmissionPredictor(AdmissionPredictor):
    """Per-request memoryless VSS. Every in-flight request is scored on its
    own solo step time:

        S_extend⁺, S_decode⁺ = _augmented_step_times(...)
        for each in-flight call c:
            if c.is_prefill:
                solo_c = estimate_step_ms([(n_c, r_c)], [])      # chunk-bounded
                predicted_VSS_c = S_extend⁺ / solo_c
            else:
                solo_c = estimate_step_ms([], [c.kv_len_now])
                predicted_VSS_c = S_decode⁺ / solo_c

    Returns {rid: predicted_VSS}.
    """

    def predict(
        self,
        active_batch: BatchSnapshot,
        arrival: ArrivalInput,
        chunked_prefill_size: Optional[int] = None,
    ) -> Dict[str, float]:
        s_extend, s_decode = _augmented_step_times(
            self.cost_model, active_batch.active_calls, arrival, chunked_prefill_size
        )
        predictions: Dict[str, float] = {}
        for call in active_batch.active_calls:
            if call.is_prefill:
                n = max(0, call.prompt_len - call.prefix_len)
                solo = self.cost_model.estimate_step_ms(
                    _bounded_extend_step(
                        [(n, max(0, call.prefix_len))], chunked_prefill_size
                    ),
                    [],
                )
                predictions[call.rid] = s_extend / max(solo, _MIN_SOLO_STEP_MS)
            else:
                solo = self.cost_model.estimate_step_ms([], [max(0, call.kv_len_now)])
                predictions[call.rid] = s_decode / max(solo, _MIN_SOLO_STEP_MS)
        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Decision wrapper
# ─────────────────────────────────────────────────────────────────────────────
def decide_admission(
    predicted_slowdowns: Dict[str, float],
    slos: Dict[str, float],
    threshold: float,
) -> AdmissionDecisionResult:
    """Apply the violation-ratio threshold to the predictor's output.

    Args:
        predicted_slowdowns: {rid: predicted slowdown after admitting the
            new arrival}.
        slos: {rid: SLO}. A missing rid is treated as always satisfied
            (its prediction is compared against itself).
        threshold: fraction (0..1). violation_ratio > threshold → REJECT.
    """
    total = len(predicted_slowdowns)
    if total == 0:
        return AdmissionDecisionResult(
            admit=True,
            reason=REASON_OK,
            violation_ratio=0.0,
            violation_count=0,
            scored_total=0,
            threshold=threshold,
            predicted_slowdowns={},
        )
    violations = sum(
        1 for uid, pred in predicted_slowdowns.items() if pred > slos.get(uid, pred)
    )
    violation_ratio = violations / total
    admit = violation_ratio <= threshold
    return AdmissionDecisionResult(
        admit=admit,
        reason=REASON_OK if admit else REASON_HALO_ADMISSION_PREDICTED,
        violation_ratio=violation_ratio,
        violation_count=violations,
        scored_total=total,
        threshold=threshold,
        predicted_slowdowns=dict(predicted_slowdowns),
    )


__all__ = [
    "REASON_OK",
    "REASON_HALO_ADMISSION_PREDICTED",
    "ActiveCallInfo",
    "BatchSnapshot",
    "ArrivalInput",
    "AdmissionDecisionResult",
    "AdmissionPredictor",
    "RequestSlowdownAdmissionPredictor",
    "decide_admission",
]
