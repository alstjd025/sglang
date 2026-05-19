"""Predictive admission decision for Halo Phase 2.

Design + rationale: ms_dev/halo_dev/admission_design.md.

Core idea (memoryless VSS — 2026-05-16):
    Reject a new arrival if admitting it would push too many *currently
    active units* over their SLO. Each active unit's predicted slowdown is
    the **virtual server slowdown (VSS)** it would experience:

        predicted_VSS_i = S_phase(i)⁺ / solo_step_i

    where S_phase(i)⁺ is the Halo Step Cost Model step time of the
    *current batch plus the new arrival* (the shared server-congestion
    numerator) and solo_step_i is unit i's *current* solo step time. This
    is memoryless: a pure function of the current batch composition and
    the arrival's first-call shape. It deliberately does NOT use a unit's
    lifetime virtual job slowdown (VJS) — VJS is correct for SLO
    *measurement* but has an integrator built in (it accumulates over the
    whole job life) and lags badly as a *control* signal. When a job
    finishes, the batch shrinks and every predicted_VSS_i drops on the
    next step — the fast, self-regulating feedback the cumulative-VJS
    gate lacked.

Two admission scopes — the difference is the *unit of the decision*:

    - JobSlowdownAdmissionPredictor   (mode "job"): the unit is a job.
      predicted_VSS_i uses the job's own current phase (prefill → EXTEND
      step, decode → DECODE step) and its own self-batched solo step.
      The decision is made once per job, at its first request.

    - RequestSlowdownAdmissionPredictor (mode "request"): the unit is a
      single request. Identical VSS math + cost model, but NO job
      aggregation — every active call is scored on its own, and the
      decision runs on *every* request (so a mid-chain request can be
      rejected). A deliberately naive baseline used to show request-scoped
      admission is worse than job-scoped; the *only* difference from mode
      "job" is the scoring unit.

    - LookaheadAdmissionPredictor (mode "level2"): 1-second slice forward
      simulation that used declared DAG / remaining lengths.
      **DEPRECATED 2026-05-15**: not selected at runtime (controller falls
      back to "job" with a WARN). Code retained for reference only.

decide_admission(...) glues a predictor's output to the violation-ratio
threshold (D3 = 0.2) and returns AdmissionDecisionResult — the controller
logs that and either passes through or raises HaloRejectError.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.managers.halo.admission_control.cost_model import HaloStepCostModel

# ─────────────────────────────────────────────────────────────────────────────
# Reject reason — kept in this module so the predictor + controller share it
# without importing from controller (avoids a circular import).
# ─────────────────────────────────────────────────────────────────────────────
REASON_OK = "OK"
REASON_HALO_ADMISSION_PREDICTED = "HALO_ADMISSION_PREDICTED"


# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ActiveCallInfo:
    """A single in-flight LLM call inside an active job.

    A per-request snapshot of *current* state only: the memoryless VSS
    predictors derive each call's solo step time from ``prompt_len /
    prefix_len`` (prefill phase) or ``kv_len_now`` (decode phase). No
    elapsed/solo history is consulted.

    ``elapsed_actual_ms`` is retained solely for the deprecated
    LookaheadAdmissionPredictor; the production predictors ignore it.
    """

    rid: str
    is_prefill: bool  # True while still doing prefill (uncached new tokens)
    prompt_len: int  # total prompt tokens (n + r)
    prefix_len: int  # cached prefix at admit time (r)
    decoded_tokens: int  # tokens already produced
    kv_len_now: int  # current KV span (= prompt_len + decoded_tokens)
    # ---- LEGACY — LookaheadAdmissionPredictor only (DEPRECATED 2026-05-15) ----
    elapsed_actual_ms: float = 0.0  # wall-clock since this call started


@dataclass(frozen=True)
class JobLookaheadInput:
    """Per-active-job snapshot consumed by AdmissionPredictor.

    The memoryless VSS predictors use only ``active_calls`` (each call's
    current prompt/prefix/KV) and ``slo``. No lifetime VJS, no elapsed
    history.

    The fields after ``active_calls`` are *legacy*: the production
    predictors ignore them. ``current_actual_elapsed_ms`` /
    ``current_solo_elapsed_ms`` and the declared structure are kept solely
    for the deprecated LookaheadAdmissionPredictor.
    """

    job_id: str
    slo: float
    # Current observed state (from RequestExecutionInfo).
    active_calls: Tuple[ActiveCallInfo, ...]
    # ---- LEGACY — LookaheadAdmissionPredictor only (DEPRECATED 2026-05-15) ----
    current_actual_elapsed_ms: float = 0.0
    current_solo_elapsed_ms: float = 0.0
    remaining_calls: Optional[int] = None
    remaining_stage_sequence: Optional[Tuple[str, ...]] = None
    expected_input_lens: Optional[Tuple[int, ...]] = None
    expected_output_lens: Optional[Tuple[int, ...]] = None
    expected_cached_prefix_lens: Optional[Tuple[int, ...]] = None


@dataclass(frozen=True)
class NewJobInput:
    """New job arrival — what the predictor needs to score admission.

    The first call's input_len / prefix_len is always known at admit time
    (it's exactly what triggered the admission decision). Everything else
    is declared via register_program and is Optional.

    **As of 2026-05-15** SnapshotAdmissionPredictor uses ONLY
    ``first_call_input_len`` and ``first_call_prefix_len``. The remaining
    fields are kept for the deprecated LookaheadAdmissionPredictor.
    """

    job_id: str
    slo: float
    # First call (always known — this is what's being admitted)
    first_call_input_len: int
    first_call_prefix_len: int = 0
    first_call_expected_output_len: int = 0
    # ---- LEGACY (not used by SnapshotAdmissionPredictor since 2026-05-15) ----
    total_calls: Optional[int] = None
    stage_sequence: Optional[Tuple[str, ...]] = None
    expected_input_lens: Optional[Tuple[int, ...]] = None
    expected_output_lens: Optional[Tuple[int, ...]] = None
    expected_cached_prefix_lens: Optional[Tuple[int, ...]] = None


# ─────────────────────────────────────────────────────────────────────────────
# Decision result
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AdmissionDecisionResult:
    """Output of decide_admission(...). Carries enough info for the JSONL
    decision log + the operator's intuition."""

    admit: bool
    reason: str  # REASON_OK or REASON_HALO_ADMISSION_PREDICTED
    violation_ratio: float  # violations / active_units_total
    violation_count: int
    active_jobs_total: int  # count of scored units (jobs or requests)
    threshold: float
    # Predicted slowdown per scored unit after admitting the new arrival.
    # Keys are job_id in mode "job" and rid in mode "request" — disambiguate
    # via the `mode` field below.
    predicted_slowdowns: Dict[str, float]
    # Optional / mode-specific diagnostic fields:
    horizon_sec: Optional[float] = None  # level2 (deprecated) only
    mode: str = ""  # "job" | "request" | "level2"


# ─────────────────────────────────────────────────────────────────────────────
# Predictor base
# ─────────────────────────────────────────────────────────────────────────────
class AdmissionPredictor(ABC):
    """Predicts each active unit's slowdown *if* the new arrival is
    admitted. Concrete subclasses differ in the *unit* of the prediction:

    - JobSlowdownAdmissionPredictor (mode "job"): unit = job. Stretches
      each job's current virtual job slowdown by its own phase's step-time
      ratio. Returns {job_id: predicted_virtual_job_slowdown}.
    - RequestSlowdownAdmissionPredictor (mode "request"): unit = request.
      Same stretch math, no job aggregation. Returns
      {rid: predicted_request_slowdown}.
    - LookaheadAdmissionPredictor (mode "level2", DEPRECATED): 1-second
      slice forward simulation. Not on the production path.
    """

    mode_name: str = "abstract"

    def __init__(self, cost_model: HaloStepCostModel) -> None:
        self.cost_model = cost_model

    @abstractmethod
    def predict(
        self,
        active_jobs: Sequence[JobLookaheadInput],
        new_job: NewJobInput,
        chunked_prefill_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Returns {unit_id: predicted_slowdown} for every active unit.

        The key is job_id (mode "job") or rid (mode "request"). The new
        arrival is not included — only existing active units are scored
        (the new arrival's own SLO is the caller's concern, not the
        admission gate's).
        """


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared by predictors
# ─────────────────────────────────────────────────────────────────────────────
def _split_current_batch_features(
    active_jobs: Sequence[JobLookaheadInput],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Returns (current_prefill, current_decode) — the (n, r) list for
    prefill-phase active calls and the KV-length list for decode-phase
    active calls. Used to compose the cost-model inputs for the two
    *separate* step types (sample env has mixed_chunk=OFF)."""
    current_prefill: List[Tuple[int, int]] = []
    current_decode: List[int] = []
    for job in active_jobs:
        for call in job.active_calls:
            if call.is_prefill:
                n = max(0, call.prompt_len - call.prefix_len)
                current_prefill.append((n, call.prefix_len))
            else:
                current_decode.append(call.kv_len_now)
    return current_prefill, current_decode


# ─────────────────────────────────────────────────────────────────────────────
# LEGACY HELPERS — kept for the deprecated LookaheadAdmissionPredictor.
# Not used by SnapshotAdmissionPredictor since 2026-05-15.
# ─────────────────────────────────────────────────────────────────────────────
def _solo_call_ms(cost_model: HaloStepCostModel, n: int, r: int, out_len: int) -> float:
    """LEGACY (2026-05-15: declared lengths no longer used by admission).
    Total solo wall-clock for one call.
    """
    n_new = max(0, n - r)
    prefill_ms = cost_model.estimate_solo_prefill_total_ms(n_new, r)
    avg_kv = n + max(0, out_len // 2)
    decode_step_ms = cost_model.estimate_solo_tbt_ms(avg_kv)
    return prefill_ms + decode_step_ms * max(0, out_len)


def _estimate_remaining_solo_ms(
    cost_model: HaloStepCostModel,
    job: JobLookaheadInput,
) -> Optional[float]:
    """LEGACY (2026-05-15: declared lengths no longer used by admission).
    Sum of per-call solo time across the job's declared remaining calls.
    """
    seq = job.remaining_stage_sequence
    if not seq:
        return None
    if not (job.expected_input_lens and job.expected_output_lens):
        return None
    n_remaining = len(seq)
    if len(job.expected_input_lens) < n_remaining:
        return None
    if len(job.expected_output_lens) < n_remaining:
        return None
    prefix_lens = job.expected_cached_prefix_lens or (0,) * n_remaining

    total = 0.0
    for i in range(n_remaining):
        total += _solo_call_ms(
            cost_model,
            int(job.expected_input_lens[i]),
            int(prefix_lens[i]) if i < len(prefix_lens) else 0,
            int(job.expected_output_lens[i]),
        )
    return total


# Floor on a solo step time — guards the predicted_VSS division.
_MIN_SOLO_STEP_MS = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Augmented step times — the shared VSS numerator for both predictors
# ─────────────────────────────────────────────────────────────────────────────
def _bounded_extend_step(
    prefill_infos: List[Tuple[int, int]],
    chunked_prefill_size: Optional[int],
) -> List[Tuple[int, int]]:
    """Cap a list of (n, r) prefill entries to ONE realistic chunked-prefill
    EXTEND step.

    결함 A (2026-05-18): with chunked prefill ON, a single forward pass
    processes at most `chunked_prefill_size` new tokens *total* (a long
    request is split across steps). Feeding the whole un-chunked waiting
    backlog into estimate_step_ms made the `Σnᵢ²` prefill term blow up —
    estimated EXTEND-step times of tens of seconds and VSS values in the
    100s–1000s. This caps the modelled step: each entry contributes at most
    the remaining budget, and entries past the budget are not in this step.

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
    active_jobs: Sequence[JobLookaheadInput],
    new_job: NewJobInput,
    chunked_prefill_size: Optional[int] = None,
) -> Tuple[float, float]:
    """Return (S_extend⁺, S_decode⁺) — the EXTEND-step and DECODE-step
    wall-clock the *current batch plus the new arrival* would cost:

        S_extend⁺ = estimate_step_ms(chunk-bounded(current_prefill ∪ new), [])
        S_decode⁺ = estimate_step_ms([], current_decode ∪ {new prompt KV})

    These are the shared numerators of every active unit's predicted VSS;
    each unit divides by its own solo step time. The new arrival is added
    to BOTH halves: it prefills now (EXTEND) and, once prefill finishes,
    joins decode (DECODE) — so a decode-heavy batch still sees the
    arrival's eventual load. This is not DAG lookahead; it is the
    inevitable consequence of admitting the request.

    The EXTEND batch is bounded to one chunked-prefill step (see
    _bounded_extend_step / 결함 A). DECODE has no such chunking — a decode
    step genuinely processes every running request — so it is not bounded.
    """
    current_prefill, current_decode = _split_current_batch_features(active_jobs)
    n_new = max(0, new_job.first_call_input_len - new_job.first_call_prefix_len)
    r_new = max(0, new_job.first_call_prefix_len)
    extend_batch = _bounded_extend_step(
        current_prefill + [(n_new, r_new)], chunked_prefill_size
    )
    s_extend = cost_model.estimate_step_ms(extend_batch, [])
    s_decode = cost_model.estimate_step_ms(
        [], current_decode + [max(0, new_job.first_call_input_len)]
    )
    return s_extend, s_decode


# ─────────────────────────────────────────────────────────────────────────────
# mode "job" — Per-job memoryless VSS (current production design)
# ─────────────────────────────────────────────────────────────────────────────
class JobSlowdownAdmissionPredictor(AdmissionPredictor):
    """Job-scoped admission — admits/rejects using only *current* batch
    state plus the arriving request's first-call shape.

    Algorithm (memoryless VSS, decided 2026-05-16):

        S_extend⁺, S_decode⁺ = _augmented_step_times(...)

        for each active job i:
            if i has any active prefill call:
                solo_i = estimate_step_ms(i's prefill (n,r) calls, [])
                predicted_VSS_i = S_extend⁺ / solo_i
            else:
                solo_i = estimate_step_ms([], i's decode-call KVs)
                predicted_VSS_i = S_decode⁺ / solo_i

    No lifetime VJS, no elapsed history, no DAG/remaining-call lookahead.
    The numerator is the shared post-admission server congestion; the
    denominator is the job running its own calls *alone*. When a job
    finishes the batch shrinks and S⁺ drops on the next step — the gate
    re-opens immediately. See ms_dev/halo_dev/admission_design.md §4.
    """

    mode_name = "job"

    def predict(
        self,
        active_jobs: Sequence[JobLookaheadInput],
        new_job: NewJobInput,
        chunked_prefill_size: Optional[int] = None,
    ) -> Dict[str, float]:
        s_extend, s_decode = _augmented_step_times(
            self.cost_model, active_jobs, new_job, chunked_prefill_size
        )

        # ── per-job: predicted VSS = augmented step / job's own solo step ──
        predictions: Dict[str, float] = {}
        for job in active_jobs:
            if not job.active_calls:
                continue
            prefill_infos = [
                (max(0, c.prompt_len - c.prefix_len), max(0, c.prefix_len))
                for c in job.active_calls
                if c.is_prefill
            ]
            if prefill_infos:
                # Prefill phase (prefill calls win if a job has both). The
                # job's own solo EXTEND step is chunk-bounded too (결함 A).
                solo = self.cost_model.estimate_step_ms(
                    _bounded_extend_step(prefill_infos, chunked_prefill_size), []
                )
                predictions[job.job_id] = s_extend / max(solo, _MIN_SOLO_STEP_MS)
            else:
                decode_kvs = [max(0, c.kv_len_now) for c in job.active_calls]
                solo = self.cost_model.estimate_step_ms([], decode_kvs)
                predictions[job.job_id] = s_decode / max(solo, _MIN_SOLO_STEP_MS)
        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# mode "request" — Per-request memoryless VSS (request-scoped baseline)
# ─────────────────────────────────────────────────────────────────────────────
class RequestSlowdownAdmissionPredictor(AdmissionPredictor):
    """Request-scoped admission baseline — the deliberately naive arm used
    to demonstrate that request-scoped admission performs worse than the
    job-scoped gate.

    The VSS math and cost model are IDENTICAL to
    JobSlowdownAdmissionPredictor; the only differences are:

      - **no job aggregation** — every in-flight request (= active call)
        across every job is scored on its own. The unit is the request.
      - the controller runs this on *every* request (not just a job's
        first), so a mid-chain request can be rejected.

    Per-request predicted VSS:

        S_extend⁺, S_decode⁺ = _augmented_step_times(...)
        for each active call c (flattened across all jobs):
            if c.is_prefill:
                solo_c = estimate_step_ms([(n_c, r_c)], [])
                predicted_VSS_c = S_extend⁺ / solo_c
            else:
                solo_c = estimate_step_ms([], [c.kv_len_now])
                predicted_VSS_c = S_decode⁺ / solo_c

    Returns {rid: predicted_VSS}.
    """

    mode_name = "request"

    def predict(
        self,
        active_jobs: Sequence[JobLookaheadInput],
        new_job: NewJobInput,
        chunked_prefill_size: Optional[int] = None,
    ) -> Dict[str, float]:
        s_extend, s_decode = _augmented_step_times(
            self.cost_model, active_jobs, new_job, chunked_prefill_size
        )

        # ── per-request: each active call scored on its own solo step ────
        predictions: Dict[str, float] = {}
        for job in active_jobs:
            for call in job.active_calls:
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
                    solo = self.cost_model.estimate_step_ms(
                        [], [max(0, call.kv_len_now)]
                    )
                    predictions[call.rid] = s_decode / max(solo, _MIN_SOLO_STEP_MS)
        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Decision wrapper
# ─────────────────────────────────────────────────────────────────────────────
def decide_admission(
    predicted_slowdowns: Dict[str, float],
    slos: Dict[str, float],
    threshold: float,
    mode: str = "",
    horizon_sec: Optional[float] = None,
) -> AdmissionDecisionResult:
    """Apply the violation-ratio threshold to a predictor's output.

    Args:
        predicted_slowdowns: {unit_id: predicted slowdown after admitting
            the new arrival}. unit_id is job_id (mode "job") or rid
            (mode "request").
        slos:        {unit_id: SLO}, keyed the same way. Falls back to the
                     prediction's value if missing (missing SLO is treated
                     as "always satisfied").
        threshold:   D3 — fraction (0..1). If violation_ratio > threshold,
                     decision is REJECT.
        mode:        for diagnostic.
        horizon_sec: level2 (deprecated) only — for diagnostic.

    Returns:
        AdmissionDecisionResult with all fields populated.
    """
    predictions = predicted_slowdowns
    total = len(predictions)
    if total == 0:
        return AdmissionDecisionResult(
            admit=True,
            reason=REASON_OK,
            violation_ratio=0.0,
            violation_count=0,
            active_jobs_total=0,
            threshold=threshold,
            predicted_slowdowns={},
            horizon_sec=horizon_sec,
            mode=mode,
        )
    violations = 0
    for uid, pred in predictions.items():
        slo = slos.get(uid, pred)
        if pred > slo:
            violations += 1
    violation_ratio = violations / total
    if violation_ratio > threshold:
        return AdmissionDecisionResult(
            admit=False,
            reason=REASON_HALO_ADMISSION_PREDICTED,
            violation_ratio=violation_ratio,
            violation_count=violations,
            active_jobs_total=total,
            threshold=threshold,
            predicted_slowdowns=dict(predictions),
            horizon_sec=horizon_sec,
            mode=mode,
        )
    return AdmissionDecisionResult(
        admit=True,
        reason=REASON_OK,
        violation_ratio=violation_ratio,
        violation_count=violations,
        active_jobs_total=total,
        threshold=threshold,
        predicted_slowdowns=dict(predictions),
        horizon_sec=horizon_sec,
        mode=mode,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — SLO-driven lookahead
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class _SimCall:
    """Mutable simulation state for one in-flight call."""

    rid: str
    is_prefill: bool
    prompt_len: int  # n + r (unchanged through decode)
    prefix_len: int  # r at start of call (unchanged)
    decoded_tokens: float  # fractional during the simulation
    expected_output_len: int  # 0 means unknown — call never auto-terminates
    elapsed_actual_ms: float  # wall-clock since this call started in sim


@dataclass
class _SimJob:
    """Mutable simulation state for one job (current + future calls)."""

    job_id: str
    slo: float
    elapsed_actual_ms: float  # cumulative across calls (sim + observed)
    solo_ms: float  # cumulative solo across calls (sim + observed)
    active_calls: List[_SimCall]
    pending_calls: List[_SimCall]


class LookaheadAdmissionPredictor(AdmissionPredictor):
    """**DEPRECATED 2026-05-15** — not used by the controller.

    Original 1-second slice forward simulation that relied on each job's
    *declared* DAG (remaining_stage_sequence, expected_input_lens,
    expected_output_lens) to project future batch composition.

    The 2026-05-15 design decision is to use *current state only* — no
    DAG, no remaining-call lookahead. ``HaloController._build_admission_predictor``
    now treats ``admission_mode='level2'`` as a soft fallback to
    JobSlowdownAdmissionPredictor with a WARN at startup. This class is kept
    in the source tree for reference (e.g. to revive once
    `expected_output_lens` becomes reliable enough to project the cumulative
    effect of new arrivals on existing jobs' slowdown), but it is not on
    the production admission path.
    """

    mode_name = "level2"
    SLICE_SEC: float = 1.0
    MAX_HORIZON_SEC: float = 600.0  # safety cap (10 minutes)
    SLICE_STEP_MS_FLOOR: float = 1.0  # protect against degenerate empty batch

    def __init__(
        self,
        cost_model: HaloStepCostModel,
        horizon_sec: float = 0.0,
    ) -> None:
        super().__init__(cost_model)
        self.horizon_sec = horizon_sec

    def predict(
        self,
        active_jobs: Sequence[JobLookaheadInput],
        new_job: NewJobInput,
        chunked_prefill_size: Optional[int] = None,  # unused (DEPRECATED)
    ) -> Dict[str, float]:
        # Build mutable simulation state from the immutable inputs.
        sim_jobs = self._build_sim_jobs(active_jobs)
        new_sim = self._build_new_sim_job(new_job)
        all_sim = sim_jobs + ([new_sim] if new_sim else [])
        if not all_sim:
            return {}

        horizon = self._resolve_horizon(active_jobs)
        slice_ms = self.SLICE_SEC * 1000.0
        t = 0.0
        # Step the simulation in 1-second increments.
        while t < horizon and self._any_work_remaining(all_sim):
            prefill_infos, decode_infos = self._collect_batch_features(all_sim)
            step_ms = self.cost_model.estimate_step_ms(prefill_infos, decode_infos)
            step_ms = max(step_ms, self.SLICE_STEP_MS_FLOOR)
            steps_in_slice = slice_ms / step_ms

            for job in all_sim:
                if not job.active_calls and not job.pending_calls:
                    continue
                job.elapsed_actual_ms += slice_ms
                self._advance_calls(job, steps_in_slice, slice_ms)
            t += self.SLICE_SEC

        # Compute predicted final slowdown_max per active job. New job is
        # excluded — admission protects existing jobs only.
        predictions: Dict[str, float] = {}
        for sim in sim_jobs:
            if sim.solo_ms > 0:
                predictions[sim.job_id] = sim.elapsed_actual_ms / sim.solo_ms
            else:
                predictions[sim.job_id] = sim.slo
        return predictions

    # ── horizon + state construction ──────────────────────────────────────
    def _resolve_horizon(self, active_jobs: Sequence[JobLookaheadInput]) -> float:
        if self.horizon_sec > 0:
            return min(self.horizon_sec, self.MAX_HORIZON_SEC)
        # SLO-driven: when would this job hit slowdown == SLO?
        best = 0.0
        for job in active_jobs:
            remaining_solo = _estimate_remaining_solo_ms(self.cost_model, job)
            if remaining_solo is None:
                # Insufficient declared info — use the current observed
                # elapsed (degenerate but not zero).
                expected_total_solo = max(job.current_solo_elapsed_ms, 1.0)
            else:
                expected_total_solo = job.current_solo_elapsed_ms + remaining_solo
            target_ms = expected_total_solo * job.slo
            best = max(best, target_ms / 1000.0)
        return min(max(best, self.SLICE_SEC), self.MAX_HORIZON_SEC)

    def _build_sim_jobs(
        self, active_jobs: Sequence[JobLookaheadInput]
    ) -> List[_SimJob]:
        out: List[_SimJob] = []
        for j in active_jobs:
            active_calls = [
                _SimCall(
                    rid=c.rid,
                    is_prefill=c.is_prefill,
                    prompt_len=c.prompt_len,
                    prefix_len=c.prefix_len,
                    decoded_tokens=float(c.decoded_tokens),
                    expected_output_len=self._lookup_expected_output_for_call(j, c),
                    elapsed_actual_ms=c.elapsed_actual_ms,
                )
                for c in j.active_calls
            ]
            pending = self._build_pending_calls(j)
            out.append(
                _SimJob(
                    job_id=j.job_id,
                    slo=j.slo,
                    elapsed_actual_ms=j.current_actual_elapsed_ms,
                    solo_ms=j.current_solo_elapsed_ms,
                    active_calls=active_calls,
                    pending_calls=pending,
                )
            )
        return out

    def _build_new_sim_job(self, new_job: NewJobInput) -> Optional[_SimJob]:
        # First-call info is required; without it the new job adds no load
        # to the simulation.
        if new_job.first_call_input_len <= 0:
            return None
        first_call = _SimCall(
            rid=f"new::{new_job.job_id}",
            is_prefill=True,
            prompt_len=new_job.first_call_input_len,
            prefix_len=new_job.first_call_prefix_len,
            decoded_tokens=0.0,
            expected_output_len=new_job.first_call_expected_output_len
            or self._derive_expected_output_for_new_first(new_job),
            elapsed_actual_ms=0.0,
        )
        pending = self._build_pending_calls_from_new(new_job)
        return _SimJob(
            job_id=new_job.job_id,
            slo=new_job.slo,
            elapsed_actual_ms=0.0,
            solo_ms=0.0,
            active_calls=[first_call],
            pending_calls=pending,
        )

    @staticmethod
    def _lookup_expected_output_for_call(
        job: JobLookaheadInput, call: ActiveCallInfo
    ) -> int:
        """Best-effort: use the FIRST expected output length when declared.
        Active calls don't carry their stage index, so this is the most
        defensible default."""
        if job.expected_output_lens:
            return int(job.expected_output_lens[0])
        return 0  # 0 ⇒ never auto-terminates within the horizon

    @staticmethod
    def _build_pending_calls(job: JobLookaheadInput) -> List[_SimCall]:
        if not job.remaining_stage_sequence:
            return []
        if not (job.expected_input_lens and job.expected_output_lens):
            return []
        n_remaining = len(job.remaining_stage_sequence)
        if len(job.expected_input_lens) < n_remaining:
            return []
        if len(job.expected_output_lens) < n_remaining:
            return []
        prefix_lens = job.expected_cached_prefix_lens or tuple(
            0 for _ in range(n_remaining)
        )
        return [
            _SimCall(
                rid=f"sim::{job.job_id}::{i}",
                is_prefill=True,
                prompt_len=int(job.expected_input_lens[i]),
                prefix_len=int(prefix_lens[i]) if i < len(prefix_lens) else 0,
                decoded_tokens=0.0,
                expected_output_len=int(job.expected_output_lens[i]),
                elapsed_actual_ms=0.0,
            )
            for i in range(n_remaining)
        ]

    @staticmethod
    def _build_pending_calls_from_new(new_job: NewJobInput) -> List[_SimCall]:
        if not (new_job.expected_input_lens and new_job.expected_output_lens):
            return []
        # Skip index 0 because it's the just-arrived (active) first call.
        n = len(new_job.expected_input_lens)
        if n <= 1:
            return []
        if len(new_job.expected_output_lens) < n:
            return []
        prefix_lens = new_job.expected_cached_prefix_lens or tuple(0 for _ in range(n))
        return [
            _SimCall(
                rid=f"new::{new_job.job_id}::{i}",
                is_prefill=True,
                prompt_len=int(new_job.expected_input_lens[i]),
                prefix_len=int(prefix_lens[i]) if i < len(prefix_lens) else 0,
                decoded_tokens=0.0,
                expected_output_len=int(new_job.expected_output_lens[i]),
                elapsed_actual_ms=0.0,
            )
            for i in range(1, n)
        ]

    @staticmethod
    def _derive_expected_output_for_new_first(new_job: NewJobInput) -> int:
        if new_job.expected_output_lens:
            return int(new_job.expected_output_lens[0])
        return 0

    # ── per-slice mechanics ───────────────────────────────────────────────
    def _collect_batch_features(
        self, sim_jobs: Sequence[_SimJob]
    ) -> Tuple[List[Tuple[int, int]], List[int]]:
        prefill_infos: List[Tuple[int, int]] = []
        decode_kvs: List[int] = []
        for job in sim_jobs:
            for call in job.active_calls:
                if call.is_prefill:
                    n_new = max(0, call.prompt_len - call.prefix_len)
                    prefill_infos.append((n_new, call.prefix_len))
                else:
                    decode_kvs.append(call.prompt_len + int(call.decoded_tokens))
        return prefill_infos, decode_kvs

    def _advance_calls(
        self,
        sim_job: _SimJob,
        steps_in_slice: float,
        slice_ms: float,
    ) -> None:
        finished: List[_SimCall] = []
        for call in sim_job.active_calls:
            call.elapsed_actual_ms += slice_ms
            if call.is_prefill:
                # Folding model: prefill resolves within the slice that
                # contained it. Charge the solo prefill cost to the job
                # and flip to decode.
                n_new = max(0, call.prompt_len - call.prefix_len)
                sim_job.solo_ms += self.cost_model.estimate_solo_prefill_total_ms(
                    n_new, call.prefix_len
                )
                call.is_prefill = False
                # Continue decoding within the same slice if budget remains.
                # Approximate: keep the slice's remaining step budget for
                # decode tokens, weighted by the fact that prefill already
                # ate some of it. We simply do not subtract here — the
                # decoded count for this slice will rest on the next loop.
                continue
            # Decode phase.
            call.decoded_tokens += steps_in_slice
            # Charge per-step solo cost (linear in current KV).
            current_kv = call.prompt_len + int(call.decoded_tokens)
            sim_job.solo_ms += (
                self.cost_model.estimate_solo_tbt_ms(current_kv) * steps_in_slice
            )
            if (
                call.expected_output_len > 0
                and call.decoded_tokens >= call.expected_output_len
            ):
                finished.append(call)
        for call in finished:
            sim_job.active_calls.remove(call)
            if sim_job.pending_calls:
                sim_job.active_calls.append(sim_job.pending_calls.pop(0))

    @staticmethod
    def _any_work_remaining(sim_jobs: Sequence[_SimJob]) -> bool:
        return any((job.active_calls or job.pending_calls) for job in sim_jobs)


__all__ = [
    "REASON_OK",
    "REASON_HALO_ADMISSION_PREDICTED",
    "ActiveCallInfo",
    "AdmissionDecisionResult",
    "AdmissionPredictor",
    "JobLookaheadInput",
    "JobSlowdownAdmissionPredictor",
    "LookaheadAdmissionPredictor",
    "NewJobInput",
    "RequestSlowdownAdmissionPredictor",
    "decide_admission",
]
