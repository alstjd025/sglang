"""Job-level admission decision for Halo Phase 2.

Design + rationale: ms_dev/halo_dev/admission_design.md.

Core idea (different from request-level admission_control):
    Reject a new job if admitting it would push too many *currently active
    jobs* over their SLO. Decision uses the Halo Step Cost Model (the same
    cost model that powers R1 slowdown tracking) to predict each active
    job's *predicted virtual job slowdown* after the new arrival.

Predictor variants:
    - SnapshotAdmissionPredictor  (Level 0): per-job stretch model. Current
      production default. Each active job's stretch is taken from its own
      current phase (prefill vs decode). No DAG / remaining-call lookahead.
    - LookaheadAdmissionPredictor (Level 2): 1-second slice forward simulation
      that used declared DAG / remaining lengths. **DEPRECATED 2026-05-15**:
      not selected at runtime (controller falls back to level0 with a WARN).
      Code retained for reference but is not on the admission path.

decide_admission(...) glues a predictor's output to the violation-ratio
threshold (D3 = 0.2) and returns AdmissionDecisionResult — the controller
logs that and either passes through or raises HaloRejectError.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.managers.admission_control.cost_model import HaloStepCostModel


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
    """A single in-flight LLM call inside an active job."""

    rid: str
    is_prefill: bool          # True while still doing prefill (uncached new tokens)
    prompt_len: int           # total prompt tokens (n + r)
    prefix_len: int           # cached prefix at admit time (r)
    decoded_tokens: int       # tokens already produced
    kv_len_now: int           # current KV span (= prompt_len + decoded_tokens)
    elapsed_actual_ms: float  # wall-clock since this call started


@dataclass(frozen=True)
class JobLookaheadInput:
    """Per-active-job snapshot consumed by AdmissionPredictor.

    All fields after ``current_solo_elapsed_ms`` are *declared* and may be
    None. **As of 2026-05-15 (per-job-stretch refactor)** the production
    SnapshotAdmissionPredictor IGNORES every declared field — they are kept
    on the dataclass solely for the deprecated LookaheadAdmissionPredictor
    and any future predictor that may reuse them.
    """

    job_id: str
    slo: float
    # Current observed state (from Halo R1 / Job + RequestExecutionInfo)
    active_calls: Tuple[ActiveCallInfo, ...]
    current_actual_elapsed_ms: float
    current_solo_elapsed_ms: float  # cost_model.solo accumulated so far
    # ---- LEGACY (not used by SnapshotAdmissionPredictor since 2026-05-15) ----
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
    reason: str                            # REASON_OK or REASON_HALO_ADMISSION_PREDICTED
    violation_ratio: float                 # violations / active_jobs_total
    violation_count: int
    active_jobs_total: int
    threshold: float
    # Renamed 2026-05-15 (was: predicted_slowdowns). Job-id → predicted
    # virtual job slowdown after admitting the new job.
    predicted_virtual_job_slowdowns: Dict[str, float]
    # Optional / mode-specific diagnostic fields:
    horizon_sec: Optional[float] = None    # Level 2 only
    mode: str = ""                          # "level0" | "level2"


# ─────────────────────────────────────────────────────────────────────────────
# Predictor base
# ─────────────────────────────────────────────────────────────────────────────
class AdmissionPredictor(ABC):
    """Predicts each active job's final slowdown_max *if* the new job is
    admitted. Concrete subclasses differ in how they extrapolate the future:

    - Level 0 (SnapshotAdmissionPredictor): treat the current load as
      time-invariant and stretch each job's declared remaining-solo time by
      a single factor (augmented_step_ms / current_step_ms).
    - Level 2 (LookaheadAdmissionPredictor): 1-second slice forward
      simulation, letting the batch composition evolve as calls finish and
      new calls of the same job become active.
    """

    mode_name: str = "abstract"

    def __init__(self, cost_model: HaloStepCostModel) -> None:
        self.cost_model = cost_model

    @abstractmethod
    def predict(
        self,
        active_jobs: Sequence[JobLookaheadInput],
        new_job: NewJobInput,
    ) -> Dict[str, float]:
        """Returns {job_id: predicted_final_slowdown_max} for every active job.

        New job not included — only existing active jobs are scored (its
        SLO is the caller's responsibility, not admission control's).
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


# ─────────────────────────────────────────────────────────────────────────────
# Level 0 — Per-job snapshot stretch (M-2, current production design)
# ─────────────────────────────────────────────────────────────────────────────
class SnapshotAdmissionPredictor(AdmissionPredictor):
    """Per-job snapshot stretch — admits/rejects using only *current* batch
    state plus the arriving request's first-call shape.

    Algorithm (M-2, decided 2026-05-15):

        # 1. Current batch composition
        current_prefill = [(n_i, r_i) for prefill-phase active calls]
        current_decode  = [kv_j      for decode-phase active calls]

        # 2. EXTEND step stretch — applies to prefill-phase active jobs
        stretch_extend = cost([prefill + new_req_first_call], []) / cost(prefill, [])
                         (= 1.0 when current_prefill is empty)

        # 3. DECODE step stretch — applies to decode-phase active jobs
        stretch_decode = cost([], [decode + new_req.prompt_len]) / cost([], decode)
                         (= 1.0 when current_decode is empty)

        # 4. Per-job: stretch by current phase
        for each active job i:
            if i has any active prefill call:
                stretch_i = stretch_extend
            else:
                stretch_i = stretch_decode
            current_VJS_i = (actual / solo) if solo > 0 else i.slo
            predicted_VJS_i = current_VJS_i × stretch_i

    No DAG, no remaining-call lookahead, no future-phase modelling. The
    rationale (sample env runs with mixed_chunk=OFF so prefill and decode
    are time-separated; admission decides only on the *immediate* impact of
    the new request on each job's *own* current step type) is described in
    ms_dev/halo_dev/admission_design.md §4.
    """

    mode_name = "level0"

    def predict(
        self,
        active_jobs: Sequence[JobLookaheadInput],
        new_job: NewJobInput,
    ) -> Dict[str, float]:
        current_prefill, current_decode = _split_current_batch_features(active_jobs)

        # ── stretch_extend (used by prefill-phase active jobs) ───────────
        n_new = max(
            0,
            new_job.first_call_input_len - new_job.first_call_prefix_len,
        )
        r_new = new_job.first_call_prefix_len
        if current_prefill:
            base_ext = self.cost_model.estimate_step_ms(current_prefill, [])
            aug_ext = self.cost_model.estimate_step_ms(
                current_prefill + [(n_new, r_new)], []
            )
            stretch_extend = aug_ext / base_ext if base_ext > 0 else 1.0
        else:
            # No prefill-phase active job will consume this anyway.
            stretch_extend = 1.0

        # ── stretch_decode (used by decode-phase active jobs) ────────────
        if current_decode:
            base_dec = self.cost_model.estimate_step_ms([], current_decode)
            aug_dec = self.cost_model.estimate_step_ms(
                [], current_decode + [new_job.first_call_input_len]
            )
            stretch_decode = aug_dec / base_dec if base_dec > 0 else 1.0
        else:
            stretch_decode = 1.0

        # ── per-job: pick stretch by current phase ───────────────────────
        predictions: Dict[str, float] = {}
        for job in active_jobs:
            job_is_in_prefill = any(call.is_prefill for call in job.active_calls)
            stretch_i = stretch_extend if job_is_in_prefill else stretch_decode

            if job.current_solo_elapsed_ms > 0:
                current_vjs = (
                    job.current_actual_elapsed_ms / job.current_solo_elapsed_ms
                )
            else:
                # Matches R1's initial slowdown_max = SLO convention.
                current_vjs = job.slo

            predictions[job.job_id] = current_vjs * stretch_i
        return predictions


# ─────────────────────────────────────────────────────────────────────────────
# Decision wrapper
# ─────────────────────────────────────────────────────────────────────────────
def decide_admission(
    predicted_virtual_job_slowdowns: Dict[str, float],
    slos: Dict[str, float],
    threshold: float,
    mode: str = "",
    horizon_sec: Optional[float] = None,
) -> AdmissionDecisionResult:
    """Apply the violation-ratio threshold to a predictor's output.

    Args:
        predicted_virtual_job_slowdowns: {job_id: predicted virtual job
            slowdown after admitting the new job}.
        slos:        {job_id: SLO}. Falls back to the prediction's value if
                     missing (treats missing SLO as "always satisfied").
        threshold:   D3 — fraction (0..1). If violation_ratio > threshold,
                     decision is REJECT.
        mode:        for diagnostic.
        horizon_sec: Level 2 only — for diagnostic.

    Returns:
        AdmissionDecisionResult with all fields populated.
    """
    predictions = predicted_virtual_job_slowdowns
    total = len(predictions)
    if total == 0:
        return AdmissionDecisionResult(
            admit=True,
            reason=REASON_OK,
            violation_ratio=0.0,
            violation_count=0,
            active_jobs_total=0,
            threshold=threshold,
            predicted_virtual_job_slowdowns={},
            horizon_sec=horizon_sec,
            mode=mode,
        )
    violations = 0
    for jid, pred in predictions.items():
        slo = slos.get(jid, pred)
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
            predicted_virtual_job_slowdowns=dict(predictions),
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
        predicted_virtual_job_slowdowns=dict(predictions),
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
    prompt_len: int            # n + r (unchanged through decode)
    prefix_len: int            # r at start of call (unchanged)
    decoded_tokens: float      # fractional during the simulation
    expected_output_len: int   # 0 means unknown — call never auto-terminates
    elapsed_actual_ms: float   # wall-clock since this call started in sim


@dataclass
class _SimJob:
    """Mutable simulation state for one job (current + future calls)."""

    job_id: str
    slo: float
    elapsed_actual_ms: float     # cumulative across calls (sim + observed)
    solo_ms: float               # cumulative solo across calls (sim + observed)
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
    SnapshotAdmissionPredictor with a WARN at startup. This class is kept
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
            step_ms = self.cost_model.estimate_step_ms(
                prefill_infos, decode_infos
            )
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
    def _resolve_horizon(
        self, active_jobs: Sequence[JobLookaheadInput]
    ) -> float:
        if self.horizon_sec > 0:
            return min(self.horizon_sec, self.MAX_HORIZON_SEC)
        # SLO-driven: when would this job hit slowdown == SLO?
        best = 0.0
        for job in active_jobs:
            remaining_solo = _estimate_remaining_solo_ms(
                self.cost_model, job
            )
            if remaining_solo is None:
                # Insufficient declared info — use the current observed
                # elapsed (degenerate but not zero).
                expected_total_solo = max(
                    job.current_solo_elapsed_ms, 1.0
                )
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
                    expected_output_len=self._lookup_expected_output_for_call(
                        j, c
                    ),
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
        prefix_lens = (
            job.expected_cached_prefix_lens or tuple(0 for _ in range(n_remaining))
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
        prefix_lens = (
            new_job.expected_cached_prefix_lens or tuple(0 for _ in range(n))
        )
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
                    decode_kvs.append(
                        call.prompt_len + int(call.decoded_tokens)
                    )
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
                self.cost_model.estimate_solo_tbt_ms(current_kv)
                * steps_in_slice
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
        return any(
            (job.active_calls or job.pending_calls) for job in sim_jobs
        )


__all__ = [
    "REASON_OK",
    "REASON_HALO_ADMISSION_PREDICTED",
    "ActiveCallInfo",
    "AdmissionDecisionResult",
    "AdmissionPredictor",
    "JobLookaheadInput",
    "LookaheadAdmissionPredictor",
    "NewJobInput",
    "SnapshotAdmissionPredictor",
    "decide_admission",
]
