"""HaloController — top-level glue for HALO Phase 1.

See managers/halo/CLAUDE.md.
"""

# HALO: Phase 1 controller. Owned by the scheduler; entirely off when
# config.enabled is False. Public methods are the only thing the scheduler
# calls. Internal modules (job_registry, slowdown_tracker, job) do not import
# scheduler state — instead the scheduler hands us a build_infos callable
# each tick.
#
# Design note: this mirrors the AdmissionController pattern so the scheduler
# integration story is uniform (init in __init__, hooks at admission/finish/
# step). Halo *does not* replace admission control — both can be active in
# parallel in Phase 1.

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from sglang.srt.managers.halo.admission_control.cost_model import (
    HaloStepCostModel,
    try_load_halo_step_cost_model,
    try_load_prefill_cost_model,
    try_load_tbt_cost_model,
)
from sglang.srt.managers.halo.admission_decision import (
    REASON_HALO_ADMISSION_PREDICTED,
    ActiveCallInfo,
    AdmissionDecisionResult,
    AdmissionPredictor,
    JobLookaheadInput,
    JobSlowdownAdmissionPredictor,
    NewJobInput,
    RequestSlowdownAdmissionPredictor,
    decide_admission,
)
from sglang.srt.managers.halo.job import Job
from sglang.srt.managers.halo.job_registry import (
    REASON_PROGRAM_NOT_REGISTERED,
    JobAdmissionResult,
    JobRegistry,
)
from sglang.srt.managers.halo.metrics import HaloMetrics
from sglang.srt.managers.halo.slowdown_tracker import (
    RequestExecutionInfo,
    SlowdownTracker,
)

# Reject reason constants returned to the scheduler. Used to build the HTTP
# error payload and the decision-log entry. Mirror of admission_control's
# REASON_* convention.
REASON_NO_JOB_ID = "HALO_NO_JOB_ID"
REASON_DISABLED = "HALO_DISABLED"
REASON_JOB_ID_ALREADY_REGISTERED = "JOB_ID_ALREADY_REGISTERED"
# Phase 2 Stage B — application-declared concurrency cap exceeded.
REASON_HALO_CONCURRENCY_CAP = "HALO_CONCURRENCY_CAP"
# Phase 2 KV cap — KV-cache pool usage at/above the configured ceiling.
REASON_HALO_KV_CAP = "HALO_KV_CAP"
# Re-export from job_registry + admission_decision so external callers only
# import controller.
__all__ = [
    "HaloConfig",
    "HaloController",
    "HaloRejectError",
    "HaloRegisterProgramResult",
    "REASON_DISABLED",
    "REASON_HALO_ADMISSION_PREDICTED",
    "REASON_HALO_CONCURRENCY_CAP",
    "REASON_HALO_KV_CAP",
    "REASON_JOB_ID_ALREADY_REGISTERED",
    "REASON_NO_JOB_ID",
    "REASON_PROGRAM_NOT_REGISTERED",
]

logger = logging.getLogger(__name__)


class HaloRejectError(Exception):
    """Raised by HaloController.register_request when a request fails the
    strict-mode admission check (per Q7 + Q12):

    - reason == REASON_NO_JOB_ID            → request has no halo_job_id
    - reason == REASON_PROGRAM_NOT_REGISTERED → job_id was not pre-registered

    Both are converted to HTTP 400 by the scheduler with a reason-specific
    message.
    """

    def __init__(self, reason: str, rid: str) -> None:
        super().__init__(f"halo reject rid={rid} reason={reason}")
        self.reason = reason
        self.rid = rid


@dataclass
class HaloRegisterProgramResult:
    """Return value of `HaloController.register_program`.

    `registered=True`   → fresh registration, http 200.
    `registered=False`  → reason set:
        REASON_DISABLED                 → halo is off; http 400
        REASON_JOB_ID_ALREADY_REGISTERED → conflict; http 409, `existing` is
                                          the current Job.to_dict()
    """

    registered: bool
    job_id: str
    reason: Optional[str] = None
    active_jobs: int = 0
    existing: Optional[Dict[str, Any]] = None


@dataclass
class HaloConfig:
    enabled: bool = False
    default_slo: float = 5.0
    tick_interval_ms: float = 100.0
    aggregator: str = (
        "max+mean"  # Phase 1 always tracks both; reserved string for future
    )
    job_log_path: Optional[str] = None
    prefill_cost_model_path: Optional[str] = None
    tbt_cost_model_path: Optional[str] = None
    # Halo Step Cost Model (post-Phase-1 follow-up; see
    # ms_dev/halo_dev/prediction_model.md). When set, supersedes
    # prefill_cost_model_path / tbt_cost_model_path.
    step_cost_model_path: Optional[str] = None
    gc_retain_seconds: float = 5.0
    # Q13: pre-registered programs that never receive an LLM request get
    # dropped after this many seconds. 0 disables idle GC.
    program_idle_timeout_seconds: float = 300.0
    # JSONL job-log write interval (seconds). Sweeps run every
    # `tick_interval_ms`, but the log captures a full active-jobs
    # snapshot which is verbose; writing it every sweep blows up the
    # file. Default to one log line every 10s; metrics gauges still
    # update every sweep. 0 → write every sweep (legacy behavior).
    job_log_interval_seconds: float = 10.0
    # Safety-net timeout: when a job is QUEUED/RUNNING but has had no
    # admit/finish activity for this many seconds, force COMPLETE so
    # gc_completed can drop it. Catches clients that crashed or forgot
    # to send halo_job_done. 0 disables. Tune up for workloads with
    # legitimately long mid-chain waits (e.g., human-in-the-loop).
    quiescent_timeout_seconds: float = 300.0
    # ── Phase 2 admission control (predictive gate) ────────────────────
    # See ms_dev/halo_dev/admission_design.md.
    #   off     — no predictive admission (Phase 1 strict mode still applies)
    #   job     — job-scoped gate, decision once per job at its first request
    #   request — request-scoped baseline, decision on every request
    #   level2  — DEPRECATED; accepted as an alias of "job" (WARN)
    admission_mode: str = "off"  # off | job | request
    admission_violation_threshold: float = 0.2  # D3 — fraction (0..1)
    admission_lookahead_horizon_sec: float = 0.0  # 0 = SLO-driven (level2, deprecated)
    admission_dry_run: bool = False  # log only; always admit
    admission_decision_log_path: Optional[str] = None  # JSONL output
    # KV-cache hard cap (Stage B′). Reject a *new job's first request* when
    # the KV-cache pool usage ratio is at/above this value — admitting it
    # would risk queueing delay / preemption the VSS gate cannot see (the
    # cost model has no eviction-cliff term). Active iff 0 < ratio < 1
    # (default 0.0 = disabled), and independent of admission_mode (works
    # even when admission_mode=off, so KV-cap-only ablation runs are
    # possible). See admission_design.md §5.
    admission_kv_cap_ratio: float = 0.0


class _JobLogger:
    """Append-only JSONL writer for periodic job snapshots. Rank-0 only."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", buffering=1)  # line-buffered
        logger.info("halo: job log opened at %s", path)

    def write(self, payload: Dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("halo: job log write failed: %s", e)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class HaloController:
    """Top-level Halo coordinator.

    Construction
    ------------
    - If `config.enabled` is False → caller should not construct this at all.
    - Cost model paths missing/malformed → tracker stays inert (no sweep math)
      but job registration / counters still work.
    - `is_rank0` gates JSONL log writes (TP-dedup; metrics use the same gate).
    """

    def __init__(self, config: HaloConfig, is_rank0: bool = True) -> None:
        self.config = config
        self.is_rank0 = is_rank0
        self.registry = JobRegistry()

        # Cost-model selection. Step model (new) wins over the legacy pair
        # when both are configured. See ms_dev/halo_dev/prediction_model.md.
        step_cost = try_load_halo_step_cost_model(config.step_cost_model_path)
        prefill_cost = None
        tbt_cost = None
        if step_cost is not None:
            if config.prefill_cost_model_path or config.tbt_cost_model_path:
                logger.info(
                    "halo: --halo-step-cost-model-path set; ignoring "
                    "--halo-prefill-cost-model-path / --halo-tbt-cost-model-path"
                )
        else:
            prefill_cost = try_load_prefill_cost_model(config.prefill_cost_model_path)
            tbt_cost = try_load_tbt_cost_model(config.tbt_cost_model_path)

        if step_cost is None and prefill_cost is None and tbt_cost is None:
            logger.warning(
                "halo: no cost model loaded — slowdown sweep will be a no-op; "
                "job counters / state still tracked"
            )
        self.tracker = SlowdownTracker(
            prefill_cost=prefill_cost,
            tbt_cost=tbt_cost,
            registry=self.registry,
            step_cost=step_cost,
        )

        self._log: Optional[_JobLogger] = None
        if is_rank0 and config.job_log_path:
            try:
                self._log = _JobLogger(config.job_log_path)
            except OSError as e:
                logger.warning("halo: job log disabled — %s", e)

        # Prometheus metrics — installed by the scheduler after construction
        # (init_halo). None when --enable-metrics is off OR rank > 0. All
        # mutators on this controller null-check before touching it.
        self.metrics: Optional[HaloMetrics] = None
        # Total slo_violation_count seen at the last sweep — diff with the
        # current sum gives the per-sweep delta the counter should bump by.
        self._last_total_violations: int = 0

        self._last_tick_monotonic: float = 0.0
        self._tick_interval_s: float = max(config.tick_interval_ms, 0.0) / 1000.0
        # Independent gate for the JSONL job log — see HaloConfig.
        self._last_log_monotonic: float = 0.0
        self._log_interval_s: float = max(config.job_log_interval_seconds, 0.0)

        # ── Phase 2 admission predictor + decision log ────────────────────
        # See ms_dev/halo_dev/admission_design.md.
        self.admission_predictor: Optional[AdmissionPredictor] = (
            self._build_admission_predictor(config, step_cost)
        )
        # KV-cache hard cap (Stage B′) — independent of admission_mode so a
        # KV-cap-only ablation (admission_mode=off) still gates.
        self.kv_cap_enabled: bool = 0.0 < config.admission_kv_cap_ratio < 1.0
        self._admission_log: Optional[_JobLogger] = None
        if (
            is_rank0
            and config.admission_decision_log_path
            and (self.admission_predictor is not None or self.kv_cap_enabled)
        ):
            try:
                self._admission_log = _JobLogger(config.admission_decision_log_path)
            except OSError as e:
                logger.warning("halo: admission decision log disabled — %s", e)

        logger.info(
            "halo: enabled (default_slo=%.2f tick_interval_ms=%.1f rank0=%s log=%s "
            "admission=%s dry_run=%s threshold=%.2f)",
            config.default_slo,
            config.tick_interval_ms,
            is_rank0,
            config.job_log_path or "off",
            config.admission_mode,
            config.admission_dry_run,
            config.admission_violation_threshold,
        )

    @staticmethod
    def _build_admission_predictor(
        config: "HaloConfig",
        step_cost: Optional[HaloStepCostModel],
    ) -> Optional[AdmissionPredictor]:
        """Construct the admission predictor matching config.admission_mode.

        Returns None when admission is off, or when the cost model is
        missing (the predictor can't function without it).
        """
        mode = (config.admission_mode or "off").lower()
        if mode == "off":
            return None
        if step_cost is None:
            logger.warning(
                "halo: admission_mode=%s requested but no Halo Step Cost Model "
                "loaded — admission disabled. Set --halo-step-cost-model-path.",
                mode,
            )
            return None
        # "level0" is the pre-2026-05-15 name for the job-scoped gate; accept
        # it as a silent alias so stale wrappers / env.local.sh keep working.
        if mode in ("job", "level0"):
            return JobSlowdownAdmissionPredictor(step_cost)
        if mode == "request":
            return RequestSlowdownAdmissionPredictor(step_cost)
        if mode == "level2":
            # DEPRECATED 2026-05-15 — see admission_decision.py docstring.
            # Lookahead used declared DAG/remaining-lengths which we now
            # ignore (admission uses current state only). Fall back to the
            # job-scoped gate with a startup warning so misconfigured runs
            # still proceed instead of silently disabling admission.
            logger.warning(
                "halo: admission_mode=level2 is DEPRECATED (2026-05-15); "
                "falling back to mode=job (snapshot per-job stretch). See "
                "ms_dev/halo_dev/admission_design.md §4."
            )
            return JobSlowdownAdmissionPredictor(step_cost)
        logger.warning("halo: unknown admission_mode=%r — admission disabled", mode)
        return None

    # ------------------------------------------------------------------
    # Admission / finish hooks
    # ------------------------------------------------------------------

    def register_request(
        self,
        rid: str,
        halo_job_id: Optional[str],
        halo_slo: Optional[float],
        *,
        prompt_len: int = 0,
        prefix_len: int = 0,
        active_request_infos: Optional[List[RequestExecutionInfo]] = None,
        kv_usage_ratio: Optional[float] = None,
        chunked_prefill_size: Optional[int] = None,
    ) -> Job:
        """Admission hook combining strict-mode (Phase 1) + predictive
        gate (Phase 2 Stage A) + KV-cache hard cap (Stage B′).

        Order of checks (the first one that fails wins):
            1. Strict mode Q7 — halo_job_id REQUIRED.
            2. Strict mode Q12 — job_id MUST be pre-registered.
            3. Stage B′ — KV cap. New job's first request only; reject when
               kv_usage_ratio >= admission_kv_cap_ratio.
            4. Stage A — Predictive VSS admission, if admission_mode != off
               and active_request_infos was supplied.
            5. Stage B — concurrency cap (declared_max_concurrency).
            6. Actually admit (rid → job mapping, counters).

        Args new in Phase 2:
            prompt_len, prefix_len      — first-call shape, for the Stage A
                                          batch composition feature.
            active_request_infos        — snapshot of currently in-flight
                                          Halo-tracked requests (the same
                                          list the periodic sweep consumes).
                                          When None or empty, Stage A is
                                          skipped.
            kv_usage_ratio              — current KV-cache pool usage ratio
                                          (0..1) for the Stage B′ KV cap.
                                          None → KV cap skipped.
            chunked_prefill_size        — server's effective chunked-prefill
                                          token budget; bounds the Stage A
                                          EXTEND-step cost (결함 A). None →
                                          no cap (chunked prefill disabled).

        Missing `halo_slo` → use `config.default_slo` (the registry's
        pre-registered slo still wins, see Q10).
        """
        # ── 1. Q7 — halo_job_id required ─────────────────────────────────
        if halo_job_id is None or halo_job_id == "":
            if self.metrics is not None:
                self.metrics.record_request_rejected(REASON_NO_JOB_ID)
            raise HaloRejectError(reason=REASON_NO_JOB_ID, rid=rid)
        slo = halo_slo if halo_slo is not None else self.config.default_slo

        # ── 2. Q12 — program must be pre-registered ──────────────────────
        # Look up the Job *before* admit_to_job so Stage A has access to its
        # declared structure (total_calls, stage_sequence, expected lens).
        new_job_record = self.registry.job_for_id(halo_job_id)
        if new_job_record is None:
            if self.metrics is not None:
                self.metrics.record_request_rejected(REASON_PROGRAM_NOT_REGISTERED)
            raise HaloRejectError(reason=REASON_PROGRAM_NOT_REGISTERED, rid=rid)

        is_first_request = new_job_record.total_request_number == 0

        # ── 3. Stage B′ — KV-cache hard cap ──────────────────────────────
        # New job's first request only. Follow-up requests of an
        # already-admitted job bypass (Phase 2: once a job is admitted its
        # subsequent requests are unconditionally admitted — there is no
        # pending queue yet to hold them, and rejecting one would strand a
        # running job). The cost-model VSS gate cannot see the KV-pressure
        # cliff (eviction / preemption), so this is a separate hard gate.
        # dry-run logs the would-reject but still admits.
        if (
            is_first_request
            and self.kv_cap_enabled
            and kv_usage_ratio is not None
            and kv_usage_ratio >= self.config.admission_kv_cap_ratio
        ):
            self._log_kv_cap_decision(rid, halo_job_id, kv_usage_ratio)
            if not self.config.admission_dry_run:
                if self.metrics is not None:
                    self.metrics.record_request_rejected(REASON_HALO_KV_CAP)
                raise HaloRejectError(reason=REASON_HALO_KV_CAP, rid=rid)

        # ── 4. Stage A — Predictive VSS admission (Phase 2) ──────────────
        # Scope depends on admission_mode (admission_design.md §3/§4):
        #   mode "job"     — decision made *once per job*, at its first
        #                    request. Follow-up requests bypass Stage A so a
        #                    chain is never rejected mid-flight (rejecting
        #                    it would waste the work in earlier calls).
        #   mode "request" — decision made on *every* request (the
        #                    request-scoped baseline). A mid-chain request
        #                    can be rejected.
        predictor = self.admission_predictor

        # Empty active list is still a valid input (auto-admit + log row);
        # only None means "scheduler did not provide the snapshot" → skip.
        run_stage_a = (
            predictor is not None
            and active_request_infos is not None
            and (predictor.mode_name == "request" or is_first_request)
        )
        if run_stage_a:
            # Job scope excludes the arriving job (it has no active calls
            # to protect yet). Request scope scores *all* in-flight
            # requests — including any already-active calls of the arriving
            # job — so nothing is excluded.
            exclude_job_id = None if predictor.mode_name == "request" else halo_job_id
            decision = self._stage_a_decide(
                new_job=new_job_record,
                new_slo=slo,
                prompt_len=prompt_len,
                prefix_len=prefix_len,
                active_request_infos=active_request_infos,
                exclude_job_id=exclude_job_id,
                chunked_prefill_size=chunked_prefill_size,
            )
            self._log_admission_decision(
                rid, halo_job_id, decision, is_first_call=is_first_request
            )
            if not decision.admit and not self.config.admission_dry_run:
                if self.metrics is not None:
                    self.metrics.record_request_rejected(
                        REASON_HALO_ADMISSION_PREDICTED
                    )
                raise HaloRejectError(reason=REASON_HALO_ADMISSION_PREDICTED, rid=rid)

        # ── 5. Stage B — Concurrency hard cap (Phase 2) ──────────────────
        # When the application declared a max in-flight count for this job,
        # reject the call if accepting it would push past that promise.
        cap = new_job_record.declared_max_concurrency
        if cap is not None and new_job_record.in_flight_count >= cap:
            if self.metrics is not None:
                self.metrics.record_request_rejected(REASON_HALO_CONCURRENCY_CAP)
            raise HaloRejectError(reason=REASON_HALO_CONCURRENCY_CAP, rid=rid)

        # ── 6. Actually admit ────────────────────────────────────────────
        result: JobAdmissionResult = self.registry.admit_to_job(halo_job_id, slo, rid)
        if not result.admitted:
            # Defensive (should not happen given step 2 above, but cheap).
            reason = result.reason or REASON_PROGRAM_NOT_REGISTERED
            if self.metrics is not None:
                self.metrics.record_request_rejected(reason)
            raise HaloRejectError(reason=reason, rid=rid)
        if self.metrics is not None:
            self.metrics.record_request_admitted()
        return result.job

    # ------------------------------------------------------------------
    # Phase 2 — Stage A helpers
    # ------------------------------------------------------------------

    def _stage_a_decide(
        self,
        new_job: Job,
        new_slo: float,
        prompt_len: int,
        prefix_len: int,
        active_request_infos: List[RequestExecutionInfo],
        exclude_job_id: Optional[str],
        chunked_prefill_size: Optional[int] = None,
    ) -> AdmissionDecisionResult:
        """Run the admission predictor + decide_admission against the
        current request-execution snapshot."""
        active_jobs_input = self._build_active_jobs_input(
            active_request_infos, exclude_job_id=exclude_job_id
        )
        new_job_input = self._build_new_job_input(
            new_job, new_slo, prompt_len, prefix_len
        )
        predictions = self.admission_predictor.predict(
            active_jobs_input, new_job_input, chunked_prefill_size
        )
        # The predictor's output keys differ by scope: job_id for the
        # job-scoped gate, rid for the request-scoped baseline. Build the
        # SLO map with the matching keys so decide_admission compares
        # like-for-like.
        if self.admission_predictor.mode_name == "request":
            slos_map = {
                call.rid: j.slo for j in active_jobs_input for call in j.active_calls
            }
        else:
            slos_map = {j.job_id: j.slo for j in active_jobs_input}
        return decide_admission(
            predictions,
            slos_map,
            threshold=self.config.admission_violation_threshold,
            mode=self.admission_predictor.mode_name,
            horizon_sec=(
                self.config.admission_lookahead_horizon_sec
                if self.admission_predictor.mode_name == "level2"
                else None
            ),
        )

    def _build_active_jobs_input(
        self,
        infos: List[RequestExecutionInfo],
        *,
        exclude_job_id: Optional[str] = None,
    ) -> List[JobLookaheadInput]:
        """Group RequestExecutionInfo by job_id and build the per-job inputs
        the predictor expects.

        Memoryless: each JobLookaheadInput carries only the job's SLO and
        its in-flight calls' *current* shape (prompt/prefix/KV). The VSS
        predictors derive every solo step time from those — no lifetime VJS,
        no elapsed history is consulted.

        Calls with `kv_len_now <= 0` are *excluded* (결함 A, 2026-05-18): a
        request with zero committed KV is sitting in the waiting queue (not
        yet prefilled, or retracted) — it is in NO forward step, so a
        step-ratio VSS cannot represent it. Including it as a fake
        prefill-phase call is what let the whole backlog inflate the EXTEND
        batch. Note: this means VSS is structurally blind to queued/
        preempted requests — an accepted limitation (see admission_design.md
        §12); the periodic VJS sweep still counts their wait."""
        by_job: Dict[str, List[RequestExecutionInfo]] = {}
        for info in infos:
            if info.job_id is None:
                continue
            if exclude_job_id is not None and info.job_id == exclude_job_id:
                continue
            # Skip pure waiting-queue requests — not in any step (결함 A).
            if info.kv_len_now <= 0:
                continue
            by_job.setdefault(info.job_id, []).append(info)

        out: List[JobLookaheadInput] = []
        for jid, info_list in by_job.items():
            job = self.registry.job_for_id(jid)
            if job is None:
                continue
            active_calls = tuple(
                ActiveCallInfo(
                    rid=i.rid,
                    # If decoded_tokens > 0 the call has finished prefill.
                    is_prefill=i.decoded_tokens_so_far <= 0,
                    prompt_len=i.prompt_len,
                    prefix_len=i.prefix_len_at_admission,
                    decoded_tokens=i.decoded_tokens_so_far,
                    kv_len_now=i.kv_len_now,
                    # elapsed_actual_ms — Lookahead (deprecated) only.
                    elapsed_actual_ms=i.elapsed_ms,
                )
                for i in info_list
            )
            out.append(
                JobLookaheadInput(
                    job_id=jid,
                    slo=job.slo,
                    active_calls=active_calls,
                )
            )
        return out

    def _build_new_job_input(
        self,
        new_job: Job,
        slo: float,
        prompt_len: int,
        prefix_len: int,
    ) -> NewJobInput:
        return NewJobInput(
            job_id=new_job.job_id,
            slo=slo,
            first_call_input_len=prompt_len,
            first_call_prefix_len=prefix_len,
            total_calls=new_job.total_calls_expected,
            stage_sequence=(
                tuple(new_job.stage_sequence)
                if new_job.stage_sequence is not None
                else None
            ),
            expected_input_lens=(
                tuple(new_job.expected_input_lens)
                if new_job.expected_input_lens is not None
                else None
            ),
            expected_output_lens=(
                tuple(new_job.expected_output_lens)
                if new_job.expected_output_lens is not None
                else None
            ),
            expected_cached_prefix_lens=None,
        )

    def _log_kv_cap_decision(
        self, rid: str, job_id: str, kv_usage_ratio: float
    ) -> None:
        """Emit a JSONL row for a Stage B′ KV-cap reject (or dry-run
        would-reject). Shares the rid/job_id/mode/decision/reason shape with
        the Stage A admission row so a single parser handles both."""
        if self._admission_log is None:
            return
        self._admission_log.write(
            {
                "ts_ns": time.time_ns(),
                "rid": rid,
                "job_id": job_id,
                "mode": "kv_cap",
                "is_first_call": True,
                "dry_run": self.config.admission_dry_run,
                "decision": ("admit" if self.config.admission_dry_run else "reject"),
                "reason": REASON_HALO_KV_CAP,
                "kv_usage_ratio": kv_usage_ratio,
                "kv_cap_ratio": self.config.admission_kv_cap_ratio,
            }
        )

    def _log_admission_decision(
        self,
        rid: str,
        job_id: str,
        decision: AdmissionDecisionResult,
        *,
        is_first_call: bool,
    ) -> None:
        if self._admission_log is None:
            return
        self._admission_log.write(
            {
                "ts_ns": time.time_ns(),
                "rid": rid,
                "job_id": job_id,
                "mode": decision.mode,
                # True when this request is its job's first LLM call. In
                # mode "request" a False here marks a mid-chain decision.
                "is_first_call": is_first_call,
                "dry_run": self.config.admission_dry_run,
                "decision": "admit" if decision.admit else "reject",
                "reason": decision.reason,
                "violation_ratio": decision.violation_ratio,
                "violation_count": decision.violation_count,
                # Count of scored units — jobs (mode "job") or requests
                # (mode "request").
                "active_units_total": decision.active_jobs_total,
                "threshold": decision.threshold,
                "horizon_sec": decision.horizon_sec,
                # Keys are job_id (mode "job") or rid (mode "request").
                "predicted_slowdowns": decision.predicted_slowdowns,
            }
        )

    def register_program(
        self,
        job_id: str,
        slo: float,
        *,
        total_calls: Optional[int] = None,
        stage_sequence: Optional[List[str]] = None,
        expected_input_lens: Optional[List[int]] = None,
        expected_output_lens: Optional[List[int]] = None,
        dag: Optional[Dict[str, Any]] = None,
        declared_max_concurrency: Optional[int] = None,
    ) -> HaloRegisterProgramResult:
        """Pre-register a job (Option A — `POST /halo/programs`).

        Behavior per §13:
        - Halo disabled (controller wouldn't be constructed, but defensive):
          returns `registered=False, reason=REASON_DISABLED`.
        - Fresh `job_id` (Q11): registered=True. http 200.
        - Duplicate `job_id` (Q11): registered=False,
          reason=REASON_JOB_ID_ALREADY_REGISTERED; carries existing job dict.
        """
        if not self.config.enabled:
            if self.metrics is not None:
                self.metrics.record_program_rejected(REASON_DISABLED)
            return HaloRegisterProgramResult(
                registered=False, job_id=job_id, reason=REASON_DISABLED
            )

        fresh, job = self.registry.register_program(
            job_id=job_id,
            slo=slo,
            total_calls=total_calls,
            stage_sequence=stage_sequence,
            expected_input_lens=expected_input_lens,
            expected_output_lens=expected_output_lens,
            dag=dag,
            declared_max_concurrency=declared_max_concurrency,
        )
        if not fresh:
            if self.metrics is not None:
                self.metrics.record_program_rejected(REASON_JOB_ID_ALREADY_REGISTERED)
            return HaloRegisterProgramResult(
                registered=False,
                job_id=job_id,
                reason=REASON_JOB_ID_ALREADY_REGISTERED,
                existing=job.to_dict(),
            )

        if self._log is not None:
            self._log.write(
                {
                    "ts": time.monotonic(),
                    "event": "register_program",
                    "job": job.to_dict(),
                }
            )
        if self.metrics is not None:
            self.metrics.record_program_registered()
        logger.info(
            "halo: registered program job_id=%s slo=%.2f total_calls=%s",
            job_id,
            slo,
            total_calls,
        )
        return HaloRegisterProgramResult(
            registered=True,
            job_id=job_id,
            active_jobs=len(self.registry.active_jobs()),
        )

    def on_request_finished(
        self,
        rid: str,
        finished_info: Optional[RequestExecutionInfo] = None,
        halo_job_done: bool = False,
    ) -> None:
        """Called from the scheduler's finish path.

        `finished_info` is the just-finished request's final snapshot. Its
        time span (admit → finish) + token features are frozen into the
        owning job's `completed_call_spans` so the call keeps contributing
        to the job's lifetime VJS after the request object is gone. This
        happens *before* record_completion pops the rid→job mapping.

        If the client set `halo_job_done=true` on this request, mark the
        owning job COMPLETE (also before record_completion). Then decrement
        counters as normal.

        Emits a `job_complete` row to the JSONL job log so the termination
        is observable even when the COMPLETE state doesn't survive long
        enough to be captured by the periodic sweep snapshot.
        """
        # Freeze the finished call's span onto its job (lifetime VJS).
        if finished_info is not None:
            job = self.registry.job_for_request(rid)
            if job is not None:
                job.record_completed_call(self.tracker.span_from_info(finished_info))
        if halo_job_done:
            job = self.registry.mark_job_done(rid)
            if job is not None and self._log is not None:
                self._log.write(
                    {
                        "ts": time.monotonic(),
                        "event": "job_complete",
                        "reason": "halo_job_done",
                        "job": job.to_dict(),
                    }
                )
        self.registry.record_completion(rid)

    # ------------------------------------------------------------------
    # Periodic tick (called from scheduler_metrics_mixin)
    # ------------------------------------------------------------------

    def should_tick(self, now_monotonic: Optional[float] = None) -> bool:
        """Returns True if at least tick_interval_ms has passed since last sweep."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return (now - self._last_tick_monotonic) >= self._tick_interval_s

    def tick(
        self,
        build_infos: Callable[[], List[RequestExecutionInfo]],
        now_monotonic: Optional[float] = None,
    ) -> None:
        """Run a sweep if enough wall-clock has elapsed. Cheap when not due."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if (now - self._last_tick_monotonic) < self._tick_interval_s:
            return
        self._last_tick_monotonic = now

        infos = build_infos()
        if infos:
            self.tracker.sweep(infos)

        # JSONL log is gated independently from the sweep — the sweep
        # itself stays high-frequency (so each job's virtual_job_slowdown
        # and the metrics gauges update every tick), but the verbose
        # per-sweep snapshot only lands on disk every
        # config.job_log_interval_seconds. Set the gate to 0 to write
        # every sweep (legacy behavior, useful for very short tests).
        if self._log is not None and (
            self._log_interval_s <= 0.0
            or (now - self._last_log_monotonic) >= self._log_interval_s
        ):
            self._log.write(
                {
                    "ts": now,
                    "active_jobs": [j.to_dict() for j in self.registry.active_jobs()],
                }
            )
            self._last_log_monotonic = now

        # Safety net BEFORE gc_completed: any job that's been quiescent
        # (no in-flight + no recent update) gets flipped to COMPLETE.
        # Then gc_completed picks them up after retain_seconds.
        flipped = self.registry.gc_quiescent_jobs(self.config.quiescent_timeout_seconds)
        if self._log is not None and flipped:
            ts = time.monotonic()
            for job in flipped:
                self._log.write(
                    {
                        "ts": ts,
                        "event": "job_complete",
                        "reason": "quiescent_timeout",
                        "job": job.to_dict(),
                    }
                )
        # GC completed jobs older than retain window — bounded memory.
        self.registry.gc_completed(self.config.gc_retain_seconds)
        # Q13: drop pre-registered programs that never received a request.
        self.registry.gc_idle_programs(self.config.program_idle_timeout_seconds)

        # Refresh Prometheus gauges from the post-GC active set. We diff the
        # cumulative slo_violation_count across the registry to drive the
        # counter increment so it never double-counts a sweep.
        if self.metrics is not None:
            actives = self.registry.active_jobs()
            total_violations = sum(
                j.slo_violation_count for j in self.registry.all_jobs()
            )
            delta = max(0, total_violations - self._last_total_violations)
            self._last_total_violations = total_violations
            self.metrics.update_from_sweep(
                active_jobs=actives,
                total_known=len(self.registry.all_jobs()),
                slo_violations_delta=delta,
            )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def snapshot(self, limit: int = 32) -> Dict[str, Any]:
        """For /server_info."""
        s = {
            "enabled": True,
            "default_slo": self.config.default_slo,
            "tick_interval_ms": self.config.tick_interval_ms,
            "aggregator": self.config.aggregator,
            "program_idle_timeout_seconds": self.config.program_idle_timeout_seconds,
            "cost_models_loaded": {
                "prefill": self.tracker.prefill_cost is not None,
                "tbt": self.tracker.tbt_cost is not None,
                "step": self.tracker.step_cost is not None,
            },
        }
        s.update(self.registry.snapshot_dict(limit=limit))
        return s

    def close(self) -> None:
        if self._log is not None:
            self._log.close()


def build_halo_controller_from_server_args(
    server_args: Any,
    is_rank0: bool,
) -> Optional[HaloController]:
    """Factory used by the scheduler. Returns None if `--halo-enabled` is off.

    Pulled into a free function so the scheduler integration is a one-line
    call and the scheduler test surface stays small.
    """
    if not getattr(server_args, "halo_enabled", False):
        return None
    config = HaloConfig(
        enabled=True,
        default_slo=float(getattr(server_args, "halo_default_slo", 5.0)),
        tick_interval_ms=float(getattr(server_args, "halo_tick_interval_ms", 100.0)),
        aggregator=getattr(server_args, "halo_aggregator", "max+mean"),
        job_log_path=getattr(server_args, "halo_job_log", None),
        prefill_cost_model_path=getattr(
            server_args, "halo_prefill_cost_model_path", None
        ),
        tbt_cost_model_path=getattr(server_args, "halo_tbt_cost_model_path", None),
        step_cost_model_path=getattr(server_args, "halo_step_cost_model_path", None),
        admission_mode=getattr(server_args, "halo_admission_mode", "off") or "off",
        admission_violation_threshold=float(
            getattr(server_args, "halo_admission_violation_threshold", 0.2)
        ),
        admission_lookahead_horizon_sec=float(
            getattr(server_args, "halo_admission_lookahead_horizon_sec", 0.0)
        ),
        admission_dry_run=bool(getattr(server_args, "halo_admission_dry_run", False)),
        admission_decision_log_path=getattr(
            server_args, "halo_admission_decision_log", None
        ),
        admission_kv_cap_ratio=float(
            getattr(server_args, "halo_admission_kv_cap_ratio", 0.0)
        ),
        program_idle_timeout_seconds=float(
            getattr(server_args, "halo_program_idle_timeout_seconds", 300.0)
        ),
        job_log_interval_seconds=float(
            getattr(server_args, "halo_job_log_interval_seconds", 10.0)
        ),
        quiescent_timeout_seconds=float(
            getattr(server_args, "halo_quiescent_timeout_seconds", 300.0)
        ),
    )
    return HaloController(config=config, is_rank0=is_rank0)
