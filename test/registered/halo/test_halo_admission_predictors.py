"""Unit tests for managers/halo/admission_decision.py (Phase 2).

Design + rationale: ms_dev/halo_dev/admission_design.md.

Coverage (per 2026-05-15 per-job-stretch design):
- decide_admission threshold logic (admit / reject / boundaries)
- JobSlowdownAdmissionPredictor (mode "job"):
    * empty active set → empty predictions
    * single decode-phase job → stretch_decode applied
    * single prefill-phase job → stretch_extend applied
    * mixed (some prefill, some decode) jobs → each gets its own phase's stretch
    * bigger arrival → larger predicted virtual job slowdown
    * fresh job (current_solo_ms = 0) → fallback to SLO
    * declared fields ignored (predictor doesn't read them anymore)
- RequestSlowdownAdmissionPredictor (mode "request"):
    * per-request keys (rid), no job aggregation
    * each active call scored by its own phase's stretch
    * per-request slowdown = elapsed / solo, fallback to SLO when solo = 0
- LookaheadAdmissionPredictor → DEPRECATED, kept callable for legacy
- Input dataclasses are frozen (defensive — we share these with the
  controller and don't want mutation across threads).
"""

import unittest

from sglang.srt.managers.admission_control.cost_model import HaloStepCostModel
from sglang.srt.managers.halo.admission_decision import (
    REASON_HALO_ADMISSION_PREDICTED,
    REASON_OK,
    ActiveCallInfo,
    AdmissionDecisionResult,
    JobLookaheadInput,
    JobSlowdownAdmissionPredictor,
    LookaheadAdmissionPredictor,
    NewJobInput,
    RequestSlowdownAdmissionPredictor,
    decide_admission,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _split_cost_model():
    """Reasonable split-form cost model for tests.

    Coefficients picked so:
        * solo prefill ≈ 70 ms baseline + ~0.013 ms/token
        * solo decode TBT ≈ 22 ms (matches §17 fit)
    """
    return HaloStepCostModel(
        theta_p1=1e-7,
        theta_p2=4e-7,
        theta_p3=0.013,
        theta_d1=1.5e-5,
        theta_d2=0.18,
        theta_c=22.0,         # mirror of θ_c_d for back-compat
        theta_c_p=70.0,
        theta_c_d=22.0,
        form="halo_step_split_v1",
        metadata={"model": "unit"},
    )


def _decode_call(rid: str, kv_len: int, decoded: int = 100,
                 elapsed_ms: float = 1000.0,
                 solo_ms: float = 0.0) -> ActiveCallInfo:
    return ActiveCallInfo(
        rid=rid,
        is_prefill=False,
        prompt_len=kv_len - decoded,
        prefix_len=0,
        decoded_tokens=decoded,
        kv_len_now=kv_len,
        elapsed_actual_ms=elapsed_ms,
        solo_elapsed_ms=solo_ms,
    )


def _job(job_id: str, slo: float = 5.0,
         current_actual_ms: float = 1000.0,
         current_solo_ms: float = 400.0,
         active: tuple = (),
         declared: bool = True) -> JobLookaheadInput:
    """Helper: pre-canned active job with optional declared structure."""
    if declared:
        return JobLookaheadInput(
            job_id=job_id, slo=slo,
            active_calls=active,
            current_actual_elapsed_ms=current_actual_ms,
            current_solo_elapsed_ms=current_solo_ms,
            remaining_calls=2,
            remaining_stage_sequence=("LOCATE", "PLAN"),
            expected_input_lens=(2000, 2500),
            expected_output_lens=(400, 600),
            expected_cached_prefix_lens=(1800, 2300),
        )
    return JobLookaheadInput(
        job_id=job_id, slo=slo,
        active_calls=active,
        current_actual_elapsed_ms=current_actual_ms,
        current_solo_elapsed_ms=current_solo_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# decide_admission threshold logic
# ─────────────────────────────────────────────────────────────────────────────
class TestDecideAdmission(unittest.TestCase):
    def test_empty_predictions_admits(self):
        r = decide_admission({}, {}, threshold=0.2)
        self.assertTrue(r.admit)
        self.assertEqual(r.reason, REASON_OK)
        self.assertEqual(r.violation_ratio, 0.0)
        self.assertEqual(r.active_jobs_total, 0)

    def test_no_violation_admits(self):
        preds = {"a": 2.0, "b": 3.0, "c": 1.5}
        slos = {"a": 5.0, "b": 5.0, "c": 5.0}
        r = decide_admission(preds, slos, threshold=0.2)
        self.assertTrue(r.admit)
        self.assertEqual(r.violation_count, 0)

    def test_at_threshold_admits(self):
        # 1/5 = 0.2 not > 0.2 → admit
        preds = {"a": 6.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0}
        slos = {k: 5.0 for k in preds}
        r = decide_admission(preds, slos, threshold=0.2)
        self.assertTrue(r.admit)
        self.assertEqual(r.violation_count, 1)
        self.assertEqual(r.violation_ratio, 0.2)

    def test_over_threshold_rejects(self):
        # 2/5 = 0.4 > 0.2 → reject
        preds = {"a": 6.0, "b": 6.0, "c": 1.0, "d": 1.0, "e": 1.0}
        slos = {k: 5.0 for k in preds}
        r = decide_admission(preds, slos, threshold=0.2)
        self.assertFalse(r.admit)
        self.assertEqual(r.reason, REASON_HALO_ADMISSION_PREDICTED)
        self.assertEqual(r.violation_count, 2)

    def test_per_job_slo_used(self):
        # Different SLO per job — only the second one violates.
        preds = {"a": 7.0, "b": 3.0}
        slos = {"a": 10.0, "b": 2.0}
        r = decide_admission(preds, slos, threshold=0.2)
        self.assertEqual(r.violation_count, 1)  # b only
        self.assertFalse(r.admit)               # 0.5 > 0.2

    def test_missing_slo_treated_as_satisfied(self):
        # If a job's SLO is unknown to the caller, do not count it as a
        # violation (defensive fallback).
        preds = {"a": 100.0, "b": 100.0}
        r = decide_admission(preds, {}, threshold=0.2)
        self.assertTrue(r.admit)
        self.assertEqual(r.violation_count, 0)

    def test_result_payload_completeness(self):
        preds = {"a": 6.0, "b": 6.0, "c": 6.0, "d": 1.0}
        slos = {k: 5.0 for k in preds}
        r = decide_admission(preds, slos, threshold=0.2, mode="job",
                             horizon_sec=23.5)
        self.assertEqual(r.active_jobs_total, 4)
        self.assertEqual(r.threshold, 0.2)
        self.assertEqual(r.predicted_slowdowns, preds)
        self.assertEqual(r.mode, "job")
        self.assertEqual(r.horizon_sec, 23.5)


# ─────────────────────────────────────────────────────────────────────────────
# JobSlowdownAdmissionPredictor (mode "job")
# ─────────────────────────────────────────────────────────────────────────────
class TestJobSlowdownPredictor(unittest.TestCase):
    def test_empty_active_returns_empty(self):
        cm = _split_cost_model()
        p = JobSlowdownAdmissionPredictor(cm)
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=1000)
        self.assertEqual(p.predict([], new_job), {})

    # ── M-2 per-job-stretch tests (2026-05-15 design) ─────────────────

    def test_single_decode_job_uses_stretch_decode(self):
        cm = _split_cost_model()
        p = JobSlowdownAdmissionPredictor(cm)
        # current_VJS = 2000 / 800 = 2.5
        job = _job(
            "a", current_actual_ms=2000.0, current_solo_ms=800.0,
            active=(_decode_call("r1", kv_len=3000),),
            declared=False,
        )
        # New arrival joins decode batch with KV = prompt_len = 1000.
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=1000)
        preds = p.predict([job], new_job)
        # Manually compute stretch_decode using the same cost_model.
        base = cm.estimate_step_ms([], [3000])
        aug = cm.estimate_step_ms([], [3000, 1000])
        expected = 2.5 * (aug / base)
        self.assertAlmostEqual(preds["a"], expected, places=4)

    def test_single_prefill_job_uses_stretch_extend(self):
        """A single active job currently in prefill phase should have its
        predicted_VJS = current_VJS × stretch_extend."""
        cm = _split_cost_model()
        p = JobSlowdownAdmissionPredictor(cm)
        prefill_call = ActiveCallInfo(
            rid="r1", is_prefill=True,
            prompt_len=4096, prefix_len=0, decoded_tokens=0,
            kv_len_now=4096, elapsed_actual_ms=500.0,
        )
        job = _job(
            "a", current_actual_ms=2000.0, current_solo_ms=800.0,
            active=(prefill_call,), declared=False,
        )
        new_job = NewJobInput(
            job_id="new", slo=5.0,
            first_call_input_len=8000, first_call_prefix_len=7500,
        )  # n_new = 500
        preds = p.predict([job], new_job)
        # stretch_extend uses current prefill batch composition.
        base = cm.estimate_step_ms([(4096, 0)], [])
        aug = cm.estimate_step_ms([(4096, 0), (500, 7500)], [])
        expected = 2.5 * (aug / base)
        self.assertAlmostEqual(preds["a"], expected, places=4)

    def test_mixed_jobs_get_phase_specific_stretch(self):
        """User example: A prefill, B/C decode. Each gets its own phase's
        stretch — A receives stretch_extend, B/C receive stretch_decode."""
        cm = _split_cost_model()
        p = JobSlowdownAdmissionPredictor(cm)

        # A: prefill phase, current VJS = 2.0
        prefill_call_A = ActiveCallInfo(
            rid="rA", is_prefill=True,
            prompt_len=4096, prefix_len=0, decoded_tokens=0,
            kv_len_now=4096, elapsed_actual_ms=1000.0,
        )
        job_A = _job(
            "A", current_actual_ms=2000.0, current_solo_ms=1000.0,
            active=(prefill_call_A,), declared=False,
        )
        # B: decode phase, current VJS = 1.5
        job_B = _job(
            "B", current_actual_ms=1500.0, current_solo_ms=1000.0,
            active=(_decode_call("rB", kv_len=3000),),
            declared=False,
        )
        # C: decode phase, current VJS = 3.0
        job_C = _job(
            "C", current_actual_ms=3000.0, current_solo_ms=1000.0,
            active=(_decode_call("rC", kv_len=7000),),
            declared=False,
        )

        new_job = NewJobInput(
            job_id="new", slo=5.0,
            first_call_input_len=8000, first_call_prefix_len=7500,
        )
        preds = p.predict([job_A, job_B, job_C], new_job)

        base_ext = cm.estimate_step_ms([(4096, 0)], [])
        aug_ext = cm.estimate_step_ms([(4096, 0), (500, 7500)], [])
        stretch_extend = aug_ext / base_ext

        base_dec = cm.estimate_step_ms([], [3000, 7000])
        aug_dec = cm.estimate_step_ms([], [3000, 7000, 8000])
        stretch_decode = aug_dec / base_dec

        self.assertAlmostEqual(preds["A"], 2.0 * stretch_extend, places=4)
        self.assertAlmostEqual(preds["B"], 1.5 * stretch_decode, places=4)
        self.assertAlmostEqual(preds["C"], 3.0 * stretch_decode, places=4)

    def test_bigger_arrival_raises_predicted_slowdown(self):
        cm = _split_cost_model()
        p = JobSlowdownAdmissionPredictor(cm)
        job = _job(
            "a", current_actual_ms=2000.0, current_solo_ms=800.0,
            active=(_decode_call("r1", kv_len=3000),),
            declared=False,
        )
        small = NewJobInput(job_id="new", slo=5.0,
                            first_call_input_len=100,
                            first_call_prefix_len=50)
        big = NewJobInput(job_id="new", slo=5.0,
                          first_call_input_len=20000,
                          first_call_prefix_len=0)
        small_preds = p.predict([job], small)
        big_preds = p.predict([job], big)
        # Bigger new arrival → larger predicted_VJS for the active job.
        self.assertGreater(big_preds["a"], small_preds["a"])

    def test_zero_current_solo_falls_back_to_slo(self):
        """A freshly admitted active job (current_solo_ms = 0) should use
        SLO as its current_VJS — matches R1's initial slowdown_max."""
        cm = _split_cost_model()
        p = JobSlowdownAdmissionPredictor(cm)
        job = _job(
            "a", current_actual_ms=0.0, current_solo_ms=0.0,
            active=(),  # no active calls — treated as decode phase
            declared=False,
        )
        new_job = NewJobInput(job_id="new", slo=5.0, first_call_input_len=100)
        preds = p.predict([job], new_job)
        # With empty active_calls + empty decode batch, stretch_decode = 1.0,
        # and current_VJS falls back to SLO → predicted = 5.0 × 1.0 = 5.0.
        self.assertAlmostEqual(preds["a"], 5.0, places=4)

    def test_declared_fields_are_ignored(self):
        """Whether declared lengths are present or not, the prediction should
        match — the 2026-05-15 design no longer consults them."""
        cm = _split_cost_model()
        p = JobSlowdownAdmissionPredictor(cm)
        active = (_decode_call("r1", kv_len=3000),)
        job_with_declared = _job(
            "a", current_actual_ms=2000.0, current_solo_ms=800.0,
            active=active, declared=True,
        )
        job_without_declared = _job(
            "a", current_actual_ms=2000.0, current_solo_ms=800.0,
            active=active, declared=False,
        )
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=1000)
        with_decl = p.predict([job_with_declared], new_job)
        without_decl = p.predict([job_without_declared], new_job)
        self.assertAlmostEqual(with_decl["a"], without_decl["a"], places=8)


# ─────────────────────────────────────────────────────────────────────────────
# RequestSlowdownAdmissionPredictor (mode "request") — request-scoped baseline
# ─────────────────────────────────────────────────────────────────────────────
class TestRequestSlowdownPredictor(unittest.TestCase):
    def test_mode_name_is_request(self):
        p = RequestSlowdownAdmissionPredictor(_split_cost_model())
        self.assertEqual(p.mode_name, "request")

    def test_empty_active_returns_empty(self):
        p = RequestSlowdownAdmissionPredictor(_split_cost_model())
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=1000)
        self.assertEqual(p.predict([], new_job), {})

    def test_keys_are_rids_no_job_aggregation(self):
        """Two in-flight calls of the SAME job must produce two separate
        entries keyed by rid — the request-scoped predictor does not
        aggregate to the job."""
        cm = _split_cost_model()
        p = RequestSlowdownAdmissionPredictor(cm)
        job = _job(
            "jobA", active=(
                _decode_call("r1", kv_len=3000, elapsed_ms=2000.0,
                             solo_ms=800.0),
                _decode_call("r2", kv_len=5000, elapsed_ms=1000.0,
                             solo_ms=800.0),
            ),
            declared=False,
        )
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=1000)
        preds = p.predict([job], new_job)
        self.assertEqual(set(preds.keys()), {"r1", "r2"})
        self.assertNotIn("jobA", preds)

    def test_per_request_decode_slowdown(self):
        """predicted_request_slowdown = (elapsed / solo) × stretch_decode,
        computed per request."""
        cm = _split_cost_model()
        p = RequestSlowdownAdmissionPredictor(cm)
        # r1 current slowdown = 2000/800 = 2.5; r2 = 1500/1000 = 1.5
        job = _job(
            "jobA", active=(
                _decode_call("r1", kv_len=3000, elapsed_ms=2000.0,
                             solo_ms=800.0),
                _decode_call("r2", kv_len=7000, elapsed_ms=1500.0,
                             solo_ms=1000.0),
            ),
            declared=False,
        )
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=8000)
        preds = p.predict([job], new_job)
        base = cm.estimate_step_ms([], [3000, 7000])
        aug = cm.estimate_step_ms([], [3000, 7000, 8000])
        stretch_decode = aug / base
        self.assertAlmostEqual(preds["r1"], 2.5 * stretch_decode, places=4)
        self.assertAlmostEqual(preds["r2"], 1.5 * stretch_decode, places=4)

    def test_each_call_scored_by_own_phase(self):
        """A prefill-phase call gets stretch_extend; a decode-phase call
        gets stretch_decode — even within the same job."""
        cm = _split_cost_model()
        p = RequestSlowdownAdmissionPredictor(cm)
        prefill_call = ActiveCallInfo(
            rid="rP", is_prefill=True, prompt_len=4096, prefix_len=0,
            decoded_tokens=0, kv_len_now=4096,
            elapsed_actual_ms=1000.0, solo_elapsed_ms=500.0,
        )
        decode_call = _decode_call("rD", kv_len=3000, elapsed_ms=1000.0,
                                   solo_ms=500.0)
        job = _job("jobA", active=(prefill_call, decode_call),
                   declared=False)
        new_job = NewJobInput(
            job_id="new", slo=5.0,
            first_call_input_len=8000, first_call_prefix_len=7500,
        )
        preds = p.predict([job], new_job)
        base_ext = cm.estimate_step_ms([(4096, 0)], [])
        aug_ext = cm.estimate_step_ms([(4096, 0), (500, 7500)], [])
        stretch_extend = aug_ext / base_ext
        base_dec = cm.estimate_step_ms([], [3000])
        aug_dec = cm.estimate_step_ms([], [3000, 8000])
        stretch_decode = aug_dec / base_dec
        # Both calls have current slowdown 1000/500 = 2.0.
        self.assertAlmostEqual(preds["rP"], 2.0 * stretch_extend, places=4)
        self.assertAlmostEqual(preds["rD"], 2.0 * stretch_decode, places=4)

    def test_zero_solo_falls_back_to_slo(self):
        """A call with no accumulated solo time uses the job SLO as its
        current slowdown — matches R1's initial convention."""
        cm = _split_cost_model()
        p = RequestSlowdownAdmissionPredictor(cm)
        job = _job(
            "jobA", slo=4.0,
            active=(_decode_call("r1", kv_len=3000, elapsed_ms=999.0,
                                 solo_ms=0.0),),
            declared=False,
        )
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=1000)
        preds = p.predict([job], new_job)
        base = cm.estimate_step_ms([], [3000])
        aug = cm.estimate_step_ms([], [3000, 1000])
        stretch_decode = aug / base
        self.assertAlmostEqual(preds["r1"], 4.0 * stretch_decode, places=4)


# ─────────────────────────────────────────────────────────────────────────────
# Frozen-dataclass sanity
# ─────────────────────────────────────────────────────────────────────────────
class TestFrozenInputs(unittest.TestCase):
    def test_active_call_info_is_frozen(self):
        info = ActiveCallInfo(
            rid="r", is_prefill=False, prompt_len=100, prefix_len=0,
            decoded_tokens=10, kv_len_now=110, elapsed_actual_ms=100.0,
        )
        with self.assertRaises(Exception):
            info.prompt_len = 200  # type: ignore[misc]

    def test_admission_decision_result_is_frozen(self):
        r = AdmissionDecisionResult(
            admit=True, reason=REASON_OK,
            violation_ratio=0.0, violation_count=0,
            active_jobs_total=0, threshold=0.2,
            predicted_slowdowns={},
        )
        with self.assertRaises(Exception):
            r.admit = False  # type: ignore[misc]


class TestLookaheadPredictor(unittest.TestCase):
    """Level 2 — SLO-driven lookahead.

    Black-box checks: we don't verify the exact slowdown number (the
    simulator's accuracy depends on horizon + slice + cost-model
    extrapolation), but we *do* verify that:
      * the predictor returns a key for every active job
      * empty active set returns empty
      * indeterminate declared length → predictor still runs, no crash
      * larger new arrival → predicted slowdown for existing jobs is
        non-decreasing (more load can't help anyone)
      * a manual config horizon caps the simulation
    """

    def _make(self, horizon_sec: float = 0.0) -> LookaheadAdmissionPredictor:
        return LookaheadAdmissionPredictor(_split_cost_model(), horizon_sec)

    def test_empty_active_returns_empty(self):
        p = self._make()
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=1000)
        self.assertEqual(p.predict([], new_job), {})

    def test_returns_one_entry_per_active_job(self):
        p = self._make(horizon_sec=5.0)  # tight horizon for fast test
        # Two declared active jobs.
        jobs = [_job("a", active=(_decode_call("r1", 4000),)),
                _job("b", active=(_decode_call("r2", 6000),))]
        new_job = NewJobInput(
            job_id="new", slo=5.0,
            first_call_input_len=2000, first_call_prefix_len=1500,
            total_calls=2,
            stage_sequence=("LOCATE", "PLAN"),
            expected_input_lens=(2000, 2500),
            expected_output_lens=(300, 400),
        )
        preds = p.predict(jobs, new_job)
        self.assertEqual(set(preds.keys()), {"a", "b"})
        # New job is excluded (admission protects existing jobs only).
        self.assertNotIn("new", preds)

    def test_undeclared_active_still_runs(self):
        p = self._make(horizon_sec=5.0)
        jobs = [_job("undeclared", active=(_decode_call("r1", 4000),),
                     declared=False)]
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=1000)
        preds = p.predict(jobs, new_job)
        self.assertIn("undeclared", preds)
        self.assertGreater(preds["undeclared"], 0)

    def test_bigger_new_load_does_not_decrease_predicted_slowdown(self):
        """Adding more work cannot *help* existing jobs."""
        p = self._make(horizon_sec=3.0)
        jobs = [_job("a", active=(_decode_call("r1", 4000),))]

        small = NewJobInput(
            job_id="new", slo=5.0,
            first_call_input_len=100, first_call_prefix_len=50,
            first_call_expected_output_len=10,
        )
        big = NewJobInput(
            job_id="new", slo=5.0,
            first_call_input_len=8000, first_call_prefix_len=0,
            first_call_expected_output_len=400,
        )
        small_preds = p.predict(jobs, small)
        big_preds = p.predict(jobs, big)
        # Big arrival should result in a >= predicted slowdown for "a".
        self.assertGreaterEqual(big_preds["a"], small_preds["a"] - 1e-6)

    def test_horizon_cap_changes_prediction(self):
        """Different horizons should produce different predictions when
        the workload has time-varying load (prefill peak then recovery).

        Note: it is NOT monotone — a short horizon that lands inside the
        new arrival's prefill slice can yield a *higher* predicted slowdown
        than a long horizon that averages over the post-prefill recovery
        period. We only check that the two values differ meaningfully —
        confirming the horizon argument is wired through."""
        p_short = self._make(horizon_sec=0.5)
        p_long = self._make(horizon_sec=10.0)
        jobs = [_job(
            "a", current_actual_ms=0.0, current_solo_ms=0.0,
            active=(_decode_call("r1", 4000, decoded=0, elapsed_ms=0.0),),
        )]
        new_job = NewJobInput(
            job_id="new", slo=5.0,
            first_call_input_len=100, first_call_prefix_len=0,
            first_call_expected_output_len=10000,
        )
        short = p_short.predict(jobs, new_job)
        long = p_long.predict(jobs, new_job)
        self.assertNotAlmostEqual(short["a"], long["a"], places=2)
        # Both should be finite positives.
        self.assertGreater(short["a"], 0.0)
        self.assertGreater(long["a"], 0.0)

    def test_slo_driven_horizon_computed_when_zero_config(self):
        """horizon_sec=0 → derive from active jobs' SLO budget. We exercise
        the path by simply running the predictor; the resolved horizon is
        not directly observable but the result must be finite."""
        p = self._make(horizon_sec=0.0)
        jobs = [_job("a", slo=3.0)]  # declared remaining → horizon derivable
        new_job = NewJobInput(
            job_id="new", slo=5.0,
            first_call_input_len=1000, first_call_prefix_len=500,
            first_call_expected_output_len=200,
        )
        preds = p.predict(jobs, new_job)
        self.assertIn("a", preds)
        import math
        self.assertTrue(math.isfinite(preds["a"]))

    def test_mode_name_is_level2(self):
        self.assertEqual(self._make().mode_name, "level2")


if __name__ == "__main__":
    unittest.main()
