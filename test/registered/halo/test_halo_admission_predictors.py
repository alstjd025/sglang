"""Unit tests for managers/halo/admission_decision.py (Phase 2).

Design + rationale: ms_dev/halo_dev/admission_design.md.

Coverage:
- decide_admission threshold logic (admit / reject / boundaries)
- SnapshotAdmissionPredictor (Level 0):
    * empty active set
    * declared insufficient → falls back to current observed slowdown
    * augmented step time stretches predicted slowdown
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
    LookaheadAdmissionPredictor,
    NewJobInput,
    SnapshotAdmissionPredictor,
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
                 elapsed_ms: float = 1000.0) -> ActiveCallInfo:
    return ActiveCallInfo(
        rid=rid,
        is_prefill=False,
        prompt_len=kv_len - decoded,
        prefix_len=0,
        decoded_tokens=decoded,
        kv_len_now=kv_len,
        elapsed_actual_ms=elapsed_ms,
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
        r = decide_admission(preds, slos, threshold=0.2, mode="level0",
                             horizon_sec=23.5)
        self.assertEqual(r.active_jobs_total, 4)
        self.assertEqual(r.threshold, 0.2)
        self.assertEqual(r.predicted_slowdowns, preds)
        self.assertEqual(r.mode, "level0")
        self.assertEqual(r.horizon_sec, 23.5)


# ─────────────────────────────────────────────────────────────────────────────
# SnapshotAdmissionPredictor (Level 0)
# ─────────────────────────────────────────────────────────────────────────────
class TestSnapshotPredictor(unittest.TestCase):
    def test_empty_active_returns_empty(self):
        cm = _split_cost_model()
        p = SnapshotAdmissionPredictor(cm)
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=1000)
        self.assertEqual(p.predict([], new_job), {})

    def test_declared_insufficient_falls_back_to_current_ratio(self):
        cm = _split_cost_model()
        p = SnapshotAdmissionPredictor(cm)
        job = _job(
            "a", current_actual_ms=2000.0, current_solo_ms=800.0,
            active=(_decode_call("r1", kv_len=3000),),
            declared=False,
        )
        new_job = NewJobInput(job_id="new", slo=5.0,
                              first_call_input_len=1000)
        preds = p.predict([job], new_job)
        self.assertIn("a", preds)
        # current ratio = 2000 / 800 = 2.5
        self.assertAlmostEqual(preds["a"], 2.5, places=4)

    def test_declared_present_uses_stretch(self):
        cm = _split_cost_model()
        p = SnapshotAdmissionPredictor(cm)
        job = _job(
            "a", current_actual_ms=2000.0, current_solo_ms=800.0,
            active=(_decode_call("r1", kv_len=3000),),
            declared=True,
        )
        new_job = NewJobInput(
            job_id="new", slo=5.0,
            first_call_input_len=4000,
            first_call_prefix_len=3500,
        )
        preds = p.predict([job], new_job)
        self.assertIn("a", preds)
        self.assertGreater(preds["a"], 0)
        # With a single decoder and a new prefill of 500 uncached tokens, the
        # step time grows — predicted slowdown should be > the current
        # observed ratio (2.5).
        self.assertGreater(preds["a"], 2.5)

    def test_adding_more_load_increases_predicted_slowdown(self):
        """Two-active-job scenario: predicted slowdown when admitting a
        bigger prefill should exceed admitting a tiny prefill."""
        cm = _split_cost_model()
        p = SnapshotAdmissionPredictor(cm)
        job_a = _job(
            "a", current_actual_ms=2000.0, current_solo_ms=800.0,
            active=(_decode_call("r1", kv_len=3000),),
        )
        job_b = _job(
            "b", current_actual_ms=1500.0, current_solo_ms=600.0,
            active=(_decode_call("r2", kv_len=5000),),
        )
        small = NewJobInput(job_id="new", slo=5.0,
                            first_call_input_len=100,
                            first_call_prefix_len=50)
        big = NewJobInput(job_id="new", slo=5.0,
                          first_call_input_len=20000,
                          first_call_prefix_len=0)
        small_preds = p.predict([job_a, job_b], small)
        big_preds = p.predict([job_a, job_b], big)
        # Both jobs should see a larger predicted slowdown under the big
        # new arrival.
        self.assertGreater(big_preds["a"], small_preds["a"])
        self.assertGreater(big_preds["b"], small_preds["b"])

    def test_zero_current_solo_no_crash(self):
        """A freshly admitted active job (current_solo_ms = 0) should be
        handled without division by zero."""
        cm = _split_cost_model()
        p = SnapshotAdmissionPredictor(cm)
        job = _job(
            "a", current_actual_ms=0.0, current_solo_ms=0.0,
            active=(),
            declared=False,
        )
        new_job = NewJobInput(job_id="new", slo=5.0, first_call_input_len=100)
        preds = p.predict([job], new_job)
        # Initial fallback in the helper is `slo` value.
        self.assertEqual(preds["a"], 5.0)


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
