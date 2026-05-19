"""Unit tests for managers/halo/admission_control/vss_predictor.py.

Covers the memoryless virtual-server-slowdown kernel used by VssPolicy:
- _bounded_extend_step — chunk-bounds the EXTEND batch (결함 A)
- _split_current_batch_features — prefill / decode split
- _augmented_step_times — the shared VSS numerator
- RequestSlowdownAdmissionPredictor.predict — per-request predicted VSS
- decide_admission — violation-ratio threshold logic
"""

import unittest

from sglang.srt.managers.halo.admission_control.vss_predictor import (
    REASON_HALO_ADMISSION_PREDICTED,
    REASON_OK,
    ActiveCallInfo,
    ArrivalInput,
    BatchSnapshot,
    RequestSlowdownAdmissionPredictor,
    _augmented_step_times,
    _bounded_extend_step,
    _split_current_batch_features,
    decide_admission,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="stage-a-test-cpu")


class _FakeStep:
    """HaloStepCostModel-like: step time grows with new-token + KV load."""

    def estimate_step_ms(self, prefill_infos, decode_infos):
        s = 5.0
        for n, _r in prefill_infos:
            s += 0.05 * n
        for kv in decode_infos:
            s += 0.002 * kv
        return s


def _prefill_call(rid, prompt_len, prefix_len=0):
    return ActiveCallInfo(
        rid=rid,
        is_prefill=True,
        prompt_len=prompt_len,
        prefix_len=prefix_len,
        decoded_tokens=0,
        kv_len_now=prompt_len,
    )


def _decode_call(rid, kv_len):
    return ActiveCallInfo(
        rid=rid,
        is_prefill=False,
        prompt_len=kv_len,
        prefix_len=0,
        decoded_tokens=8,
        kv_len_now=kv_len,
    )


# ───────────────────────────────────────────────────────────────────────────
class TestBoundedExtendStep(unittest.TestCase):
    def test_no_cap_when_size_none(self):
        infos = [(5000, 0), (3000, 0)]
        self.assertEqual(_bounded_extend_step(infos, None), infos)

    def test_caps_to_chunked_prefill_size(self):
        # budget 4096: first entry takes 4096, rest dropped.
        out = _bounded_extend_step([(5000, 0), (3000, 0)], 4096)
        self.assertEqual(out, [(4096, 0)])

    def test_spreads_budget_across_entries(self):
        out = _bounded_extend_step([(1000, 0), (1000, 0), (5000, 0)], 2500)
        self.assertEqual(out, [(1000, 0), (1000, 0), (500, 0)])


class TestSplitBatchFeatures(unittest.TestCase):
    def test_split(self):
        calls = [_prefill_call("p", 1000, 100), _decode_call("d", 2000)]
        prefill, decode = _split_current_batch_features(calls)
        self.assertEqual(prefill, [(900, 100)])  # n = prompt - prefix
        self.assertEqual(decode, [2000])


class TestAugmentedStepTimes(unittest.TestCase):
    def test_arrival_adds_to_both_halves(self):
        cm = _FakeStep()
        calls = [_decode_call("d", 1000)]
        arrival = ArrivalInput(
            slo=5.0, first_call_input_len=2000, first_call_prefix_len=0
        )
        s_extend, s_decode = _augmented_step_times(cm, calls, arrival)
        # extend sees the 2000-token arrival; decode sees 1000 (running) + 2000.
        self.assertGreater(s_extend, 5.0)
        self.assertGreater(s_decode, 5.0)


class TestRequestPredictor(unittest.TestCase):
    def test_empty_batch_returns_empty(self):
        pred = RequestSlowdownAdmissionPredictor(_FakeStep())
        out = pred.predict(
            BatchSnapshot(slo=5.0, active_calls=()),
            ArrivalInput(slo=5.0, first_call_input_len=1000),
        )
        self.assertEqual(out, {})

    def test_scores_every_inflight_request(self):
        pred = RequestSlowdownAdmissionPredictor(_FakeStep())
        batch = BatchSnapshot(
            slo=5.0,
            active_calls=(_decode_call("a", 1000), _decode_call("b", 3000)),
        )
        out = pred.predict(batch, ArrivalInput(slo=5.0, first_call_input_len=500))
        self.assertEqual(set(out), {"a", "b"})
        # every predicted VSS is a finite positive ratio.
        for v in out.values():
            self.assertGreater(v, 0.0)


class TestDecideAdmission(unittest.TestCase):
    def test_empty_admits(self):
        r = decide_admission({}, {}, threshold=0.2)
        self.assertTrue(r.admit)
        self.assertEqual(r.reason, REASON_OK)

    def test_reject_when_violation_ratio_over_threshold(self):
        # 3/4 requests over SLO → ratio 0.75 > 0.2 → reject.
        pred = {"a": 9.0, "b": 9.0, "c": 9.0, "d": 1.0}
        slos = {"a": 5.0, "b": 5.0, "c": 5.0, "d": 5.0}
        r = decide_admission(pred, slos, threshold=0.2)
        self.assertFalse(r.admit)
        self.assertEqual(r.reason, REASON_HALO_ADMISSION_PREDICTED)
        self.assertEqual(r.violation_count, 3)
        self.assertAlmostEqual(r.violation_ratio, 0.75)

    def test_admit_at_or_below_threshold(self):
        # 1/5 over → ratio 0.2, not > 0.2 → admit.
        pred = {"a": 9.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0}
        slos = {k: 5.0 for k in pred}
        self.assertTrue(decide_admission(pred, slos, threshold=0.2).admit)

    def test_missing_slo_treated_as_satisfied(self):
        # "b" has no SLO → never a violation; only "a" counts → 1/2 = 0.5.
        r = decide_admission({"a": 9.0, "b": 9.0}, {"a": 5.0}, threshold=0.2)
        self.assertEqual(r.violation_count, 1)


if __name__ == "__main__":
    unittest.main()
