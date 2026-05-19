"""Unit + integration tests for admission_control module.

Step 1 covers PrefillCostModel, TBTCostModel, TBTEwmaTracker only. Controller
and integration tests are added in later steps.

See test/registered/admission/CLAUDE.md and
python/sglang/srt/managers/halo/admission_control/CLAUDE.md.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sglang.srt.managers.halo.admission_control.controller import (
    REASON_ADMIT,
    REASON_DISABLED,
    REASON_TBT_PREDICTED,
    REASON_TBT_RATIO,
    REASON_TBT_REACTIVE,
    REASON_TTFT_PREDICTED,
    REASON_TTFT_RATIO,
    AdmissionConfig,
    AdmissionController,
    AdmissionDecision,
    SchedulerSnapshot,
)
from sglang.srt.managers.halo.admission_control.cost_model import (
    CostModelLoadError,
    PrefillCostModel,
    TBTCostModel,
    try_load_prefill_cost_model,
    try_load_tbt_cost_model,
)
from sglang.srt.managers.halo.admission_control.decision_log import DecisionLogger
from sglang.srt.managers.halo.admission_control.metrics import AdmissionMetrics
from sglang.srt.managers.halo.admission_control.tbt_tracker import TBTEwmaTracker
from sglang.test.ci.ci_register import register_cpu_ci

# Step 1 is pure-Python; CPU suite is enough.
register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestPrefillCostModel(unittest.TestCase):
    def test_estimate_quadratic(self):
        m = PrefillCostModel(alpha=2.0, beta=3.0, gamma=5.0)
        # d = max(0, 10 - 4) = 6 → 2*36 + 3*6 + 5 + 0*4 = 95  (delta defaults to 0)
        self.assertAlmostEqual(m.estimate_ms(10, 4), 95.0)

    def test_estimate_with_delta(self):
        # T = 1*d² + 2*d + 10 + 0.05*p
        m = PrefillCostModel(alpha=1.0, beta=2.0, gamma=10.0, delta=0.05)
        # n=100, p=80 → d=20 → 400 + 40 + 10 + 4.0 = 454.0
        self.assertAlmostEqual(m.estimate_ms(100, 80), 454.0)
        # p=0 → no delta contribution
        self.assertAlmostEqual(m.estimate_ms(20, 0), 1 * 400 + 2 * 20 + 10)

    def test_back_compat_load_no_delta(self):
        # Pre-delta JSONs must still load (delta defaults to 0).
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.json"
            path.write_text(json.dumps({"alpha": 1.0, "beta": 2.0, "gamma": 3.0}))
            m = PrefillCostModel.from_json(path)
            self.assertEqual(m.delta, 0.0)
            self.assertAlmostEqual(m.estimate_ms(10, 5), 1 * 25 + 2 * 5 + 3)

    def test_estimate_clamps_negative_d(self):
        # prefix longer than prompt should not produce negative time.
        m = PrefillCostModel(alpha=1.0, beta=1.0, gamma=10.0)
        self.assertEqual(m.estimate_ms(5, 100), 10.0)  # d clamped to 0 → just gamma

    def test_estimate_zero_prompt(self):
        m = PrefillCostModel(alpha=1.0, beta=2.0, gamma=7.0)
        self.assertEqual(m.estimate_ms(0, 0), 7.0)

    def test_json_roundtrip(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "prefill.json"
            original = PrefillCostModel(
                alpha=1.5e-5,
                beta=0.045,
                gamma=12.3,
                metadata={"model": "test", "samples": 42},
            )
            original.to_json(path)
            loaded = PrefillCostModel.from_json(path)
            self.assertAlmostEqual(loaded.alpha, original.alpha)
            self.assertAlmostEqual(loaded.beta, original.beta)
            self.assertAlmostEqual(loaded.gamma, original.gamma)
            self.assertEqual(loaded.metadata.get("model"), "test")

    def test_from_json_missing_file(self):
        with self.assertRaises(CostModelLoadError):
            PrefillCostModel.from_json("/no/such/path.json")

    def test_from_json_malformed_payload(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("{ not valid json")
            with self.assertRaises(CostModelLoadError):
                PrefillCostModel.from_json(path)

    def test_from_json_missing_keys(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            path.write_text(json.dumps({"alpha": 1.0, "beta": 2.0}))  # no gamma
            with self.assertRaises(CostModelLoadError):
                PrefillCostModel.from_json(path)

    def test_from_json_non_object(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "list.json"
            path.write_text(json.dumps([1, 2, 3]))
            with self.assertRaises(CostModelLoadError):
                PrefillCostModel.from_json(path)


class TestTBTCostModel(unittest.TestCase):
    def test_estimate_linear(self):
        m = TBTCostModel(a=10.0, b=2.0, c=0.001)
        # 10 + 2*8 + 0.001*5000 = 10 + 16 + 5 = 31
        self.assertAlmostEqual(m.estimate_ms(8, 5000), 31.0)

    def test_estimate_zero_batch(self):
        m = TBTCostModel(a=5.0, b=1.0, c=0.0)
        self.assertEqual(m.estimate_ms(0, 0), 5.0)

    def test_json_roundtrip(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "tbt.json"
            TBTCostModel(a=35.0, b=1.5, c=0.0008).to_json(path)
            loaded = TBTCostModel.from_json(path)
            self.assertAlmostEqual(loaded.a, 35.0)
            self.assertAlmostEqual(loaded.b, 1.5)
            self.assertAlmostEqual(loaded.c, 0.0008)


class TestLenientLoaders(unittest.TestCase):
    """try_load_* return None and log on failure (not raise)."""

    def test_none_path_returns_none(self):
        self.assertIsNone(try_load_prefill_cost_model(None))
        self.assertIsNone(try_load_prefill_cost_model(""))
        self.assertIsNone(try_load_tbt_cost_model(None))
        self.assertIsNone(try_load_tbt_cost_model(""))

    def test_missing_file_returns_none(self):
        self.assertIsNone(try_load_prefill_cost_model("/no/such/file.json"))
        self.assertIsNone(try_load_tbt_cost_model("/no/such/file.json"))

    def test_malformed_returns_none(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("not json")
            self.assertIsNone(try_load_prefill_cost_model(str(path)))
            self.assertIsNone(try_load_tbt_cost_model(str(path)))

    def test_valid_returns_model(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.json"
            path.write_text(json.dumps({"alpha": 1.0, "beta": 2.0, "gamma": 3.0}))
            m = try_load_prefill_cost_model(str(path))
            self.assertIsNotNone(m)
            self.assertEqual(m.alpha, 1.0)


class TestTBTEwmaTracker(unittest.TestCase):
    def test_initial_state(self):
        t = TBTEwmaTracker(alpha=0.1, warm_up_steps=5)
        self.assertEqual(t.get(), 0.0)
        self.assertFalse(t.is_warm())
        self.assertEqual(t.sample_count(), 0)

    def test_first_update_seeds_value(self):
        t = TBTEwmaTracker(alpha=0.1, warm_up_steps=5)
        t.update(100.0)
        # First sample should be the seed, not blended with 0.
        self.assertAlmostEqual(t.get(), 100.0)
        self.assertEqual(t.sample_count(), 1)
        self.assertFalse(t.is_warm())

    def test_smoothing(self):
        t = TBTEwmaTracker(alpha=0.5, warm_up_steps=0)
        t.update(100.0)
        # 0.5*200 + 0.5*100 = 150
        t.update(200.0)
        self.assertAlmostEqual(t.get(), 150.0)
        # 0.5*0 + 0.5*150 = 75
        t.update(0.0)
        self.assertAlmostEqual(t.get(), 75.0)

    def test_warm_up_threshold(self):
        t = TBTEwmaTracker(alpha=0.5, warm_up_steps=3)
        self.assertFalse(t.is_warm())
        t.update(10.0)
        t.update(10.0)
        self.assertFalse(t.is_warm())
        t.update(10.0)
        self.assertTrue(t.is_warm())

    def test_warm_up_zero_means_immediately_warm(self):
        t = TBTEwmaTracker(alpha=0.1, warm_up_steps=0)
        # No samples yet, but warm_up=0 means is_warm = (n >= 0) = True.
        # That is intentional: caller can disable warm-up entirely.
        self.assertTrue(t.is_warm())

    def test_negative_latency_ignored(self):
        t = TBTEwmaTracker(alpha=0.5, warm_up_steps=0)
        t.update(100.0)
        t.update(-50.0)  # garbage
        self.assertAlmostEqual(t.get(), 100.0)
        self.assertEqual(t.sample_count(), 1)

    def test_reset(self):
        t = TBTEwmaTracker(alpha=0.5, warm_up_steps=2)
        t.update(100.0)
        t.update(200.0)
        self.assertTrue(t.is_warm())
        t.reset()
        self.assertEqual(t.get(), 0.0)
        self.assertEqual(t.sample_count(), 0)
        self.assertFalse(t.is_warm())

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            TBTEwmaTracker(alpha=0.0)
        with self.assertRaises(ValueError):
            TBTEwmaTracker(alpha=1.5)
        with self.assertRaises(ValueError):
            TBTEwmaTracker(alpha=-0.1)

    def test_invalid_warm_up(self):
        with self.assertRaises(ValueError):
            TBTEwmaTracker(alpha=0.1, warm_up_steps=-1)

    def test_alpha_one_means_no_smoothing(self):
        # alpha=1 → always equals latest sample.
        t = TBTEwmaTracker(alpha=1.0, warm_up_steps=0)
        t.update(50.0)
        t.update(123.0)
        self.assertAlmostEqual(t.get(), 123.0)


# ---------------------------------------------------------------------------
# AdmissionController decision matrix
# ---------------------------------------------------------------------------


def _snapshot(
    queue_ms: float = 0.0,
    bs: int = 0,
    kv: int = 0,
    null: bool = True,
) -> SchedulerSnapshot:
    return SchedulerSnapshot(
        waiting_queue_predicted_prefill_ms=queue_ms,
        running_batch_size=bs,
        running_batch_total_kv_tokens=kv,
        disaggregation_mode_is_null=null,
    )


def _prefill(alpha: float = 0.0, beta: float = 1.0, gamma: float = 0.0) -> PrefillCostModel:
    """Default linear: T = beta·d + gamma  (1ms per missed token)."""
    return PrefillCostModel(alpha=alpha, beta=beta, gamma=gamma)


def _tbt(a: float = 0.0, b: float = 1.0, c: float = 0.0) -> TBTCostModel:
    """Default: TBT = a + b·bs + c·kv  (1ms per request in batch)."""
    return TBTCostModel(a=a, b=b, c=c)


class TestAdmissionControllerDisabled(unittest.TestCase):
    """When neither SLO is set, decide() short-circuits to ADMIT(DISABLED)."""

    def test_no_slos_admits(self):
        ctrl = AdmissionController(AdmissionConfig())
        d = ctrl.decide(prompt_len=1000, prefix_match_len=0, snapshot=_snapshot())
        self.assertTrue(d.admit)
        self.assertEqual(d.reason, REASON_DISABLED)
        self.assertFalse(ctrl.is_active())

    def test_zero_slos_treated_as_disabled(self):
        ctrl = AdmissionController(AdmissionConfig(ttft_slo_ms=0, tbt_slo_ms=0))
        d = ctrl.decide(prompt_len=10, prefix_match_len=0, snapshot=_snapshot())
        self.assertTrue(d.admit)
        self.assertEqual(d.reason, REASON_DISABLED)


class TestAdmissionControllerStage1(unittest.TestCase):
    """TTFT predicted policy."""

    def test_ttft_under_slo_admits(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=1000),
            prefill_cost=_prefill(),  # T = 1ms per token
        )
        d = ctrl.decide(prompt_len=500, prefix_match_len=0, snapshot=_snapshot())
        self.assertTrue(d.admit)
        self.assertEqual(d.reason, REASON_ADMIT)
        self.assertAlmostEqual(d.predicted_ttft_ms, 500.0)
        self.assertAlmostEqual(d.predicted_prefill_ms, 500.0)

    def test_ttft_over_slo_rejects(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=1000),
            prefill_cost=_prefill(),
        )
        d = ctrl.decide(prompt_len=2000, prefix_match_len=0, snapshot=_snapshot())
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_TTFT_PREDICTED)
        self.assertAlmostEqual(d.predicted_ttft_ms, 2000.0)
        self.assertAlmostEqual(d.predicted_prefill_ms, 2000.0)

    def test_queue_pushes_over_slo(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=1000),
            prefill_cost=_prefill(),
        )
        # 800ms in queue + 500ms self = 1300ms > 1000ms slo
        d = ctrl.decide(
            prompt_len=500,
            prefix_match_len=0,
            snapshot=_snapshot(queue_ms=800.0),
        )
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_TTFT_PREDICTED)
        self.assertAlmostEqual(d.queue_predicted_ms, 800.0)

    def test_prefix_match_reduces_ttft(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=1000),
            prefill_cost=_prefill(),
        )
        # 2000-tok prompt would normally be 2000ms; with 1500-tok cache hit → 500ms
        d = ctrl.decide(
            prompt_len=2000, prefix_match_len=1500, snapshot=_snapshot()
        )
        self.assertTrue(d.admit)
        self.assertAlmostEqual(d.predicted_prefill_ms, 500.0)

    def test_no_cost_model_means_stage1_disabled(self):
        # Stage 1 is configured but no cost model loaded — controller silently
        # skips Stage 1 (lenient mode, server still runs).
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=1),  # absurdly tight
            prefill_cost=None,
        )
        d = ctrl.decide(prompt_len=99999, prefix_match_len=0, snapshot=_snapshot())
        self.assertTrue(d.admit)
        self.assertEqual(d.reason, REASON_ADMIT)


class TestAdmissionControllerStage2(unittest.TestCase):
    """TBT predicted (current-batch approximation)."""

    def test_tbt_under_slo_admits(self):
        ctrl = AdmissionController(
            AdmissionConfig(tbt_slo_ms=200),
            tbt_cost=_tbt(a=10.0, b=5.0, c=0.0),  # 10 + 5·bs
        )
        d = ctrl.decide(
            prompt_len=10,
            prefix_match_len=0,
            snapshot=_snapshot(bs=10, kv=1000),
        )
        # bs after = 11 → 10 + 55 = 65ms < 200ms
        self.assertTrue(d.admit)
        self.assertAlmostEqual(d.predicted_tbt_ms, 65.0)

    def test_tbt_over_slo_rejects(self):
        ctrl = AdmissionController(
            AdmissionConfig(tbt_slo_ms=200),
            tbt_cost=_tbt(a=0.0, b=10.0, c=0.0),
        )
        d = ctrl.decide(
            prompt_len=10, prefix_match_len=0, snapshot=_snapshot(bs=30)
        )
        # bs after = 31 → 310ms > 200ms slo
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_TBT_PREDICTED)

    def test_kv_term_can_trip_slo(self):
        ctrl = AdmissionController(
            AdmissionConfig(tbt_slo_ms=200),
            tbt_cost=_tbt(a=0.0, b=0.0, c=0.01),  # 0.01ms per kv token
        )
        d = ctrl.decide(
            prompt_len=5000,
            prefix_match_len=0,
            snapshot=_snapshot(bs=0, kv=20000),
        )
        # kv after = 25000 → 250ms > 200ms
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_TBT_PREDICTED)


class TestAdmissionControllerStage3(unittest.TestCase):
    """TBT reactive EWMA safety net."""

    def test_cold_ewma_does_not_trip(self):
        # Tracker not warm yet → Stage 3 skipped even if value is high.
        tracker = TBTEwmaTracker(alpha=0.5, warm_up_steps=10)
        tracker.update(9999.0)  # one sample, not warm
        ctrl = AdmissionController(
            AdmissionConfig(tbt_slo_ms=200, tbt_reactive_ratio=0.9),
            tbt_tracker=tracker,
        )
        d = ctrl.decide(prompt_len=1, prefix_match_len=0, snapshot=_snapshot())
        self.assertTrue(d.admit)

    def test_warm_ewma_over_threshold_rejects(self):
        tracker = TBTEwmaTracker(alpha=1.0, warm_up_steps=1)  # snap to latest
        tracker.update(250.0)  # warm, ewma = 250
        ctrl = AdmissionController(
            AdmissionConfig(tbt_slo_ms=200, tbt_reactive_ratio=0.9),
            tbt_tracker=tracker,
        )
        # threshold = 200 * 0.9 = 180; ewma=250 > 180 → reject
        d = ctrl.decide(prompt_len=1, prefix_match_len=0, snapshot=_snapshot())
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_TBT_REACTIVE)
        self.assertAlmostEqual(d.tbt_ewma_ms, 250.0)

    def test_warm_ewma_under_threshold_admits(self):
        tracker = TBTEwmaTracker(alpha=1.0, warm_up_steps=1)
        tracker.update(150.0)
        ctrl = AdmissionController(
            AdmissionConfig(tbt_slo_ms=200, tbt_reactive_ratio=0.9),
            tbt_tracker=tracker,
        )
        # threshold = 180; ewma=150 < 180 → admit
        d = ctrl.decide(prompt_len=1, prefix_match_len=0, snapshot=_snapshot())
        self.assertTrue(d.admit)
        self.assertAlmostEqual(d.tbt_ewma_ms, 150.0)

    def test_no_tracker_means_stage3_disabled(self):
        ctrl = AdmissionController(
            AdmissionConfig(tbt_slo_ms=200, tbt_reactive_ratio=0.9),
            tbt_tracker=None,
        )
        d = ctrl.decide(prompt_len=1, prefix_match_len=0, snapshot=_snapshot())
        self.assertTrue(d.admit)


class TestAdmissionControllerRatioSlo(unittest.TestCase):
    """Per-request slowdown-ratio policy (Stage 1b / 2b)."""

    def test_ttft_ratio_only_admit(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ratio=2.0),  # solo×2 ceiling
            prefill_cost=_prefill(beta=1.0, gamma=10.0),  # T = d + 10
        )
        # n=100, p=0 → solo_ttft=110ms, queue=0 → pred=110, ratio=1.0 < 2.0 → admit
        d = ctrl.decide(prompt_len=100, prefix_match_len=0, snapshot=_snapshot())
        self.assertTrue(d.admit)
        self.assertEqual(d.reason, REASON_ADMIT)
        self.assertAlmostEqual(d.solo_ttft_ms, 110.0)

    def test_ttft_ratio_only_reject_due_to_queue(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ratio=2.0),
            prefill_cost=_prefill(beta=1.0, gamma=10.0),
        )
        # solo=110, queue=300 → pred=410, ratio=410/110≈3.7 > 2.0 → reject
        d = ctrl.decide(
            prompt_len=100, prefix_match_len=0,
            snapshot=_snapshot(queue_ms=300.0),
        )
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_TTFT_RATIO)
        self.assertAlmostEqual(d.solo_ttft_ms, 110.0)

    def test_ttft_ratio_skipped_when_solo_below_floor(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ratio=2.0),
            prefill_cost=_prefill(beta=0.0, gamma=0.5),  # solo=0.5ms < 1ms floor
        )
        # Even with huge queue, ratio check is skipped.
        d = ctrl.decide(
            prompt_len=100, prefix_match_len=0,
            snapshot=_snapshot(queue_ms=99999.0),
        )
        self.assertTrue(d.admit)

    def test_tbt_ratio_reject(self):
        ctrl = AdmissionController(
            AdmissionConfig(tbt_slo_ratio=2.0),
            tbt_cost=_tbt(a=0.0, b=10.0, c=0.0),  # bs-only model
        )
        # solo=10*1=10, with bs=5 → pred=10*6=60, ratio=6 > 2 → reject
        d = ctrl.decide(
            prompt_len=10, prefix_match_len=0, snapshot=_snapshot(bs=5),
        )
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_TBT_RATIO)
        self.assertAlmostEqual(d.solo_tbt_ms, 10.0)

    def test_absolute_wins_over_ratio(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=200, ttft_slo_ratio=2.0),
            prefill_cost=_prefill(beta=1.0, gamma=10.0),
        )
        # solo=510, queue=2000 → pred=2510. Both abs(>200) and ratio(~5>2) fire.
        d = ctrl.decide(
            prompt_len=500, prefix_match_len=0,
            snapshot=_snapshot(queue_ms=2000.0),
        )
        self.assertFalse(d.admit)
        # Absolute is checked first.
        self.assertEqual(d.reason, REASON_TTFT_PREDICTED)


class TestAdmissionControllerStageOrder(unittest.TestCase):
    """First-violation-wins ordering: Stage 1 > Stage 2 > Stage 3."""

    def test_stage1_wins_when_all_would_reject(self):
        tracker = TBTEwmaTracker(alpha=1.0, warm_up_steps=1)
        tracker.update(9999.0)
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=10, tbt_slo_ms=10, tbt_reactive_ratio=0.9),
            prefill_cost=_prefill(),  # 1ms/token, will exceed 10ms easily
            tbt_cost=_tbt(b=999.0),
            tbt_tracker=tracker,
        )
        d = ctrl.decide(prompt_len=100, prefix_match_len=0, snapshot=_snapshot())
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_TTFT_PREDICTED)

    def test_stage2_wins_when_stage1_passes(self):
        tracker = TBTEwmaTracker(alpha=1.0, warm_up_steps=1)
        tracker.update(9999.0)
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=10000, tbt_slo_ms=10),
            prefill_cost=_prefill(),
            tbt_cost=_tbt(b=999.0),
            tbt_tracker=tracker,
        )
        d = ctrl.decide(prompt_len=100, prefix_match_len=0, snapshot=_snapshot())
        self.assertFalse(d.admit)
        self.assertEqual(d.reason, REASON_TBT_PREDICTED)


class TestAdmissionControllerDryRun(unittest.TestCase):
    """Dry-run admits even when policy says reject; flag is set for logging."""

    def test_dry_run_admits_with_would_reject_flag(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=10, dry_run=True),
            prefill_cost=_prefill(),
        )
        d = ctrl.decide(prompt_len=500, prefix_match_len=0, snapshot=_snapshot())
        self.assertTrue(d.admit)  # dry-run admits
        self.assertEqual(d.reason, REASON_TTFT_PREDICTED)  # but reason is preserved
        self.assertTrue(d.dry_run_would_reject)
        self.assertAlmostEqual(d.predicted_prefill_ms, 500.0)

    def test_dry_run_admit_path_no_would_reject(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=10000, dry_run=True),
            prefill_cost=_prefill(),
        )
        d = ctrl.decide(prompt_len=100, prefix_match_len=0, snapshot=_snapshot())
        self.assertTrue(d.admit)
        self.assertEqual(d.reason, REASON_ADMIT)
        self.assertFalse(d.dry_run_would_reject)


class TestAdmissionControllerDisaggregation(unittest.TestCase):
    """Phase A only supports DisaggregationMode.NULL."""

    def test_non_null_mode_admits_with_disabled_reason(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=10),
            prefill_cost=_prefill(),
        )
        d = ctrl.decide(
            prompt_len=99999,
            prefix_match_len=0,
            snapshot=_snapshot(null=False),
        )
        self.assertTrue(d.admit)
        self.assertEqual(d.reason, REASON_DISABLED)

    def test_disagg_warning_emitted_once(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=10),
            prefill_cost=_prefill(),
        )
        snap = _snapshot(null=False)
        ctrl.decide(prompt_len=1, prefix_match_len=0, snapshot=snap)
        ctrl.decide(prompt_len=1, prefix_match_len=0, snapshot=snap)
        ctrl.decide(prompt_len=1, prefix_match_len=0, snapshot=snap)
        # No assertion on log content; just exercise the path that gates warns.
        self.assertTrue(ctrl._disagg_warned)


class TestAdmissionControllerRecentDecisions(unittest.TestCase):
    """Ring buffer for introspection, capped at 32."""

    def test_records_decisions(self):
        ctrl = AdmissionController(
            AdmissionConfig(ttft_slo_ms=1000), prefill_cost=_prefill()
        )
        for _ in range(3):
            ctrl.decide(prompt_len=10, prefix_match_len=0, snapshot=_snapshot())
        self.assertEqual(len(ctrl.recent_decisions()), 3)

    def test_caps_at_max(self):
        ctrl = AdmissionController(AdmissionConfig())
        for _ in range(100):
            ctrl.decide(prompt_len=1, prefix_match_len=0, snapshot=_snapshot())
        recent = ctrl.recent_decisions()
        self.assertEqual(len(recent), 32)


class TestAdmissionMetrics(unittest.TestCase):
    """Smoke tests for AdmissionMetrics — uses an isolated CollectorRegistry."""

    def _new_metrics(self) -> AdmissionMetrics:
        from prometheus_client import CollectorRegistry

        return AdmissionMetrics(
            labels={"model_name": "test", "tp_rank": "0"},
            registry=CollectorRegistry(),
        )

    def test_record_admit(self):
        m = self._new_metrics()
        d = AdmissionDecision(
            admit=True,
            reason=REASON_ADMIT,
            predicted_ttft_ms=150.0,
            predicted_tbt_ms=50.0,
            queue_predicted_ms=100.0,
            tbt_ewma_ms=45.0,
        )
        m.record_decision(d)  # should not raise

    def test_record_reject(self):
        m = self._new_metrics()
        d = AdmissionDecision(
            admit=False,
            reason=REASON_TTFT_PREDICTED,
            predicted_ttft_ms=99999.0,
            queue_predicted_ms=80000.0,
        )
        m.record_decision(d)

    def test_record_dryrun(self):
        m = self._new_metrics()
        d = AdmissionDecision(
            admit=True,
            reason=REASON_TBT_REACTIVE,
            tbt_ewma_ms=300.0,
            dry_run_would_reject=True,
        )
        m.record_decision(d)

    def test_record_minimal_decision(self):
        # disabled-path decisions carry no predicted values
        m = self._new_metrics()
        m.record_decision(AdmissionDecision(admit=True, reason=REASON_DISABLED))

    def test_update_tbt_ewma(self):
        m = self._new_metrics()
        m.update_tbt_ewma(125.5)
        m.update_tbt_ewma(None)  # no-op

    def test_counter_label_distinguishes_decisions(self):
        # Verify the counter actually carries the decision/reason labels.
        from prometheus_client import CollectorRegistry

        reg = CollectorRegistry()
        m = AdmissionMetrics(labels={"model_name": "test"}, registry=reg)
        m.record_decision(AdmissionDecision(admit=True, reason=REASON_ADMIT))
        m.record_decision(
            AdmissionDecision(admit=False, reason=REASON_TTFT_PREDICTED)
        )
        m.record_decision(
            AdmissionDecision(admit=False, reason=REASON_TTFT_PREDICTED)
        )

        admit_value = reg.get_sample_value(
            "sglang:admission_decisions_total",
            {"model_name": "test", "decision": "admit", "reason": REASON_ADMIT},
        )
        reject_value = reg.get_sample_value(
            "sglang:admission_decisions_total",
            {
                "model_name": "test",
                "decision": "reject",
                "reason": REASON_TTFT_PREDICTED,
            },
        )
        self.assertEqual(admit_value, 1.0)
        self.assertEqual(reject_value, 2.0)


class TestDecisionLogger(unittest.TestCase):
    """JSONL persistence of admission decisions."""

    def test_writes_jsonl_lines(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "decisions.jsonl"
            log = DecisionLogger(path)
            self.assertTrue(log.enabled)
            decision = AdmissionDecision(
                admit=False, reason=REASON_TTFT_PREDICTED,
                predicted_ttft_ms=4500.0, ttft_slo_ms=2000.0,
                predicted_prefill_ms=320.0, queue_predicted_ms=4180.0,
            )
            snap = SchedulerSnapshot(
                waiting_queue_predicted_prefill_ms=4180.0,
                running_batch_size=12,
                running_batch_total_kv_tokens=145000,
                disaggregation_mode_is_null=True,
            )
            log.record(rid="abc", decision=decision, prompt_len=18432,
                       prefix_len=16384, snapshot=snap)
            log.close()
            content = path.read_text().strip().splitlines()
            self.assertEqual(len(content), 1)
            row = json.loads(content[0])
            self.assertEqual(row["rid"], "abc")
            self.assertFalse(row["admit"])
            self.assertEqual(row["reason"], REASON_TTFT_PREDICTED)
            self.assertEqual(row["prompt_len"], 18432)
            self.assertEqual(row["prefix_len"], 16384)
            self.assertEqual(row["running_batch_size"], 12)

    def test_disabled_on_open_failure(self):
        # /proc is read-only; the open should fail and the logger should
        # disable itself rather than raise.
        log = DecisionLogger("/proc/this/cannot/be/created.jsonl")
        self.assertFalse(log.enabled)
        # record() must not raise even when disabled.
        log.record(
            rid="x",
            decision=AdmissionDecision(admit=True, reason=REASON_ADMIT),
            prompt_len=0, prefix_len=0,
            snapshot=SchedulerSnapshot(0.0, 0, 0, True),
        )


class TestReplayLogic(unittest.TestCase):
    """Verifies the offline replay tool's policy reconstruction matches the
    online controller. The replay module lives under tools/, so we import it
    by relative path discovery rather than installation.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util
        import sys
        from pathlib import Path

        repo = Path(__file__).resolve().parents[3]
        spec = importlib.util.spec_from_file_location(
            "replay_admission",
            repo / "tools" / "admission_control" / "replay_admission.py",
        )
        cls.replay = importlib.util.module_from_spec(spec)
        # dataclass decorator needs the module visible in sys.modules.
        sys.modules["replay_admission"] = cls.replay
        spec.loader.exec_module(cls.replay)

    def _row(self, **kw) -> "Any":
        defaults = dict(
            rid="r", ts=0.0, admit=True, reason=REASON_ADMIT,
            dry_run_would_reject=False,
            predicted_ttft_ms=None, predicted_tbt_ms=None,
            solo_ttft_ms=None, solo_tbt_ms=None,
            tbt_ewma_ms=None, queue_predicted_ms=None,
            ttft_slo_ms=None, tbt_slo_ms=None,
            ttft_slo_ratio=None, tbt_slo_ratio=None,
            prompt_len=0, prefix_len=0,
            running_batch_size=0, running_batch_total_kv_tokens=0,
        )
        defaults.update(kw)
        return self.replay.DecisionRow(**defaults)

    # ---- Absolute SLO -----------------------------------------------------

    def test_disabled_passes_through(self):
        row = self._row(reason=REASON_DISABLED)
        verdict = self.replay.evaluate_row(row, 1000, 100, 1.0, 1.0, 0.9)
        self.assertEqual(verdict, REASON_DISABLED)

    def test_ttft_threshold_applied(self):
        row = self._row(predicted_ttft_ms=1500.0, predicted_tbt_ms=10.0)
        self.assertEqual(
            self.replay.evaluate_row(row, 2000, 100, 1.0, 1.0, 0.9), REASON_ADMIT
        )
        self.assertEqual(
            self.replay.evaluate_row(row, 1000, 100, 1.0, 1.0, 0.9),
            REASON_TTFT_PREDICTED,
        )

    def test_tbt_predicted_priority(self):
        row = self._row(predicted_ttft_ms=10.0, predicted_tbt_ms=300.0, tbt_ewma_ms=400.0)
        self.assertEqual(
            self.replay.evaluate_row(row, 5000, 200, 1.0, 1.0, 0.9),
            REASON_TBT_PREDICTED,
        )

    def test_tbt_reactive_when_predicted_passes(self):
        row = self._row(predicted_ttft_ms=10.0, predicted_tbt_ms=20.0, tbt_ewma_ms=400.0)
        self.assertEqual(
            self.replay.evaluate_row(row, 5000, 200, 1.0, 1.0, 0.9),
            REASON_TBT_REACTIVE,
        )

    # ---- Ratio SLO --------------------------------------------------------

    def test_ttft_ratio_admit(self):
        # 200ms / 100ms = 2.0 ratio; SLO ratio 3.0 → admit.
        row = self._row(predicted_ttft_ms=200.0, solo_ttft_ms=100.0)
        self.assertEqual(
            self.replay.evaluate_row(row, 0, 0, 3.0, 1.0, 0.9), REASON_ADMIT
        )

    def test_ttft_ratio_reject(self):
        # 400ms / 100ms = 4.0 ratio; SLO ratio 2.0 → reject.
        row = self._row(predicted_ttft_ms=400.0, solo_ttft_ms=100.0)
        self.assertEqual(
            self.replay.evaluate_row(row, 0, 0, 2.0, 1.0, 0.9), REASON_TTFT_RATIO
        )

    def test_tbt_ratio_reject(self):
        row = self._row(predicted_tbt_ms=80.0, solo_tbt_ms=20.0)
        # 4.0 ratio; SLO 3.0 → reject
        self.assertEqual(
            self.replay.evaluate_row(row, 0, 0, 1.0, 3.0, 0.9), REASON_TBT_RATIO
        )

    def test_ratio_skips_when_solo_below_floor(self):
        # solo_ttft = 0.5ms < 1ms floor → ratio check skipped (admit)
        row = self._row(predicted_ttft_ms=10.0, solo_ttft_ms=0.5)
        self.assertEqual(
            self.replay.evaluate_row(row, 0, 0, 2.0, 1.0, 0.9), REASON_ADMIT
        )

    def test_absolute_wins_over_ratio_when_both_fire(self):
        # pred=5000, solo=1000 → both abs(5000>2000) and ratio(5>2) fire.
        # Absolute is checked first → reason=TTFT_PREDICTED.
        row = self._row(predicted_ttft_ms=5000.0, solo_ttft_ms=1000.0)
        self.assertEqual(
            self.replay.evaluate_row(row, 2000, 0, 2.0, 1.0, 0.9),
            REASON_TTFT_PREDICTED,
        )

    def test_evaluate_slo_counts_with_ratio(self):
        rows = [
            self._row(predicted_ttft_ms=10.0),                     # admit (no slo)
            self._row(predicted_ttft_ms=5000.0, solo_ttft_ms=100), # rej_ttft (abs)
            self._row(predicted_ttft_ms=300.0, solo_ttft_ms=100),  # rej_ttft_ratio (3>2)
            self._row(predicted_tbt_ms=300.0, solo_tbt_ms=20),     # rej_tbt_pred (abs)
            self._row(predicted_tbt_ms=80.0, solo_tbt_ms=20),      # rej_tbt_ratio (4>3)
            self._row(predicted_ttft_ms=10.0, predicted_tbt_ms=20.0,
                      solo_tbt_ms=20.0, tbt_ewma_ms=400.0),        # rej_tbt_react
            self._row(reason=REASON_DISABLED),                     # skipped
        ]
        out = self.replay.evaluate_slo(rows, 1000, 200, 2.0, 3.0, 0.9)
        self.assertEqual(out.total, 7)
        self.assertEqual(out.admit, 1)
        self.assertEqual(out.rej_ttft, 1)
        self.assertEqual(out.rej_ttft_ratio, 1)
        self.assertEqual(out.rej_tbt_pred, 1)
        self.assertEqual(out.rej_tbt_ratio, 1)
        self.assertEqual(out.rej_tbt_react, 1)
        self.assertEqual(out.skipped_disabled, 1)


if __name__ == "__main__":
    unittest.main()
