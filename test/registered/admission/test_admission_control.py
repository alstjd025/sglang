"""Unit + integration tests for admission_control module.

Step 1 covers PrefillCostModel, TBTCostModel, TBTEwmaTracker only. Controller
and integration tests are added in later steps.

See test/registered/admission/CLAUDE.md and
python/sglang/srt/managers/admission_control/CLAUDE.md.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sglang.srt.managers.admission_control.cost_model import (
    CostModelLoadError,
    PrefillCostModel,
    TBTCostModel,
    try_load_prefill_cost_model,
    try_load_tbt_cost_model,
)
from sglang.srt.managers.admission_control.tbt_tracker import TBTEwmaTracker
from sglang.test.ci.ci_register import register_cpu_ci

# Step 1 is pure-Python; CPU suite is enough.
register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestPrefillCostModel(unittest.TestCase):
    def test_estimate_quadratic(self):
        m = PrefillCostModel(alpha=2.0, beta=3.0, gamma=5.0)
        # d = max(0, 10 - 4) = 6 → 2*36 + 3*6 + 5 = 72 + 18 + 5 = 95
        self.assertAlmostEqual(m.estimate_ms(10, 4), 95.0)

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


if __name__ == "__main__":
    unittest.main()
