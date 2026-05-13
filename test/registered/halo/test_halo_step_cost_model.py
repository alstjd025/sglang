"""Unit tests for HaloStepCostModel (post-Phase-1 cost-model follow-up).

Design + rationale: ms_dev/halo_dev/prediction_model.md.

Coverage:
- estimate_solo_prefill_total_ms / estimate_solo_tbt_ms exact arithmetic
- estimate_step_ms (batched) — empty / single-element / mixed prefill+decode
- solo ↔ batched equivalence when the supplied batch is a single request
- to_json / from_json roundtrip preserves coefficients
- from_json rejects: wrong form, missing coefficient, malformed JSON, missing file
- try_load_halo_step_cost_model lenient contract:
  * None / empty / missing path → None
  * malformed file → None + WARN (not raise)
  * valid file → instance
"""

import json
import os
import tempfile
import unittest

from sglang.srt.managers.admission_control.cost_model import (
    HALO_STEP_FORM_SPLIT_V1,
    HALO_STEP_FORM_V1,
    CostModelLoadError,
    HaloStepCostModel,
    try_load_halo_step_cost_model,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def _make(**overrides):
    """A reusable fixture with easy-to-mental-math coefficients."""
    base = dict(
        theta_p1=1e-7,   # Σnᵢ²
        theta_p2=4e-7,   # Σ(nᵢ·rᵢ)
        theta_p3=0.08,   # Σnᵢ
        theta_d1=5e-5,   # Σrⱼ
        theta_d2=0.3,    # bs_d
        theta_c=10.0,    # constant
        metadata={"model": "unit", "hw": "unit"},
    )
    base.update(overrides)
    return HaloStepCostModel(**base)


class TestHaloStepCostModelMath(unittest.TestCase):
    def test_solo_prefill_total_ms_exact(self):
        m = _make()
        # (n=1000, r=5000):
        # 1e-7·1_000_000 + 4e-7·5_000_000 + 0.08·1000 + 10.0
        # = 0.1 + 2.0 + 80.0 + 10.0 = 92.1
        self.assertAlmostEqual(
            m.estimate_solo_prefill_total_ms(1000, 5000), 92.1, places=9
        )

    def test_solo_prefill_zero_inputs_falls_back_to_constant(self):
        m = _make()
        # n=0, r=0 → every term zero, only θ_c
        self.assertAlmostEqual(
            m.estimate_solo_prefill_total_ms(0, 0), 10.0, places=9
        )

    def test_solo_tbt_exact(self):
        m = _make()
        # r=5000, bs=1: 5e-5·5000 + 0.3·1 + 10 = 0.25 + 0.3 + 10 = 10.55
        self.assertAlmostEqual(m.estimate_solo_tbt_ms(5000), 10.55, places=9)

    def test_solo_tbt_zero_kv(self):
        m = _make()
        # r=0: only θ_d2 + θ_c
        self.assertAlmostEqual(m.estimate_solo_tbt_ms(0), 10.3, places=9)

    def test_batched_empty_returns_constant(self):
        m = _make()
        self.assertAlmostEqual(m.estimate_step_ms([], []), 10.0, places=9)

    def test_batched_single_prefill_matches_solo(self):
        m = _make()
        n, r = 1000, 5000
        self.assertAlmostEqual(
            m.estimate_step_ms([(n, r)], []),
            m.estimate_solo_prefill_total_ms(n, r),
            places=9,
        )

    def test_batched_single_decode_matches_solo(self):
        m = _make()
        r = 5000
        self.assertAlmostEqual(
            m.estimate_step_ms([], [r]),
            m.estimate_solo_tbt_ms(r),
            places=9,
        )

    def test_batched_mixed_step_exact(self):
        m = _make()
        # prefill [(500, 2000)], decode [3000, 4000]:
        #   Σnᵢ²  = 250_000
        #   Σnᵢrᵢ = 1_000_000
        #   Σnᵢ   = 500
        #   Σrⱼ   = 7_000
        #   bs_d  = 2
        # → 1e-7·250000 + 4e-7·1e6 + 0.08·500 + 5e-5·7000 + 0.3·2 + 10
        # = 0.025 + 0.4 + 40 + 0.35 + 0.6 + 10 = 51.375
        self.assertAlmostEqual(
            m.estimate_step_ms([(500, 2000)], [3000, 4000]),
            51.375,
            places=9,
        )

    def test_prefill_self_attention_is_quadratic_sum_not_squared_sum(self):
        # Σnᵢ² != (Σnᵢ)² — two requests of 500 should give 2·500²·θ_p1,
        # not (1000)²·θ_p1.
        m = _make(
            theta_p1=1.0, theta_p2=0.0, theta_p3=0.0,
            theta_d1=0.0, theta_d2=0.0, theta_c=0.0,
        )
        two_reqs = m.estimate_step_ms([(500, 0), (500, 0)], [])
        single_merged = m.estimate_step_ms([(1000, 0)], [])
        self.assertAlmostEqual(two_reqs, 2 * 500 * 500, places=9)
        self.assertAlmostEqual(single_merged, 1000 * 1000, places=9)
        # And the merged-request prediction is *strictly larger* than the
        # two-separate-request prediction — physically expected for self-
        # attention.
        self.assertGreater(single_merged, two_reqs)

    def test_decode_kv_summed_not_averaged(self):
        # The headline cliff fix: Σrⱼ scales with batch SUM, not per-request.
        m = _make(
            theta_p1=0.0, theta_p2=0.0, theta_p3=0.0,
            theta_d1=1.0, theta_d2=0.0, theta_c=0.0,
        )
        # Solo: r=10000 → 10000
        solo = m.estimate_solo_tbt_ms(10000)
        # Two decodes, each r=10000 → 20000 (sum), not 10000 (avg)
        batched = m.estimate_step_ms([], [10000, 10000])
        self.assertAlmostEqual(solo, 10000.0, places=9)
        self.assertAlmostEqual(batched, 20000.0, places=9)


class TestHaloStepCostModelJson(unittest.TestCase):
    def test_roundtrip_preserves_coefficients(self):
        m = _make()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            m.to_json(p)
            m2 = HaloStepCostModel.from_json(p)
        for k in ("theta_p1", "theta_p2", "theta_p3",
                  "theta_d1", "theta_d2", "theta_c"):
            self.assertEqual(getattr(m2, k), getattr(m, k), k)
        self.assertEqual(m2.metadata, m.metadata)

    def test_to_json_writes_form_field(self):
        m = _make()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.json")
            m.to_json(p)
            with open(p) as f:
                data = json.load(f)
        self.assertEqual(data["form"], HALO_STEP_FORM_V1)
        # fit_metadata always present (empty dict if missing).
        self.assertIn("fit_metadata", data)

    def test_from_json_rejects_wrong_form(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w") as f:
                json.dump(
                    {
                        "form": "halo_step_v0",
                        "theta_p1": 0.0, "theta_p2": 0.0, "theta_p3": 0.0,
                        "theta_d1": 0.0, "theta_d2": 0.0, "theta_c": 0.0,
                    },
                    f,
                )
            with self.assertRaises(CostModelLoadError):
                HaloStepCostModel.from_json(p)

    def test_from_json_rejects_missing_coefficient(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w") as f:
                json.dump({"form": HALO_STEP_FORM_V1, "theta_p1": 0.0}, f)
            with self.assertRaises(CostModelLoadError):
                HaloStepCostModel.from_json(p)

    def test_from_json_rejects_malformed_json(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w") as f:
                f.write("{this is not json")
            with self.assertRaises(CostModelLoadError):
                HaloStepCostModel.from_json(p)

    def test_from_json_rejects_missing_file(self):
        with self.assertRaises(CostModelLoadError):
            HaloStepCostModel.from_json("/nonexistent/path.json")


class TestHaloStepCostModelLenientLoader(unittest.TestCase):
    """try_load_halo_step_cost_model: should never raise."""

    def test_none_path_returns_none(self):
        self.assertIsNone(try_load_halo_step_cost_model(None))

    def test_empty_path_returns_none(self):
        self.assertIsNone(try_load_halo_step_cost_model(""))

    def test_missing_file_returns_none_with_warn(self):
        with self.assertLogs("sglang.srt.managers.admission_control.cost_model",
                             level="WARNING"):
            self.assertIsNone(
                try_load_halo_step_cost_model("/nonexistent/path.json")
            )

    def test_malformed_file_returns_none_with_warn(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w") as f:
                f.write("not-json")
            with self.assertLogs(
                "sglang.srt.managers.admission_control.cost_model",
                level="WARNING",
            ):
                self.assertIsNone(try_load_halo_step_cost_model(p))

    def test_wrong_form_returns_none_with_warn(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w") as f:
                json.dump(
                    {
                        "form": "halo_step_v0",
                        "theta_p1": 0.0, "theta_p2": 0.0, "theta_p3": 0.0,
                        "theta_d1": 0.0, "theta_d2": 0.0, "theta_c": 0.0,
                    },
                    f,
                )
            with self.assertLogs(
                "sglang.srt.managers.admission_control.cost_model",
                level="WARNING",
            ):
                self.assertIsNone(try_load_halo_step_cost_model(p))

    def test_valid_file_returns_instance_with_info(self):
        m = _make()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "good.json")
            m.to_json(p)
            with self.assertLogs(
                "sglang.srt.managers.admission_control.cost_model",
                level="INFO",
            ):
                loaded = try_load_halo_step_cost_model(p)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.theta_p1, m.theta_p1)
        self.assertEqual(loaded.theta_c, m.theta_c)


def _make_split(**overrides):
    """Split-form fixture with two distinct intercepts."""
    base = dict(
        theta_p1=1e-7,
        theta_p2=4e-7,
        theta_p3=0.08,
        theta_d1=5e-5,
        theta_d2=0.3,
        theta_c=20.0,       # mirror of theta_c_d in split mode (decode dominant)
        theta_c_p=70.0,     # prefill intercept (≫ decode)
        theta_c_d=20.0,     # decode intercept
        form=HALO_STEP_FORM_SPLIT_V1,
        metadata={"model": "unit", "hw": "unit"},
    )
    base.update(overrides)
    return HaloStepCostModel(**base)


class TestSplitModeArithmetic(unittest.TestCase):
    """Validate that split-form model uses θ_c_p/θ_c_d correctly."""

    def test_is_split_flag(self):
        self.assertTrue(_make_split().is_split)
        self.assertFalse(_make().is_split)

    def test_solo_prefill_uses_theta_c_p(self):
        m = _make_split()
        # n=1000, r=5000:
        # 1e-7·1_000_000 + 4e-7·5_000_000 + 0.08·1000 + 70.0
        # = 0.1 + 2.0 + 80.0 + 70.0 = 152.1
        self.assertAlmostEqual(
            m.estimate_solo_prefill_total_ms(1000, 5000), 152.1, places=9
        )

    def test_solo_tbt_uses_theta_c_d(self):
        m = _make_split()
        # r=5000, bs=1: 5e-5·5000 + 0.3·1 + 20.0 = 0.25 + 0.3 + 20.0 = 20.55
        self.assertAlmostEqual(m.estimate_solo_tbt_ms(5000), 20.55, places=9)

    def test_step_prefill_only_uses_theta_c_p(self):
        m = _make_split()
        # Same as solo prefill, since decode_infos is empty.
        self.assertAlmostEqual(
            m.estimate_step_ms([(1000, 5000)], []), 152.1, places=9
        )

    def test_step_decode_only_uses_theta_c_d(self):
        m = _make_split()
        # bs=2, sum_r=8000: 5e-5·8000 + 0.3·2 + 20.0 = 0.4 + 0.6 + 20.0 = 21.0
        self.assertAlmostEqual(
            m.estimate_step_ms([], [3000, 5000]), 21.0, places=9
        )

    def test_step_mixed_uses_sum_of_intercepts(self):
        """Mixed batches add both intercepts — documented behavior."""
        m = _make_split()
        # (500, 2000) prefill + [3000] decode:
        # prefill_part = 1e-7·250000 + 4e-7·1_000_000 + 0.08·500
        #              = 0.025 + 0.4 + 40 = 40.425
        # decode_part  = 5e-5·3000 + 0.3·1 = 0.15 + 0.3 = 0.45
        # intercept    = 70.0 + 20.0 = 90.0
        # → 40.425 + 0.45 + 90.0 = 130.875
        self.assertAlmostEqual(
            m.estimate_step_ms([(500, 2000)], [3000]), 130.875, places=9
        )

    def test_unified_solo_prefill_uses_theta_c(self):
        """Sanity: unified form's solo_prefill uses θ_c (no θ_c_p)."""
        m = _make()  # unified
        # n=1000, r=5000: 0.1 + 2.0 + 80.0 + 10.0 (θ_c=10) = 92.1
        self.assertAlmostEqual(
            m.estimate_solo_prefill_total_ms(1000, 5000), 92.1, places=9
        )


class TestSplitModeJson(unittest.TestCase):
    def test_split_roundtrip_preserves_intercepts(self):
        m = _make_split()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "split.json")
            m.to_json(p)
            m2 = HaloStepCostModel.from_json(p)
        self.assertTrue(m2.is_split)
        self.assertEqual(m2.form, HALO_STEP_FORM_SPLIT_V1)
        self.assertAlmostEqual(m2._prefill_const(), 70.0)
        self.assertAlmostEqual(m2._decode_const(), 20.0)
        # Slope coefficients identical.
        for k in ("theta_p1", "theta_p2", "theta_p3", "theta_d1", "theta_d2"):
            self.assertEqual(getattr(m2, k), getattr(m, k), k)

    def test_split_json_writes_two_intercepts(self):
        m = _make_split()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "split.json")
            m.to_json(p)
            with open(p) as f:
                data = json.load(f)
        self.assertEqual(data["form"], HALO_STEP_FORM_SPLIT_V1)
        self.assertEqual(data["theta_c_p"], 70.0)
        self.assertEqual(data["theta_c_d"], 20.0)
        self.assertNotIn("theta_c", data)

    def test_from_json_split_missing_intercept_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            with open(p, "w") as f:
                json.dump(
                    {
                        "form": HALO_STEP_FORM_SPLIT_V1,
                        "theta_p1": 0.0, "theta_p2": 0.0, "theta_p3": 0.0,
                        "theta_d1": 0.0, "theta_d2": 0.0,
                        "theta_c_p": 1.0,
                        # theta_c_d missing
                    },
                    f,
                )
            with self.assertRaises(CostModelLoadError):
                HaloStepCostModel.from_json(p)

    def test_loader_split_form_returns_instance_with_info(self):
        m = _make_split()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "split.json")
            m.to_json(p)
            with self.assertLogs(
                "sglang.srt.managers.admission_control.cost_model",
                level="INFO",
            ) as captured:
                loaded = try_load_halo_step_cost_model(p)
        self.assertIsNotNone(loaded)
        self.assertTrue(loaded.is_split)
        # The info log mentions [split form] to make this visible at startup.
        self.assertTrue(
            any("[split form]" in m for m in captured.output),
            captured.output,
        )


if __name__ == "__main__":
    unittest.main()
