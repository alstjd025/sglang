"""Unit tests for tools/halo/fit_halo_cost_model.py.

Design + rationale: ms_dev/halo_dev/prediction_model.md §10.

The headline test is `test_ols_recovers_known_coefficients` — we synthesize a
JSONL stream from a known θ vector, run the fit script, and verify the
recovered coefficients are within a small tolerance of the truth. That is the
strongest sanity check for the whole pipeline (sample format ↔ OLS ↔ output
JSON ↔ HaloStepCostModel loader).

Additional coverage:
- filter by forward_mode
- filter by step_time bounds
- refuses when too few samples remain
- robust to malformed JSONL lines
"""

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest

# Local import — the script lives under tools/halo/ which isn't a sglang
# package; we exercise it via subprocess + sys.path, mirroring how a user
# would invoke it.
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
sys.path.insert(0, REPO_ROOT)

from sglang.srt.managers.admission_control.cost_model import (
    HALO_STEP_FORM_SPLIT_V1,
    HaloStepCostModel,
)
from sglang.test.ci.ci_register import register_cpu_ci
from tools.halo import fit_halo_cost_model as fit

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


# True coefficients used to generate synthetic data.
TRUE_THETA = dict(
    theta_p1=2.0e-7,
    theta_p2=5.0e-7,
    theta_p3=0.090,
    theta_d1=6.0e-5,
    theta_d2=0.40,
    theta_c=11.5,
)


def _synthesize_jsonl(
    path: str, n_rows: int, noise_ms: float = 0.0, seed: int = 42
) -> None:
    """Generate a JSONL log under the assumption that step_time follows the
    Halo step cost model with TRUE_THETA + Gaussian noise."""
    rng = random.Random(seed)
    truth = HaloStepCostModel(**TRUE_THETA, metadata={})
    with open(path, "w") as f:
        for i in range(n_rows):
            # Mix of decode-heavy and prefill-heavy rows.
            mode_dice = rng.random()
            if mode_dice < 0.5:
                # decode-only
                bs = rng.randint(1, 24)
                r_each = rng.randint(500, 30000)
                sum_r = bs * r_each + rng.randint(-bs, bs)
                fm = "DECODE"
                prefill_infos = []
                decode_infos = [r_each] * bs
                sum_n_sq = 0.0
                sum_nr = 0.0
                sum_n = 0
                num_prefill = 0
                max_kv = max(r_each, 0)
            elif mode_dice < 0.9:
                # extend-heavy (one large prefill request)
                n = rng.randint(64, 4096)
                r = rng.randint(0, 20000)
                bs = rng.randint(0, 8)
                r_each = rng.randint(500, 20000) if bs else 0
                sum_r = bs * r_each
                fm = "EXTEND"
                sum_n_sq = float(n * n)
                sum_nr = float(n * r)
                sum_n = n
                num_prefill = 1
                max_kv = max(n + r, r_each)
                decode_infos = [r_each] * bs
                prefill_infos = [(n, r)]
            else:
                # mixed: small prefill chunk + many decoders
                n = rng.randint(8, 256)
                r = rng.randint(1000, 20000)
                bs = rng.randint(4, 24)
                r_each = rng.randint(1000, 30000)
                sum_r = bs * r_each
                fm = "MIXED"
                sum_n_sq = float(n * n)
                sum_nr = float(n * r)
                sum_n = n
                num_prefill = 1
                max_kv = max(n + r, r_each)
                decode_infos = [r_each] * bs
                prefill_infos = [(n, r)]

            true_step = truth.estimate_step_ms(prefill_infos, decode_infos)
            noise = rng.gauss(0.0, noise_ms) if noise_ms > 0 else 0.0
            step_time = max(0.1, true_step + noise)

            row = {
                "step_idx": i,
                "ts_ns": i * 1_000_000,
                "sum_n_sq": sum_n_sq,
                "sum_nr": sum_nr,
                "sum_n": sum_n,
                "sum_r": sum_r,
                "bs_d": bs,
                "step_time_ms": step_time,
                "num_prefill_reqs": num_prefill,
                "num_decode_reqs": bs,
                "max_kv": max_kv,
                "forward_mode": fm,
            }
            f.write(json.dumps(row) + "\n")


class TestRecoverCoefficients(unittest.TestCase):
    def test_ols_recovers_known_theta_noiseless(self):
        """With zero noise the OLS solver should recover θ exactly."""
        with tempfile.TemporaryDirectory() as d:
            samples = os.path.join(d, "s.jsonl")
            output = os.path.join(d, "out.json")
            _synthesize_jsonl(samples, n_rows=1000, noise_ms=0.0)
            rc = fit.main([
                "--samples", samples,
                "--output", output,
                "--form", "unified",
                "--min-samples", "100",
            ])
            self.assertEqual(rc, 0)
            loaded = HaloStepCostModel.from_json(output)
        for k, true_v in TRUE_THETA.items():
            got = getattr(loaded, k)
            # Allow small relative tolerance for numerical drift.
            self.assertAlmostEqual(
                got, true_v,
                delta=max(abs(true_v) * 1e-4, 1e-8),
                msg=f"{k}: got {got} expected {true_v}",
            )

    def test_ols_recovers_theta_with_small_noise(self):
        """With ~1 ms noise the recovered θ should be close (RMSE small)."""
        with tempfile.TemporaryDirectory() as d:
            samples = os.path.join(d, "s.jsonl")
            output = os.path.join(d, "out.json")
            _synthesize_jsonl(samples, n_rows=2000, noise_ms=1.0)
            rc = fit.main([
                "--samples", samples,
                "--output", output,
                "--form", "unified",
                "--min-samples", "100",
            ])
            self.assertEqual(rc, 0)
            with open(output) as f:
                data = json.load(f)
        # RMSE should be roughly the noise level.
        self.assertLess(data["fit_metadata"]["rmse_ms"], 2.0)
        # R² should be high.
        self.assertGreater(data["fit_metadata"]["r_squared"], 0.99)


class TestFiltering(unittest.TestCase):
    def test_filter_forward_mode_keeps_only_matching(self):
        with tempfile.TemporaryDirectory() as d:
            samples = os.path.join(d, "s.jsonl")
            output = os.path.join(d, "out.json")
            _synthesize_jsonl(samples, n_rows=2000, noise_ms=0.0)
            rc = fit.main([
                "--samples", samples,
                "--output", output,
                "--filter-forward-mode", "DECODE",
                "--min-samples", "10",
            ])
            self.assertEqual(rc, 0)
            with open(output) as f:
                data = json.load(f)
        self.assertEqual(data["fit_metadata"]["filter_forward_modes"],
                         ["DECODE"])
        # Decode-only set should still recover theta_d1 / theta_d2 / theta_c
        # accurately. Prefill θ's may be ill-conditioned (no signal) — only
        # check decode part.
        self.assertAlmostEqual(
            data["theta_d1"], TRUE_THETA["theta_d1"],
            delta=abs(TRUE_THETA["theta_d1"]) * 1e-3,
        )

    def test_filter_step_time_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            samples = os.path.join(d, "s.jsonl")
            # Inject some out-of-range rows manually.
            with open(samples, "w") as f:
                base = {
                    "sum_n_sq": 0.0, "sum_nr": 0.0, "sum_n": 0,
                    "sum_r": 1000, "bs_d": 1, "forward_mode": "DECODE",
                }
                # 5 in-range
                for t in [10.0, 12.0, 14.0, 16.0, 18.0]:
                    row = dict(base, step_time_ms=t)
                    f.write(json.dumps(row) + "\n")
                # 5 out-of-range
                for t in [0.001, 0.05, 50000.0, 99999.0, 99999.9]:
                    row = dict(base, step_time_ms=t)
                    f.write(json.dumps(row) + "\n")
            output = os.path.join(d, "out.json")
            # min_samples=5 to allow degenerate-rank fit (all rows have
            # identical features, so the fit will be ill-conditioned but
            # it should still run and produce a valid JSON).
            rc = fit.main([
                "--samples", samples,
                "--output", output,
                "--min-step-time-ms", "1.0",
                "--max-step-time-ms", "1000.0",
                "--min-samples", "5",
            ])
            self.assertEqual(rc, 0)


class TestRobustness(unittest.TestCase):
    def test_refuses_when_too_few_samples(self):
        with tempfile.TemporaryDirectory() as d:
            samples = os.path.join(d, "s.jsonl")
            output = os.path.join(d, "out.json")
            _synthesize_jsonl(samples, n_rows=20, noise_ms=0.0)
            rc = fit.main([
                "--samples", samples,
                "--output", output,
                "--form", "unified",
                "--min-samples", "100",
            ])
            self.assertNotEqual(rc, 0)
            self.assertFalse(os.path.exists(output))

    def test_skips_malformed_lines(self):
        with tempfile.TemporaryDirectory() as d:
            samples = os.path.join(d, "s.jsonl")
            _synthesize_jsonl(samples, n_rows=300, noise_ms=0.0)
            # Inject malformed lines.
            with open(samples, "a") as f:
                f.write("{not json\n")
                f.write("\n")
                f.write("garbage line without braces\n")
            output = os.path.join(d, "out.json")
            rc = fit.main([
                "--samples", samples,
                "--output", output,
                "--form", "unified",
                "--min-samples", "100",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(output))

    def test_glob_expansion(self):
        with tempfile.TemporaryDirectory() as d:
            a = os.path.join(d, "a.jsonl")
            b = os.path.join(d, "b.jsonl")
            _synthesize_jsonl(a, n_rows=150, noise_ms=0.0, seed=1)
            _synthesize_jsonl(b, n_rows=150, noise_ms=0.0, seed=2)
            output = os.path.join(d, "out.json")
            rc = fit.main([
                "--samples", os.path.join(d, "*.jsonl"),
                "--output", output,
                "--form", "unified",
                "--min-samples", "100",
            ])
            self.assertEqual(rc, 0)
            with open(output) as f:
                data = json.load(f)
        self.assertEqual(data["fit_metadata"]["n_samples"], 300)


class TestSplitMode(unittest.TestCase):
    """Verify --split produces a halo_step_split_v1 JSON that recovers θ."""

    TRUE_PREFILL = dict(theta_p1=2.0e-7, theta_p2=5.0e-7, theta_p3=0.090,
                        theta_c_p=70.0)
    TRUE_DECODE = dict(theta_d1=6.0e-5, theta_d2=0.40, theta_c_d=20.0)

    def _make_split_jsonl(self, path, n_prefill=400, n_decode=2000, seed=7):
        rng = random.Random(seed)
        with open(path, "w") as f:
            # Pure prefill (EXTEND) rows.
            for i in range(n_prefill):
                n = rng.randint(64, 4096)
                r = rng.randint(0, 20000)
                t = (self.TRUE_PREFILL["theta_p1"] * n * n
                     + self.TRUE_PREFILL["theta_p2"] * n * r
                     + self.TRUE_PREFILL["theta_p3"] * n
                     + self.TRUE_PREFILL["theta_c_p"])
                row = {"sum_n_sq": float(n*n), "sum_nr": float(n*r),
                       "sum_n": n, "sum_r": 0, "bs_d": 0,
                       "step_time_ms": t, "forward_mode": "EXTEND"}
                f.write(json.dumps(row) + "\n")
            # Pure decode (DECODE) rows.
            for i in range(n_decode):
                bs = rng.randint(1, 24)
                r_each = rng.randint(500, 30000)
                sum_r = bs * r_each
                t = (self.TRUE_DECODE["theta_d1"] * sum_r
                     + self.TRUE_DECODE["theta_d2"] * bs
                     + self.TRUE_DECODE["theta_c_d"])
                row = {"sum_n_sq": 0.0, "sum_nr": 0.0, "sum_n": 0,
                       "sum_r": sum_r, "bs_d": bs,
                       "step_time_ms": t, "forward_mode": "DECODE"}
                f.write(json.dumps(row) + "\n")

    def test_split_fit_recovers_known_coefficients(self):
        with tempfile.TemporaryDirectory() as d:
            samples = os.path.join(d, "s.jsonl")
            output = os.path.join(d, "out.json")
            self._make_split_jsonl(samples)
            rc = fit.main([
                "--samples", samples,
                "--output", output,
                "--split",
                "--min-samples", "100",
            ])
            self.assertEqual(rc, 0)
            loaded = HaloStepCostModel.from_json(output)
        self.assertTrue(loaded.is_split)
        self.assertEqual(loaded.form, HALO_STEP_FORM_SPLIT_V1)
        # Recover prefill coefs.
        for k, true_v in self.TRUE_PREFILL.items():
            if k == "theta_c_p":
                got = loaded._prefill_const()
            else:
                got = getattr(loaded, k)
            self.assertAlmostEqual(
                got, true_v,
                delta=max(abs(true_v) * 1e-4, 1e-8),
                msg=f"{k}: got {got} expected {true_v}",
            )
        # Recover decode coefs.
        for k, true_v in self.TRUE_DECODE.items():
            if k == "theta_c_d":
                got = loaded._decode_const()
            else:
                got = getattr(loaded, k)
            self.assertAlmostEqual(
                got, true_v,
                delta=max(abs(true_v) * 1e-4, 1e-8),
                msg=f"{k}: got {got} expected {true_v}",
            )

    def test_split_fit_metadata_records_per_side_stats(self):
        with tempfile.TemporaryDirectory() as d:
            samples = os.path.join(d, "s.jsonl")
            output = os.path.join(d, "out.json")
            self._make_split_jsonl(samples)
            rc = fit.main([
                "--samples", samples,
                "--output", output,
                "--split",
                "--min-samples", "100",
            ])
            self.assertEqual(rc, 0)
            with open(output) as f:
                data = json.load(f)
        meta = data["fit_metadata"]
        self.assertIn("prefill", meta)
        self.assertIn("decode", meta)
        self.assertEqual(meta["prefill"]["n_samples"], 400)
        self.assertEqual(meta["decode"]["n_samples"], 2000)
        self.assertIn("combined_rmse_ms", meta)


if __name__ == "__main__":
    unittest.main()
