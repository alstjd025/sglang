"""Unit tests for managers/halo/cost_model_sampler.py.

See ms_dev/halo_dev/prediction_model.md §9 for the design.

Coverage:
- _extract_features classifies decode / prefill / mixed batches correctly
- _extract_features returns None for non-fit forward modes
- observe_step subsamples by sample_every
- background flush thread writes JSONL rows that round-trip through json.loads
- queue-full triggers drop counter (we don't lose program correctness)
- close() drains pending samples and joins the flush thread
- build_halo_cost_sampler_from_server_args TP-dedups (rank != 0 → None)
- build_halo_cost_sampler_from_server_args returns None when path is unset
"""

import json
import os
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.managers.halo.cost_model_sampler import (
    HaloCostModelSampler,
    build_halo_cost_sampler_from_server_args,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Fake batch fixtures — mimic the subset of ScheduleBatch that the sampler
# touches, without dragging in ScheduleBatch construction overhead.
# ─────────────────────────────────────────────────────────────────────────────
class _FakeForwardMode:
    def __init__(self, name, decode=False, extend=False):
        self.name = name
        self._decode = decode
        self._extend = extend

    def is_decode(self):
        return self._decode

    def is_extend(self):
        return self._extend


def _fake_decode_batch(seq_lens):
    """forward_mode=DECODE; everyone decoding with given KV lengths."""
    return SimpleNamespace(
        forward_mode=_FakeForwardMode("DECODE", decode=True),
        reqs=[SimpleNamespace(fill_ids=[0] * s) for s in seq_lens],
        seq_lens_cpu=[SimpleNamespace(item=lambda v=s: v) for s in seq_lens],
        prefix_lens=None,
        extend_lens=None,
    )


def _fake_extend_batch(extend_lens, prefix_lens, seq_lens=None):
    """forward_mode=EXTEND. extend_lens[i] > 1 → prefill; ==1 → decode."""
    reqs = [SimpleNamespace(fill_ids=[0] * (e + p), extend_input_len=e)
            for e, p in zip(extend_lens, prefix_lens)]
    if seq_lens is None:
        seq_lens = prefix_lens  # decode reqs in MIXED use their existing KV
    return SimpleNamespace(
        forward_mode=_FakeForwardMode("EXTEND", extend=True),
        reqs=reqs,
        extend_lens=list(extend_lens),
        prefix_lens=list(prefix_lens),
        seq_lens_cpu=[SimpleNamespace(item=lambda v=s: v) for s in seq_lens],
    )


def _fake_idle_batch():
    return SimpleNamespace(
        forward_mode=_FakeForwardMode("IDLE"),  # neither decode nor extend
        reqs=[],
        seq_lens_cpu=None,
        prefix_lens=None,
        extend_lens=None,
    )


class _SamplerHarness(unittest.TestCase):
    """Helper to instantiate a sampler in a temp dir + clean up."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self._tmp.name, "samples.jsonl")
        self.sampler = None

    def tearDown(self):
        if self.sampler is not None:
            self.sampler.close()
        self._tmp.cleanup()

    def _new_sampler(self, **kwargs):
        kwargs.setdefault("flush_interval_ms", 5)
        self.sampler = HaloCostModelSampler(self.log_path, **kwargs)
        return self.sampler

    def _read_jsonl(self):
        # sampler holds the fh open in append+buffered mode; we need to wait
        # for the flush thread to write, then read from disk.
        time.sleep(0.05)
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]


class TestExtractFeatures(_SamplerHarness):
    def test_decode_batch_sums_kv(self):
        s = self._new_sampler()
        # Two decoders, KV 1000 and 3000 → Σrⱼ=4000, bs_d=2, prefill terms=0
        f = s._extract_features(_fake_decode_batch([1000, 3000]), 50.0)
        self.assertEqual(f.sum_r, 4000)
        self.assertEqual(f.bs_d, 2)
        self.assertEqual(f.sum_n_sq, 0.0)
        self.assertEqual(f.sum_nr, 0.0)
        self.assertEqual(f.sum_n, 0)
        self.assertEqual(f.num_prefill_reqs, 0)
        self.assertEqual(f.num_decode_reqs, 2)
        self.assertEqual(f.max_kv, 3000)
        self.assertEqual(f.forward_mode, "DECODE")
        self.assertAlmostEqual(f.step_time_ms, 50.0)

    def test_extend_batch_classifies_prefill(self):
        s = self._new_sampler()
        # Two prefill reqs: (n=500, r=2000), (n=300, r=1000)
        f = s._extract_features(
            _fake_extend_batch([500, 300], [2000, 1000]),
            12.5,
        )
        # Σnᵢ² = 500² + 300² = 250000 + 90000 = 340000
        # Σnᵢrᵢ = 500·2000 + 300·1000 = 1_000_000 + 300_000 = 1_300_000
        # Σnᵢ = 800
        # No decode → sum_r=0, bs_d=0
        self.assertEqual(f.sum_n_sq, 340000)
        self.assertEqual(f.sum_nr, 1_300_000)
        self.assertEqual(f.sum_n, 800)
        self.assertEqual(f.sum_r, 0)
        self.assertEqual(f.bs_d, 0)
        self.assertEqual(f.num_prefill_reqs, 2)
        self.assertEqual(f.num_decode_reqs, 0)
        # max_kv accounts for prefix+extend after step: max(2500, 1300)=2500
        self.assertEqual(f.max_kv, 2500)

    def test_mixed_batch_splits_by_extend_len(self):
        """A chunked-prefill step: one prefill req (extend_len=200) + two
        decoders (extend_len=1 each)."""
        s = self._new_sampler()
        # extend_lens [200, 1, 1], prefix_lens [3000, 1500, 2500]
        # → prefill: (n=200, r=3000)
        # → decode:  rⱼ ∈ {1500, 2500} (extend_len==1 path)
        f = s._extract_features(
            _fake_extend_batch([200, 1, 1], [3000, 1500, 2500]),
            18.0,
        )
        self.assertEqual(f.sum_n_sq, 200 * 200)
        self.assertEqual(f.sum_nr, 200 * 3000)
        self.assertEqual(f.sum_n, 200)
        self.assertEqual(f.sum_r, 1500 + 2500)
        self.assertEqual(f.bs_d, 2)
        self.assertEqual(f.num_prefill_reqs, 1)
        self.assertEqual(f.num_decode_reqs, 2)
        # max_kv: 3000+200 (prefill after step) vs 2500 (decode)
        self.assertEqual(f.max_kv, 3200)

    def test_non_fit_forward_mode_returns_none(self):
        s = self._new_sampler()
        self.assertIsNone(s._extract_features(_fake_idle_batch(), 10.0))

    def test_empty_batch_returns_none(self):
        s = self._new_sampler()
        empty = _fake_decode_batch([])
        empty.reqs = []
        self.assertIsNone(s._extract_features(empty, 10.0))


class TestObserveAndFlush(_SamplerHarness):
    def test_observe_writes_jsonl_row(self):
        s = self._new_sampler()
        s.observe_step(_fake_decode_batch([1000, 2000]), 51.2)
        rows = self._read_jsonl()
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["sum_r"], 3000)
        self.assertEqual(r["bs_d"], 2)
        self.assertEqual(r["sum_n_sq"], 0.0)
        self.assertEqual(r["forward_mode"], "DECODE")
        self.assertAlmostEqual(r["step_time_ms"], 51.2)
        self.assertIn("ts_ns", r)
        self.assertIn("step_idx", r)

    def test_sample_every_subsamples(self):
        s = self._new_sampler(sample_every=3)
        for _ in range(10):
            s.observe_step(_fake_decode_batch([100]), 1.0)
        rows = self._read_jsonl()
        # step_counter increments every call; sampled when counter % 3 == 0
        # → indices 3, 6, 9 → 3 rows
        self.assertEqual(len(rows), 3)

    def test_sample_every_must_be_positive(self):
        with self.assertRaises(ValueError):
            HaloCostModelSampler(self.log_path, sample_every=0)
        with self.assertRaises(ValueError):
            HaloCostModelSampler(self.log_path, sample_every=-1)

    def test_close_drains_pending_samples(self):
        s = HaloCostModelSampler(
            self.log_path, sample_every=1, flush_interval_ms=10000
        )
        # Push faster than the flush interval; close() must drain them.
        try:
            for _ in range(5):
                s.observe_step(_fake_decode_batch([100]), 1.0)
        finally:
            s.close()
        with open(self.log_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(rows), 5)
        # close() is idempotent.
        s.close()

    def test_queue_full_increments_dropped(self):
        # max_queue=2; flush_interval very long so the queue can't drain.
        s = HaloCostModelSampler(
            self.log_path,
            sample_every=1,
            max_queue=2,
            flush_interval_ms=10000,
        )
        try:
            for _ in range(5):
                s.observe_step(_fake_decode_batch([100]), 1.0)
            # 5 pushes, max_queue=2: at least 3 should be dropped.
            self.assertGreaterEqual(s.dropped_total, 3)
            # Queue should be at capacity.
            self.assertEqual(s._queue_len_for_test(), 2)
        finally:
            s.close()


class TestFactory(unittest.TestCase):
    def test_returns_none_when_path_unset(self):
        sa = SimpleNamespace(
            halo_cost_model_sample_log=None,
            halo_cost_model_sample_every=1,
        )
        self.assertIsNone(build_halo_cost_sampler_from_server_args(sa, 0))

    def test_returns_none_for_non_rank0(self):
        with tempfile.TemporaryDirectory() as d:
            sa = SimpleNamespace(
                halo_cost_model_sample_log=os.path.join(d, "x.jsonl"),
                halo_cost_model_sample_every=1,
            )
            self.assertIsNone(build_halo_cost_sampler_from_server_args(sa, 1))

    def test_returns_sampler_for_rank0_with_path(self):
        with tempfile.TemporaryDirectory() as d:
            sa = SimpleNamespace(
                halo_cost_model_sample_log=os.path.join(d, "x.jsonl"),
                halo_cost_model_sample_every=2,
            )
            s = build_halo_cost_sampler_from_server_args(sa, 0)
            try:
                self.assertIsNotNone(s)
                self.assertEqual(s.sample_every, 2)
            finally:
                s.close()


if __name__ == "__main__":
    unittest.main()
