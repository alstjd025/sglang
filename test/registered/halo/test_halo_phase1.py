"""Unit tests for managers/halo/ (Project Halo Phase 1).

See test/registered/halo/CLAUDE.md.
"""

import time
import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.admission_control.cost_model import (
    PrefillCostModel,
    TBTCostModel,
)
from sglang.srt.managers.halo import (
    HaloConfig,
    HaloController,
    HaloRejectError,
    Job,
    JobRegistry,
    JobState,
    RequestExecutionInfo,
    SlowdownTracker,
    build_halo_controller_from_server_args,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


class TestJob(unittest.TestCase):
    def test_initial_slowdown_equals_slo(self):
        """Per spec: initial slowdown_max/mean = slo (worst-case fallback)."""
        job = Job(job_id="agent-1", slo=5.0)
        self.assertEqual(job.slowdown_max, 5.0)
        self.assertEqual(job.slowdown_mean, 5.0)
        self.assertEqual(job.state, JobState.QUEUED)

    def test_admission_completion_lifecycle(self):
        job = Job(job_id="agent-1", slo=3.0)
        job.on_request_admitted("rid-a")
        self.assertEqual(job.total_request_number, 1)
        self.assertEqual(job.remaining_request_number, 1)
        self.assertIn("rid-a", job.request_ids)

        job.on_request_admitted("rid-b")
        self.assertEqual(job.total_request_number, 2)
        self.assertEqual(job.remaining_request_number, 2)

        job.on_request_completed("rid-a")
        self.assertEqual(job.remaining_request_number, 1)
        self.assertNotIn("rid-a", job.request_ids)
        self.assertNotEqual(job.state, JobState.COMPLETE)

        job.on_request_completed("rid-b")
        self.assertEqual(job.remaining_request_number, 0)
        self.assertEqual(job.state, JobState.COMPLETE)

    def test_record_sweep_promotes_to_running_and_counts_violations(self):
        job = Job(job_id="agent-1", slo=2.0)
        job.on_request_admitted("rid-a")
        job.record_sweep(max_ratio=1.5, mean_ratio=1.2)
        self.assertEqual(job.state, JobState.RUNNING)
        self.assertEqual(job.slowdown_max, 1.5)
        self.assertEqual(job.slowdown_mean, 1.2)
        self.assertEqual(job.slo_violation_count, 0)

        job.record_sweep(max_ratio=2.5, mean_ratio=1.9)
        self.assertEqual(job.slo_violation_count, 1)
        self.assertEqual(len(job.slowdown_history), 2)


class TestJobRegistry(unittest.TestCase):
    def test_lazy_creation_and_rid_mapping(self):
        reg = JobRegistry()
        job = reg.record_admission("agent-1", 5.0, "rid-a")
        self.assertEqual(job.job_id, "agent-1")
        self.assertIs(reg.job_for_request("rid-a"), job)
        self.assertEqual(len(reg.active_jobs()), 1)

    def test_completion_decrements_and_removes_mapping(self):
        reg = JobRegistry()
        job = reg.record_admission("agent-1", 5.0, "rid-a")
        affected = reg.record_completion("rid-a")
        self.assertIs(affected, job)
        self.assertEqual(job.remaining_request_number, 0)
        self.assertIsNone(reg.job_for_request("rid-a"))

    def test_record_completion_unknown_rid_returns_none(self):
        reg = JobRegistry()
        self.assertIsNone(reg.record_completion("nope"))

    def test_gc_completed_after_retain_window(self):
        reg = JobRegistry()
        reg.record_admission("agent-1", 5.0, "rid-a")
        reg.record_completion("rid-a")
        # Force completion timestamp older than retain window.
        job = reg.job_for_id("agent-1")
        self.assertIsNotNone(job)
        job.last_update_ts = time.monotonic() - 100.0
        dropped = reg.gc_completed(retain_seconds=1.0)
        self.assertEqual(dropped, 1)
        self.assertIsNone(reg.job_for_id("agent-1"))


def _make_cost_models() -> tuple:
    prefill = PrefillCostModel(alpha=0.0, beta=0.1, gamma=10.0, delta=0.0)
    tbt = TBTCostModel(a=20.0, b=0.5, c=0.0)
    return prefill, tbt


class TestSlowdownTracker(unittest.TestCase):
    def test_compute_request_slowdown_with_both_models(self):
        prefill, tbt = _make_cost_models()
        reg = JobRegistry()
        tracker = SlowdownTracker(prefill, tbt, reg)
        info = RequestExecutionInfo(
            rid="rid-a",
            job_id="agent-1",
            prompt_len=100,
            prefix_len_at_admission=0,
            decoded_tokens_so_far=10,
            kv_len_now=200,
            elapsed_ms=400.0,
        )
        # solo_prefill = 0 + 0.1*100 + 10 = 20 ms
        # solo_tbt = 20.5 ms; solo_total = 20 + 20.5*10 = 225 ms
        # ratio = 400 / 225 ≈ 1.778
        ratio = tracker.compute_request_slowdown(info)
        self.assertIsNotNone(ratio)
        self.assertAlmostEqual(ratio, 400.0 / 225.0, places=3)

    def test_no_cost_models_returns_none(self):
        reg = JobRegistry()
        tracker = SlowdownTracker(None, None, reg)
        info = RequestExecutionInfo(
            rid="rid-a", job_id="agent-1", prompt_len=10,
            prefix_len_at_admission=0, decoded_tokens_so_far=5,
            kv_len_now=10, elapsed_ms=50.0,
        )
        self.assertIsNone(tracker.compute_request_slowdown(info))

    def test_sweep_aggregates_max_and_mean(self):
        prefill, tbt = _make_cost_models()
        reg = JobRegistry()
        reg.record_admission("agent-1", 5.0, "rid-a")
        reg.record_admission("agent-1", 5.0, "rid-b")
        tracker = SlowdownTracker(prefill, tbt, reg)
        # Build infos with different elapsed → different ratios.
        infos = [
            RequestExecutionInfo(
                rid="rid-a", job_id="agent-1", prompt_len=100,
                prefix_len_at_admission=0, decoded_tokens_so_far=10,
                kv_len_now=200, elapsed_ms=225.0,   # ratio = 1.0
            ),
            RequestExecutionInfo(
                rid="rid-b", job_id="agent-1", prompt_len=100,
                prefix_len_at_admission=0, decoded_tokens_so_far=10,
                kv_len_now=200, elapsed_ms=450.0,   # ratio = 2.0
            ),
        ]
        tracker.sweep(infos)
        job = reg.job_for_id("agent-1")
        self.assertIsNotNone(job)
        self.assertAlmostEqual(job.slowdown_max, 2.0, places=3)
        self.assertAlmostEqual(job.slowdown_mean, 1.5, places=3)

    def test_sweep_skips_unknown_jobs(self):
        prefill, tbt = _make_cost_models()
        reg = JobRegistry()
        tracker = SlowdownTracker(prefill, tbt, reg)
        # job_id not in registry — should be silently skipped.
        info = RequestExecutionInfo(
            rid="rid-a", job_id="ghost-job", prompt_len=10,
            prefix_len_at_admission=0, decoded_tokens_so_far=5,
            kv_len_now=10, elapsed_ms=50.0,
        )
        tracker.sweep([info])  # no exception expected
        self.assertEqual(len(reg.all_jobs()), 0)


class TestHaloController(unittest.TestCase):
    def _config(self, **overrides) -> HaloConfig:
        kwargs = dict(enabled=True, default_slo=5.0, tick_interval_ms=100.0)
        kwargs.update(overrides)
        return HaloConfig(**kwargs)

    def test_register_request_strict_mode_rejects_missing_job_id(self):
        c = HaloController(self._config(), is_rank0=True)
        with self.assertRaises(HaloRejectError) as cm:
            c.register_request(rid="rid-1", halo_job_id=None, halo_slo=None)
        self.assertEqual(cm.exception.reason, "HALO_NO_JOB_ID")
        self.assertEqual(cm.exception.rid, "rid-1")

    def test_register_request_uses_default_slo_when_omitted(self):
        c = HaloController(self._config(default_slo=7.5), is_rank0=True)
        job = c.register_request(rid="rid-1", halo_job_id="J", halo_slo=None)
        self.assertEqual(job.slo, 7.5)

    def test_register_then_finish_decrements_remaining(self):
        c = HaloController(self._config(), is_rank0=True)
        job = c.register_request(rid="rid-1", halo_job_id="J", halo_slo=3.0)
        self.assertEqual(job.remaining_request_number, 1)
        c.on_request_finished("rid-1")
        self.assertEqual(job.remaining_request_number, 0)

    def test_should_tick_respects_wall_clock_gate(self):
        c = HaloController(self._config(tick_interval_ms=100.0), is_rank0=True)
        # Immediately after construction, last_tick = 0 (epoch), so should_tick True.
        self.assertTrue(c.should_tick())
        now = time.monotonic()
        c._last_tick_monotonic = now
        # 50ms after a tick → not yet.
        self.assertFalse(c.should_tick(now + 0.05))
        # 150ms after a tick → due.
        self.assertTrue(c.should_tick(now + 0.15))

    def test_tick_calls_build_infos_only_when_due(self):
        c = HaloController(self._config(tick_interval_ms=100.0), is_rank0=True)
        c._last_tick_monotonic = time.monotonic()  # mark "just ticked"
        build = MagicMock(return_value=[])
        c.tick(build)
        build.assert_not_called()
        # Force the gate open.
        c._last_tick_monotonic = time.monotonic() - 1.0
        c.tick(build)
        build.assert_called_once()

    def test_snapshot_returns_active_jobs(self):
        c = HaloController(self._config(), is_rank0=True)
        c.register_request(rid="rid-a", halo_job_id="agent-1", halo_slo=2.0)
        snap = c.snapshot()
        self.assertTrue(snap["enabled"])
        self.assertEqual(snap["default_slo"], 5.0)
        self.assertEqual(snap["active_jobs"], 1)
        self.assertEqual(snap["jobs"][0]["job_id"], "agent-1")


class TestFactory(unittest.TestCase):
    def test_returns_none_when_disabled(self):
        class Args:
            halo_enabled = False
        self.assertIsNone(build_halo_controller_from_server_args(Args(), True))

    def test_builds_controller_when_enabled(self):
        class Args:
            halo_enabled = True
            halo_default_slo = 4.0
            halo_tick_interval_ms = 50.0
            halo_aggregator = "max+mean"
            halo_job_log = None
            halo_prefill_cost_model_path = None
            halo_tbt_cost_model_path = None
        c = build_halo_controller_from_server_args(Args(), True)
        self.assertIsNotNone(c)
        self.assertEqual(c.config.default_slo, 4.0)
        self.assertEqual(c.config.tick_interval_ms, 50.0)


if __name__ == "__main__":
    unittest.main()
