"""Unit tests for managers/halo/ (Project Halo Phase 1, Option A + B).

See test/registered/halo/CLAUDE.md and ms_dev/halo_dev/CLAUDE.md §13.

Coverage:
- Job dataclass + lifecycle + initial slowdown==slo
- JobRegistry: register_program, admit_to_job strict mode (Q12),
  re-register 409 (Q11), SLO conflict WARN (Q10), idle GC (Q13), completion GC
- SlowdownTracker: per-request slowdown math + max/mean aggregation
- HaloController: register_program/register_request flow, factory, ticks
"""

import logging
import time
import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.admission_control.cost_model import (
    PrefillCostModel,
    TBTCostModel,
)
from sglang.srt.managers.halo import (
    REASON_DISABLED,
    REASON_JOB_ID_ALREADY_REGISTERED,
    REASON_NO_JOB_ID,
    REASON_PROGRAM_NOT_REGISTERED,
    JobAdmissionResult,
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
        job = Job(job_id="agent-1", slo=5.0)
        self.assertEqual(job.slowdown_max, 5.0)
        self.assertEqual(job.slowdown_mean, 5.0)
        self.assertEqual(job.state, JobState.QUEUED)
        # Option A fields default None / False.
        self.assertIsNone(job.total_calls_expected)
        self.assertIsNone(job.stage_sequence)
        self.assertIsNone(job.dag)
        self.assertFalse(job.from_program)

    def test_admission_completion_lifecycle(self):
        job = Job(job_id="agent-1", slo=3.0)
        job.on_request_admitted("rid-a")
        self.assertEqual(job.total_request_number, 1)
        self.assertEqual(job.remaining_request_number, 1)

        job.on_request_admitted("rid-b")
        self.assertEqual(job.remaining_request_number, 2)

        job.on_request_completed("rid-a")
        self.assertNotEqual(job.state, JobState.COMPLETE)

        job.on_request_completed("rid-b")
        self.assertEqual(job.state, JobState.COMPLETE)

    def test_record_sweep_promotes_to_running_and_counts_violations(self):
        job = Job(job_id="agent-1", slo=2.0)
        job.on_request_admitted("rid-a")
        job.record_sweep(max_ratio=1.5, mean_ratio=1.2)
        self.assertEqual(job.state, JobState.RUNNING)
        self.assertEqual(job.slo_violation_count, 0)

        job.record_sweep(max_ratio=2.5, mean_ratio=1.9)
        self.assertEqual(job.slo_violation_count, 1)
        self.assertEqual(len(job.slowdown_history), 2)

    def test_is_idle(self):
        """A pre-registered job with zero requests is idle."""
        job = Job(job_id="agent-1", slo=5.0, from_program=True)
        self.assertTrue(job.is_idle())
        job.on_request_admitted("rid-a")
        self.assertFalse(job.is_idle())


class TestJobRegistry(unittest.TestCase):
    # ---- Option A: register_program ----

    def test_register_program_fresh(self):
        reg = JobRegistry()
        fresh, job = reg.register_program("agent-1", 5.0, total_calls=4)
        self.assertTrue(fresh)
        self.assertEqual(job.job_id, "agent-1")
        self.assertEqual(job.slo, 5.0)
        self.assertEqual(job.total_calls_expected, 4)
        self.assertTrue(job.from_program)
        self.assertEqual(len(reg.active_jobs()), 1)

    def test_register_program_duplicate_returns_existing(self):
        """Q11: duplicate job_id returns (False, existing_job). Caller renders HTTP 409."""
        reg = JobRegistry()
        _, first = reg.register_program("agent-1", 5.0, total_calls=3)
        fresh, second = reg.register_program("agent-1", 8.0, total_calls=99)
        self.assertFalse(fresh)
        self.assertIs(second, first)
        self.assertEqual(second.slo, 5.0)
        self.assertEqual(second.total_calls_expected, 3)

    def test_register_program_invalid_slo_raises(self):
        reg = JobRegistry()
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                reg.register_program("agent-1", bad)

    # ---- Option B: admit_to_job strict mode ----

    def test_admit_to_job_rejects_unregistered_program(self):
        """Q12: lazy-create is gone. Unregistered job_id → reject."""
        reg = JobRegistry()
        result = reg.admit_to_job("ghost", 5.0, "rid-a")
        self.assertIsInstance(result, JobAdmissionResult)
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, REASON_PROGRAM_NOT_REGISTERED)
        self.assertIsNone(result.job)
        # No mapping established.
        self.assertIsNone(reg.job_for_request("rid-a"))
        self.assertEqual(len(reg.all_jobs()), 0)

    def test_admit_to_job_succeeds_after_register_program(self):
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        result = reg.admit_to_job("agent-1", 5.0, "rid-a")
        self.assertTrue(result.admitted)
        self.assertIs(result.job, reg.job_for_id("agent-1"))
        self.assertEqual(result.job.remaining_request_number, 1)

    def test_admit_to_job_slo_conflict_keeps_pre_registered(self):
        """Q10: pre-registered SLO wins, request slo is ignored (WARN)."""
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        with self.assertLogs("sglang.srt.managers.halo.job_registry", level="WARNING") as cm:
            result = reg.admit_to_job("agent-1", 99.0, "rid-a")
        self.assertTrue(result.admitted)
        self.assertEqual(result.job.slo, 5.0)  # pre-registered SLO retained
        self.assertTrue(any("ignoring request slo=99" in m for m in cm.output))

    # ---- completion / GC ----

    def test_completion_decrements_and_removes_mapping(self):
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        reg.admit_to_job("agent-1", 5.0, "rid-a")
        affected = reg.record_completion("rid-a")
        self.assertEqual(affected.remaining_request_number, 0)
        self.assertIsNone(reg.job_for_request("rid-a"))

    def test_record_completion_unknown_rid_returns_none(self):
        reg = JobRegistry()
        self.assertIsNone(reg.record_completion("nope"))

    def test_gc_completed_after_retain_window(self):
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        reg.admit_to_job("agent-1", 5.0, "rid-a")
        reg.record_completion("rid-a")
        job = reg.job_for_id("agent-1")
        job.last_update_ts = time.monotonic() - 100.0
        dropped = reg.gc_completed(retain_seconds=1.0)
        self.assertEqual(dropped, 1)
        self.assertIsNone(reg.job_for_id("agent-1"))

    def test_gc_idle_programs_drops_unused_pre_registered(self):
        """Q13: pre-registered programs with zero requests get GC'd after timeout."""
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        reg.register_program("agent-2", 5.0)
        reg.admit_to_job("agent-2", 5.0, "rid-a")  # agent-2 has a request now
        # Force agent-1 to look old.
        idle_job = reg.job_for_id("agent-1")
        idle_job.first_seen_ts = time.monotonic() - 1000.0
        dropped = reg.gc_idle_programs(idle_seconds=60.0)
        self.assertEqual(dropped, 1)
        self.assertIsNone(reg.job_for_id("agent-1"))
        self.assertIsNotNone(reg.job_for_id("agent-2"))  # has active request

    def test_gc_idle_programs_disabled_when_zero(self):
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        idle_job = reg.job_for_id("agent-1")
        idle_job.first_seen_ts = time.monotonic() - 10000.0
        self.assertEqual(reg.gc_idle_programs(idle_seconds=0), 0)
        self.assertIsNotNone(reg.job_for_id("agent-1"))


def _make_cost_models():
    prefill = PrefillCostModel(alpha=0.0, beta=0.1, gamma=10.0, delta=0.0)
    tbt = TBTCostModel(a=20.0, b=0.5, c=0.0)
    return prefill, tbt


class TestSlowdownTracker(unittest.TestCase):
    def test_compute_request_slowdown_with_both_models(self):
        prefill, tbt = _make_cost_models()
        reg = JobRegistry()
        tracker = SlowdownTracker(prefill, tbt, reg)
        info = RequestExecutionInfo(
            rid="rid-a", job_id="agent-1", prompt_len=100,
            prefix_len_at_admission=0, decoded_tokens_so_far=10,
            kv_len_now=200, elapsed_ms=400.0,
        )
        # solo_total = 20 + 20.5*10 = 225 → ratio = 400/225
        ratio = tracker.compute_request_slowdown(info)
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
        reg.register_program("agent-1", 5.0)
        reg.admit_to_job("agent-1", 5.0, "rid-a")
        reg.admit_to_job("agent-1", 5.0, "rid-b")
        tracker = SlowdownTracker(prefill, tbt, reg)
        infos = [
            RequestExecutionInfo(
                rid="rid-a", job_id="agent-1", prompt_len=100,
                prefix_len_at_admission=0, decoded_tokens_so_far=10,
                kv_len_now=200, elapsed_ms=225.0,  # ratio = 1.0
            ),
            RequestExecutionInfo(
                rid="rid-b", job_id="agent-1", prompt_len=100,
                prefix_len_at_admission=0, decoded_tokens_so_far=10,
                kv_len_now=200, elapsed_ms=450.0,  # ratio = 2.0
            ),
        ]
        tracker.sweep(infos)
        job = reg.job_for_id("agent-1")
        self.assertAlmostEqual(job.slowdown_max, 2.0, places=3)
        self.assertAlmostEqual(job.slowdown_mean, 1.5, places=3)

    def test_sweep_skips_unknown_jobs(self):
        prefill, tbt = _make_cost_models()
        reg = JobRegistry()
        tracker = SlowdownTracker(prefill, tbt, reg)
        info = RequestExecutionInfo(
            rid="rid-a", job_id="ghost", prompt_len=10,
            prefix_len_at_admission=0, decoded_tokens_so_far=5,
            kv_len_now=10, elapsed_ms=50.0,
        )
        tracker.sweep([info])  # no exception
        self.assertEqual(len(reg.all_jobs()), 0)


class TestHaloController(unittest.TestCase):
    def _config(self, **overrides) -> HaloConfig:
        kwargs = dict(enabled=True, default_slo=5.0, tick_interval_ms=100.0)
        kwargs.update(overrides)
        return HaloConfig(**kwargs)

    def test_register_request_rejects_missing_job_id(self):
        c = HaloController(self._config(), is_rank0=True)
        with self.assertRaises(HaloRejectError) as cm:
            c.register_request(rid="rid-1", halo_job_id=None, halo_slo=None)
        self.assertEqual(cm.exception.reason, REASON_NO_JOB_ID)

    def test_register_request_rejects_unregistered_program(self):
        """Q12: even with halo_job_id, unregistered program → reject."""
        c = HaloController(self._config(), is_rank0=True)
        with self.assertRaises(HaloRejectError) as cm:
            c.register_request(rid="rid-1", halo_job_id="unknown", halo_slo=5.0)
        self.assertEqual(cm.exception.reason, REASON_PROGRAM_NOT_REGISTERED)

    def test_register_program_then_request_succeeds(self):
        c = HaloController(self._config(), is_rank0=True)
        result = c.register_program("agent-1", slo=3.0, total_calls=4)
        self.assertTrue(result.registered)
        self.assertEqual(result.active_jobs, 1)
        job = c.register_request(rid="rid-1", halo_job_id="agent-1", halo_slo=3.0)
        self.assertEqual(job.remaining_request_number, 1)
        self.assertEqual(job.slo, 3.0)
        self.assertTrue(job.from_program)

    def test_register_program_duplicate_returns_409_payload(self):
        c = HaloController(self._config(), is_rank0=True)
        c.register_program("agent-1", slo=3.0)
        second = c.register_program("agent-1", slo=99.0)
        self.assertFalse(second.registered)
        self.assertEqual(second.reason, REASON_JOB_ID_ALREADY_REGISTERED)
        self.assertIsNotNone(second.existing)
        self.assertEqual(second.existing["slo"], 3.0)

    def test_register_program_carries_all_optional_fields(self):
        c = HaloController(self._config(), is_rank0=True)
        c.register_program(
            "agent-1", slo=5.0,
            total_calls=6,
            stage_sequence=["U", "L", "P"],
            expected_input_lens=[100, 200, 300],
            expected_output_lens=[50, 70, 90],
            dag={"type": "linear"},
        )
        job = c.registry.job_for_id("agent-1")
        self.assertEqual(job.stage_sequence, ["U", "L", "P"])
        self.assertEqual(job.expected_input_lens, [100, 200, 300])
        self.assertEqual(job.dag, {"type": "linear"})

    def test_register_program_disabled_controller(self):
        c = HaloController(HaloConfig(enabled=False), is_rank0=True)
        r = c.register_program("agent-1", slo=5.0)
        self.assertFalse(r.registered)
        self.assertEqual(r.reason, REASON_DISABLED)

    def test_should_tick_respects_wall_clock_gate(self):
        c = HaloController(self._config(tick_interval_ms=100.0), is_rank0=True)
        self.assertTrue(c.should_tick())
        now = time.monotonic()
        c._last_tick_monotonic = now
        self.assertFalse(c.should_tick(now + 0.05))
        self.assertTrue(c.should_tick(now + 0.15))

    def test_tick_calls_build_infos_only_when_due(self):
        c = HaloController(self._config(tick_interval_ms=100.0), is_rank0=True)
        c._last_tick_monotonic = time.monotonic()
        build = MagicMock(return_value=[])
        c.tick(build)
        build.assert_not_called()
        c._last_tick_monotonic = time.monotonic() - 1.0
        c.tick(build)
        build.assert_called_once()

    def test_tick_runs_idle_program_gc(self):
        c = HaloController(
            self._config(tick_interval_ms=0.0),  # always tick
            is_rank0=True,
        )
        c.config.program_idle_timeout_seconds = 0.001
        c.register_program("agent-1", slo=5.0)
        time.sleep(0.005)
        c.tick(build_infos=lambda: [])
        self.assertIsNone(c.registry.job_for_id("agent-1"))

    def test_halo_bypass_field_does_not_affect_controller(self):
        """halo_bypass is a scheduler-level concern (skip the gate). The
        controller itself doesn't see the flag — admit_to_job still gets
        called only for non-bypassed requests. We just confirm here that
        a bypassed-request never reaches `register_request` and therefore
        never touches the registry counters."""
        c = HaloController(self._config(), is_rank0=True)
        c.register_program("agent-1", slo=2.0)
        # No call to register_request — emulates the scheduler shortcut.
        snap = c.snapshot()
        self.assertEqual(snap["active_jobs"], 1)
        job = c.registry.job_for_id("agent-1")
        self.assertEqual(job.total_request_number, 0)  # bypass skips counter

    def test_snapshot_returns_active_jobs(self):
        c = HaloController(self._config(), is_rank0=True)
        c.register_program("agent-1", slo=2.0, total_calls=5)
        c.register_request(rid="rid-a", halo_job_id="agent-1", halo_slo=2.0)
        snap = c.snapshot()
        self.assertTrue(snap["enabled"])
        self.assertEqual(snap["active_jobs"], 1)
        self.assertEqual(snap["jobs"][0]["job_id"], "agent-1")
        self.assertEqual(snap["jobs"][0]["total_calls_expected"], 5)
        self.assertEqual(snap["program_idle_timeout_seconds"], 300.0)


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
            halo_program_idle_timeout_seconds = 200.0
        c = build_halo_controller_from_server_args(Args(), True)
        self.assertIsNotNone(c)
        self.assertEqual(c.config.default_slo, 4.0)
        self.assertEqual(c.config.program_idle_timeout_seconds, 200.0)


if __name__ == "__main__":
    unittest.main()
