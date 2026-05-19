"""Unit tests for managers/halo/ (Project Halo Phase 1, Option A + B).

See test/registered/halo/CLAUDE.md and ms_dev/halo_dev/CLAUDE.md §13.

Coverage:
- Job dataclass + lifecycle + initial slowdown==slo
- JobRegistry: register_program, admit_to_job strict mode (Q12),
  re-register 409 (Q11), SLO conflict WARN (Q10), idle GC (Q13), completion GC
- SlowdownTracker: per-request slowdown math + max/mean aggregation
- HaloController: register_program/register_request flow, factory, ticks
"""

import os
import time
import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.halo.admission_control.cost_model import (
    PrefillCostModel,
    TBTCostModel,
)
from sglang.srt.managers.halo import (
    REASON_DISABLED,
    REASON_JOB_ID_ALREADY_REGISTERED,
    REASON_NO_JOB_ID,
    REASON_PROGRAM_NOT_REGISTERED,
    HaloConfig,
    HaloController,
    HaloRejectError,
    Job,
    JobAdmissionResult,
    JobCallSpan,
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
        self.assertEqual(job.virtual_job_slowdown, 5.0)
        self.assertEqual(job.state, JobState.QUEUED)
        # Option A fields default None / False.
        self.assertIsNone(job.total_calls_expected)
        self.assertIsNone(job.stage_sequence)
        self.assertIsNone(job.dag)
        self.assertFalse(job.from_program)

    def test_admission_completion_lifecycle(self):
        """Counters track per-call admit/finish. State does NOT auto-flip
        to COMPLETE — the client must send halo_job_done (or the
        quiescent safety net trips). See halo_api_reference.md.
        """
        job = Job(job_id="agent-1", slo=3.0)
        job.on_request_admitted("rid-a")
        self.assertEqual(job.total_request_number, 1)
        self.assertEqual(job.remaining_request_number, 1)

        job.on_request_admitted("rid-b")
        self.assertEqual(job.remaining_request_number, 2)

        job.on_request_completed("rid-a")
        self.assertNotEqual(job.state, JobState.COMPLETE)

        job.on_request_completed("rid-b")
        # All in-flight done but state still NOT COMPLETE — explicit
        # halo_job_done required. (Old auto-COMPLETE on remaining==0 was
        # too eager and broke mid-chain pauses; see commit f25fd41ea
        # and follow-up rollback.)
        self.assertNotEqual(job.state, JobState.COMPLETE)
        job.mark_done()
        self.assertEqual(job.state, JobState.COMPLETE)

    def test_record_vjs_promotes_to_running_and_counts_violations(self):
        job = Job(job_id="agent-1", slo=2.0)
        job.on_request_admitted("rid-a")
        job.record_vjs(1.5)
        self.assertEqual(job.state, JobState.RUNNING)
        self.assertEqual(job.virtual_job_slowdown, 1.5)
        self.assertEqual(job.slo_violation_count, 0)

        job.record_vjs(2.5)
        self.assertEqual(job.virtual_job_slowdown, 2.5)
        self.assertEqual(job.slo_violation_count, 1)
        self.assertEqual(len(job.slowdown_history), 2)

    def test_completion_does_not_mark_complete_mid_chain(self):
        """Multi-round chains (e.g. parallel_tool_delay) finish each round
        then sleep before the next. `remaining=0` between rounds must NOT
        flip the job to COMPLETE. Even after every expected call has
        finished, the state stays RUNNING/QUEUED until the client
        explicitly signals via `halo_job_done`. The total_calls_expected
        count is advisory only (Phase 1 — workloads with dynamic DAGs
        won't know it up front).
        """
        job = Job(job_id="agent-1", slo=5.0, total_calls_expected=4)
        # Round 1 — both calls finish, remaining briefly 0.
        job.on_request_admitted("r1")
        job.on_request_admitted("r2")
        job.on_request_completed("r1")
        job.on_request_completed("r2")
        self.assertEqual(job.remaining_request_number, 0)
        self.assertNotEqual(job.state, JobState.COMPLETE)
        # Round 2.
        job.on_request_admitted("r3")
        job.on_request_admitted("r4")
        job.on_request_completed("r3")
        job.on_request_completed("r4")
        # Even though total_request_number now == total_calls_expected,
        # state stays not-COMPLETE — explicit signal required.
        self.assertNotEqual(job.state, JobState.COMPLETE)
        job.mark_done()
        self.assertEqual(job.state, JobState.COMPLETE)

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
        with self.assertLogs(
            "sglang.srt.managers.halo.job_registry", level="WARNING"
        ) as cm:
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
        """gc_completed only fires on jobs in COMPLETE state. The
        explicit signal pattern is: client sends halo_job_done=true on
        the last call → controller.on_request_finished marks job COMPLETE
        *before* dropping the rid mapping → gc_completed picks it up
        after retain_seconds.
        """
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        reg.admit_to_job("agent-1", 5.0, "rid-a")
        # Client signals "this is the last call" on its finish.
        reg.mark_job_done("rid-a")  # ← order matters: mark first
        reg.record_completion("rid-a")  # ← then drop the rid mapping
        job = reg.job_for_id("agent-1")
        self.assertEqual(job.state, JobState.COMPLETE)
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

    def test_mark_job_done_via_registry(self):
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        reg.admit_to_job("agent-1", 5.0, "rid-a")
        affected = reg.mark_job_done("rid-a")
        self.assertIsNotNone(affected)
        self.assertEqual(affected.state, JobState.COMPLETE)
        # mark_job_done is decoupled from record_completion; both can
        # run in either order without inconsistency.
        reg.record_completion("rid-a")
        self.assertEqual(reg.job_for_id("agent-1").remaining_request_number, 0)

    def test_gc_quiescent_jobs_flips_long_idle(self):
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        reg.admit_to_job("agent-1", 5.0, "rid-a")
        reg.record_completion("rid-a")
        job = reg.job_for_id("agent-1")
        self.assertNotEqual(job.state, JobState.COMPLETE)
        # Force last_update_ts to be old.
        job.last_update_ts = time.monotonic() - 100.0
        flipped = reg.gc_quiescent_jobs(quiescent_seconds=1.0)
        self.assertEqual(len(flipped), 1)
        self.assertIs(flipped[0], job)
        self.assertEqual(job.state, JobState.COMPLETE)

    def test_gc_quiescent_jobs_skips_in_flight(self):
        """A job with active in-flight requests must not be force-completed
        even if last_update_ts is old (e.g. a long slow generation)."""
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        reg.admit_to_job("agent-1", 5.0, "rid-a")
        job = reg.job_for_id("agent-1")
        job.last_update_ts = time.monotonic() - 1000.0  # very stale
        flipped = reg.gc_quiescent_jobs(quiescent_seconds=1.0)
        self.assertEqual(flipped, [])
        self.assertNotEqual(job.state, JobState.COMPLETE)

    def test_gc_quiescent_disabled_when_zero(self):
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        reg.admit_to_job("agent-1", 5.0, "rid-a")
        reg.record_completion("rid-a")
        job = reg.job_for_id("agent-1")
        job.last_update_ts = time.monotonic() - 1000.0
        self.assertEqual(reg.gc_quiescent_jobs(quiescent_seconds=0.0), [])
        self.assertNotEqual(job.state, JobState.COMPLETE)

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


def _step_model():
    """Step cost model with trivial coefficients for predictable arithmetic:
    prefill_solo(n, r) = (prompt_len - prefix_len)   [θ_p3=1, rest 0, θ_c_p=0]
    decode_step(kvs)   = batch_size + 10             [θ_d2=1, θ_c_d=10, rest 0]
    """
    from sglang.srt.managers.halo.admission_control.cost_model import HaloStepCostModel

    return HaloStepCostModel(
        theta_p1=0.0,
        theta_p2=0.0,
        theta_p3=1.0,
        theta_d1=0.0,
        theta_d2=1.0,
        theta_c=0.0,
        theta_c_p=0.0,
        theta_c_d=10.0,
        form="halo_step_split_v1",
    )


def _span(admitted, end, prompt=100, prefix=0, decoded=10, kv=110):
    """A JobCallSpan with sensible defaults. ts are monotonic seconds."""
    return JobCallSpan(
        admitted_ts=admitted,
        end_ts=end,
        prompt_len=prompt,
        prefix_len=prefix,
        decoded_tokens=decoded,
        kv_len=kv,
    )


class TestSlowdownTrackerPrimitives(unittest.TestCase):
    """prefill_solo_ms / decode_step_ms — the per-call cost-model primitives."""

    def test_prefill_solo_excludes_cached_prefix(self):
        # n = prompt_len - prefix_len: the radix-cache hit is not prefilled.
        t = SlowdownTracker(None, None, JobRegistry(), step_cost=_step_model())
        self.assertAlmostEqual(t.prefill_solo_ms(1000, 600), 400.0, places=6)
        self.assertAlmostEqual(t.prefill_solo_ms(1000, 0), 1000.0, places=6)
        # prefix >= prompt → clamped to 0.
        self.assertAlmostEqual(t.prefill_solo_ms(100, 500), 0.0, places=6)

    def test_decode_step_scales_with_batch(self):
        t = SlowdownTracker(None, None, JobRegistry(), step_cost=_step_model())
        self.assertAlmostEqual(t.decode_step_ms([110]), 11.0, places=6)
        self.assertAlmostEqual(t.decode_step_ms([110, 110]), 12.0, places=6)
        self.assertAlmostEqual(t.decode_step_ms([110, 110, 110]), 13.0, places=6)
        self.assertAlmostEqual(t.decode_step_ms([]), 0.0, places=6)

    def test_no_cost_model_primitives_return_zero(self):
        t = SlowdownTracker(None, None, JobRegistry())
        self.assertFalse(t.has_cost_model)
        self.assertEqual(t.prefill_solo_ms(100, 0), 0.0)
        self.assertEqual(t.decode_step_ms([100]), 0.0)

    def test_step_cost_implies_has_cost_model(self):
        t = SlowdownTracker(None, None, JobRegistry(), step_cost=_step_model())
        self.assertTrue(t.has_cost_model)


class TestComputeJobVjs(unittest.TestCase):
    """compute_job_vjs — job-lifetime VJS via the stage-merge model.

    With _step_model(): prefill_solo(100,0)=100, decode_step([110])=11,
    decode_step([110,110])=12. A single decode call's solo (decoded=10) is
    100 + 10*11 = 210.
    """

    def setUp(self):
        self.t = SlowdownTracker(None, None, JobRegistry(), step_cost=_step_model())

    def test_empty_spans_returns_fallback(self):
        self.assertEqual(self.t.compute_job_vjs([], fallback_slo=5.0), 5.0)

    def test_no_cost_model_returns_fallback(self):
        bare = SlowdownTracker(None, None, JobRegistry())
        self.assertEqual(bare.compute_job_vjs([_span(0.0, 1.0)], fallback_slo=7.0), 7.0)

    def test_single_call(self):
        # span [0,1]s → actual 1000 ms; solo = 100 + 10*11 = 210.
        vjs = self.t.compute_job_vjs([_span(0.0, 1.0)], fallback_slo=5.0)
        self.assertAlmostEqual(vjs, 1000.0 / 210.0, places=4)

    def test_sequential_calls_sum(self):
        # Two non-overlapping calls → two stages → actual + solo both sum.
        spans = [_span(0.0, 1.0), _span(2.0, 3.0)]
        vjs = self.t.compute_job_vjs(spans, fallback_slo=5.0)
        self.assertAlmostEqual(vjs, 2000.0 / 420.0, places=4)

    def test_tool_delay_gap_excluded(self):
        # A long gap between two calls is NOT counted — identical to the
        # back-to-back sequential case.
        gapped = [_span(0.0, 1.0), _span(100.0, 101.0)]
        vjs = self.t.compute_job_vjs(gapped, fallback_slo=5.0)
        self.assertAlmostEqual(vjs, 2000.0 / 420.0, places=4)

    def test_concurrent_calls_critical_path(self):
        # Overlapping [0,2] and [1,3] → one stage.
        # actual = 3 - 0 = 3 s = 3000 ms.
        # decode_step batch-2 = 12; each call_solo = 100 + 10*12 = 220;
        # stage_solo = max = 220.
        spans = [_span(0.0, 2.0), _span(1.0, 3.0)]
        vjs = self.t.compute_job_vjs(spans, fallback_slo=5.0)
        self.assertAlmostEqual(vjs, 3000.0 / 220.0, places=4)

    def test_concurrent_solo_uses_batch_k_decode_step(self):
        # The concurrent stage charges decode steps at the batch-of-2 step
        # time (12), not batch-of-1 (11) — the job's own self-batching.
        # If it used batch-1, stage_solo would be 210, not 220.
        spans = [_span(0.0, 2.0), _span(1.0, 3.0)]
        vjs = self.t.compute_job_vjs(spans, fallback_slo=5.0)
        self.assertAlmostEqual(vjs, 3000.0 / 220.0, places=4)
        self.assertNotAlmostEqual(vjs, 3000.0 / 210.0, places=4)

    def test_prefill_only_call_excludes_cached_prefix(self):
        # decoded=0 → no decode term. n = 1000 - 600 = 400.
        sp = _span(0.0, 1.0, prompt=1000, prefix=600, decoded=0, kv=1000)
        vjs = self.t.compute_job_vjs([sp], fallback_slo=5.0)
        # actual 1000 ms / solo 400 = 2.5 (would be 1.0 if prefix not excluded).
        self.assertAlmostEqual(vjs, 2.5, places=4)

    def test_completed_plus_inflight(self):
        # One completed span [0,1] + one in-flight span [5,6.5], sequential.
        # stage1: actual 1000, solo 210. stage2: actual 1500, solo 210.
        spans = [_span(0.0, 1.0), _span(5.0, 6.5)]
        vjs = self.t.compute_job_vjs(spans, fallback_slo=5.0)
        self.assertAlmostEqual(vjs, 2500.0 / 420.0, places=4)


class TestSweep(unittest.TestCase):
    """SlowdownTracker.sweep — recompute each job's virtual_job_slowdown."""

    def test_sweep_records_vjs(self):
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        reg.admit_to_job("agent-1", 5.0, "rid-a")
        t = SlowdownTracker(None, None, reg, step_cost=_step_model())
        info = RequestExecutionInfo(
            rid="rid-a",
            job_id="agent-1",
            prompt_len=100,
            prefix_len_at_admission=0,
            decoded_tokens_so_far=10,
            kv_len_now=110,
            elapsed_ms=1000.0,
            admitted_ts=0.0,
        )
        t.sweep([info])
        job = reg.job_for_id("agent-1")
        # in-flight span: end_ts = 0 + 1000/1000 = 1.0 s → actual 1000, solo 210.
        self.assertAlmostEqual(job.virtual_job_slowdown, 1000.0 / 210.0, places=3)

    def test_sweep_skips_unknown_jobs(self):
        reg = JobRegistry()
        t = SlowdownTracker(None, None, reg, step_cost=_step_model())
        info = RequestExecutionInfo(
            rid="r",
            job_id="ghost",
            prompt_len=10,
            prefix_len_at_admission=0,
            decoded_tokens_so_far=5,
            kv_len_now=10,
            elapsed_ms=50.0,
            admitted_ts=0.0,
        )
        t.sweep([info])  # no exception
        self.assertEqual(len(reg.all_jobs()), 0)

    def test_sweep_includes_completed_calls(self):
        reg = JobRegistry()
        reg.register_program("agent-1", 5.0)
        reg.admit_to_job("agent-1", 5.0, "rid-2")
        job = reg.job_for_id("agent-1")
        # A finished call already frozen onto the job.
        job.record_completed_call(_span(0.0, 1.0))
        t = SlowdownTracker(None, None, reg, step_cost=_step_model())
        # In-flight second call, sequential after the completed one.
        info = RequestExecutionInfo(
            rid="rid-2",
            job_id="agent-1",
            prompt_len=100,
            prefix_len_at_admission=0,
            decoded_tokens_so_far=10,
            kv_len_now=110,
            elapsed_ms=1000.0,
            admitted_ts=5.0,
        )
        t.sweep([info])
        # completed [0,1] + in-flight [5,6] → 2 stages → 2000/420.
        self.assertAlmostEqual(job.virtual_job_slowdown, 2000.0 / 420.0, places=3)

    def test_step_cost_supersedes_legacy(self):
        # With both step + legacy models, compute_job_vjs uses the step model.
        prefill, tbt = _make_cost_models()
        t = SlowdownTracker(
            prefill_cost=prefill,
            tbt_cost=tbt,
            registry=JobRegistry(),
            step_cost=_step_model(),
        )
        vjs = t.compute_job_vjs([_span(0.0, 1.0)], fallback_slo=5.0)
        # Step model wins: 1000 / 210.
        self.assertAlmostEqual(vjs, 1000.0 / 210.0, places=4)

    def test_legacy_cost_model_path(self):
        # Legacy PrefillCostModel + TBTCostModel still drive compute_job_vjs.
        prefill, tbt = _make_cost_models()
        t = SlowdownTracker(prefill, tbt, JobRegistry())
        self.assertTrue(t.has_cost_model)
        vjs = t.compute_job_vjs([_span(0.0, 1.0)], fallback_slo=5.0)
        self.assertGreater(vjs, 0.0)

    def test_no_cost_models_has_cost_model_false(self):
        self.assertFalse(SlowdownTracker(None, None, JobRegistry()).has_cost_model)


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
            "agent-1",
            slo=5.0,
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

    def test_on_request_finished_with_halo_job_done_signal(self):
        """Client sends halo_job_done=True on the last call → controller
        flips the job to COMPLETE on its finish hook."""
        c = HaloController(self._config(), is_rank0=True)
        c.register_program("agent-1", slo=2.0, total_calls=3)
        c.register_request(rid="rid-a", halo_job_id="agent-1", halo_slo=2.0)
        # Plain finish — state stays RUNNING/QUEUED.
        c.on_request_finished("rid-a", halo_job_done=False)
        job = c.registry.job_for_id("agent-1")
        self.assertNotEqual(job.state, JobState.COMPLETE)
        # Now the last call comes with halo_job_done=True.
        c.register_request(rid="rid-b", halo_job_id="agent-1", halo_slo=2.0)
        c.on_request_finished("rid-b", halo_job_done=True)
        self.assertEqual(job.state, JobState.COMPLETE)

    def test_jsonl_emits_job_complete_event_on_done_signal(self):
        """The job_complete row in halo_jobs.jsonl with reason=halo_job_done
        is what makes job termination visible to post-hoc analysis even
        when the COMPLETE state doesn't survive the next sweep snapshot."""
        import json as _json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "halo_jobs.jsonl")
            c = HaloController(self._config(job_log_path=path), is_rank0=True)
            c.register_program("agent-1", slo=2.0, total_calls=2)
            c.register_request(rid="rid-a", halo_job_id="agent-1", halo_slo=2.0)
            c.on_request_finished("rid-a", halo_job_done=True)
            c.close()  # flush
            with open(path) as f:
                events = [_json.loads(l) for l in f]
        kinds = [e.get("event") for e in events]
        self.assertIn("register_program", kinds)
        self.assertIn("job_complete", kinds)
        # The complete event carries reason and a job dict.
        completes = [e for e in events if e.get("event") == "job_complete"]
        self.assertEqual(len(completes), 1)
        self.assertEqual(completes[0]["reason"], "halo_job_done")
        self.assertEqual(completes[0]["job"]["job_id"], "agent-1")
        self.assertEqual(completes[0]["job"]["state"], "complete")

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


class TestPhase2AdmissionIntegration(unittest.TestCase):
    """Phase 2: register_request runs Stage A (predictive admission) when
    admission_mode != off and active_request_infos is supplied.

    See ms_dev/halo_dev/admission_design.md §9 (workflow)."""

    def _step_cost(self):
        from sglang.srt.managers.halo.admission_control.cost_model import (
            HaloStepCostModel,
        )

        # Same fixture as the cost-model unit tests.
        return HaloStepCostModel(
            theta_p1=1e-7,
            theta_p2=4e-7,
            theta_p3=0.013,
            theta_d1=1.5e-5,
            theta_d2=0.18,
            theta_c=22.0,
            theta_c_p=70.0,
            theta_c_d=22.0,
            form="halo_step_split_v1",
        )

    def _config(self, **overrides) -> HaloConfig:
        kwargs = dict(
            enabled=True,
            default_slo=5.0,
            tick_interval_ms=100.0,
        )
        kwargs.update(overrides)
        return HaloConfig(**kwargs)

    def _make_controller(self, *, step_path=None, **cfg_overrides) -> HaloController:
        """Build a controller wired with the test step-cost-model.

        If step_path is None, write the fixture to a temp file. Caller can
        supply step_path to share a single JSON across operations (e.g. to
        let the controller's normal __init__ path build the admission log).
        """
        if step_path is None:
            import tempfile

            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
            tmp.close()
            self._step_cost().to_json(tmp.name)
            step_path = tmp.name
            self.addCleanup(lambda p=tmp.name: os.unlink(p))
        cfg = self._config(step_cost_model_path=step_path, **cfg_overrides)
        return HaloController(cfg, is_rank0=True)

    def test_admission_off_skips_stage_a(self):
        c = self._make_controller(admission_mode="off")
        c.register_program("agent-1", slo=5.0, total_calls=2)
        # No active_request_infos passed — strict-mode-only path.
        job = c.register_request(
            rid="rid-a",
            halo_job_id="agent-1",
            halo_slo=5.0,
        )
        self.assertEqual(job.job_id, "agent-1")

    def test_waiting_queue_requests_excluded_from_vss(self):
        """결함 A fix: a request with kv_len_now == 0 (sitting in the waiting
        queue / retracted — in NO forward step) is excluded from the Stage A
        VSS batch. A request with committed KV is kept."""
        c = self._make_controller(admission_mode="job")
        c.register_program("running-1", slo=5.0)
        c.register_program("waiting-1", slo=5.0)
        c.registry.admit_to_job("running-1", 5.0, "r-run")
        c.registry.admit_to_job("waiting-1", 5.0, "r-wait")
        infos = [
            RequestExecutionInfo(
                rid="r-run",
                job_id="running-1",
                prompt_len=1000,
                prefix_len_at_admission=0,
                decoded_tokens_so_far=5,
                kv_len_now=1005,
                elapsed_ms=100.0,
                admitted_ts=0.0,
            ),
            RequestExecutionInfo(
                rid="r-wait",
                job_id="waiting-1",
                prompt_len=1000,
                prefix_len_at_admission=0,
                decoded_tokens_so_far=0,
                kv_len_now=0,
                elapsed_ms=100.0,
                admitted_ts=0.0,
            ),
        ]
        built = c._build_active_jobs_input(infos, exclude_job_id=None)
        ids = {j.job_id for j in built}
        self.assertIn("running-1", ids)
        self.assertNotIn("waiting-1", ids)  # kv_len_now == 0 → excluded

    def test_admission_job_mode_admits_when_below_threshold(self):
        """No active jobs → violation_ratio = 0 → admit."""
        c = self._make_controller(
            admission_mode="job", admission_violation_threshold=0.2
        )
        c.register_program("agent-1", slo=5.0, total_calls=2)
        job = c.register_request(
            rid="rid-a",
            halo_job_id="agent-1",
            halo_slo=5.0,
            prompt_len=1000,
            prefix_len=500,
            active_request_infos=[],  # empty → ADMIT
        )
        self.assertEqual(job.job_id, "agent-1")

    def test_admission_job_mode_rejects_when_predictions_violate(self):
        """An active job whose predicted VSS exceeds its SLO ⇒
        violation_ratio = 1.0 > 0.2 ⇒ reject. The VSS arithmetic itself is
        unit-tested in test_halo_admission_predictors.py; here we inject a
        predictor that reports a violating value so the integration path
        (predict → decide_admission → HaloRejectError) is exercised
        deterministically."""
        c = self._make_controller(
            admission_mode="job",
            admission_violation_threshold=0.2,
        )
        c.register_program("active-1", slo=2.0, total_calls=2)
        c.register_program("new-1", slo=5.0, total_calls=2)
        c.registry.admit_to_job("active-1", 2.0, "rid-x")

        from sglang.srt.managers.halo.admission_decision import (
            AdmissionPredictor,
        )

        class _RejectAllJobPredictor(AdmissionPredictor):
            mode_name = "job"

            def predict(self, active_jobs, new_job, chunked_prefill_size=None):
                return {j.job_id: 999.0 for j in active_jobs}

        c.admission_predictor = _RejectAllJobPredictor(c.tracker.step_cost)
        active_info = RequestExecutionInfo(
            rid="rid-x",
            job_id="active-1",
            prompt_len=2000,
            prefix_len_at_admission=1800,
            decoded_tokens_so_far=200,
            kv_len_now=2200,
            elapsed_ms=200_000.0,
            admitted_ts=0.0,
        )
        with self.assertRaises(HaloRejectError) as cm:
            c.register_request(
                rid="rid-b",
                halo_job_id="new-1",
                halo_slo=5.0,
                prompt_len=4000,
                prefix_len=0,
                active_request_infos=[active_info],
            )
        self.assertEqual(cm.exception.reason, "HALO_ADMISSION_PREDICTED")

    def test_admission_dry_run_admits_despite_reject_decision(self):
        c = self._make_controller(
            admission_mode="job",
            admission_violation_threshold=0.2,
            admission_dry_run=True,
        )
        c.register_program("active-1", slo=2.0, total_calls=2)
        c.register_program("new-1", slo=5.0, total_calls=2)
        c.registry.admit_to_job("active-1", 2.0, "rid-x")

        from sglang.srt.managers.halo.admission_decision import (
            AdmissionPredictor,
        )

        class _RejectAllJobPredictor(AdmissionPredictor):
            mode_name = "job"

            def predict(self, active_jobs, new_job, chunked_prefill_size=None):
                return {j.job_id: 999.0 for j in active_jobs}

        c.admission_predictor = _RejectAllJobPredictor(c.tracker.step_cost)
        active_info = RequestExecutionInfo(
            rid="rid-x",
            job_id="active-1",
            prompt_len=2000,
            prefix_len_at_admission=1800,
            decoded_tokens_so_far=200,
            kv_len_now=2200,
            elapsed_ms=200_000.0,
            admitted_ts=0.0,
        )
        # Dry-run: decision is REJECT but controller admits anyway.
        job = c.register_request(
            rid="rid-b",
            halo_job_id="new-1",
            halo_slo=5.0,
            prompt_len=4000,
            prefix_len=0,
            active_request_infos=[active_info],
        )
        self.assertEqual(job.job_id, "new-1")

    def test_admission_decision_log_path_written(self):
        import json as _json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "admission_decisions.jsonl")
            c = self._make_controller(
                admission_mode="job",
                admission_decision_log_path=path,
            )
            c.register_program("new-1", slo=5.0, total_calls=2)
            c.register_request(
                rid="rid-a",
                halo_job_id="new-1",
                halo_slo=5.0,
                prompt_len=1000,
                prefix_len=500,
                active_request_infos=[],
            )
            c.close()
            with open(path) as f:
                rows = [_json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rid"], "rid-a")
        self.assertEqual(rows[0]["job_id"], "new-1")
        self.assertEqual(rows[0]["decision"], "admit")
        self.assertEqual(rows[0]["mode"], "job")

    def test_level2_falls_back_to_job_with_warn(self):
        """DEPRECATED 2026-05-15 — admission_mode=level2 must fall back to
        mode=job (JobSlowdownAdmissionPredictor) with a startup warning.
        The lookahead class lingers in the source tree but is not on the
        admission path."""
        with self.assertLogs(
            "sglang.srt.managers.halo.controller", level="WARNING"
        ) as captured:
            c = self._make_controller(admission_mode="level2")
        self.assertIsNotNone(c.admission_predictor)
        self.assertEqual(c.admission_predictor.mode_name, "job")
        self.assertTrue(
            any("DEPRECATED" in m for m in captured.output),
            captured.output,
        )

    def test_followup_requests_skip_stage_a(self):
        """Job-level reject (2026-05-15): admission decision is taken
        *once per job* — at the first request. Follow-up requests inside
        the same job must bypass Stage A, even when active load would
        trip it. We verify this by stuffing a violating predictor (mock)
        and confirming the *second* register_request succeeds anyway."""
        c = self._make_controller(
            admission_mode="job", admission_violation_threshold=0.2
        )
        c.register_program("agent-1", slo=5.0, total_calls=2)
        # First request admits.
        c.register_request(
            rid="rid-1",
            halo_job_id="agent-1",
            halo_slo=5.0,
            prompt_len=1000,
            prefix_len=500,
            active_request_infos=[],
        )
        # Force-replace the predictor with one that always rejects. If
        # Stage A were called, this would raise.
        from sglang.srt.managers.halo.admission_decision import (
            AdmissionPredictor,
        )

        class _AlwaysRejectPredictor(AdmissionPredictor):
            mode_name = "test_reject_all"

            def predict(self, active_jobs, new_job, chunked_prefill_size=None):
                return {j.job_id: 999.0 for j in active_jobs}

        c.admission_predictor = _AlwaysRejectPredictor(c.tracker.step_cost)
        # Build a fake active info so the controller would otherwise run
        # Stage A. The second call belongs to the same job → must skip.
        info = RequestExecutionInfo(
            rid="rid-1",
            job_id="agent-1",
            prompt_len=1000,
            prefix_len_at_admission=500,
            decoded_tokens_so_far=20,
            kv_len_now=1020,
            elapsed_ms=200.0,
            admitted_ts=0.0,
        )
        job = c.register_request(
            rid="rid-2",
            halo_job_id="agent-1",
            halo_slo=5.0,
            prompt_len=2000,
            prefix_len=1500,
            active_request_infos=[info],
        )
        self.assertEqual(job.job_id, "agent-1")

    def test_request_mode_admits_when_below_threshold(self):
        """mode=request: empty active set → violation_ratio 0 → admit."""
        c = self._make_controller(
            admission_mode="request", admission_violation_threshold=0.2
        )
        c.register_program("agent-1", slo=5.0, total_calls=2)
        job = c.register_request(
            rid="rid-a",
            halo_job_id="agent-1",
            halo_slo=5.0,
            prompt_len=1000,
            prefix_len=500,
            active_request_infos=[],
        )
        self.assertEqual(job.job_id, "agent-1")

    def test_request_mode_runs_stage_a_on_every_request(self):
        """Request-scoped baseline: unlike mode=job, mode=request runs
        Stage A on *every* request — including follow-ups, so a mid-chain
        request CAN be rejected. Counterpart to
        test_followup_requests_skip_stage_a."""
        c = self._make_controller(
            admission_mode="request", admission_violation_threshold=0.2
        )
        c.register_program("agent-1", slo=5.0, total_calls=3)
        # First request admits (empty active set).
        c.register_request(
            rid="rid-1",
            halo_job_id="agent-1",
            halo_slo=5.0,
            prompt_len=1000,
            prefix_len=500,
            active_request_infos=[],
        )
        # Force a predictor that always rejects. In request mode the gate
        # runs even for a follow-up → the second request must be rejected.
        from sglang.srt.managers.halo.admission_decision import (
            AdmissionPredictor,
        )

        class _AlwaysRejectRequestPredictor(AdmissionPredictor):
            mode_name = "request"

            def predict(self, active_jobs, new_job, chunked_prefill_size=None):
                return {
                    call.rid: 999.0 for job in active_jobs for call in job.active_calls
                }

        c.admission_predictor = _AlwaysRejectRequestPredictor(c.tracker.step_cost)
        info = RequestExecutionInfo(
            rid="rid-1",
            job_id="agent-1",
            prompt_len=1000,
            prefix_len_at_admission=500,
            decoded_tokens_so_far=20,
            kv_len_now=1020,
            elapsed_ms=200.0,
            admitted_ts=0.0,
        )
        with self.assertRaises(HaloRejectError) as cm:
            c.register_request(
                rid="rid-2",
                halo_job_id="agent-1",
                halo_slo=5.0,
                prompt_len=2000,
                prefix_len=1500,
                active_request_infos=[info],
            )
        self.assertEqual(cm.exception.reason, "HALO_ADMISSION_PREDICTED")

    def test_request_mode_decision_log_has_is_first_call(self):
        """The decision-log row carries is_first_call so post-hoc analysis
        can separate chain-start rejects from mid-chain rejects."""
        import json as _json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "admission_decisions.jsonl")
            c = self._make_controller(
                admission_mode="request",
                admission_decision_log_path=path,
            )
            c.register_program("new-1", slo=5.0, total_calls=2)
            c.register_request(
                rid="rid-a",
                halo_job_id="new-1",
                halo_slo=5.0,
                prompt_len=1000,
                prefix_len=500,
                active_request_infos=[],
            )
            c.close()
            with open(path) as f:
                rows = [_json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "request")
        self.assertTrue(rows[0]["is_first_call"])


class TestPhase2KvCap(unittest.TestCase):
    """Phase 2 Stage B′ — KV-cache hard cap.

    Reject a new job's first request when the KV-cache pool usage ratio is
    at/above admission_kv_cap_ratio. Independent of admission_mode; skipped
    for follow-up requests (no pending queue yet — Phase 2 admits a job's
    subsequent requests unconditionally). See admission_design.md §5."""

    def _config(self, **overrides) -> HaloConfig:
        kwargs = dict(enabled=True, default_slo=5.0)
        kwargs.update(overrides)
        return HaloConfig(**kwargs)

    def test_disabled_by_default(self):
        """Default ratio 0.0 → KV cap inert even at 99% usage."""
        c = HaloController(self._config(), is_rank0=True)
        self.assertFalse(c.kv_cap_enabled)
        c.register_program("a", slo=5.0)
        job = c.register_request(
            rid="r1",
            halo_job_id="a",
            halo_slo=5.0,
            kv_usage_ratio=0.99,
        )
        self.assertEqual(job.job_id, "a")

    def test_rejects_when_pool_at_cap(self):
        c = HaloController(self._config(admission_kv_cap_ratio=0.90), is_rank0=True)
        self.assertTrue(c.kv_cap_enabled)
        c.register_program("a", slo=5.0)
        with self.assertRaises(HaloRejectError) as cm:
            c.register_request(
                rid="r1",
                halo_job_id="a",
                halo_slo=5.0,
                kv_usage_ratio=0.95,
            )
        self.assertEqual(cm.exception.reason, "HALO_KV_CAP")

    def test_admits_when_pool_below_cap(self):
        c = HaloController(self._config(admission_kv_cap_ratio=0.90), is_rank0=True)
        c.register_program("a", slo=5.0)
        job = c.register_request(
            rid="r1",
            halo_job_id="a",
            halo_slo=5.0,
            kv_usage_ratio=0.50,
        )
        self.assertEqual(job.job_id, "a")

    def test_none_ratio_skips_cap(self):
        """kv_usage_ratio=None (scheduler did not supply it) → cap skipped."""
        c = HaloController(self._config(admission_kv_cap_ratio=0.90), is_rank0=True)
        c.register_program("a", slo=5.0)
        job = c.register_request(
            rid="r1",
            halo_job_id="a",
            halo_slo=5.0,
            kv_usage_ratio=None,
        )
        self.assertEqual(job.job_id, "a")

    def test_skips_followup_requests(self):
        """Follow-up requests of an already-admitted job bypass the KV cap
        even when the pool is full."""
        c = HaloController(self._config(admission_kv_cap_ratio=0.90), is_rank0=True)
        c.register_program("a", slo=5.0)
        # First request at low usage → admits.
        c.register_request(
            rid="r1",
            halo_job_id="a",
            halo_slo=5.0,
            kv_usage_ratio=0.10,
        )
        # Second (follow-up) request at 99% → still admitted (bypass).
        job = c.register_request(
            rid="r2",
            halo_job_id="a",
            halo_slo=5.0,
            kv_usage_ratio=0.99,
        )
        self.assertEqual(job.job_id, "a")

    def test_independent_of_admission_mode(self):
        """KV cap fires even when admission_mode='off' — a KV-cap-only
        ablation run."""
        c = HaloController(
            self._config(admission_mode="off", admission_kv_cap_ratio=0.90),
            is_rank0=True,
        )
        self.assertIsNone(c.admission_predictor)
        self.assertTrue(c.kv_cap_enabled)
        c.register_program("a", slo=5.0)
        with self.assertRaises(HaloRejectError) as cm:
            c.register_request(
                rid="r1",
                halo_job_id="a",
                halo_slo=5.0,
                kv_usage_ratio=0.95,
            )
        self.assertEqual(cm.exception.reason, "HALO_KV_CAP")

    def test_dry_run_admits_despite_cap(self):
        c = HaloController(
            self._config(admission_kv_cap_ratio=0.90, admission_dry_run=True),
            is_rank0=True,
        )
        c.register_program("a", slo=5.0)
        job = c.register_request(
            rid="r1",
            halo_job_id="a",
            halo_slo=5.0,
            kv_usage_ratio=0.95,
        )
        self.assertEqual(job.job_id, "a")

    def test_decision_logged(self):
        """A KV-cap reject writes a 'kv_cap' row to the admission log."""
        import json as _json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "admission_decisions.jsonl")
            c = HaloController(
                self._config(
                    admission_kv_cap_ratio=0.90,
                    admission_decision_log_path=path,
                ),
                is_rank0=True,
            )
            c.register_program("a", slo=5.0)
            with self.assertRaises(HaloRejectError):
                c.register_request(
                    rid="r1",
                    halo_job_id="a",
                    halo_slo=5.0,
                    kv_usage_ratio=0.95,
                )
            c.close()
            with open(path) as f:
                rows = [_json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["mode"], "kv_cap")
        self.assertEqual(rows[0]["reason"], "HALO_KV_CAP")
        self.assertEqual(rows[0]["decision"], "reject")
        self.assertAlmostEqual(rows[0]["kv_usage_ratio"], 0.95)


class TestPhase2StageBConcurrencyCap(unittest.TestCase):
    """Phase 2 Stage B — per-job declared concurrency cap.

    Stage B is independent of Stage A: it triggers even when admission_mode
    is off. The cap is enforced strictly (D4 — no queue, P2 — reject).
    """

    def _config(self) -> HaloConfig:
        return HaloConfig(enabled=True, default_slo=5.0)

    def test_no_declared_cap_means_no_limit(self):
        c = HaloController(self._config(), is_rank0=True)
        c.register_program("a", slo=5.0)  # no declared_max_concurrency
        # Admit five requests in a row — all should pass.
        for i in range(5):
            c.register_request(rid=f"r{i}", halo_job_id="a", halo_slo=5.0)
        job = c.registry.job_for_id("a")
        self.assertEqual(job.in_flight_count, 5)

    def test_cap_enforced_at_third_admit(self):
        c = HaloController(self._config(), is_rank0=True)
        c.register_program("a", slo=5.0, declared_max_concurrency=2)
        c.register_request(rid="r1", halo_job_id="a", halo_slo=5.0)
        c.register_request(rid="r2", halo_job_id="a", halo_slo=5.0)
        with self.assertRaises(HaloRejectError) as cm:
            c.register_request(rid="r3", halo_job_id="a", halo_slo=5.0)
        self.assertEqual(cm.exception.reason, "HALO_CONCURRENCY_CAP")
        # Counters did not advance on the rejected admit.
        job = c.registry.job_for_id("a")
        self.assertEqual(job.in_flight_count, 2)

    def test_in_flight_decremented_on_finish_reopens_slot(self):
        c = HaloController(self._config(), is_rank0=True)
        c.register_program("a", slo=5.0, declared_max_concurrency=1)
        c.register_request(rid="r1", halo_job_id="a", halo_slo=5.0)
        # r2 is blocked while r1 is in-flight.
        with self.assertRaises(HaloRejectError):
            c.register_request(rid="r2", halo_job_id="a", halo_slo=5.0)
        # Finishing r1 frees the slot.
        c.on_request_finished("r1")
        # r3 now succeeds.
        c.register_request(rid="r3", halo_job_id="a", halo_slo=5.0)
        job = c.registry.job_for_id("a")
        self.assertEqual(job.in_flight_count, 1)

    def test_cap_independent_of_admission_mode(self):
        """Stage B fires even when admission_mode is off — it's a hard
        contract, not a predictive gate."""
        cfg = HaloConfig(enabled=True, default_slo=5.0, admission_mode="off")
        c = HaloController(cfg, is_rank0=True)
        c.register_program("a", slo=5.0, declared_max_concurrency=0)
        # cap=0 means *nothing admitted*. Useful sanity probe.
        with self.assertRaises(HaloRejectError) as cm:
            c.register_request(rid="r1", halo_job_id="a", halo_slo=5.0)
        self.assertEqual(cm.exception.reason, "HALO_CONCURRENCY_CAP")


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

    def test_step_cost_model_path_propagates_through_factory(self):
        """End-to-end: server_args.halo_step_cost_model_path → HaloConfig →
        loader → SlowdownTracker.step_cost."""
        import os
        import tempfile

        from sglang.srt.managers.halo.admission_control.cost_model import (
            HaloStepCostModel,
        )

        # Write a valid step-cost JSON to disk.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "step.json")
            HaloStepCostModel(
                theta_p1=1e-7,
                theta_p2=4e-7,
                theta_p3=0.08,
                theta_d1=5e-5,
                theta_d2=0.3,
                theta_c=10.0,
            ).to_json(p)

            class Args:
                halo_enabled = True
                halo_default_slo = 5.0
                halo_tick_interval_ms = 100.0
                halo_aggregator = "max+mean"
                halo_job_log = None
                # Legacy paths intentionally set to verify they get ignored.
                halo_prefill_cost_model_path = "/does/not/exist/prefill.json"
                halo_tbt_cost_model_path = "/does/not/exist/tbt.json"
                halo_step_cost_model_path = p
                halo_program_idle_timeout_seconds = 300.0

            c = build_halo_controller_from_server_args(Args(), True)
        self.assertIsNotNone(c)
        # Step model loaded, legacy pair left None (their paths were ignored).
        self.assertIsNotNone(c.tracker.step_cost)
        self.assertIsNone(c.tracker.prefill_cost)
        self.assertIsNone(c.tracker.tbt_cost)
        # snapshot exposes the new flag.
        snap = c.snapshot()
        self.assertTrue(snap["cost_models_loaded"]["step"])
        self.assertFalse(snap["cost_models_loaded"]["prefill"])
        self.assertFalse(snap["cost_models_loaded"]["tbt"])


if __name__ == "__main__":
    unittest.main()
