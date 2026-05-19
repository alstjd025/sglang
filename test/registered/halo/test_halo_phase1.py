"""Unit tests for managers/halo/ — request-level admission control + tracking.

See test/registered/halo/CLAUDE.md and managers/halo/CLAUDE.md.

Coverage:
- RequestRecord / RequestTracker — the single per-request state store
- HaloAdmissionGate — KV-cache hard cap + policy dispatch
- MooncakePolicy / VssPolicy — the request-level admission policies
- HaloController — register/finish/tick plumbing, dry-run, off-by-default
"""

import unittest

from sglang.srt.managers.halo import (
    HaloConfig,
    HaloController,
    HaloRejectError,
    RequestState,
    RequestTracker,
    build_halo_controller_from_server_args,
)
from sglang.srt.managers.halo.admission_control import (
    HaloAdmissionGate,
    MooncakePolicy,
    NewRequestInput,
    VssPolicy,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


# ── Duck-typed fake cost models — keep the unit tests model-file-free. ──────
class _FakePrefill:
    """PrefillCostModel-like: estimate_ms(prompt_len, prefix_len)."""

    def estimate_ms(self, prompt_len, prefix_len):
        return 10.0 + 0.1 * max(0, prompt_len - prefix_len)


class _FakeTbt:
    """TBTCostModel-like: estimate_ms(batch_size, per_req_kv)."""

    def estimate_ms(self, batch_size, per_req_kv):
        return 20.0 + 2.0 * batch_size + 0.001 * per_req_kv


class _FakeStep:
    """HaloStepCostModel-like."""

    def estimate_step_ms(self, prefill_infos, decode_infos):
        s = 5.0
        for n, _r in prefill_infos:
            s += 0.05 * n
        for kv in decode_infos:
            s += 0.002 * kv
        return s

    def estimate_solo_prefill_total_ms(self, n, r):
        return 5.0 + 0.05 * n

    def estimate_solo_tbt_ms(self, r):
        return 20.0 + 0.002 * r


# ───────────────────────────────────────────────────────────────────────────
class TestRequestTracker(unittest.TestCase):
    def test_on_admitted_creates_queued_record(self):
        t = RequestTracker()
        rec = t.on_admitted(
            "r1",
            ttft_slo=2.0,
            tbt_slo=3.0,
            e2e_slo=5.0,
            prompt_len=1000,
            prefix_len=100,
            ts=0.0,
        )
        self.assertIs(rec.state, RequestState.QUEUED)
        self.assertEqual(rec.prompt_len, 1000)
        self.assertEqual(t.active_count(), 1)
        self.assertIs(t.get("r1"), rec)

    def test_on_step_stamps_first_token_and_runs(self):
        t = RequestTracker()
        t.on_admitted(
            "r1",
            ttft_slo=None,
            tbt_slo=None,
            e2e_slo=None,
            prompt_len=100,
            prefix_len=0,
            ts=0.0,
        )
        t.on_step("r1", decoded_tokens=1, kv_len=101, ts=0.5)
        rec = t.get("r1")
        self.assertIs(rec.state, RequestState.RUNNING)
        self.assertEqual(rec.first_token_ts, 0.5)
        self.assertAlmostEqual(rec.ttft_ms, 500.0)

    def test_tbt_mean(self):
        t = RequestTracker()
        t.on_admitted(
            "r1",
            ttft_slo=None,
            tbt_slo=None,
            e2e_slo=None,
            prompt_len=100,
            prefix_len=0,
            ts=0.0,
        )
        t.on_step("r1", decoded_tokens=1, kv_len=101, ts=1.0)
        t.on_step("r1", decoded_tokens=5, kv_len=105, ts=2.0)
        # span 1.0s over (5-1) inter-token intervals → 250 ms.
        self.assertAlmostEqual(t.get("r1").tbt_mean_ms, 250.0)

    def test_on_finished_computes_e2e_and_slowdown(self):
        t = RequestTracker(step_cost=_FakeStep())
        t.on_admitted(
            "r1",
            ttft_slo=None,
            tbt_slo=None,
            e2e_slo=None,
            prompt_len=200,
            prefix_len=0,
            ts=0.0,
        )
        t.on_step("r1", decoded_tokens=10, kv_len=210, ts=1.0)
        rec = t.on_finished("r1", ts=4.0)
        self.assertIs(rec.state, RequestState.FINISHED)
        self.assertAlmostEqual(rec.e2e_ms, 4000.0)
        self.assertIsNotNone(rec.solo_e2e_ms)
        self.assertGreater(rec.e2e_slowdown, 1.0)
        # finished record left the active set.
        self.assertEqual(t.active_count(), 0)

    def test_snapshot_partitions_queued_and_running(self):
        t = RequestTracker()
        t.on_admitted(
            "q",
            ttft_slo=None,
            tbt_slo=None,
            e2e_slo=None,
            prompt_len=1,
            prefix_len=0,
            ts=0.0,
        )
        t.on_admitted(
            "run",
            ttft_slo=None,
            tbt_slo=None,
            e2e_slo=None,
            prompt_len=1,
            prefix_len=0,
            ts=0.0,
        )
        t.on_step("run", decoded_tokens=1, kv_len=2, ts=1.0)
        snap = t.snapshot(kv_usage_ratio=0.42)
        self.assertEqual(snap.queue_depth, 1)
        self.assertEqual(snap.running_batch_size, 1)
        self.assertEqual(snap.kv_usage_ratio, 0.42)

    def test_on_rejected_records_terminal(self):
        t = RequestTracker()
        t.on_rejected("r1", "HALO_KV_CAP")
        rec = t.get("r1")
        self.assertIs(rec.state, RequestState.REJECTED)
        self.assertEqual(rec.reject_reason, "HALO_KV_CAP")
        self.assertEqual(t.active_count(), 0)

    def test_solo_primitives_zero_without_cost_model(self):
        t = RequestTracker()
        self.assertEqual(t.prefill_solo_ms(100, 0), 0.0)
        self.assertEqual(t.decode_step_ms([100]), 0.0)


# ───────────────────────────────────────────────────────────────────────────
class TestHaloAdmissionGate(unittest.TestCase):
    def _req(self):
        return NewRequestInput(rid="new", prompt_len=100, prefix_len=0)

    def test_kv_cap_rejects_at_threshold(self):
        gate = HaloAdmissionGate(policy=None, kv_cap_ratio=0.9)
        snap = RequestTracker().snapshot(kv_usage_ratio=0.95)
        result = gate.decide(self._req(), snap)
        self.assertFalse(result.admit)
        self.assertEqual(result.reason, "HALO_KV_CAP")

    def test_kv_cap_disabled_when_zero(self):
        gate = HaloAdmissionGate(policy=None, kv_cap_ratio=0.0)
        snap = RequestTracker().snapshot(kv_usage_ratio=1.0)
        self.assertTrue(gate.decide(self._req(), snap).admit)

    def test_none_policy_admits(self):
        gate = HaloAdmissionGate(policy=None, kv_cap_ratio=0.9)
        snap = RequestTracker().snapshot(kv_usage_ratio=0.1)
        result = gate.decide(self._req(), snap)
        self.assertTrue(result.admit)
        self.assertEqual(result.reason, "ADMIT")


# ───────────────────────────────────────────────────────────────────────────
class TestMooncakePolicy(unittest.TestCase):
    def test_admits_without_cost_models(self):
        policy = MooncakePolicy(prefill_cost=None, tbt_cost=None)
        snap = RequestTracker().snapshot()
        req = NewRequestInput(
            rid="r", prompt_len=100, prefix_len=0, ttft_slo=2.0, tbt_slo=2.0
        )
        self.assertTrue(policy.decide(req, snap).admit)

    def test_ttft_ratio_reject(self):
        # A long queue backlog pushes pred_ttft / solo_ttft over the ratio.
        t = RequestTracker()
        for i in range(20):
            t.on_admitted(
                f"q{i}",
                ttft_slo=None,
                tbt_slo=None,
                e2e_slo=None,
                prompt_len=5000,
                prefix_len=0,
                ts=0.0,
            )
        policy = MooncakePolicy(
            prefill_cost=_FakePrefill(), tbt_cost=None, slo_mode="ratio"
        )
        req = NewRequestInput(rid="r", prompt_len=100, prefix_len=0, ttft_slo=2.0)
        decision = policy.decide(req, t.snapshot())
        self.assertFalse(decision.admit)
        self.assertEqual(decision.reason, "MOONCAKE_TTFT")

    def test_absolute_mode_ttft(self):
        policy = MooncakePolicy(
            prefill_cost=_FakePrefill(), tbt_cost=None, slo_mode="absolute"
        )
        req = NewRequestInput(
            rid="r", prompt_len=100000, prefix_len=0, ttft_slo=50.0
        )  # 50 ms cap; solo prefill far over
        self.assertFalse(policy.decide(req, RequestTracker().snapshot()).admit)


# ───────────────────────────────────────────────────────────────────────────
class TestVssPolicy(unittest.TestCase):
    def test_admits_empty_batch(self):
        policy = VssPolicy(cost_model=_FakeStep(), violation_threshold=0.2)
        req = NewRequestInput(rid="r", prompt_len=100, prefix_len=0, e2e_slo=5.0)
        self.assertTrue(policy.decide(req, RequestTracker().snapshot()).admit)

    def test_decide_runs_with_running_batch(self):
        t = RequestTracker()
        t.on_admitted(
            "a",
            ttft_slo=None,
            tbt_slo=None,
            e2e_slo=5.0,
            prompt_len=2000,
            prefix_len=0,
            ts=0.0,
        )
        t.on_step("a", decoded_tokens=4, kv_len=2004, ts=1.0)
        policy = VssPolicy(cost_model=_FakeStep(), violation_threshold=0.2)
        req = NewRequestInput(rid="r", prompt_len=100, prefix_len=0, e2e_slo=5.0)
        decision = policy.decide(req, t.snapshot())
        self.assertIn("violation_ratio", decision.detail)


# ───────────────────────────────────────────────────────────────────────────
class TestHaloController(unittest.TestCase):
    def test_factory_returns_none_when_disabled(self):
        class _SA:
            halo_enabled = False

        self.assertIsNone(build_halo_controller_from_server_args(_SA(), True))

    def test_register_request_admit_creates_record(self):
        c = HaloController(HaloConfig(enabled=True, admission_policy="off"))
        c.register_request(
            "r1",
            ttft_slo=2.0,
            tbt_slo=2.0,
            e2e_slo=5.0,
            prompt_len=100,
            prefix_len=0,
            kv_usage_ratio=0.1,
        )
        rec = c.tracker.get("r1")
        self.assertIsNotNone(rec)
        self.assertIs(rec.state, RequestState.QUEUED)

    def test_register_request_kv_cap_rejects(self):
        c = HaloController(
            HaloConfig(enabled=True, admission_policy="off", admission_kv_cap_ratio=0.9)
        )
        with self.assertRaises(HaloRejectError) as ctx:
            c.register_request(
                "r1",
                ttft_slo=None,
                tbt_slo=None,
                e2e_slo=None,
                prompt_len=100,
                prefix_len=0,
                kv_usage_ratio=0.95,
            )
        self.assertEqual(ctx.exception.reason, "HALO_KV_CAP")
        self.assertIs(c.tracker.get("r1").state, RequestState.REJECTED)

    def test_dry_run_admits_despite_would_reject(self):
        c = HaloController(
            HaloConfig(
                enabled=True,
                admission_policy="off",
                admission_kv_cap_ratio=0.9,
                admission_dry_run=True,
            )
        )
        # Would-reject on KV cap, but dry-run admits + still tracks.
        c.register_request(
            "r1",
            ttft_slo=None,
            tbt_slo=None,
            e2e_slo=None,
            prompt_len=100,
            prefix_len=0,
            kv_usage_ratio=0.95,
        )
        self.assertIs(c.tracker.get("r1").state, RequestState.QUEUED)

    def test_on_request_finished(self):
        c = HaloController(HaloConfig(enabled=True, admission_policy="off"))
        c.register_request(
            "r1",
            ttft_slo=None,
            tbt_slo=None,
            e2e_slo=None,
            prompt_len=100,
            prefix_len=0,
        )
        c.on_request_finished("r1", decoded_tokens=10, kv_len=110)
        self.assertIs(c.tracker.get("r1").state, RequestState.FINISHED)

    def test_tick_updates_tracker(self):
        c = HaloController(
            HaloConfig(enabled=True, admission_policy="off", tick_interval_ms=0.0)
        )
        c.register_request(
            "r1",
            ttft_slo=None,
            tbt_slo=None,
            e2e_slo=None,
            prompt_len=100,
            prefix_len=0,
        )
        c.tick([("r1", 7, 107)], now_monotonic=1.0)
        rec = c.tracker.get("r1")
        self.assertEqual(rec.decoded_tokens, 7)
        self.assertIs(rec.state, RequestState.RUNNING)

    def test_snapshot_shape(self):
        c = HaloController(HaloConfig(enabled=True, admission_policy="mooncake"))
        snap = c.snapshot()
        self.assertEqual(snap["admission_policy"], "mooncake")
        self.assertIn("active_requests", snap)


if __name__ == "__main__":
    unittest.main()
