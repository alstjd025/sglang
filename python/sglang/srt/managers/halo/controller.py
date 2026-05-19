"""HaloController — request-level admission control + tracking glue.

See managers/halo/CLAUDE.md.

The controller is the scheduler-owned top of the halo umbrella. It wires
the two halo components together:

  - request_tracker     — the single per-request state store.
  - admission_control   — the policy-pluggable admission gate.

When `config.enabled` is False the scheduler never builds this object and
every halo hook is a one-line null check. The scheduler calls only the
public methods: `register_request`, `on_request_step`,
`on_request_finished`, `tick`, `snapshot`, `close`.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

from sglang.srt.managers.halo.admission_control.cost_model import (
    try_load_halo_step_cost_model,
    try_load_prefill_cost_model,
    try_load_tbt_cost_model,
)
from sglang.srt.managers.halo.admission_control.gate import (
    HaloAdmissionGate,
)
from sglang.srt.managers.halo.admission_control.policy import NewRequestInput
from sglang.srt.managers.halo.admission_control.policy_mooncake import MooncakePolicy
from sglang.srt.managers.halo.admission_control.policy_vss import VssPolicy
from sglang.srt.managers.halo.admission_control.tbt_tracker import TBTEwmaTracker
from sglang.srt.managers.halo.metrics import HaloMetrics
from sglang.srt.managers.halo.request_tracker import RequestTracker

logger = logging.getLogger(__name__)

REASON_DISABLED = "HALO_DISABLED"

__all__ = [
    "HaloConfig",
    "HaloController",
    "HaloRejectError",
    "REASON_DISABLED",
    "build_halo_controller_from_server_args",
]


class HaloRejectError(Exception):
    """Raised by `HaloController.register_request` when the admission gate
    rejects an arrival. The scheduler converts it to an HTTP-400 abort
    carrying `reason`.
    """

    def __init__(self, reason: str, rid: str) -> None:
        super().__init__(f"halo reject rid={rid} reason={reason}")
        self.reason = reason
        self.rid = rid


@dataclass
class HaloConfig:
    """Snapshot of the `--halo-*` CLI flags driving the controller."""

    enabled: bool = False

    # ── Admission ──────────────────────────────────────────────────────
    # off | mooncake | vss | reactive
    admission_policy: str = "off"
    # ttft_slo / tbt_slo interpretation: "ratio" (vs solo) or "absolute" (ms).
    slo_mode: str = "ratio"
    # KV-cache hard cap (Stage B′). 0 disables; 0 < r ≤ 1 ⇒ reject at ≥ r usage.
    admission_kv_cap_ratio: float = 0.0
    # vss policy — violation-ratio threshold.
    admission_violation_threshold: float = 0.2
    # mooncake policy — Stage-3 reactive EWMA trips at tbt_slo * this.
    tbt_reactive_ratio: float = 0.9
    admission_dry_run: bool = False
    admission_decision_log_path: Optional[str] = None

    # ── Cost models ────────────────────────────────────────────────────
    prefill_cost_model_path: Optional[str] = None
    tbt_cost_model_path: Optional[str] = None
    step_cost_model_path: Optional[str] = None

    # ── Misc ───────────────────────────────────────────────────────────
    tick_interval_ms: float = 100.0
    # Server's effective chunked-prefill token budget — bounds the VSS
    # EXTEND-step cost (결함 A). None ⇒ chunked prefill disabled.
    chunked_prefill_size: Optional[int] = None


class _DecisionLogger:
    """Append-only JSONL writer for admission decisions. Rank-0 only."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", buffering=1)  # line-buffered
        logger.info("halo: admission decision log opened at %s", path)

    def write(self, payload: Dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("halo: decision log write failed: %s", e)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class HaloController:
    """Request-level admission control + tracking coordinator."""

    def __init__(self, config: HaloConfig, is_rank0: bool = True) -> None:
        self.config = config
        self.is_rank0 = is_rank0

        # ── Cost models — the step model wins over the legacy pair. ──────
        step_cost = try_load_halo_step_cost_model(config.step_cost_model_path)
        prefill_cost = None
        tbt_cost = None
        if step_cost is None:
            prefill_cost = try_load_prefill_cost_model(config.prefill_cost_model_path)
            tbt_cost = try_load_tbt_cost_model(config.tbt_cost_model_path)
        if step_cost is None and prefill_cost is None and tbt_cost is None:
            logger.warning(
                "halo: no cost model loaded — solo baselines and any "
                "cost-model policy will be inert; tracking still works"
            )

        # ── request_tracker — the single per-request state store. ────────
        self.tracker = RequestTracker(
            step_cost=step_cost, prefill_cost=prefill_cost, tbt_cost=tbt_cost
        )

        # ── admission_control — gate + selected policy. ──────────────────
        self.tbt_tracker = TBTEwmaTracker()
        self.gate = HaloAdmissionGate(
            policy=self._build_policy(config, step_cost, prefill_cost, tbt_cost),
            kv_cap_ratio=config.admission_kv_cap_ratio,
        )

        # Prometheus metrics — installed by the scheduler post-construction
        # (init_halo). None when --enable-metrics off OR rank > 0.
        self.metrics: Optional[HaloMetrics] = None

        self._decision_log: Optional[_DecisionLogger] = None
        if is_rank0 and config.admission_decision_log_path:
            try:
                self._decision_log = _DecisionLogger(config.admission_decision_log_path)
            except OSError as e:
                logger.warning("halo: decision log disabled — %s", e)

        # Recent decisions ring for /halo/status introspection.
        self._recent_decisions: Deque[Dict[str, Any]] = deque(maxlen=32)

        self._tick_interval_s = max(config.tick_interval_ms, 0.0) / 1000.0
        self._last_tick_monotonic = 0.0

        logger.info(
            "halo: enabled (policy=%s slo_mode=%s kv_cap=%.2f dry_run=%s "
            "cost_model=%s rank0=%s)",
            config.admission_policy,
            config.slo_mode,
            config.admission_kv_cap_ratio,
            config.admission_dry_run,
            "step" if step_cost else ("legacy" if prefill_cost else "none"),
            is_rank0,
        )

    def _build_policy(self, config, step_cost, prefill_cost, tbt_cost):
        """Construct the admission policy named by config.admission_policy.

        Returns None for "off" (the gate then runs only the KV cap) or when
        a policy's cost model is missing (a policy can't function without it).
        """
        policy = (config.admission_policy or "off").lower()
        if policy == "off":
            return None
        if policy == "mooncake":
            return MooncakePolicy(
                prefill_cost=prefill_cost,
                tbt_cost=tbt_cost,
                slo_mode=config.slo_mode,
                tbt_tracker=self.tbt_tracker,
                tbt_reactive_ratio=config.tbt_reactive_ratio,
            )
        if policy == "vss":
            if step_cost is None:
                logger.warning(
                    "halo: admission_policy=vss needs the Halo Step Cost "
                    "Model — admission disabled. Set --halo-step-cost-model-path."
                )
                return None
            return VssPolicy(
                cost_model=step_cost,
                violation_threshold=config.admission_violation_threshold,
                chunked_prefill_size=config.chunked_prefill_size,
            )
        if policy == "reactive":
            # Phase 3 — not yet implemented.
            logger.warning(
                "halo: admission_policy=reactive is not implemented yet "
                "(Phase 3) — admission disabled."
            )
            return None
        logger.warning("halo: unknown admission_policy=%r — admission disabled", policy)
        return None

    # ------------------------------------------------------------------
    # Admission hook
    # ------------------------------------------------------------------

    def register_request(
        self,
        rid: str,
        *,
        ttft_slo: Optional[float],
        tbt_slo: Optional[float],
        e2e_slo: Optional[float],
        prompt_len: int,
        prefix_len: int,
        kv_usage_ratio: float = 0.0,
    ) -> None:
        """Admission hook — run the gate, then either record the admitted
        request in the tracker or raise `HaloRejectError`.

        In dry-run mode a would-reject is logged but the request is still
        admitted (and tracked).
        """
        new_req = NewRequestInput(
            rid=rid,
            prompt_len=max(0, prompt_len),
            prefix_len=max(0, prefix_len),
            ttft_slo=ttft_slo,
            tbt_slo=tbt_slo,
            e2e_slo=e2e_slo,
        )
        snapshot = self.tracker.snapshot(kv_usage_ratio=kv_usage_ratio)
        result = self.gate.decide(new_req, snapshot)
        self._log_decision(rid, result)

        rejected = not result.admit
        if rejected and not self.config.admission_dry_run:
            if self.metrics is not None:
                self.metrics.record_request_rejected(result.reason)
            self.tracker.on_rejected(rid, result.reason)
            raise HaloRejectError(reason=result.reason, rid=rid)

        # Admitted (or dry-run would-reject) — create the tracker record.
        pd = result.policy_decision
        self.tracker.on_admitted(
            rid,
            ttft_slo=ttft_slo,
            tbt_slo=tbt_slo,
            e2e_slo=e2e_slo,
            prompt_len=prompt_len,
            prefix_len=prefix_len,
            ts=time.monotonic(),
            predicted_ttft_ms=pd.predicted_ttft_ms if pd else None,
            predicted_tbt_ms=pd.predicted_tbt_ms if pd else None,
        )
        if self.metrics is not None:
            self.metrics.record_request_admitted()

    # ------------------------------------------------------------------
    # Progress / finish hooks
    # ------------------------------------------------------------------

    def on_request_step(self, rid: str, *, decoded_tokens: int, kv_len: int) -> None:
        """Per-step progress update for a running request."""
        self.tracker.on_step(
            rid, decoded_tokens=decoded_tokens, kv_len=kv_len, ts=time.monotonic()
        )

    def on_request_finished(
        self,
        rid: str,
        *,
        decoded_tokens: Optional[int] = None,
        kv_len: Optional[int] = None,
    ) -> None:
        """Finish hook — apply the final progress update, finalize the
        record, and emit terminal metrics."""
        if decoded_tokens is not None:
            self.tracker.on_step(
                rid,
                decoded_tokens=decoded_tokens,
                kv_len=kv_len if kv_len is not None else 0,
                ts=time.monotonic(),
            )
        record = self.tracker.on_finished(rid, ts=time.monotonic())
        if record is not None and self.metrics is not None:
            self.metrics.observe_finished(record)

    def on_tbt_sample(self, per_step_ms: float) -> None:
        """Feed the mooncake Stage-3 reactive EWMA with a measured decode
        step latency. Called from the scheduler's per-step metrics hook."""
        self.tbt_tracker.update(per_step_ms)

    # ------------------------------------------------------------------
    # Periodic tick
    # ------------------------------------------------------------------

    def tick(
        self,
        running_infos: List[Tuple[str, int, int]],
        now_monotonic: Optional[float] = None,
    ) -> None:
        """Periodic sweep. `running_infos` is (rid, decoded_tokens, kv_len)
        for each request currently in the running batch. Cheap when the
        tick interval has not elapsed.
        """
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if (now - self._last_tick_monotonic) < self._tick_interval_s:
            return
        self._last_tick_monotonic = now

        for rid, decoded, kv in running_infos:
            self.tracker.on_step(rid, decoded_tokens=decoded, kv_len=kv, ts=now)

        if self.metrics is not None:
            self.metrics.update_gauges(
                active_requests=self.tracker.active_count(),
            )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def _log_decision(self, rid: str, result) -> None:
        pd = result.policy_decision
        row = {
            "ts_ns": time.time_ns(),
            "rid": rid,
            "policy": self.config.admission_policy,
            "dry_run": self.config.admission_dry_run,
            "decision": "admit" if result.admit else "reject",
            "reason": result.reason,
            "predicted_ttft_ms": pd.predicted_ttft_ms if pd else None,
            "predicted_tbt_ms": pd.predicted_tbt_ms if pd else None,
            "detail": pd.detail if pd else {},
        }
        self._recent_decisions.append(row)
        if self._decision_log is not None:
            self._decision_log.write(row)

    def snapshot(self, limit: int = 32) -> Dict[str, Any]:
        """For /halo/status and /server_info."""
        return {
            "enabled": True,
            "admission_policy": self.config.admission_policy,
            "slo_mode": self.config.slo_mode,
            "kv_cap_ratio": self.config.admission_kv_cap_ratio,
            "dry_run": self.config.admission_dry_run,
            "tick_interval_ms": self.config.tick_interval_ms,
            "active_requests": self.tracker.active_count(),
            "recent_decisions": list(self._recent_decisions)[-limit:],
        }

    def close(self) -> None:
        if self._decision_log is not None:
            self._decision_log.close()


def build_halo_controller_from_server_args(
    server_args: Any,
    is_rank0: bool,
) -> Optional[HaloController]:
    """Factory used by the scheduler. Returns None when `--halo-enabled` is off."""
    if not getattr(server_args, "halo_enabled", False):
        return None
    config = HaloConfig(
        enabled=True,
        admission_policy=getattr(server_args, "halo_admission_policy", "off") or "off",
        slo_mode=getattr(server_args, "halo_slo_mode", "ratio") or "ratio",
        admission_kv_cap_ratio=float(
            getattr(server_args, "halo_admission_kv_cap_ratio", 0.0)
        ),
        admission_violation_threshold=float(
            getattr(server_args, "halo_admission_violation_threshold", 0.2)
        ),
        tbt_reactive_ratio=float(getattr(server_args, "halo_tbt_reactive_ratio", 0.9)),
        admission_dry_run=bool(getattr(server_args, "halo_admission_dry_run", False)),
        admission_decision_log_path=getattr(
            server_args, "halo_admission_decision_log", None
        ),
        prefill_cost_model_path=getattr(
            server_args, "halo_prefill_cost_model_path", None
        ),
        tbt_cost_model_path=getattr(server_args, "halo_tbt_cost_model_path", None),
        step_cost_model_path=getattr(server_args, "halo_step_cost_model_path", None),
        tick_interval_ms=float(getattr(server_args, "halo_tick_interval_ms", 100.0)),
        chunked_prefill_size=getattr(server_args, "chunked_prefill_size", None),
    )
    return HaloController(config=config, is_rank0=is_rank0)
