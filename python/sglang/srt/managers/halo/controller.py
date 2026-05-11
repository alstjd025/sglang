"""HaloController — top-level glue for HALO Phase 1.

See managers/halo/CLAUDE.md.
"""

# HALO: Phase 1 controller. Owned by the scheduler; entirely off when
# config.enabled is False. Public methods are the only thing the scheduler
# calls. Internal modules (job_registry, slowdown_tracker, job) do not import
# scheduler state — instead the scheduler hands us a build_infos callable
# each tick.
#
# Design note: this mirrors the AdmissionController pattern so the scheduler
# integration story is uniform (init in __init__, hooks at admission/finish/
# step). Halo *does not* replace admission control — both can be active in
# parallel in Phase 1.

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from sglang.srt.managers.admission_control.cost_model import (
    try_load_prefill_cost_model,
    try_load_tbt_cost_model,
)
from sglang.srt.managers.halo.job import Job, JobState
from sglang.srt.managers.halo.job_registry import JobRegistry
from sglang.srt.managers.halo.slowdown_tracker import (
    RequestExecutionInfo,
    SlowdownTracker,
)

logger = logging.getLogger(__name__)


class HaloRejectError(Exception):
    """Raised by HaloController.register_request when the request lacks a
    halo_job_id while Halo is enabled (strict-mode rejection — per spec).

    The scheduler catches this and converts it into an HTTP 400 abort.
    """

    def __init__(self, reason: str, rid: str) -> None:
        super().__init__(f"halo reject rid={rid} reason={reason}")
        self.reason = reason
        self.rid = rid


@dataclass
class HaloConfig:
    enabled: bool = False
    default_slo: float = 5.0
    tick_interval_ms: float = 100.0
    aggregator: str = "max+mean"     # Phase 1 always tracks both; reserved string for future
    job_log_path: Optional[str] = None
    prefill_cost_model_path: Optional[str] = None
    tbt_cost_model_path: Optional[str] = None
    gc_retain_seconds: float = 5.0


class _JobLogger:
    """Append-only JSONL writer for periodic job snapshots. Rank-0 only."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", buffering=1)  # line-buffered
        logger.info("halo: job log opened at %s", path)

    def write(self, payload: Dict[str, Any]) -> None:
        try:
            self._fh.write(json.dumps(payload, default=str) + "\n")
        except Exception as e:  # pragma: no cover — defensive
            logger.warning("halo: job log write failed: %s", e)

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class HaloController:
    """Top-level Halo coordinator.

    Construction
    ------------
    - If `config.enabled` is False → caller should not construct this at all.
    - Cost model paths missing/malformed → tracker stays inert (no sweep math)
      but job registration / counters still work.
    - `is_rank0` gates JSONL log writes (TP-dedup; metrics use the same gate).
    """

    def __init__(self, config: HaloConfig, is_rank0: bool = True) -> None:
        self.config = config
        self.is_rank0 = is_rank0
        self.registry = JobRegistry()

        prefill_cost = try_load_prefill_cost_model(config.prefill_cost_model_path)
        tbt_cost = try_load_tbt_cost_model(config.tbt_cost_model_path)
        if prefill_cost is None and tbt_cost is None:
            logger.warning(
                "halo: both cost models missing — slowdown sweep will be a no-op; "
                "job counters / state still tracked"
            )
        self.tracker = SlowdownTracker(prefill_cost, tbt_cost, self.registry)

        self._log: Optional[_JobLogger] = None
        if is_rank0 and config.job_log_path:
            try:
                self._log = _JobLogger(config.job_log_path)
            except OSError as e:
                logger.warning("halo: job log disabled — %s", e)

        self._last_tick_monotonic: float = 0.0
        self._tick_interval_s: float = max(config.tick_interval_ms, 0.0) / 1000.0

        logger.info(
            "halo: enabled (default_slo=%.2f tick_interval_ms=%.1f rank0=%s log=%s)",
            config.default_slo,
            config.tick_interval_ms,
            is_rank0,
            config.job_log_path or "off",
        )

    # ------------------------------------------------------------------
    # Admission / finish hooks
    # ------------------------------------------------------------------

    def register_request(
        self,
        rid: str,
        halo_job_id: Optional[str],
        halo_slo: Optional[float],
    ) -> Job:
        """Strict-mode admission hook.

        Per spec (decision Q7): Halo enabled ⇒ `halo_job_id` is REQUIRED.
        Missing → raise HaloRejectError (scheduler converts to HTTP 400).

        Missing `halo_slo` → use `config.default_slo`.
        """
        if halo_job_id is None or halo_job_id == "":
            raise HaloRejectError(reason="HALO_NO_JOB_ID", rid=rid)
        slo = halo_slo if halo_slo is not None else self.config.default_slo
        job = self.registry.record_admission(halo_job_id, slo, rid)
        return job

    def on_request_finished(self, rid: str) -> None:
        """Called from the scheduler's finish path."""
        self.registry.record_completion(rid)

    # ------------------------------------------------------------------
    # Periodic tick (called from scheduler_metrics_mixin)
    # ------------------------------------------------------------------

    def should_tick(self, now_monotonic: Optional[float] = None) -> bool:
        """Returns True if at least tick_interval_ms has passed since last sweep."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return (now - self._last_tick_monotonic) >= self._tick_interval_s

    def tick(
        self,
        build_infos: Callable[[], List[RequestExecutionInfo]],
        now_monotonic: Optional[float] = None,
    ) -> None:
        """Run a sweep if enough wall-clock has elapsed. Cheap when not due."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        if (now - self._last_tick_monotonic) < self._tick_interval_s:
            return
        self._last_tick_monotonic = now

        infos = build_infos()
        if infos:
            self.tracker.sweep(infos)

        if self._log is not None:
            self._log.write(
                {
                    "ts": now,
                    "active_jobs": [j.to_dict() for j in self.registry.active_jobs()],
                }
            )

        # GC completed jobs older than retain window — bounded memory.
        self.registry.gc_completed(self.config.gc_retain_seconds)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def snapshot(self, limit: int = 32) -> Dict[str, Any]:
        """For /server_info."""
        s = {
            "enabled": True,
            "default_slo": self.config.default_slo,
            "tick_interval_ms": self.config.tick_interval_ms,
            "aggregator": self.config.aggregator,
            "cost_models_loaded": {
                "prefill": self.tracker.prefill_cost is not None,
                "tbt": self.tracker.tbt_cost is not None,
            },
        }
        s.update(self.registry.snapshot_dict(limit=limit))
        return s

    def close(self) -> None:
        if self._log is not None:
            self._log.close()


def build_halo_controller_from_server_args(
    server_args: Any,
    is_rank0: bool,
) -> Optional[HaloController]:
    """Factory used by the scheduler. Returns None if `--halo-enabled` is off.

    Pulled into a free function so the scheduler integration is a one-line
    call and the scheduler test surface stays small.
    """
    if not getattr(server_args, "halo_enabled", False):
        return None
    config = HaloConfig(
        enabled=True,
        default_slo=float(getattr(server_args, "halo_default_slo", 5.0)),
        tick_interval_ms=float(
            getattr(server_args, "halo_tick_interval_ms", 100.0)
        ),
        aggregator=getattr(server_args, "halo_aggregator", "max+mean"),
        job_log_path=getattr(server_args, "halo_job_log", None),
        prefill_cost_model_path=getattr(
            server_args, "halo_prefill_cost_model_path", None
        ),
        tbt_cost_model_path=getattr(
            server_args, "halo_tbt_cost_model_path", None
        ),
    )
    return HaloController(config=config, is_rank0=is_rank0)
