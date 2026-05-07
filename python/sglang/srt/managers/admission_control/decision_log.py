"""Per-decision JSONL logger for admission control.

Each line captures everything replay_admission.py needs to re-evaluate the
same decision under different SLOs:

    {
      "ts": 1730000123.456,
      "rid": "abc123",
      "admit": true,
      "reason": "ADMIT" | "TTFT_PREDICTED" | "TBT_PREDICTED" | "TBT_REACTIVE" | "DISABLED",
      "dry_run_would_reject": false,
      "predicted_ttft_ms": 320.5,
      "predicted_tbt_ms": 18.2,
      "tbt_ewma_ms": 14.1,
      "queue_predicted_ms": 50.0,
      "ttft_slo_ms": 30000.0,
      "tbt_slo_ms": 200.0,
      "predicted_prefill_ms": 270.5,
      "prompt_len": 18432,
      "prefix_len": 16384,
      "running_batch_size": 12,
      "running_batch_total_kv_tokens": 145000
    }

See managers/admission_control/CLAUDE.md and tools/admission_control/CLAUDE.md
for the replay tool that consumes this log.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # pragma: no cover
    from sglang.srt.managers.admission_control.controller import (
        AdmissionDecision,
        SchedulerSnapshot,
    )

logger = logging.getLogger(__name__)


class DecisionLogger:
    """Append-only JSONL writer for admission decisions.

    Designed for the scheduler hot path: opens the file once, writes one line
    per decision, never raises into the request path. Failures degrade to
    a single warning log and a disabled state for the remainder of the
    process.
    """

    def __init__(self, path: str | os.PathLike) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._fp = None
        self._disabled = False
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Line-buffered so partial logs survive ungraceful exit.
            self._fp = self._path.open("a", encoding="utf-8", buffering=1)
        except OSError as e:
            logger.warning(
                "admission decision log: open failed at %s — disabling (%s)",
                self._path, e,
            )
            self._disabled = True

    @property
    def path(self) -> Path:
        return self._path

    @property
    def enabled(self) -> bool:
        return not self._disabled and self._fp is not None

    def record(
        self,
        *,
        rid: str,
        decision: "AdmissionDecision",
        prompt_len: int,
        prefix_len: int,
        snapshot: "SchedulerSnapshot",
    ) -> None:
        if self._disabled or self._fp is None:
            return
        rec = {
            "ts": time.time(),
            "rid": rid,
            "admit": decision.admit,
            "reason": decision.reason,
            "dry_run_would_reject": decision.dry_run_would_reject,
            "predicted_ttft_ms": decision.predicted_ttft_ms,
            "predicted_tbt_ms": decision.predicted_tbt_ms,
            "solo_ttft_ms": decision.solo_ttft_ms,
            "solo_tbt_ms": decision.solo_tbt_ms,
            "tbt_ewma_ms": decision.tbt_ewma_ms,
            "queue_predicted_ms": decision.queue_predicted_ms,
            "ttft_slo_ms": decision.ttft_slo_ms,
            "tbt_slo_ms": decision.tbt_slo_ms,
            "ttft_slo_ratio": decision.ttft_slo_ratio,
            "tbt_slo_ratio": decision.tbt_slo_ratio,
            "predicted_prefill_ms": decision.predicted_prefill_ms,
            "prompt_len": prompt_len,
            "prefix_len": prefix_len,
            "running_batch_size": snapshot.running_batch_size,
            "running_batch_total_kv_tokens": snapshot.running_batch_total_kv_tokens,
        }
        try:
            with self._lock:
                self._fp.write(json.dumps(rec, ensure_ascii=True) + "\n")
        except OSError as e:
            logger.warning(
                "admission decision log: write failed — disabling (%s)", e
            )
            self._disabled = True

    def close(self) -> None:
        if self._fp is None:
            return
        try:
            with self._lock:
                self._fp.close()
        except OSError:
            pass
        self._fp = None
