"""Per-step JSONL sampling for Halo Step Cost Model fitting.

Design + rationale: ms_dev/halo_dev/prediction_model.md §9.

The scheduler hot path calls `sampler.observe_step(batch, step_time_ms)` once
per forward step. The sampler extracts a minimal 5-tuple of features
(Σnᵢ², Σnᵢrᵢ, Σnᵢ, Σrⱼ, bs_d) plus the measured step time, pushes them onto a
bounded in-memory queue, and a background thread flushes the queue to JSONL
every `flush_interval_ms`. This keeps the scheduler thread's overhead at
microsecond scale (no I/O on the hot path).

Output is consumed offline by `tools/halo/fit_halo_cost_model.py`. The sampler
is ONLY active when `--halo-cost-model-sample-log <path>` is set; otherwise the
scheduler `__init__` leaves `self.halo_cost_sampler = None` and the hot-path
hook short-circuits on a single `is None` check.

TP-dedup: only `attn_tp_rank == 0` instantiates a sampler — other ranks keep it
None.

Per-req prefill/decode classification:
    Inside an EXTEND/MIXED batch a req with `extend_input_len > 1` is treated
    as prefill (its uncached new tokens for this step) and a req with
    `extend_input_len == 1` is treated as decode (chunked-prefill stepping
    alongside; physically that step is a decode against its own existing KV).
    DECODE-mode batches set everyone to decode. Special forward modes
    (DRAFT_EXTEND, SPLIT_PREFILL, IDLE, ...) are skipped — fitting data should
    come from the normal decode/extend/mixed steady-state.
"""

from __future__ import annotations

import collections
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# Internal payload pushed from hot path to flusher thread.
# Plain ints/floats only — no references to ScheduleBatch / Req objects so the
# scheduler is free to mutate the batch in the next step.
@dataclass(frozen=True)
class _Sample:
    step_idx: int
    ts_ns: int
    sum_n_sq: float
    sum_nr: float
    sum_n: int
    sum_r: int
    bs_d: int
    step_time_ms: float
    num_prefill_reqs: int
    num_decode_reqs: int
    max_kv: int
    forward_mode: str


class HaloCostModelSampler:
    """Bounded-queue per-step JSONL sampler with background flush.

    Threading model:
      - Hot path (scheduler thread): observe_step → in-memory deque append.
      - Flusher thread (daemon): pops batches every `flush_interval_ms`,
        writes JSONL with buffered I/O, flushes the file handle.

    When the queue is full, the OLDEST samples are dropped. We log a periodic
    WARN with the drop count so operators see the data loss without flooding
    logs.
    """

    DEFAULT_MAX_QUEUE = 10_000
    DEFAULT_FLUSH_INTERVAL_MS = 10
    DROP_LOG_INTERVAL_S = 5.0

    def __init__(
        self,
        log_path: str,
        sample_every: int = 1,
        max_queue: int = DEFAULT_MAX_QUEUE,
        flush_interval_ms: int = DEFAULT_FLUSH_INTERVAL_MS,
    ):
        if sample_every < 1:
            raise ValueError(
                f"sample_every must be >= 1, got {sample_every}"
            )
        self.log_path = log_path
        self.sample_every = sample_every
        self._queue: collections.deque[_Sample] = collections.deque(
            maxlen=max_queue
        )
        self._lock = threading.Lock()
        self._step_counter = 0
        self._dropped_total = 0
        self._last_drop_log = time.monotonic()
        self._stop = threading.Event()
        self._flush_interval_s = max(flush_interval_ms / 1000.0, 0.001)

        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        # Append mode — multiple runs to the same path concatenate cleanly.
        self._fh = open(log_path, "a", buffering=8192, encoding="utf-8")

        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="halo-cost-sampler-flush",
            daemon=True,
        )
        self._flush_thread.start()
        logger.info(
            "halo cost-model sampler: writing to %s (sample_every=%d, "
            "max_queue=%d, flush_interval_ms=%d)",
            log_path, sample_every, max_queue, flush_interval_ms,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Hot path — must stay cheap (microsecond scale).
    # ─────────────────────────────────────────────────────────────────────
    def observe_step(self, batch, step_time_ms: float) -> None:
        """Called from scheduler thread after each forward step.

        Cheap path: counter increment + (every Nth call) feature extraction
        + bounded-queue append. No I/O.
        """
        self._step_counter += 1
        if self._step_counter % self.sample_every != 0:
            return

        sample = self._extract_features(batch, step_time_ms)
        if sample is None:
            return

        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self._dropped_total += 1
                self._maybe_log_drops_locked()
            self._queue.append(sample)

    def _maybe_log_drops_locked(self) -> None:
        now = time.monotonic()
        if now - self._last_drop_log >= self.DROP_LOG_INTERVAL_S:
            logger.warning(
                "halo cost-model sampler: queue full — %d samples dropped "
                "(total since start). Increase --halo-cost-model-sample-every "
                "or raise max_queue if this persists.",
                self._dropped_total,
            )
            self._last_drop_log = now

    def _extract_features(self, batch, step_time_ms: float) -> Optional[_Sample]:
        """Extract the 5-tuple of Σ features + bookkeeping from ScheduleBatch.

        Returns None for batches we don't want to fit on (idle, speculative
        draft, split-prefill, etc.)
        """
        fm = batch.forward_mode
        if fm is None:
            return None

        # Use is_decode / is_extend (both also cover MIXED via is_extend).
        # Speculative + idle + split-prefill modes are skipped.
        if not (fm.is_decode() or fm.is_extend()):
            return None

        reqs = batch.reqs or []
        if not reqs:
            return None

        sum_n_sq = 0.0
        sum_nr = 0.0
        sum_n = 0
        sum_r = 0
        bs_d = 0
        num_prefill = 0
        max_kv = 0

        # DECODE batch: everyone is decoding. Use seq_lens_cpu when available,
        # otherwise len(req.fill_ids) as a fallback.
        if fm.is_decode():
            seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
            for i, req in enumerate(reqs):
                if seq_lens_cpu is not None:
                    r = int(seq_lens_cpu[i].item())
                else:
                    r = len(req.fill_ids)
                sum_r += r
                bs_d += 1
                if r > max_kv:
                    max_kv = r
        else:
            # EXTEND / MIXED: per-req classification by extend_input_len.
            #   extend_input_len  > 1  → prefill contribution (n=extend_input_len, r=prefix_len)
            #   extend_input_len == 1  → decode contribution  (r=seq_len_before this step)
            prefix_lens = batch.prefix_lens or [0] * len(reqs)
            extend_lens = batch.extend_lens or [
                getattr(r, "extend_input_len", 0) for r in reqs
            ]
            seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
            for i, req in enumerate(reqs):
                n = int(extend_lens[i]) if i < len(extend_lens) else 0
                r_prefix = (
                    int(prefix_lens[i]) if i < len(prefix_lens) else 0
                )
                if n > 1:
                    sum_n_sq += n * n
                    sum_nr += n * r_prefix
                    sum_n += n
                    num_prefill += 1
                    # max_kv across all reqs (prefix + extend tokens after this step)
                    candidate = r_prefix + n
                    if candidate > max_kv:
                        max_kv = candidate
                else:
                    # decode-equivalent step for this req
                    if seq_lens_cpu is not None and i < len(seq_lens_cpu):
                        r_decode = int(seq_lens_cpu[i].item())
                    else:
                        r_decode = r_prefix
                    sum_r += r_decode
                    bs_d += 1
                    if r_decode > max_kv:
                        max_kv = r_decode

        return _Sample(
            step_idx=self._step_counter,
            ts_ns=time.time_ns(),
            sum_n_sq=sum_n_sq,
            sum_nr=sum_nr,
            sum_n=sum_n,
            sum_r=sum_r,
            bs_d=bs_d,
            step_time_ms=step_time_ms,
            num_prefill_reqs=num_prefill,
            num_decode_reqs=bs_d,
            max_kv=max_kv,
            forward_mode=fm.name,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Background flush thread — owns the file handle.
    # ─────────────────────────────────────────────────────────────────────
    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(self._flush_interval_s)
            self._drain_once()
        # final flush after stop signal
        self._drain_once()
        try:
            self._fh.flush()
        except Exception:
            pass

    def _drain_once(self) -> None:
        # Move out the queue contents under lock, then write outside.
        with self._lock:
            if not self._queue:
                return
            samples = list(self._queue)
            self._queue.clear()
        for s in samples:
            row = {
                "step_idx": s.step_idx,
                "ts_ns": s.ts_ns,
                "sum_n_sq": s.sum_n_sq,
                "sum_nr": s.sum_nr,
                "sum_n": s.sum_n,
                "sum_r": s.sum_r,
                "bs_d": s.bs_d,
                "step_time_ms": s.step_time_ms,
                "num_prefill_reqs": s.num_prefill_reqs,
                "num_decode_reqs": s.num_decode_reqs,
                "max_kv": s.max_kv,
                "forward_mode": s.forward_mode,
            }
            try:
                self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            except Exception as e:
                # Don't crash the scheduler over a logging failure.
                logger.warning(
                    "halo cost-model sampler: write failed — %s", e
                )
                return
        try:
            self._fh.flush()
        except Exception:
            pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._flush_thread.join(timeout=2.0)
        finally:
            try:
                self._fh.close()
            except Exception:
                pass

    # For tests / introspection.
    def _queue_len_for_test(self) -> int:
        with self._lock:
            return len(self._queue)

    @property
    def dropped_total(self) -> int:
        return self._dropped_total


def build_halo_cost_sampler_from_server_args(
    server_args, attn_tp_rank: int
) -> Optional[HaloCostModelSampler]:
    """Factory: returns a sampler iff this rank should record and a path is set.

    Mirrors the existing `build_halo_controller_from_server_args` factory
    pattern. Only attn_tp_rank == 0 records (TP-dedup).
    """
    if attn_tp_rank != 0:
        return None
    path = getattr(server_args, "halo_cost_model_sample_log", None)
    if not path:
        return None
    sample_every = int(
        getattr(server_args, "halo_cost_model_sample_every", 1) or 1
    )
    try:
        return HaloCostModelSampler(log_path=path, sample_every=sample_every)
    except OSError as e:
        logger.warning(
            "halo cost-model sampler: failed to open %s — sampling disabled (%s)",
            path, e,
        )
        return None
