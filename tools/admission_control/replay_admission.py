#!/usr/bin/env python3
"""Offline replay tool for admission control decision logs.

See tools/admission_control/CLAUDE.md.

Reads JSONL produced by --admission-decision-log and reports, for each
candidate (TTFT_SLO, TBT_SLO) pair, what would have been admitted/rejected.
Lets you tune SLOs against real traffic before turning enforcement on.

Caveat: replay assumes the predicted_* values stored in each log row are
"what the predictor output at decision time". It does NOT recompute
predictions from a different cost model — only re-thresholds the same
predictions. It also does NOT simulate the feedback effect of a different
reject decision on subsequent queue depth (the queue_predicted_ms in
later rows reflects the actual admission outcome, not the
counterfactual). Treat reported reject rates as upper bounds.

Usage
-----

  python tools/admission_control/replay_admission.py \\
      --decision-log ms_dev/runtime/sessions/<sess>/admission_decisions.jsonl \\
      --ttft-slo 1000 2000 5000 \\
      --tbt-slo 50 100 200

  python tools/admission_control/replay_admission.py \\
      --decision-log <log> --ttft-slo 2000 --tbt-slo 100 --csv-out report.csv

  # Single SLO pair, also dump per-decision counterfactual table:
  python tools/admission_control/replay_admission.py \\
      --decision-log <log> --ttft-slo 2000 --tbt-slo 100 \\
      --details-out details.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# Reasons must stay in sync with managers/admission_control/controller.py.
REASON_ADMIT = "ADMIT"
REASON_DISABLED = "DISABLED"
REASON_TTFT_PREDICTED = "TTFT_PREDICTED"
REASON_TTFT_RATIO = "TTFT_RATIO"
REASON_TBT_PREDICTED = "TBT_PREDICTED"
REASON_TBT_RATIO = "TBT_RATIO"
REASON_TBT_REACTIVE = "TBT_REACTIVE"

_RATIO_SOLO_FLOOR_MS = 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class DecisionRow:
    rid: str
    ts: float
    admit: bool
    reason: str
    dry_run_would_reject: bool
    predicted_ttft_ms: Optional[float]
    predicted_tbt_ms: Optional[float]
    solo_ttft_ms: Optional[float]
    solo_tbt_ms: Optional[float]
    tbt_ewma_ms: Optional[float]
    queue_predicted_ms: Optional[float]
    ttft_slo_ms: Optional[float]
    tbt_slo_ms: Optional[float]
    ttft_slo_ratio: Optional[float]
    tbt_slo_ratio: Optional[float]
    prompt_len: int
    prefix_len: int
    running_batch_size: int
    running_batch_total_kv_tokens: int

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DecisionRow":
        return cls(
            rid=str(d.get("rid", "")),
            ts=float(d.get("ts", 0.0)),
            admit=bool(d.get("admit", True)),
            reason=str(d.get("reason", "")),
            dry_run_would_reject=bool(d.get("dry_run_would_reject", False)),
            predicted_ttft_ms=_optfloat(d.get("predicted_ttft_ms")),
            predicted_tbt_ms=_optfloat(d.get("predicted_tbt_ms")),
            solo_ttft_ms=_optfloat(d.get("solo_ttft_ms")),
            solo_tbt_ms=_optfloat(d.get("solo_tbt_ms")),
            tbt_ewma_ms=_optfloat(d.get("tbt_ewma_ms")),
            queue_predicted_ms=_optfloat(d.get("queue_predicted_ms")),
            ttft_slo_ms=_optfloat(d.get("ttft_slo_ms")),
            tbt_slo_ms=_optfloat(d.get("tbt_slo_ms")),
            ttft_slo_ratio=_optfloat(d.get("ttft_slo_ratio")),
            tbt_slo_ratio=_optfloat(d.get("tbt_slo_ratio")),
            prompt_len=int(d.get("prompt_len", 0)),
            prefix_len=int(d.get("prefix_len", 0)),
            running_batch_size=int(d.get("running_batch_size", 0)),
            running_batch_total_kv_tokens=int(d.get("running_batch_total_kv_tokens", 0)),
        )


def _optfloat(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_log(path: Path) -> List[DecisionRow]:
    rows: List[DecisionRow] = []
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    bad += 1
                    continue
                rows.append(DecisionRow.from_dict(obj))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"[replay] WARN: skipped {bad} unparseable lines", file=sys.stderr)
    return rows


# ---------------------------------------------------------------------------
# Counterfactual evaluation
# ---------------------------------------------------------------------------


@dataclass
class SloOutcome:
    ttft_slo_ms: float
    tbt_slo_ms: float
    ttft_slo_ratio: float
    tbt_slo_ratio: float
    reactive_ratio: float
    total: int
    admit: int
    rej_ttft: int
    rej_ttft_ratio: int
    rej_tbt_pred: int
    rej_tbt_ratio: int
    rej_tbt_react: int
    skipped_disabled: int

    @property
    def reject_total(self) -> int:
        return (
            self.rej_ttft
            + self.rej_ttft_ratio
            + self.rej_tbt_pred
            + self.rej_tbt_ratio
            + self.rej_tbt_react
        )

    @property
    def reject_pct(self) -> float:
        n = self.total - self.skipped_disabled
        return (self.reject_total / n * 100.0) if n > 0 else 0.0


def evaluate_row(
    row: DecisionRow,
    ttft_slo_ms: float,
    tbt_slo_ms: float,
    ttft_slo_ratio: float,
    tbt_slo_ratio: float,
    reactive_ratio: float,
) -> str:
    """Apply the same hybrid policy at the candidate SLOs.

    Set any threshold to 0 (or below 1.0 for ratios) to disable that stage.
    Returns one of REASON_* strings. DISABLED rows pass through.
    """
    if row.reason == REASON_DISABLED:
        return REASON_DISABLED

    # Stage 1a: TTFT absolute
    if (
        ttft_slo_ms > 0
        and row.predicted_ttft_ms is not None
        and row.predicted_ttft_ms > ttft_slo_ms
    ):
        return REASON_TTFT_PREDICTED

    # Stage 1b: TTFT ratio
    if (
        ttft_slo_ratio > 1.0
        and row.predicted_ttft_ms is not None
        and row.solo_ttft_ms is not None
        and row.solo_ttft_ms > _RATIO_SOLO_FLOOR_MS
        and row.predicted_ttft_ms / row.solo_ttft_ms > ttft_slo_ratio
    ):
        return REASON_TTFT_RATIO

    # Stage 2a: TBT absolute
    if (
        tbt_slo_ms > 0
        and row.predicted_tbt_ms is not None
        and row.predicted_tbt_ms > tbt_slo_ms
    ):
        return REASON_TBT_PREDICTED

    # Stage 2b: TBT ratio
    if (
        tbt_slo_ratio > 1.0
        and row.predicted_tbt_ms is not None
        and row.solo_tbt_ms is not None
        and row.solo_tbt_ms > _RATIO_SOLO_FLOOR_MS
        and row.predicted_tbt_ms / row.solo_tbt_ms > tbt_slo_ratio
    ):
        return REASON_TBT_RATIO

    # Stage 3: reactive (only meaningful with absolute TBT SLO)
    if (
        tbt_slo_ms > 0
        and row.tbt_ewma_ms is not None
        and row.tbt_ewma_ms > tbt_slo_ms * reactive_ratio
    ):
        return REASON_TBT_REACTIVE

    return REASON_ADMIT


def evaluate_slo(
    rows: List[DecisionRow],
    ttft_slo_ms: float,
    tbt_slo_ms: float,
    ttft_slo_ratio: float,
    tbt_slo_ratio: float,
    reactive_ratio: float,
) -> SloOutcome:
    out = SloOutcome(
        ttft_slo_ms=ttft_slo_ms,
        tbt_slo_ms=tbt_slo_ms,
        ttft_slo_ratio=ttft_slo_ratio,
        tbt_slo_ratio=tbt_slo_ratio,
        reactive_ratio=reactive_ratio,
        total=len(rows),
        admit=0,
        rej_ttft=0,
        rej_ttft_ratio=0,
        rej_tbt_pred=0,
        rej_tbt_ratio=0,
        rej_tbt_react=0,
        skipped_disabled=0,
    )
    for r in rows:
        verdict = evaluate_row(
            r, ttft_slo_ms, tbt_slo_ms, ttft_slo_ratio, tbt_slo_ratio, reactive_ratio
        )
        if verdict == REASON_ADMIT:
            out.admit += 1
        elif verdict == REASON_TTFT_PREDICTED:
            out.rej_ttft += 1
        elif verdict == REASON_TTFT_RATIO:
            out.rej_ttft_ratio += 1
        elif verdict == REASON_TBT_PREDICTED:
            out.rej_tbt_pred += 1
        elif verdict == REASON_TBT_RATIO:
            out.rej_tbt_ratio += 1
        elif verdict == REASON_TBT_REACTIVE:
            out.rej_tbt_react += 1
        else:  # DISABLED
            out.skipped_disabled += 1
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(outcomes: Iterable[SloOutcome], stream=sys.stdout) -> None:
    rows = list(outcomes)
    if not rows:
        print("(no SLO combinations evaluated)", file=stream)
        return
    print(file=stream)
    print(
        f"{'TTFT_SLO':>9}  {'TBT_SLO':>8}  {'TTFTrx':>6}  {'TBTrx':>6}  "
        f"{'total':>6}  {'admit':>6}  {'rTTFT':>6}  {'rTTFTr':>7}  "
        f"{'rTBTp':>6}  {'rTBTr':>6}  {'rTBTrx':>7}  {'rej_pct':>8}",
        file=stream,
    )
    print("-" * 105, file=stream)
    for r in rows:
        print(
            f"{r.ttft_slo_ms:>9.0f}  {r.tbt_slo_ms:>8.0f}  "
            f"{r.ttft_slo_ratio:>6.2f}  {r.tbt_slo_ratio:>6.2f}  "
            f"{r.total:>6d}  {r.admit:>6d}  "
            f"{r.rej_ttft:>6d}  {r.rej_ttft_ratio:>7d}  "
            f"{r.rej_tbt_pred:>6d}  {r.rej_tbt_ratio:>6d}  "
            f"{r.rej_tbt_react:>7d}  {r.reject_pct:>7.2f}%",
            file=stream,
        )
    print(file=stream)


def write_csv(outcomes: Iterable[SloOutcome], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ttft_slo_ms", "tbt_slo_ms",
            "ttft_slo_ratio", "tbt_slo_ratio", "reactive_ratio",
            "total", "admit",
            "rej_ttft", "rej_ttft_ratio",
            "rej_tbt_pred", "rej_tbt_ratio", "rej_tbt_react",
            "skipped_disabled", "reject_pct",
        ])
        for r in outcomes:
            w.writerow([
                r.ttft_slo_ms, r.tbt_slo_ms,
                r.ttft_slo_ratio, r.tbt_slo_ratio, r.reactive_ratio,
                r.total, r.admit,
                r.rej_ttft, r.rej_ttft_ratio,
                r.rej_tbt_pred, r.rej_tbt_ratio, r.rej_tbt_react,
                r.skipped_disabled, f"{r.reject_pct:.4f}",
            ])


def write_details_csv(
    rows: List[DecisionRow],
    ttft_slo_ms: float,
    tbt_slo_ms: float,
    ttft_slo_ratio: float,
    tbt_slo_ratio: float,
    reactive_ratio: float,
    path: Path,
) -> None:
    """Per-decision CSV with both the original verdict and the counterfactual."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "ts", "rid", "prompt_len", "prefix_len",
            "running_batch_size", "running_batch_total_kv_tokens",
            "predicted_ttft_ms", "predicted_tbt_ms",
            "solo_ttft_ms", "solo_tbt_ms",
            "tbt_ewma_ms", "queue_predicted_ms",
            "original_admit", "original_reason",
            "replay_verdict",
            "ttft_slo_ms", "tbt_slo_ms",
            "ttft_slo_ratio", "tbt_slo_ratio", "reactive_ratio",
        ])
        for r in rows:
            verdict = evaluate_row(
                r, ttft_slo_ms, tbt_slo_ms,
                ttft_slo_ratio, tbt_slo_ratio, reactive_ratio,
            )
            w.writerow([
                r.ts, r.rid, r.prompt_len, r.prefix_len,
                r.running_batch_size, r.running_batch_total_kv_tokens,
                r.predicted_ttft_ms, r.predicted_tbt_ms,
                r.solo_ttft_ms, r.solo_tbt_ms,
                r.tbt_ewma_ms, r.queue_predicted_ms,
                r.admit, r.reason,
                verdict,
                ttft_slo_ms, tbt_slo_ms,
                ttft_slo_ratio, tbt_slo_ratio, reactive_ratio,
            ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decision-log", type=Path, required=True,
                   help="JSONL file produced by --admission-decision-log")
    p.add_argument("--ttft-slo", type=float, nargs="+", default=[0.0],
                   help="candidate TTFT SLO(s) in ms; pass 0 to disable absolute TTFT check")
    p.add_argument("--tbt-slo", type=float, nargs="+", default=[0.0],
                   help="candidate TBT SLO(s) in ms; pass 0 to disable absolute TBT check")
    p.add_argument("--ttft-slo-ratio", type=float, nargs="+", default=[1.0],
                   help="candidate TTFT slowdown ratio(s) vs solo-run; pass 1 to disable")
    p.add_argument("--tbt-slo-ratio", type=float, nargs="+", default=[1.0],
                   help="candidate TBT slowdown ratio(s) vs solo-run; pass 1 to disable")
    p.add_argument("--reactive-ratio", type=float, default=0.9,
                   help="Stage 3 reactive threshold = tbt_slo * this (default 0.9)")
    p.add_argument("--csv-out", type=Path, default=None,
                   help="write summary table to CSV")
    p.add_argument("--details-out", type=Path, default=None,
                   help="write per-decision counterfactual CSV (only meaningful with a single SLO combo)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if not args.decision_log.exists():
        print(f"[replay] decision log not found: {args.decision_log}", file=sys.stderr)
        return 2

    rows = load_log(args.decision_log)
    if not rows:
        print(f"[replay] no rows loaded from {args.decision_log}", file=sys.stderr)
        return 2

    print(
        f"[replay] loaded {len(rows)} rows from {args.decision_log}",
        file=sys.stderr,
    )

    combos: List[Tuple[float, float, float, float]] = [
        (t, b, tr, br)
        for t in args.ttft_slo
        for b in args.tbt_slo
        for tr in args.ttft_slo_ratio
        for br in args.tbt_slo_ratio
    ]
    outcomes = [
        evaluate_slo(rows, t, b, tr, br, args.reactive_ratio)
        for (t, b, tr, br) in combos
    ]

    print_summary(outcomes)

    if args.csv_out is not None:
        write_csv(outcomes, args.csv_out)
        print(f"[replay] summary CSV: {args.csv_out}", file=sys.stderr)

    if args.details_out is not None:
        if len(combos) != 1:
            print(
                "[replay] WARN: --details-out is only meaningful with a single "
                f"SLO combo; skipping (got {len(combos)} combos)",
                file=sys.stderr,
            )
        else:
            t, b, tr, br = combos[0]
            write_details_csv(
                rows, t, b, tr, br, args.reactive_ratio, args.details_out
            )
            print(f"[replay] per-decision CSV: {args.details_out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
