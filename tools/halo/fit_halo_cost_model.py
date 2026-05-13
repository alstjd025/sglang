#!/usr/bin/env python3
"""Fit the Halo Step Cost Model (6 coefficients) from a per-step JSONL sample log.

Reads JSONL lines emitted by `managers/halo/cost_model_sampler.py` (each line
holds the 5-tuple of Σ features + measured step_time_ms) and writes the fitted
JSON consumed by `try_load_halo_step_cost_model`.

Design + rationale: ms_dev/halo_dev/prediction_model.md §10.

Form (per ms_dev/halo_dev/prediction_model.md §3):

    T_step_ms ≈ θ_p1·Σnᵢ² + θ_p2·Σ(nᵢ·rᵢ) + θ_p3·Σnᵢ
              + θ_d1·Σrⱼ + θ_d2·bs_d + θ_c

Algorithm: ordinary least squares via numpy.linalg.lstsq.

Usage:

    python tools/halo/fit_halo_cost_model.py \\
        --samples ms_dev/runtime/sessions/<id>/halo_cost_samples.jsonl \\
        --output  ms_dev/runtime/cost_models/halo_step_llama3-70b_b200x4.json \\
        --model   "Llama-3.3-70B-Instruct" \\
        --hw      "B200x4 TP=4"

Optional filters (useful for sanity-checking subsets):

    --filter-forward-mode DECODE              # only decode steps
    --filter-forward-mode EXTEND,MIXED        # only prefill-containing steps
    --min-step-time-ms 1.0                    # drop noisy near-zero rows
    --max-step-time-ms 1000.0                 # drop outliers
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("halo.fit")


# JSON form identifiers — keep in sync with managers/admission_control/cost_model.py.
HALO_STEP_FORM_V1 = "halo_step_v1"
HALO_STEP_FORM_SPLIT_V1 = "halo_step_split_v1"

# Feature order — keep in sync with HaloStepCostModel coefficients.
FEATURE_NAMES = ("sum_n_sq", "sum_nr", "sum_n", "sum_r", "bs_d")
THETA_NAMES = ("theta_p1", "theta_p2", "theta_p3", "theta_d1", "theta_d2")
PREFILL_FEATURES = ("sum_n_sq", "sum_nr", "sum_n")
PREFILL_THETAS = ("theta_p1", "theta_p2", "theta_p3")
DECODE_FEATURES = ("sum_r", "bs_d")
DECODE_THETAS = ("theta_d1", "theta_d2")


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────
def iter_samples(paths: Sequence[str]) -> Iterable[Dict[str, Any]]:
    """Yield JSONL rows from one or more files. Bad lines logged + skipped."""
    n_lines = 0
    n_bad = 0
    for p in paths:
        with open(p) as f:
            for line in f:
                n_lines += 1
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    n_bad += 1
                    continue
    if n_bad:
        logger.warning(
            "fit: skipped %d malformed JSON lines out of %d total",
            n_bad, n_lines,
        )


def expand_paths(samples_args: Sequence[str]) -> List[str]:
    """Expand glob patterns and validate that each path exists."""
    out: List[str] = []
    for a in samples_args:
        matched = glob.glob(a) if any(c in a for c in "*?[") else [a]
        if not matched:
            raise FileNotFoundError(f"No samples file matched: {a}")
        for m in matched:
            if not Path(m).is_file():
                raise FileNotFoundError(f"Not a file: {m}")
            out.append(m)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────
def make_filter(
    forward_modes: Optional[Sequence[str]],
    min_step_time_ms: float,
    max_step_time_ms: float,
):
    fm_set = set(forward_modes) if forward_modes else None

    def keep(row: Dict[str, Any]) -> bool:
        if fm_set is not None and row.get("forward_mode") not in fm_set:
            return False
        t = row.get("step_time_ms")
        if t is None or not isinstance(t, (int, float)):
            return False
        if t < min_step_time_ms or t > max_step_time_ms:
            return False
        # Defensive: every feature must be a finite number.
        for k in FEATURE_NAMES:
            v = row.get(k)
            if v is None or not isinstance(v, (int, float)):
                return False
            if not math.isfinite(v):
                return False
        return True

    return keep


# ─────────────────────────────────────────────────────────────────────────────
# OLS fit
# ─────────────────────────────────────────────────────────────────────────────
def build_design_matrix(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) with X shape (n, 6), y shape (n,)."""
    n = len(rows)
    X = np.empty((n, 6), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, 0] = r["sum_n_sq"]
        X[i, 1] = r["sum_nr"]
        X[i, 2] = r["sum_n"]
        X[i, 3] = r["sum_r"]
        X[i, 4] = r["bs_d"]
        X[i, 5] = 1.0  # constant term
        y[i] = r["step_time_ms"]
    return X, y


def fit_ols(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Returns (theta, rmse_ms, r_squared)."""
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_pred = X @ theta
    residuals = y - y_pred
    rmse = float(np.sqrt(np.mean(residuals * residuals)))
    ss_res = float(np.sum(residuals * residuals))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return theta, rmse, r2


def build_prefill_design_matrix(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Split-mode prefill fit: only Σnᵢ², Σ(nᵢrᵢ), Σnᵢ + constant."""
    n = len(rows)
    X = np.empty((n, 4), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, 0] = r["sum_n_sq"]
        X[i, 1] = r["sum_nr"]
        X[i, 2] = r["sum_n"]
        X[i, 3] = 1.0
        y[i] = r["step_time_ms"]
    return X, y


def build_decode_design_matrix(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Split-mode decode fit: only Σrⱼ, bs_d + constant."""
    n = len(rows)
    X = np.empty((n, 3), dtype=np.float64)
    y = np.empty(n, dtype=np.float64)
    for i, r in enumerate(rows):
        X[i, 0] = r["sum_r"]
        X[i, 1] = r["bs_d"]
        X[i, 2] = 1.0
        y[i] = r["step_time_ms"]
    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────
def _base_metadata(
    args: argparse.Namespace, samples_paths: Sequence[str]
) -> Dict[str, Any]:
    # json.dumps emits non-standard "Infinity" for float('inf'); serialize as
    # None so the JSON stays strictly RFC-compliant.
    max_ms: Any = args.max_step_time_ms
    if not math.isfinite(max_ms):
        max_ms = None
    return {
        "model": args.model or "",
        "hw": args.hw or "",
        "fit_at": datetime.now(timezone.utc).isoformat(),
        "source_jsonl": list(samples_paths),
        "filter_forward_modes": (
            args.filter_forward_mode.split(",")
            if args.filter_forward_mode
            else None
        ),
        "min_step_time_ms": args.min_step_time_ms,
        "max_step_time_ms": max_ms,
    }


def write_unified_model_json(
    output_path: str,
    theta: np.ndarray,
    rmse_ms: float,
    r_squared: float,
    n_samples: int,
    args: argparse.Namespace,
    samples_paths: Sequence[str],
) -> None:
    meta = _base_metadata(args, samples_paths)
    meta.update(
        {"n_samples": n_samples, "rmse_ms": rmse_ms, "r_squared": r_squared}
    )
    payload = {
        "form": HALO_STEP_FORM_V1,
        "theta_p1": float(theta[0]),
        "theta_p2": float(theta[1]),
        "theta_p3": float(theta[2]),
        "theta_d1": float(theta[3]),
        "theta_d2": float(theta[4]),
        "theta_c":  float(theta[5]),
        "fit_metadata": meta,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2))


def write_split_model_json(
    output_path: str,
    theta_p: np.ndarray,
    theta_d: np.ndarray,
    prefill_stats: Dict[str, float],
    decode_stats: Dict[str, float],
    combined_rmse: float,
    args: argparse.Namespace,
    samples_paths: Sequence[str],
) -> None:
    meta = _base_metadata(args, samples_paths)
    meta.update(
        {
            "n_samples_total": prefill_stats["n"] + decode_stats["n"],
            "combined_rmse_ms": combined_rmse,
            "prefill": {
                "n_samples": prefill_stats["n"],
                "rmse_ms": prefill_stats["rmse"],
                "r_squared": prefill_stats["r2"],
            },
            "decode": {
                "n_samples": decode_stats["n"],
                "rmse_ms": decode_stats["rmse"],
                "r_squared": decode_stats["r2"],
            },
        }
    )
    payload = {
        "form": HALO_STEP_FORM_SPLIT_V1,
        "theta_p1": float(theta_p[0]),
        "theta_p2": float(theta_p[1]),
        "theta_p3": float(theta_p[2]),
        "theta_c_p": float(theta_p[3]),
        "theta_d1": float(theta_d[0]),
        "theta_d2": float(theta_d[1]),
        "theta_c_d": float(theta_d[2]),
        "fit_metadata": meta,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit Halo Step Cost Model from per-step JSONL samples."
    )
    p.add_argument(
        "--samples", nargs="+", required=True,
        help="One or more JSONL files (glob patterns OK).",
    )
    p.add_argument(
        "--output", required=True,
        help="Output JSON path (halo_step_v1).",
    )
    p.add_argument("--model", default="", help="Model name for fit_metadata.")
    p.add_argument("--hw", default="", help="Hardware/TP descriptor for fit_metadata.")
    p.add_argument(
        "--filter-forward-mode", default=None,
        help="Comma-separated forward modes to keep (e.g. DECODE or EXTEND,MIXED). Default: all.",
    )
    p.add_argument(
        "--min-step-time-ms", type=float, default=0.0,
        help="Drop rows with step_time_ms below this value (filter noise).",
    )
    p.add_argument(
        "--max-step-time-ms", type=float, default=float("inf"),
        help="Drop rows with step_time_ms above this value (filter outliers).",
    )
    p.add_argument(
        "--min-samples", type=int, default=200,
        help="Refuse to fit if fewer than this many rows remain after filtering.",
    )
    # ── Form selection ──────────────────────────────────────────────────────
    # Default is "split" since 2026-05-14 (validated as best-balanced on this
    # workload; see prediction_model.md §17). Use --form unified (or the
    # deprecated --unified alias) to fall back to the joint 6-coefficient fit.
    p.add_argument(
        "--form", choices=("split", "unified"), default="split",
        help="Which form to fit. 'split' (default since 2026-05-14, "
             "recommended) writes halo_step_split_v1 — prefill and decode "
             "fit separately with their own intercepts; preferred when MIXED "
             "steps are absent (--enable-mixed-chunk off). 'unified' writes "
             "halo_step_v1 — joint 6-coef fit; preferred when MIXED steps "
             "appear. See prediction_model.md §17.",
    )
    p.add_argument(
        "--split", action="store_true",
        help="Alias for --form split (kept for backward compatibility).",
    )
    p.add_argument(
        "--unified", action="store_true",
        help="Alias for --form unified.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging.",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    paths = expand_paths(args.samples)
    logger.info("fit: reading %d sample file(s)", len(paths))

    forward_modes = (
        [s.strip() for s in args.filter_forward_mode.split(",") if s.strip()]
        if args.filter_forward_mode
        else None
    )
    keep = make_filter(
        forward_modes=forward_modes,
        min_step_time_ms=args.min_step_time_ms,
        max_step_time_ms=args.max_step_time_ms,
    )

    rows: List[Dict[str, Any]] = []
    for r in iter_samples(paths):
        if keep(r):
            rows.append(r)
    n = len(rows)
    logger.info("fit: %d rows kept after filtering", n)
    if n < args.min_samples:
        logger.error(
            "fit: only %d rows — below --min-samples=%d. Collect more samples "
            "or relax filters.",
            n, args.min_samples,
        )
        return 2

    # Resolve form: --form takes precedence; --split / --unified are aliases.
    if args.split and args.unified:
        logger.error("fit: --split and --unified are mutually exclusive")
        return 2
    form = args.form
    if args.split:
        form = "split"
    elif args.unified:
        form = "unified"

    if form == "unified":
        # Unified fit (halo_step_v1) — single shared θ_c.
        X, y = build_design_matrix(rows)
        theta, rmse, r2 = fit_ols(X, y)
        write_unified_model_json(args.output, theta, rmse, r2, n, args, paths)
        logger.info(
            "fit: wrote %s [unified] (rmse=%.3f ms, R²=%.4f, n=%d)",
            args.output, rmse, r2, n,
        )
        logger.info(
            "fit: coefficients θ_p1=%.4g θ_p2=%.4g θ_p3=%.4g "
            "θ_d1=%.4g θ_d2=%.4g θ_c=%.4g",
            *theta,
        )
        return 0

    # Split fit (halo_step_split_v1).
    prefill_rows = [r for r in rows if r.get("forward_mode") == "EXTEND"]
    decode_rows = [r for r in rows if r.get("forward_mode") == "DECODE"]
    other_rows = [
        r for r in rows
        if r.get("forward_mode") not in ("EXTEND", "DECODE")
    ]
    if other_rows:
        logger.warning(
            "fit --split: %d rows have forward_mode not in {EXTEND,DECODE} "
            "(e.g. MIXED) and will be DROPPED from the fit. Split mode assumes "
            "these are absent. If they're frequent, use unified fit instead.",
            len(other_rows),
        )
    if len(prefill_rows) < args.min_samples // 4 or len(decode_rows) < args.min_samples // 4:
        logger.warning(
            "fit --split: prefill rows %d / decode rows %d — one side has very "
            "few samples; fit there will be noisy.",
            len(prefill_rows), len(decode_rows),
        )

    Xp, yp = build_prefill_design_matrix(prefill_rows)
    theta_p, rmse_p, r2_p = fit_ols(Xp, yp)
    Xd, yd = build_decode_design_matrix(decode_rows)
    theta_d, rmse_d, r2_d = fit_ols(Xd, yd)

    # Combined RMSE across both fits (same denominator as unified for
    # easy comparison).
    total_n = len(prefill_rows) + len(decode_rows)
    ss_res = float(
        np.sum(((Xp @ theta_p) - yp) ** 2)
        + np.sum(((Xd @ theta_d) - yd) ** 2)
    )
    combined_rmse = float(np.sqrt(ss_res / total_n)) if total_n else float("nan")

    write_split_model_json(
        args.output,
        theta_p, theta_d,
        {"n": len(prefill_rows), "rmse": rmse_p, "r2": r2_p},
        {"n": len(decode_rows), "rmse": rmse_d, "r2": r2_d},
        combined_rmse,
        args, paths,
    )
    logger.info(
        "fit: wrote %s [split] (combined_rmse=%.3f ms, "
        "prefill: rmse=%.3f R²=%.4f n=%d, "
        "decode: rmse=%.3f R²=%.4f n=%d)",
        args.output, combined_rmse,
        rmse_p, r2_p, len(prefill_rows),
        rmse_d, r2_d, len(decode_rows),
    )
    logger.info(
        "fit prefill: θ_p1=%.4g θ_p2=%.4g θ_p3=%.4g θ_c_p=%.4g",
        *theta_p,
    )
    logger.info(
        "fit decode:  θ_d1=%.4g θ_d2=%.4g θ_c_d=%.4g",
        *theta_d,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
