#!/usr/bin/env python3
"""Fit admission-control cost models against a running SGLang server.

See tools/admission_control/CLAUDE.md for the full design and
managers/admission_control/CLAUDE.md for how the resulting JSONs are consumed.

Usage
-----

  Prefill model (T_prefill ≈ α·d² + β·d + γ, d = n − p):

    python tools/admission_control/fit_cost_model.py prefill \\
        --server http://localhost:31000 \\
        --output ms_dev/runtime/cost_models/prefill_llama3-70b.json

  TBT model (TBT ≈ a + b·batch_size + c·total_kv_tokens):

    python tools/admission_control/fit_cost_model.py tbt \\
        --server http://localhost:31000 \\
        --output ms_dev/runtime/cost_models/tbt_llama3-70b.json

The server must be reachable, idle (no other traffic), and started without
admission control. The script flushes the radix cache between cells to keep
samples independent.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.error import URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _post(url: str, payload: Dict[str, Any], timeout: float = 120.0) -> bytes:
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get(url: str, timeout: float = 30.0) -> bytes:
    with urlopen(url, timeout=timeout) as resp:
        return resp.read()


def _flush_cache(server: str) -> None:
    try:
        _post(urljoin(server, "/flush_cache"), {})
    except Exception as e:
        print(f"[fit] WARN: /flush_cache failed: {e}", file=sys.stderr)


def _get_server_info(server: str) -> Dict[str, Any]:
    raw = _get(urljoin(server, "/get_server_info"))
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Random token id generation
# ---------------------------------------------------------------------------


def _make_random_ids(rng: random.Random, n: int, vocab: int = 128_000) -> List[int]:
    """Generate n pseudo-random token ids in [10, vocab).

    Avoids 0..9 to stay clear of common BOS/EOS/PAD ids. Pure RNG → no
    semantic content, no tokenizer round-trip needed.
    """
    return [rng.randrange(10, vocab) for _ in range(n)]


# ---------------------------------------------------------------------------
# TTFT measurement (streaming /generate)
# ---------------------------------------------------------------------------


def _measure_ttft_ms(
    server: str,
    input_ids: List[int],
    max_new_tokens: int = 1,
    timeout: float = 600.0,
) -> Tuple[float, Dict[str, Any]]:
    """POST a streaming /generate request, return (TTFT_ms, last_meta_info)."""
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": max_new_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "log_metrics": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        urljoin(server, "/generate"),
        data=body,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    ttft_ms: Optional[float] = None
    last_meta: Dict[str, Any] = {}

    with urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload_str = line[len("data:"):].strip()
            if payload_str == "[DONE]":
                break
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - t0) * 1000.0
            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            mi = chunk.get("meta_info") if isinstance(chunk, dict) else None
            if isinstance(mi, dict):
                last_meta = mi

    if ttft_ms is None:
        raise RuntimeError("no streaming chunks received from /generate")
    return ttft_ms, last_meta


# ---------------------------------------------------------------------------
# Prefill fit
# ---------------------------------------------------------------------------


def _default_prefill_grid() -> List[Tuple[int, int]]:
    """Default (n, p) grid tuned to the production workload distribution.

    Workload survey (parallel_tool sweep, lambda 0.05/0.1/0.2):
      prompt_tokens median 18-30k, p99 50-77k
      cache hit ratio median 93-96%
      d = n - p median ~1500, p90 ~3000-3500, p99 10-15k

    The grid covers d ∈ {0, 256, 1k, 2k, 4k, 8k, 16k} at three prefix
    sizes (0, 8k, 32k) so the polynomial sees enough variation in d while
    holding n + p configurations close to production.
    """
    cells: List[Tuple[int, int]] = []
    for p in (0, 8192, 32_768):
        for d in (0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384):
            n = p + d
            if n == 0:
                continue
            cells.append((n, p))
    return cells


def _measure_prefill_cell(
    server: str,
    rng: random.Random,
    n: int,
    p: int,
    repeats: int,
    flush_each: bool,
) -> List[Tuple[int, int, float, int]]:
    """Measure repeats samples for one (n, p) cell.

    Returns list of (n, p, ttft_ms, observed_cached_tokens).
    """
    samples: List[Tuple[int, int, float, int]] = []
    for r in range(repeats):
        if flush_each:
            _flush_cache(server)

        # Build prefix (cached after warmup) + fresh suffix.
        prefix = _make_random_ids(rng, p) if p > 0 else []
        suffix_len = n - p
        suffix = _make_random_ids(rng, suffix_len) if suffix_len > 0 else []

        if prefix:
            # Warm-up call: ensures prefix lands in radix cache.
            try:
                _measure_ttft_ms(server, prefix, max_new_tokens=1)
            except Exception as e:
                print(f"[fit] warmup failed for n={n} p={p}: {e}", file=sys.stderr)
                continue

        full = prefix + suffix
        try:
            ttft_ms, meta = _measure_ttft_ms(server, full, max_new_tokens=1)
        except Exception as e:
            print(f"[fit] measure failed for n={n} p={p}: {e}", file=sys.stderr)
            continue

        cached = int(meta.get("cached_tokens", 0))
        prompt_tokens = int(meta.get("prompt_tokens", len(full)))
        # Sanity check — observed n must match what we sent.
        if prompt_tokens != n:
            print(
                f"[fit] WARN: prompt_tokens={prompt_tokens} != n={n} "
                f"(possible page rounding); using observed",
                file=sys.stderr,
            )
        samples.append((prompt_tokens, cached, ttft_ms, cached))
        print(
            f"[fit] prefill n={n:>6d} p={p:>6d} d={n-p:>6d} "
            f"observed_cached={cached:>6d} ttft={ttft_ms:>8.2f}ms ({r+1}/{repeats})",
            file=sys.stderr,
        )
    return samples


def _fit_prefill_4var(
    d_list: Sequence[float], p_list: Sequence[float], t_list: Sequence[float]
) -> Tuple[float, float, float, float]:
    """Fit T = α·d² + β·d + γ + δ·p via numpy.linalg.lstsq with pure-Python fallback.

    Returns (alpha, beta, gamma, delta).
    """
    try:
        import numpy as np  # type: ignore

        d = np.asarray(d_list, dtype=np.float64)
        p = np.asarray(p_list, dtype=np.float64)
        X = np.column_stack([d * d, d, np.ones_like(d), p])
        y = np.asarray(t_list, dtype=np.float64)
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(coeffs[0]), float(coeffs[1]), float(coeffs[2]), float(coeffs[3])
    except ImportError:
        pass

    if len(t_list) < 4:
        raise RuntimeError("need at least 4 samples for the 4-coef fit")
    # Build X^T X (4x4) and X^T y for the design matrix [d², d, 1, p].
    cols = [
        [d * d for d in d_list],
        list(d_list),
        [1.0] * len(d_list),
        list(p_list),
    ]
    A = [[0.0] * 4 for _ in range(4)]
    b = [0.0] * 4
    for i in range(4):
        for j in range(4):
            A[i][j] = sum(cols[i][k] * cols[j][k] for k in range(len(t_list)))
        b[i] = sum(cols[i][k] * t_list[k] for k in range(len(t_list)))
    coeffs = _solve_nxn(A, b)
    return coeffs[0], coeffs[1], coeffs[2], coeffs[3]


def _solve_nxn(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(A)
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[pivot] = M[pivot], M[i]
        if abs(M[i][i]) < 1e-18:
            raise RuntimeError("singular matrix")
        for j in range(i + 1, n):
            factor = M[j][i] / M[i][i]
            for k in range(i, n + 1):
                M[j][k] -= factor * M[i][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][k] * x[k] for k in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def cmd_prefill(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    server = args.server.rstrip("/")
    info = _get_server_info(server)
    model = info.get("served_model_name") or info.get("model_path") or "unknown"

    if args.n_list and args.p_list:
        cells = [
            (n, p) for n in args.n_list for p in args.p_list if n >= p > -1 and n > 0
        ]
    else:
        cells = _default_prefill_grid()

    print(
        f"[fit] prefill: server={server} model={model} cells={len(cells)} "
        f"repeats={args.repeats} flush_each={args.flush_each}",
        file=sys.stderr,
    )

    _flush_cache(server)
    all_samples: List[Tuple[int, int, float, int]] = []
    for n, p in cells:
        all_samples.extend(
            _measure_prefill_cell(
                server=server,
                rng=rng,
                n=n,
                p=p,
                repeats=args.repeats,
                flush_each=args.flush_each,
            )
        )

    if len(all_samples) < 4:
        print(f"[fit] ERROR: only {len(all_samples)} samples; need >= 4", file=sys.stderr)
        return 2

    # Use the observed cached_tokens (after page rounding) as p; n is what we sent.
    ds = [float(n - cached) for (n, _, _, cached) in all_samples]
    ps = [float(cached) for (_, _, _, cached) in all_samples]
    ts = [float(t) for (_, _, t, _) in all_samples]
    alpha, beta, gamma, delta = _fit_prefill_4var(ds, ps, ts)
    rmse = math.sqrt(
        sum(
            (alpha * d * d + beta * d + gamma + delta * p - t) ** 2
            for d, p, t in zip(ds, ps, ts)
        )
        / len(ds)
    )

    payload = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "delta": delta,
        "fit_metadata": {
            "model": model,
            "server": server,
            "samples": len(all_samples),
            "rmse_ms": rmse,
            "form": "T_ms = alpha*d^2 + beta*d + gamma + delta*p (d = n - p)",
            "fit_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_samples": [
                {"n": n, "p": p, "d": n - p, "ttft_ms": t, "cached": c}
                for (n, p, t, c) in all_samples
            ],
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"[fit] prefill model written: {out}\n"
        f"  alpha={alpha:.6g} beta={beta:.6g} gamma={gamma:.6g} delta={delta:.6g} "
        f"RMSE={rmse:.2f}ms n_samples={len(all_samples)}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# TBT fit
# ---------------------------------------------------------------------------


def _default_tbt_grid() -> List[Tuple[int, int]]:
    """Default (batch_size, per_request_prompt_len) grid.

    Cell -> per_req_kv at decode start ≈ prompt_len; total_kv ≈ bs * prompt_len.
    Cells with total_kv > _TBT_TOTAL_KV_CAP are skipped to stay within VRAM
    and bracket the production envelope (production p99 total_kv ≈ 921k,
    p99 per_req_kv ≈ 61k).
    """
    cells: List[Tuple[int, int]] = []
    for bs in (1, 4, 8, 16, 24, 32):
        for prompt_len in (1024, 4096, 16384, 32768, 65536):
            if bs * prompt_len > _TBT_TOTAL_KV_CAP:
                continue
            cells.append((bs, prompt_len))
    return cells


_TBT_TOTAL_KV_CAP = 1_200_000


def _stream_tbt_samples(
    server: str,
    input_ids: List[int],
    measure_tokens: int,
    timeout: float = 600.0,
) -> List[float]:
    """Open a streaming generate, return inter-token intervals (ms)."""
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": measure_tokens,
            "ignore_eos": True,
        },
        "stream": True,
        "log_metrics": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        urljoin(server, "/generate"),
        data=body,
        headers={"Content-Type": "application/json"},
    )
    intervals: List[float] = []
    last: Optional[float] = None
    with urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload_str = line[len("data:"):].strip()
            if payload_str == "[DONE]":
                break
            try:
                chunk = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            if last is not None:
                intervals.append((now - last) * 1000.0)
            last = now
    return intervals


def _measure_tbt_cell(
    server: str,
    rng: random.Random,
    bs: int,
    prompt_len: int,
    measure_tokens: int,
    warmup_tokens: int,
) -> Optional[Tuple[int, int, float]]:
    """Run bs concurrent generations; report (bs, kv_total_estimate, median_tbt_ms)."""

    # Build distinct prompts so they don't collapse into one cache line.
    prompts = [_make_random_ids(rng, prompt_len) for _ in range(bs)]
    total_tokens = bs * prompt_len  # approximation of total KV at decode start

    # Each thread streams one request; we discard the first `warmup_tokens`
    # intervals to skip prefill and ramp-up effects, then keep the next
    # `measure_tokens` intervals.
    def _worker(prompt: List[int]) -> List[float]:
        intervals = _stream_tbt_samples(
            server, prompt, measure_tokens=measure_tokens + warmup_tokens
        )
        return intervals[warmup_tokens : warmup_tokens + measure_tokens]

    with ThreadPoolExecutor(max_workers=bs) as pool:
        results = list(pool.map(_worker, prompts))

    flat: List[float] = [v for lst in results for v in lst if v >= 0]
    if not flat:
        print(f"[fit] tbt cell bs={bs} prompt_len={prompt_len} no samples", file=sys.stderr)
        return None
    flat.sort()
    median = flat[len(flat) // 2]
    print(
        f"[fit] tbt bs={bs:>3d} prompt_len={prompt_len:>6d} "
        f"kv≈{total_tokens:>7d} median_tbt={median:>7.2f}ms n={len(flat)}",
        file=sys.stderr,
    )
    return (bs, total_tokens, median)


def _solve_3x3_general(A: List[List[float]], b: List[float]) -> List[float]:
    n = len(A)
    M = [row[:] + [bi] for row, bi in zip(A, b)]
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(M[r][i]))
        M[i], M[pivot] = M[pivot], M[i]
        if abs(M[i][i]) < 1e-18:
            raise RuntimeError("singular matrix")
        for j in range(i + 1, n):
            factor = M[j][i] / M[i][i]
            for k in range(i, n + 1):
                M[j][k] -= factor * M[i][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = M[i][n] - sum(M[i][k] * x[k] for k in range(i + 1, n))
        x[i] = s / M[i][i]
    return x


def _fit_linear_2var(
    bs_list: Sequence[int], per_req_kv_list: Sequence[int], t_list: Sequence[float]
) -> Tuple[float, float, float]:
    """Fit t = a + b·bs + c·per_req_kv via normal equations."""
    try:
        import numpy as np  # type: ignore

        X = np.column_stack(
            [np.ones(len(bs_list)), np.asarray(bs_list, dtype=np.float64),
             np.asarray(per_req_kv_list, dtype=np.float64)]
        )
        y = np.asarray(t_list, dtype=np.float64)
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    except ImportError:
        pass

    n = len(bs_list)
    s0 = float(n)
    sb = float(sum(bs_list))
    sk = float(sum(per_req_kv_list))
    sbb = float(sum(b * b for b in bs_list))
    skk = float(sum(k * k for k in per_req_kv_list))
    sbk = float(sum(b * k for b, k in zip(bs_list, per_req_kv_list)))
    st = float(sum(t_list))
    sbt = float(sum(b * t for b, t in zip(bs_list, t_list)))
    skt = float(sum(k * t for k, t in zip(per_req_kv_list, t_list)))
    A = [[s0, sb, sk], [sb, sbb, sbk], [sk, sbk, skk]]
    rhs = [st, sbt, skt]
    a, b, c = _solve_3x3_general(A, rhs)
    return a, b, c


def cmd_tbt(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    server = args.server.rstrip("/")
    info = _get_server_info(server)
    model = info.get("served_model_name") or info.get("model_path") or "unknown"

    if args.batch_sizes and args.prompt_lens:
        cells = [(bs, p) for bs in args.batch_sizes for p in args.prompt_lens]
    else:
        cells = _default_tbt_grid()

    print(
        f"[fit] tbt: server={server} model={model} cells={len(cells)} "
        f"warmup_tokens={args.warmup_tokens} measure_tokens={args.measure_tokens}",
        file=sys.stderr,
    )

    bs_list: List[int] = []
    kv_list: List[int] = []
    per_req_kv_list: List[int] = []
    t_list: List[float] = []
    raw: List[Dict[str, Any]] = []
    for bs, prompt_len in cells:
        _flush_cache(server)
        result = _measure_tbt_cell(
            server=server,
            rng=rng,
            bs=bs,
            prompt_len=prompt_len,
            measure_tokens=args.measure_tokens,
            warmup_tokens=args.warmup_tokens,
        )
        if result is None:
            continue
        b_, k_, t_ = result
        per_req = k_ // b_
        bs_list.append(b_)
        kv_list.append(k_)
        per_req_kv_list.append(per_req)
        t_list.append(t_)
        raw.append({"bs": b_, "total_kv": k_, "per_req_kv": per_req, "median_tbt_ms": t_})

    if len(t_list) < 3:
        print(f"[fit] ERROR: only {len(t_list)} cells; need >= 3", file=sys.stderr)
        return 2

    a, b, c = _fit_linear_2var(bs_list, per_req_kv_list, t_list)
    rmse = math.sqrt(
        sum((a + b * bv + c * pk - t) ** 2 for bv, pk, t in zip(bs_list, per_req_kv_list, t_list))
        / len(t_list)
    )

    payload = {
        "a": a,
        "b": b,
        "c": c,
        "fit_metadata": {
            "model": model,
            "server": server,
            "samples": len(t_list),
            "rmse_ms": rmse,
            "form": "TBT_ms = a + b*bs + c*per_req_kv",
            "fit_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "raw_samples": raw,
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(
        f"[fit] tbt model written: {out}\n"
        f"  a={a:.6g} b={b:.6g} c={c:.6g} RMSE={rmse:.2f}ms n_samples={len(t_list)}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="target", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--server", required=True, help="SGLang server base URL")
    common.add_argument("--output", required=True, help="output JSON path")
    common.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("prefill", parents=[common], help="fit prefill cost model")
    p.add_argument("--n-list", type=int, nargs="*", default=None,
                   help="prompt lengths to sweep (default: production-tuned grid)")
    p.add_argument("--p-list", type=int, nargs="*", default=None,
                   help="prefix lengths to sweep (default: production-tuned grid)")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--flush-each", action="store_true",
                   help="flush radix cache before every sample (slower, cleaner)")
    p.set_defaults(func=cmd_prefill)

    t = sub.add_parser("tbt", parents=[common], help="fit TBT cost model")
    t.add_argument("--batch-sizes", type=int, nargs="*", default=None)
    t.add_argument("--prompt-lens", type=int, nargs="*", default=None,
                   help="per-request prompt length; total_kv ≈ bs * prompt_len")
    t.add_argument("--warmup-tokens", type=int, default=10,
                   help="discard this many initial decode intervals per request")
    t.add_argument("--measure-tokens", type=int, default=30,
                   help="number of decode intervals to keep per request")
    t.set_defaults(func=cmd_tbt)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (URLError, RuntimeError) as e:
        print(f"[fit] ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
