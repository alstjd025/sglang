"""Cost models for admission control.

See managers/admission_control/CLAUDE.md for the full design.

PrefillCostModel:  T_prefill_ms ≈ α·d² + β·d + γ        where d = max(0, n - p)
TBTCostModel:      T_tbt_ms     ≈ a + b·batch_size + c·total_kv_tokens

Both load coefficients from a JSON file produced by
tools/admission_control/fit_cost_model.py. Malformed / missing files raise a
typed error that the controller treats as "stage disabled" rather than fatal.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CostModelLoadError(Exception):
    """Raised when a cost-model JSON cannot be loaded or validated."""


@dataclass(frozen=True)
class PrefillCostModel:
    """Estimate prefill latency from prompt length and prefix-cache match length.

    Form:    T_ms ≈ α·d² + β·d + γ + δ·p           where d = max(0, n - p)

    - α, β capture the actual prefill compute (Mooncake Eq. 1's quadratic-in-
      prompt-length term collapsed to fit coefficients per
      (model, hardware, kernel config)).
    - γ is the fixed per-request overhead.
    - δ captures the cost of loading the matched prefix from radix cache —
      empirically ~7 µs/token on B200×4 with Llama-3.3-70B. Older models
      that omit `delta` from JSON load with δ=0 (backward-compatible).
    """

    alpha: float
    beta: float
    gamma: float
    delta: float = 0.0
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def estimate_ms(self, prompt_len: int, prefix_len: int) -> float:
        d = max(0, prompt_len - prefix_len)
        return (
            self.alpha * d * d
            + self.beta * d
            + self.gamma
            + self.delta * prefix_len
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "PrefillCostModel":
        data = _read_json(path, "prefill cost model")
        try:
            return cls(
                alpha=float(data["alpha"]),
                beta=float(data["beta"]),
                gamma=float(data["gamma"]),
                delta=float(data.get("delta", 0.0)),  # back-compat: 0 if absent
                metadata=data.get("fit_metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CostModelLoadError(
                f"Invalid prefill cost model at {path}: missing/non-numeric "
                f"alpha/beta/gamma ({e})"
            ) from e

    def to_json(self, path: str | Path) -> None:
        payload = {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "fit_metadata": self.metadata or {},
        }
        Path(path).write_text(json.dumps(payload, indent=2))


@dataclass(frozen=True)
class TBTCostModel:
    """Estimate per-step decode latency (TBT) given current batch composition.

    Linear approximation: TBT grows with batch size (compute) and total KV
    tokens (memory bandwidth). Higher-order terms are absorbed by re-fitting
    when hardware/config changes.
    """

    a: float
    b: float
    c: float
    metadata: Dict[str, Any] = None  # type: ignore[assignment]

    def estimate_ms(self, batch_size: int, total_kv_tokens: int) -> float:
        return self.a + self.b * batch_size + self.c * total_kv_tokens

    @classmethod
    def from_json(cls, path: str | Path) -> "TBTCostModel":
        data = _read_json(path, "TBT cost model")
        try:
            return cls(
                a=float(data["a"]),
                b=float(data["b"]),
                c=float(data["c"]),
                metadata=data.get("fit_metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise CostModelLoadError(
                f"Invalid TBT cost model at {path}: missing/non-numeric "
                f"a/b/c ({e})"
            ) from e

    def to_json(self, path: str | Path) -> None:
        payload = {
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "fit_metadata": self.metadata or {},
        }
        Path(path).write_text(json.dumps(payload, indent=2))


def _read_json(path: str | Path, what: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise CostModelLoadError(f"{what} JSON not found: {p}")
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise CostModelLoadError(f"{what} JSON malformed at {p}: {e}") from e
    if not isinstance(data, dict):
        raise CostModelLoadError(f"{what} JSON at {p} must be an object")
    return data


def try_load_prefill_cost_model(
    path: Optional[str],
) -> Optional[PrefillCostModel]:
    """Lenient loader: returns None and logs WARN on failure (controller path)."""
    if not path:
        return None
    try:
        model = PrefillCostModel.from_json(path)
    except CostModelLoadError as e:
        logger.warning(
            "admission control: prefill cost model disabled — %s", e
        )
        return None
    logger.info(
        "admission control: prefill cost model loaded from %s "
        "(alpha=%.4g beta=%.4g gamma=%.4g)",
        path, model.alpha, model.beta, model.gamma,
    )
    return model


def try_load_tbt_cost_model(path: Optional[str]) -> Optional[TBTCostModel]:
    """Lenient loader: returns None and logs WARN on failure (controller path)."""
    if not path:
        return None
    try:
        model = TBTCostModel.from_json(path)
    except CostModelLoadError as e:
        logger.warning("admission control: TBT cost model disabled — %s", e)
        return None
    logger.info(
        "admission control: TBT cost model loaded from %s "
        "(a=%.4g b=%.4g c=%.4g)",
        path, model.a, model.b, model.c,
    )
    return model
